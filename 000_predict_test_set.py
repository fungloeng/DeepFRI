#!/usr/bin/env python
"""
使用已训练的模型对测试集进行预测

用法:
    python predict_test_set.py \
        --model_name trained_models/GalaxyModel_CC \
        --test_list galaxy_deepfri/test.csv \
        --test_npz_dir npz_source/ \
        --cmap_thresh 10.0 \
        --output_file predictions.pckl
"""

import csv
import pickle
import numpy as np
import argparse
import os
import tensorflow as tf

from deepfrier.DeepFRI import DeepFRI
from deepfrier.utils import seq2onehot
from deepfrier.layers import FuncPredictor, SumPooling
from deepfrier.layers import GraphConv, MultiGraphConv, SAGEConv, ChebConv, NoGraphConv, GAT


def load_model_and_params(model_name, annot_fn=None, ontology='cc', 
                          gc_dims=None, fc_dims=None, gc_layer=None,
                          lm_model_name=None):
    """加载模型和参数
    
    如果model_params.json不存在，从注释文件中提取GO terms信息
    """
    import json
    
    params_file = model_name + '_model_params.json'
    
    if os.path.exists(params_file):
        # 从文件加载参数
        print(f"从文件加载模型参数: {params_file}")
        with open(params_file, 'r') as f:
            params = json.load(f)
    else:
        # 从注释文件提取GO terms信息
        print(f"警告: 找不到 {params_file}")
        if annot_fn and os.path.exists(annot_fn):
            print(f"从注释文件提取GO terms信息: {annot_fn}")
            from deepfrier.utils import load_GO_annot, load_EC_annot
            
            if ontology == 'ec':
                _, goterms = load_EC_annot(annot_fn)
                gonames = []
            else:
                _, goterms_dict, gonames_dict, _ = load_GO_annot(annot_fn)
                goterms = goterms_dict[ontology]
                gonames = gonames_dict[ontology]
            
            # 使用默认参数或用户提供的参数
            params = {
                'goterms': goterms,
                'gonames': gonames,
                'gc_dims': gc_dims if gc_dims else [128, 128, 256],
                'fc_dims': fc_dims if fc_dims else [],
                'gc_layer': gc_layer if gc_layer else 'GraphConv',
                'lr': 0.0002,
                'dropout': 0.3,
                'l2_reg': 1e-4,
                'lm_model_name': lm_model_name
            }
            print(f"  从注释文件提取到 {len(goterms)} 个GO terms")
        else:
            raise FileNotFoundError(
                f"找不到模型参数文件 {params_file}，且未提供注释文件。\n"
                f"请提供 --annot_fn 参数，或确保模型参数文件存在。"
            )
    
    # 获取模型架构参数
    output_dim = len(params['goterms'])
    gc_dims = params.get('gc_dims', [128, 128, 256])
    fc_dims = params.get('fc_dims', [])
    gc_layer = params.get('gc_layer', 'GraphConv')
    lr = params.get('lr', 0.0002)
    dropout = params.get('dropout', 0.3)
    l2_reg = params.get('l2_reg', 1e-4)
    lm_model_name = params.get('lm_model_name', None)
    
    print(f"\n模型架构参数:")
    print(f"  output_dim: {output_dim}")
    print(f"  gc_dims: {gc_dims}")
    print(f"  fc_dims: {fc_dims}")
    print(f"  gc_layer: {gc_layer}")
    
    # 创建模型（需要先构建才能加载权重）
    model = DeepFRI(
        output_dim=output_dim,
        n_channels=26,
        gc_dims=gc_dims,
        fc_dims=fc_dims,
        lr=lr,
        drop=dropout,
        l2_reg=l2_reg,
        gc_layer=gc_layer,
        lm_model_name=lm_model_name,
        model_name_prefix=model_name
    )
    
    # 加载模型权重
    model_path = model_name + '.hdf5'
    if not os.path.exists(model_path):
        # 尝试加载最佳模型
        model_path = model_name + '_best_train_model.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_name}.hdf5 或 {model_name}_best_train_model.h5")
    
    print(f"\n加载模型权重: {model_path}")
    model.model.load_weights(model_path)
    
    return model, params


def predict_test_set(model, params, test_list, test_npz_dir, cmap_thresh=10.0, 
                     annot_fn=None, ontology='cc'):
    """对测试集进行预测"""
    proteins = []
    Y_pred = []
    Y_true = []  # 如果有注释文件，可以计算真实标签
    
    goterms = params['goterms']
    gonames = params.get('gonames', [])
    output_dim = len(goterms)
    
    # 如果有注释文件，加载真实标签
    prot2annot = None
    if annot_fn and os.path.exists(annot_fn):
        print(f"加载注释文件: {annot_fn}")
        from deepfrier.utils import load_GO_annot, load_EC_annot
        if ontology == 'ec':
            prot2annot, _, _ = load_EC_annot(annot_fn)
        else:
            prot2annot, _, _, _ = load_GO_annot(annot_fn)
    
    # 确保路径以/结尾
    if test_npz_dir and not test_npz_dir.endswith('/') and not test_npz_dir.endswith('\\'):
        test_npz_dir = test_npz_dir + '/'
    
    print(f"开始预测，测试集列表: {test_list}")
    print(f"NPZ文件目录: {test_npz_dir}")
    print(f"距离阈值: {cmap_thresh}")
    
    missing_files = []
    with open(test_list, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # 跳过表头
        
        for row_num, row in enumerate(csv_reader, start=2):
            if len(row) == 0:
                continue
            prot = row[0].strip()
            if not prot:
                continue
            
            npz_file = test_npz_dir + prot + '.npz'
            
            if not os.path.exists(npz_file):
                missing_files.append(prot)
                if len(missing_files) <= 10:  # 只显示前10个
                    print(f"警告: 找不到文件 {npz_file}")
                continue
            
            try:
                cmap = np.load(npz_file)
                sequence = str(cmap['seqres'])
                Ca_dist = cmap['C_alpha']
                
                A = np.double(Ca_dist < cmap_thresh)
                S = seq2onehot(sequence)
                
                # 重塑为batch格式
                S = S.reshape(1, *S.shape)
                A = A.reshape(1, *A.shape)
                
                # 预测（与原始代码一致：使用model.predict方法）
                # model.predict返回的是[0][:, 0]，即第一个样本的所有类别的正类概率
                y_pred = model.predict([A, S]).reshape(1, output_dim)
                
                proteins.append(prot)
                Y_pred.append(y_pred)
                
                # 如果有真实标签，也保存
                if prot2annot and prot in prot2annot:
                    if ontology == 'ec':
                        y_true = prot2annot[prot].reshape(1, output_dim)
                    else:
                        y_true = prot2annot[prot][ontology].reshape(1, output_dim)
                    Y_true.append(y_true)
                else:
                    # 如果没有真实标签，创建零数组
                    Y_true.append(np.zeros((1, output_dim), dtype=np.int64))
                
                if (row_num - 1) % 100 == 0:
                    print(f"已处理: {row_num - 1} 个蛋白质...")
                    
            except Exception as e:
                print(f"处理 {prot} 时出错: {str(e)}")
                continue
    
    if missing_files:
        print(f"\n警告: 共 {len(missing_files)} 个文件缺失")
    
    print(f"\n预测完成!")
    print(f"成功预测: {len(proteins)} 个蛋白质")
    
    return {
        'proteins': np.asarray(proteins),
        'Y_pred': np.concatenate(Y_pred, axis=0) if Y_pred else np.array([]),
        'Y_true': np.concatenate(Y_true, axis=0) if Y_true else np.array([]),
        'ontology': ontology,
        'goterms': goterms,
        'gonames': gonames,
        'has_true_labels': prot2annot is not None
    }


def main():
    parser = argparse.ArgumentParser(
        description='使用已训练的模型对测试集进行预测',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_name', type=str, required=True,
                       help='模型名称（不含扩展名，例如: trained_models/GalaxyModel_CC）')
    parser.add_argument('--test_list', type=str, required=True,
                       help='测试集列表文件（CSV格式，包含PDB-chain列）')
    parser.add_argument('--test_npz_dir', type=str, required=True,
                       help='测试集npz文件目录')
    parser.add_argument('--cmap_thresh', type=float, default=10.0,
                       help='距离阈值（默认: 10.0）')
    parser.add_argument('--annot_fn', type=str, default=None,
                       help='注释文件（可选，用于计算真实标签和评估指标）')
    parser.add_argument('--ontology', type=str, default='cc', choices=['mf', 'bp', 'cc', 'pf','ec'],
                       help='Ontology类型')
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出文件路径（默认: {model_name}_predictions.pckl）')
    parser.add_argument('--gc_dims', type=int, nargs='+', default=None,
                       help='GraphConv层维度（如果model_params.json不存在时需要）')
    parser.add_argument('--fc_dims', type=int, nargs='+', default=None,
                       help='全连接层维度（如果model_params.json不存在时需要）')
    parser.add_argument('--gc_layer', type=str, default=None,
                       choices=['GraphConv', 'MultiGraphConv', 'SAGEConv', 'ChebConv', 'GAT', 'NoGraphConv'],
                       help='图卷积层类型（如果model_params.json不存在时需要）')
    parser.add_argument('--lm_model_name', type=str, default=None,
                       help='预训练LSTM语言模型路径（如果model_params.json不存在时需要）')
    
    args = parser.parse_args()
    
    # 检查必需参数
    params_file = args.model_name + '_model_params.json'
    if not os.path.exists(params_file) and not args.annot_fn:
        parser.error(
            f"找不到模型参数文件 {params_file}，且未提供 --annot_fn 参数。\n"
            f"请提供 --annot_fn 参数以从注释文件提取GO terms信息。"
        )
    
    # 设置输出文件
    if args.output_file is None:
        args.output_file = args.model_name + '_predictions.pckl'
    
    print("=" * 80)
    print("使用已训练模型进行预测")
    print("=" * 80)
    print(f"模型: {args.model_name}")
    print(f"测试集列表: {args.test_list}")
    print(f"NPZ目录: {args.test_npz_dir}")
    print(f"Ontology: {args.ontology}")
    if args.annot_fn:
        print(f"注释文件: {args.annot_fn}")
    print()
    
    # 加载模型
    print("步骤1: 加载模型...")
    try:
        model, params = load_model_and_params(
            args.model_name,
            annot_fn=args.annot_fn,
            ontology=args.ontology,
            gc_dims=args.gc_dims,
            fc_dims=args.fc_dims,
            gc_layer=args.gc_layer,
            lm_model_name=args.lm_model_name
        )
        print(f"  模型输出维度: {len(params['goterms'])}")
        print(f"  GO terms数量: {len(params['goterms'])}")
    except Exception as e:
        print(f"错误: 加载模型失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 进行预测
    print("\n步骤2: 对测试集进行预测...")
    results = predict_test_set(
        model, params, args.test_list, args.test_npz_dir,
        args.cmap_thresh, args.annot_fn, args.ontology
    )
    
    # 保存结果
    print(f"\n步骤3: 保存结果到 {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "=" * 80)
    print("预测完成！")
    print("=" * 80)
    print(f"\n结果文件: {args.output_file}")
    print(f"预测的蛋白质数量: {len(results['proteins'])}")
    print(f"预测结果形状: {results['Y_pred'].shape}")
    if results['has_true_labels']:
        print(f"包含真实标签: 是")
        print(f"\n可以使用 read_results.py 分析结果:")
        print(f"  python read_results.py --results_file {args.output_file}")
    else:
        print(f"包含真实标签: 否（仅预测结果）")
    
    print("\n提示: 可以使用 read_results.py 查看预测结果")


if __name__ == "__main__":
    main()

