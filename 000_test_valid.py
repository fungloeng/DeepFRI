#!/usr/bin/env python
"""
使用train/valid/test集作为测试集进行预测并生成报告（支持MF、CC、BP）

步骤：
1. 将蛋白质列表文件转换为CSV格式
2. 使用predict_test_set.py进行预测
3. 使用000_export_results.py导出结果
"""

import os
import csv
import subprocess
import sys
from pathlib import Path


def txt_to_csv(txt_file, csv_file):
    """将TXT格式的蛋白质列表转换为CSV格式"""
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"找不到文件: {txt_file}")
    
    print(f"转换 {txt_file} -> {csv_file}")
    
    with open(txt_file, 'r', encoding='utf-8') as f_in, open(csv_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['PDB-chain'])  # 表头
        
        for line in f_in:
            prot_id = line.strip()
            if prot_id:  # 跳过空行
                writer.writerow([prot_id])
    
    # 统计数量
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        count = sum(1 for row in reader)
    
    print(f"  转换完成，共 {count} 个蛋白质")
    return csv_file


def run_prediction(model_name, test_csv, test_npz_dir, annot_fn, ontology='mf', 
                   output_file=None, gc_layer='GraphConv'):
    """运行预测"""
    if output_file is None:
        output_file = f"results/org_galaxy/{ontology}_predictions.pckl"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 检查必需文件是否存在
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"找不到测试集列表文件: {test_csv}")
    if not os.path.exists(annot_fn):
        raise FileNotFoundError(f"找不到注释文件: {annot_fn}")
    if not os.path.exists(model_name + '.hdf5') and not os.path.exists(model_name + '_best_train_model.h5'):
        raise FileNotFoundError(f"找不到模型文件: {model_name}.hdf5 或 {model_name}_best_train_model.h5")
    
    cmd = [
        sys.executable, '003_predict_test_set.py',
        '--model_name', model_name,
        '--test_list', test_csv,
        '--test_npz_dir', test_npz_dir,
        '--annot_fn', annot_fn,
        '--ontology', ontology,
        '--output_file', output_file,
        '--gc_layer', gc_layer
    ]
    
    print("\n" + "=" * 80)
    print("步骤2: 运行预测")
    print("=" * 80)
    print(f"命令: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"错误: 预测失败，返回码: {result.returncode}")
        return None
    
    if not os.path.exists(output_file):
        print(f"错误: 预测结果文件未生成: {output_file}")
        return None
    
    return output_file


def export_results(results_file, output_prefix):
    """导出结果"""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"找不到预测结果文件: {results_file}")
    
    cmd = [
        sys.executable, '000_export_results.py',
        '--results_file', results_file,
        '--output_prefix', output_prefix,
        '--threshold', '0.0',
        '--metrics_threshold', '0.5'
    ]
    
    print("\n" + "=" * 80)
    print("步骤3: 导出结果")
    print("=" * 80)
    print(f"命令: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"错误: 导出失败，返回码: {result.returncode}")
        return False
    
    # 检查输出文件是否生成
    metrics_file = output_prefix + '.txt'
    predictions_file = output_prefix + '.tsv'
    if not os.path.exists(metrics_file) or not os.path.exists(predictions_file):
        print(f"警告: 部分输出文件未生成")
        return False
    
    return True


def get_default_paths(ontology, dataset_type='valid', run_number=1):
    """根据ontology和数据集类型获取默认路径
    
    Args:
        ontology: 'mf', 'bp', 或 'cc'
        dataset_type: 'train', 'valid', 或 'test'
        run_number: 运行编号（默认1）
    """
    ontology_upper = ontology.upper()
    ontology_lower = ontology.lower()
    
    # 数据集类型映射（注意：valid -> validation用于文件名）
    dataset_map = {
        'train': {
            'txt_suffix': 'train_proteins.txt',
            'name': 'train',
            'file_name': 'train'
        },
        'valid': {
            'txt_suffix': 'validation_proteins.txt',
            'name': 'valid',
            'file_name': 'validation'  # 文件名中使用validation
        },
        'test': {
            'txt_suffix': 'test_proteins.txt',
            'name': 'test',
            'file_name': 'test'
        }
    }
    
    if dataset_type not in dataset_map:
        raise ValueError(f"dataset_type必须是 'train', 'valid', 或 'test'，当前: {dataset_type}")
    
    dataset_info = dataset_map[dataset_type]
    
    # 生成符合要求的文件名格式: {ontology}_{dataset_type}_preds_deepfri_run{run_number}
    output_base = f'{ontology_lower}_{dataset_info["file_name"]}_preds_deepfri_run{run_number}'
    
    return {
        'txt_file': f'org_galaxy/{ontology_upper}_{dataset_info["txt_suffix"]}',
        'annot_fn': f'galaxy_deepfri/galaxy_{ontology_lower}_annot.tsv',
        'csv_file': f'{ontology_upper}_{dataset_info["name"]}.csv',
        'predictions': f'{ontology_lower}_{dataset_info["name"]}_predictions.pckl',
        'output_base': output_base,  # 用于生成最终的tsv和txt文件名
        'output_prefix': output_base  # 传递给000_export_results.py的前缀
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用train/valid/test集作为测试集进行预测并生成报告（支持MF、CC、BP）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_name', type=str, required=True,
                       help='模型名称（例如: trained_models/GalaxyModel_MF）')
    parser.add_argument('--ontology', type=str, required=True,
                       choices=['mf', 'bp', 'cc'],
                       help='Ontology类型（mf, bp, 或 cc）')
    parser.add_argument('--dataset_type', type=str, default='valid',
                       choices=['train', 'valid', 'test'],
                       help='数据集类型（train, valid, 或 test）')
    parser.add_argument('--txt_file', type=str, default=None,
                       help='蛋白质列表TXT文件路径（默认: org_galaxy/{ONTOLOGY}_{dataset_type}_proteins.txt）')
    parser.add_argument('--test_npz_dir', type=str, required=True,
                       help='NPZ文件目录')
    parser.add_argument('--annot_fn', type=str, default=None,
                       help='注释文件路径（默认: galaxy_deepfri/galaxy_{ontology}_annot.tsv）')
    parser.add_argument('--output_dir', type=str, default='results/galaxy',
                       help='输出目录')
    parser.add_argument('--gc_layer', type=str, default='GraphConv',
                       help='图卷积层类型')
    parser.add_argument('--skip_prediction', action='store_true',
                       help='跳过预测步骤（如果已有预测结果）')
    parser.add_argument('--skip_export', action='store_true',
                       help='跳过导出步骤')
    parser.add_argument('--run_number', type=int, default=1,
                       help='运行编号（用于文件名，默认1）')
    
    args = parser.parse_args()
    
    # 获取默认路径
    default_paths = get_default_paths(args.ontology, args.dataset_type, args.run_number)
    
    # 使用用户提供的路径或默认路径
    txt_file = args.txt_file if args.txt_file else default_paths['txt_file']
    annot_fn = args.annot_fn if args.annot_fn else default_paths['annot_fn']
    
    dataset_name = args.dataset_type.upper()
    print("=" * 80)
    print(f"使用{args.ontology.upper()} {dataset_name}集作为测试集进行预测和评估")
    print("=" * 80)
    print(f"Ontology: {args.ontology.upper()}")
    print(f"数据集类型: {dataset_name}")
    print(f"模型: {args.model_name}")
    print(f"蛋白质列表: {txt_file}")
    print(f"NPZ目录: {args.test_npz_dir}")
    print(f"注释文件: {annot_fn}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 步骤1: 转换TXT为CSV
    print("=" * 80)
    print(f"步骤1: 转换{dataset_name}集为CSV格式")
    print("=" * 80)
    
    csv_file = os.path.join(args.output_dir, default_paths['csv_file'])
    if not os.path.exists(csv_file):
        txt_to_csv(txt_file, csv_file)
    else:
        print(f"CSV文件已存在: {csv_file}")
    
    # 步骤2: 运行预测
    results_file = None
    if not args.skip_prediction:
        results_file = run_prediction(
            args.model_name,
            csv_file,
            args.test_npz_dir,
            annot_fn,
            ontology=args.ontology,
            output_file=os.path.join(args.output_dir, default_paths['predictions']),
            gc_layer=args.gc_layer
        )
    else:
        results_file = os.path.join(args.output_dir, default_paths['predictions'])
        if not os.path.exists(results_file):
            print(f"错误: 找不到预测结果文件: {results_file}")
            return
        print(f"使用已有预测结果: {results_file}")
    
    # 步骤3: 导出结果
    if not args.skip_export and results_file:
        # 使用新的命名格式: {ontology}_{dataset_type}_preds_deepfri_run{run_number}
        output_base = default_paths['output_base']
        output_prefix = os.path.join(args.output_dir, output_base)
        success = export_results(results_file, output_prefix)
        
        if success:
            print("\n" + "=" * 80)
            print("完成！生成的文件：")
            print("=" * 80)
            print(f"  预测结果: {results_file}")
            print(f"  评估指标: {output_prefix}.txt")
            print(f"  预测详情: {output_prefix}.tsv")
            print()
            print("可以使用以下命令查看结果：")
            print(f"  cat {output_prefix}.txt")
            print(f"  head -n 20 {output_prefix}.tsv")


if __name__ == "__main__":
    main()

