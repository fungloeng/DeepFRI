#!/usr/bin/env python
"""
导出DeepFRI预测结果为指定格式

生成两种文件：
1. metrics.txt - 包含评估指标（threshold, Smin, Fmax, AUPR）
2. predictions.tsv - 包含每个蛋白质的预测结果（protein_id, go_id, score）

用法:
    python export_results.py \
        --results_file predictions.pckl \
        --output_prefix results \
        --threshold 0.5
"""

import pickle
import numpy as np
import argparse
import csv
import os
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score
from collections import defaultdict
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 简单的进度条替代
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(desc)
        return iterable


def load_results(results_file):
    """加载结果文件"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def calculate_metrics_adaptive_threshold(Y_true, Y_pred, thresholds=None):
    """
    参考 evaluate_deepgoplus2.py 的方式，遍历阈值计算 Fmax、Smin 和 AUPR
    
    返回: (fmax, fmax_threshold, smin, smin_threshold, aupr, precisions, recalls)
    """
    if thresholds is None:
        # 参考 evaluate_deepgoplus2.py，使用 0.01 到 1.00 的阈值
        thresholds = np.arange(0.01, 1.01, 0.01)
    
    fmax = 0.0
    fmax_threshold = 0.0
    smin = float('inf')
    smin_threshold = 0.0
    precisions = []
    recalls = []
    
    y_true_flat = Y_true.flatten()
    total_positives = np.sum(Y_true == 1)
    total_negatives = np.sum(Y_true == 0)
    
    if total_positives == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, [], []
    
    # 遍历所有阈值
    for threshold in tqdm(thresholds, desc="计算评估指标", disable=not HAS_TQDM):
        Y_pred_binary = (Y_pred >= threshold).astype(int)
        y_pred_flat = Y_pred_binary.flatten()
        
        # 计算 Precision 和 Recall
        tp = np.sum((Y_true == 1) & (Y_pred_binary == 1))
        fp = np.sum((Y_true == 0) & (Y_pred_binary == 1))
        fn = np.sum((Y_true == 1) & (Y_pred_binary == 0))
        
        # Precision
        if tp + fp > 0:
            prec = tp / (tp + fp)
        else:
            prec = 0.0
        
        # Recall
        if tp + fn > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        
        # 计算 F1 分数
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        
        # 更新 Fmax
        if f1 > fmax:
            fmax = f1
            fmax_threshold = threshold
        
        # 计算 Smin（语义距离）
        # RU = Remaining Uncertainty（剩余不确定性）
        ru = fn / (total_positives + 1e-10)
        # MI = Misinformation（错误信息）
        mi = fp / (total_negatives + 1e-10)
        # S = sqrt(RU^2 + MI^2) - 参考 evaluate_deepgoplus2.py 的 evaluate_annotations
        s = np.sqrt(ru * ru + mi * mi)
        
        # 更新 Smin
        if s < smin:
            smin = s
            smin_threshold = threshold
    
    # 计算 AUPR（Area Under Precision-Recall Curve）
    # 需要按 recall 排序后计算曲线下面积
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # 按 recall 排序
    sorted_index = np.argsort(recalls)
    recalls_sorted = recalls[sorted_index]
    precisions_sorted = precisions[sorted_index]
    
    # 使用梯形法则计算 AUPR
    aupr = np.trapz(precisions_sorted, recalls_sorted)
    
    return fmax, fmax_threshold, smin, smin_threshold, aupr, precisions, recalls


def export_metrics(results, output_file, threshold=0.5):
    """
    导出评估指标到txt文件
    参考 evaluate_deepgoplus2.py 的方式，使用自适应阈值计算指标
    
    输出格式：
    threshold: <fmax_threshold>
    Smin: <smin_value>
    Fmax: <fmax_value>
    AUPR: <aupr_value>
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    Y_pred = results['Y_pred']
    Y_true = results.get('Y_true', None)
    has_true_labels = results.get('has_true_labels', Y_true is not None and Y_true.size > 0)
    
    if not has_true_labels or Y_true is None:
        print("警告: 没有真实标签，无法计算评估指标")
        print("将使用默认值或仅输出预测结果")
        with open(output_file, 'w') as f:
            f.write(f"threshold: N/A\n")
            f.write(f"Smin: N/A\n")
            f.write(f"Fmax: N/A\n")
            f.write(f"AUPR: N/A\n")
        return
    
    # 计算指标（使用自适应阈值）
    print("计算评估指标（遍历阈值寻找最优值）...")
    
    fmax, fmax_threshold, smin, smin_threshold, aupr, precisions, recalls = \
        calculate_metrics_adaptive_threshold(Y_true, Y_pred)
    
    print(f"  Fmax: {fmax:.4f} (threshold: {fmax_threshold:.4f})")
    print(f"  Smin: {smin:.4f} (threshold: {smin_threshold:.4f})")
    print(f"  AUPR: {aupr:.4f}")
    
    # 使用 Fmax 对应的阈值作为主要阈值（参考 evaluate_deepgoplus2.py）
    best_threshold = fmax_threshold
    
    # 写入文件（参考 evaluate_deepgoplus2.py 的格式）
    with open(output_file, 'w') as f:
        f.write(f"threshold: {best_threshold}\n")
        f.write(f"Smin: {smin:.3f}\n")
        f.write(f"Fmax: {fmax:.3f}\n")
        f.write(f"AUPR: {aupr:.3f}\n")
    
    print(f"\n评估指标已保存到: {output_file}")
    print(f"  使用阈值: {best_threshold:.4f} (Fmax对应的阈值)")


def export_predictions_tsv(results, output_file, threshold=0.0):
    """
    导出预测结果到TSV文件
    
    注意：此函数使用 threshold=0.0 来输出所有预测结果（不进行过滤）
    这与计算指标时使用的自适应阈值不同
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    proteins = results['proteins']
    Y_pred = results['Y_pred']
    goterms = results['goterms']
    
    print(f"导出预测结果到TSV文件...")
    print(f"  蛋白质数量: {len(proteins)}")
    print(f"  GO terms数量: {len(goterms)}")
    print(f"  阈值: {threshold} (输出所有预测结果，不进行过滤)")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # 写入表头
        writer.writerow(['protein_id', 'go_id', 'score'])
        
        # 写入每个预测结果（按分数排序）
        n_predictions = 0
        for i, prot in enumerate(tqdm(proteins, desc="处理蛋白质", disable=not HAS_TQDM)):
            y_pred = Y_pred[i]
            
            # 获取所有GO terms的预测分数，按分数排序
            indices = np.argsort(y_pred)[::-1]  # 降序排序
            
            for j in indices:
                goterm = goterms[j]
                score = float(y_pred[j])
                # 使用 threshold=0.0，输出所有预测结果（不进行过滤）
                if score >= threshold:
                    writer.writerow([prot, goterm, score])
                    n_predictions += 1
    
    print(f"  已写入 {n_predictions} 条预测结果")
    print(f"预测结果已保存到: {output_file}")


def export_predictions_tsv_all(results, output_file):
    """导出所有预测结果（不设阈值，按分数排序）"""
    proteins = results['proteins']
    Y_pred = results['Y_pred']
    goterms = results['goterms']
    
    print(f"导出所有预测结果到TSV文件...")
    print(f"  蛋白质数量: {len(proteins)}")
    print(f"  GO terms数量: {len(goterms)}")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # 写入表头
        writer.writerow(['protein_id', 'go_id', 'score'])
        
        # 写入每个预测结果（按分数排序）
        n_predictions = 0
        for i, prot in enumerate(proteins):
            y_pred = Y_pred[i]
            
            # 获取所有GO terms的预测分数，按分数排序
            indices = np.argsort(y_pred)[::-1]  # 降序排序
            
            for j in indices:
                goterm = goterms[j]
                score = float(y_pred[j])
                writer.writerow([prot, goterm, score])
                n_predictions += 1
    
    print(f"  已写入 {n_predictions} 条预测结果")
    print(f"预测结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='导出DeepFRI预测结果为指定格式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--results_file', type=str, required=True,
                       help='结果文件路径 (*.pckl)')
    parser.add_argument('--output_prefix', type=str, default='results',
                       help='输出文件前缀（将生成 {prefix}.txt 和 {prefix}.tsv）')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='TSV输出阈值（用于过滤predictions，默认0.0输出所有分数）')
    # parser.add_argument('--metrics_threshold', type=float, default=None,
                       # help='已废弃：指标计算现在使用自适应阈值，此参数将被忽略')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("导出DeepFRI预测结果")
    print("=" * 80)
    print(f"结果文件: {args.results_file}")
    print(f"输出前缀: {args.output_prefix}")
    print()
    
    # 加载结果
    print("加载结果文件...")
    results = load_results(args.results_file)
    
    # 导出评估指标（新格式: {prefix}.txt）
    # 注意：现在使用自适应阈值计算指标，不再使用 metrics_threshold 参数
    metrics_file = args.output_prefix + '.txt'
    print(f"\n步骤1: 导出评估指标（使用自适应阈值）...")
    export_metrics(results, metrics_file, threshold=None)  # threshold 参数已废弃
    
    # 导出预测结果（新格式: {prefix}.tsv）
    predictions_file = args.output_prefix + '.tsv'
    print(f"\n步骤2: 导出预测结果...")
    export_predictions_tsv(results, predictions_file, args.threshold)
    
    print("\n" + "=" * 80)
    print("导出完成！")
    print("=" * 80)
    print(f"\n生成的文件:")
    print(f"  评估指标: {metrics_file}")
    print(f"  预测结果: {predictions_file}")


if __name__ == "__main__":
    main()

