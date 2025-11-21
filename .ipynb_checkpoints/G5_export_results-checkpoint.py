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


def calculate_fmax(Y_true, Y_pred, thresholds=None):
    """
    计算Fmax（在不同阈值下的最大F1分数）
    
    Fmax = max_t F1(t)，其中F1(t)是在阈值t下的F1分数
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)
    
    fmax = 0.0
    best_threshold = 0.0
    
    y_true_flat = Y_true.flatten()
    
    for threshold in tqdm(thresholds, desc="计算Fmax", disable=not HAS_TQDM):
        Y_pred_binary = (Y_pred >= threshold).astype(int)
        
        # 计算micro-average F1
        y_pred_flat = Y_pred_binary.flatten()
        
        if np.sum(y_pred_flat) > 0 or np.sum(y_true_flat) > 0:
            f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
            if f1 > fmax:
                fmax = f1
                best_threshold = threshold
    
    return fmax, best_threshold


def calculate_smin(Y_true, Y_pred, thresholds=None):
    """
    计算Smin（语义距离的最小值，Minimum Semantic Distance）
    
    Smin = min_t [RU(t) + MI(t)]，其中：
    - RU(t) = Remaining Uncertainty（剩余不确定性）：真实为正但预测为负的比例
    - MI(t) = Misinformation（错误信息）：真实为负但预测为正的比例
    
    这是CAFA评估中使用的指标
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)
    
    # 计算剩余不确定性和错误信息
    smin = float('inf')
    best_threshold = 0.0
    
    total_positives = np.sum(Y_true == 1)
    total_negatives = np.sum(Y_true == 0)
    
    if total_positives == 0:
        return 0.0, 0.0
    
    for threshold in tqdm(thresholds, desc="计算Smin", disable=not HAS_TQDM):
        Y_pred_binary = (Y_pred >= threshold).astype(int)
        
        # 剩余不确定性：真实为正但预测为负的数量 / 总正样本数
        ru = np.sum((Y_true == 1) & (Y_pred_binary == 0)) / (total_positives + 1e-10)
        
        # 错误信息：真实为负但预测为正的数量 / 总负样本数
        mi = np.sum((Y_true == 0) & (Y_pred_binary == 1)) / (total_negatives + 1e-10)
        
        # Smin = RU + MI
        s = ru + mi
        
        if s < smin:
            smin = s
            best_threshold = threshold
    
    return smin, best_threshold


def calculate_aupr(Y_true, Y_pred):
    """计算AUPR（Area Under Precision-Recall Curve）"""
    try:
        aupr = average_precision_score(Y_true.flatten(), Y_pred.flatten())
        return aupr
    except:
        return 0.0


def export_metrics(results, output_file, threshold=0.5):
    """导出评估指标到txt文件"""
    Y_pred = results['Y_pred']
    Y_true = results.get('Y_true', None)
    has_true_labels = results.get('has_true_labels', Y_true is not None and Y_true.size > 0)
    
    if not has_true_labels or Y_true is None:
        print("警告: 没有真实标签，无法计算评估指标")
        print("将使用默认值或仅输出预测结果")
        with open(output_file, 'w') as f:
            f.write(f"threshold\t{threshold}\n")
            f.write(f"Smin\tN/A\n")
            f.write(f"Fmax\tN/A\n")
            f.write(f"AUPR\tN/A\n")
        return
    
    # 计算指标
    print("计算评估指标...")
    
    # AUPR
    aupr = calculate_aupr(Y_true, Y_pred)
    print(f"  AUPR: {aupr:.4f}")
    
    # Fmax
    fmax, fmax_threshold = calculate_fmax(Y_true, Y_pred)
    print(f"  Fmax: {fmax:.4f} (threshold: {fmax_threshold:.4f})")
    
    # Smin
    smin, smin_threshold = calculate_smin(Y_true, Y_pred)
    print(f"  Smin: {smin:.4f} (threshold: {smin_threshold:.4f})")
    
    # 写入文件（一行格式，制表符分隔）
    with open(output_file, 'w') as f:
        f.write(f"threshold\tSmin\tFmax\tAUPR\n")
        f.write(f"{threshold:.6f}\t{smin:.6f}\t{fmax:.6f}\t{aupr:.6f}\n")
    
    print(f"\n评估指标已保存到: {output_file}")


def export_predictions_tsv(results, output_file, threshold=0.0):
    """导出预测结果到TSV文件（只输出超过阈值的预测）"""
    proteins = results['proteins']
    Y_pred = results['Y_pred']
    goterms = results['goterms']
    
    print(f"导出预测结果到TSV文件...")
    print(f"  蛋白质数量: {len(proteins)}")
    print(f"  GO terms数量: {len(goterms)}")
    print(f"  阈值: {threshold}")
    
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
                # 只输出超过阈值的预测
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
                       help='预测阈值（用于过滤predictions，默认0.0输出所有分数）')
    parser.add_argument('--metrics_threshold', type=float, default=0.5,
                       help='用于计算metrics的阈值（默认0.5）')
    
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
    metrics_file = args.output_prefix + '.txt'
    print(f"\n步骤1: 导出评估指标...")
    export_metrics(results, metrics_file, args.metrics_threshold)
    
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

