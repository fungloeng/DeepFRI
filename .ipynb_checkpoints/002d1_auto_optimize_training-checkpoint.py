#!/usr/bin/env python
"""
auto_optimize_training.py

根据数据集自动计算最优的训练参数（pad_len, batch_size, gc_dims等）
以避免内存不足的问题。
"""

import argparse
import os
import numpy as np
import subprocess
import sys


def load_protein_list(path: str):
    """加载蛋白质列表"""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def estimate_memory_per_sample(pad_len, n_goterms, gc_dims, fc_dims, batch_size=1):
    """
    估算每个样本的内存占用（单位：MB）
    
    参数:
        pad_len: 序列填充长度
        n_goterms: GO terms数量
        gc_dims: GraphConv层维度列表
        fc_dims: 全连接层维度列表
        batch_size: 批次大小
    """
    # Contact map: pad_len x pad_len x 4 bytes (float32)
    cmap_memory = pad_len * pad_len * 4 / (1024**2)  # MB
    
    # Sequence: pad_len x 26 x 4 bytes (float32)
    seq_memory = pad_len * 26 * 4 / (1024**2)  # MB
    
    # 模型中间层（粗略估算）
    # GraphConv layers
    gc_memory = sum([pad_len * dim * 4 / (1024**2) for dim in gc_dims])
    
    # FC layers
    fc_memory = sum([dim * 4 / (1024**2) for dim in fc_dims]) if fc_dims else 0
    
    # Output layer: n_goterms * 2 (pos/neg) * 4 bytes
    output_memory = n_goterms * 2 * 4 / (1024**2)
    
    # 单个样本总内存（包括中间激活）
    single_sample = cmap_memory + seq_memory + gc_memory + fc_memory + output_memory
    
    # Batch内存
    batch_memory = single_sample * batch_size
    
    # Shuffle buffer内存（保守估计buffer_size=500）
    buffer_memory = single_sample * 500
    
    return {
        'single_sample_mb': single_sample,
        'batch_memory_mb': batch_memory,
        'buffer_memory_mb': buffer_memory,
        'total_estimated_mb': batch_memory + buffer_memory
    }


def collect_lengths(proteins, npz_dir):
    """收集序列长度"""
    lengths = []
    missing = []
    failures = []
    
    for prot in proteins:
        npz_path = os.path.join(npz_dir, prot + '.npz')
        if not os.path.exists(npz_path):
            missing.append(prot)
            continue
        try:
            import numpy as np
            data = np.load(npz_path)
            seq = data['seqres']
            lengths.append(len(str(seq)))
        except Exception as exc:
            failures.append(f"{prot}: {exc}")
    return lengths, missing, failures


def find_optimal_pad_len(prot_list_file, npz_dir, max_memory_gb=8, target_percentile=95):
    """
    找到最优的pad_len，确保内存使用在限制内
    
    参数:
        prot_list_file: 蛋白质列表文件
        npz_dir: npz文件目录
        max_memory_gb: 最大可用内存（GB）
        target_percentile: 目标分位数
    """
    proteins = load_protein_list(prot_list_file)
    lengths, missing, failures = collect_lengths(proteins, npz_dir)
    
    if not lengths:
        print("错误: 无法读取任何npz文件")
        return None
    
    arr = np.array(lengths)
    max_len = int(arr.max())
    p95_len = int(np.percentile(arr, target_percentile))
    p90_len = int(np.percentile(arr, 90))
    
    print(f"\n序列长度统计:")
    print(f"  最大长度: {max_len}")
    print(f"  95%分位数: {p95_len}")
    print(f"  90%分位数: {p90_len}")
    
    # 尝试不同的pad_len，找到在内存限制内的最大值
    # 注意：pad_len必须 >= 最大序列长度，否则会报错
    # 所以我们从最大长度开始，如果内存不够就减小模型参数
    test_lengths = [max_len, p95_len, p90_len]
    optimal = max_len  # 默认使用最大长度（必须）
    
    print(f"\n注意: pad_len必须 >= 最大序列长度({max_len})，否则训练会失败")
    print(f"将尝试在内存限制内使用最大长度，如果不够则减小模型参数")
    
    for test_len in test_lengths:
        # 粗略估算内存（假设1714个GO terms，默认模型参数）
        mem_est = estimate_memory_per_sample(
            pad_len=test_len,
            n_goterms=1714,  # CC的GO terms数量
            gc_dims=[256, 256, 256],
            fc_dims=[512],
            batch_size=1
        )
        
        total_gb = mem_est['total_estimated_mb'] / 1024
        print(f"\n  pad_len={test_len}: 估算内存={total_gb:.2f}GB")
        
        # 如果使用最大长度且内存可接受，就使用最大长度
        if test_len == max_len and total_gb <= max_memory_gb * 0.8:
            optimal = max_len
            break
        elif test_len < max_len:
            # 如果使用分位数长度，需要警告用户可能会失败
            print(f"  警告: 使用pad_len={test_len}可能导致训练失败（有序列长度>{test_len}）")
            if total_gb <= max_memory_gb * 0.8:
                optimal = test_len
                break
    
    # 如果最优值小于最大长度，强制使用最大长度并警告
    if optimal < max_len:
        print(f"\n警告: 推荐的pad_len({optimal})小于最大序列长度({max_len})")
        print(f"将使用最大长度{max_len}，但可能需要减小模型参数以适应内存限制")
        optimal = max_len
    
    return optimal


def generate_training_command(prot_list_file, npz_dir, annot_file, test_list, 
                              ontology, model_name, max_memory_gb=8):
    """生成优化的训练命令"""
    
    # 1. 找到最优pad_len
    print("=" * 70)
    print("步骤1: 计算最优pad_len")
    print("=" * 70)
    optimal_pad_len = find_optimal_pad_len(prot_list_file, npz_dir, max_memory_gb)
    
    if optimal_pad_len is None:
        print("错误: 无法计算pad_len")
        return None
    
    print(f"\n推荐pad_len: {optimal_pad_len}")
    
    # 2. 根据内存限制调整模型参数
    print("\n" + "=" * 70)
    print("步骤2: 优化模型参数")
    print("=" * 70)
    
    # 估算内存并调整参数
    # 由于pad_len必须使用最大长度，可能需要大幅减小模型参数
    gc_dims = [256, 256, 256]
    fc_dims = [512]
    
    mem_est = estimate_memory_per_sample(
        pad_len=optimal_pad_len,
        n_goterms=1714,  # 需要从注释文件读取实际值
        gc_dims=gc_dims,
        fc_dims=fc_dims,
        batch_size=1
    )
    
    total_gb = mem_est['total_estimated_mb'] / 1024
    print(f"初始估算内存（gc_dims={gc_dims}, fc_dims={fc_dims}）: {total_gb:.2f}GB")
    
    # 如果内存太大，逐步减小模型参数
    if total_gb > max_memory_gb * 0.7:
        print(f"警告: 估算内存({total_gb:.2f}GB)超过限制，减小模型参数...")
        gc_dims = [128, 128, 256]
        fc_dims = [256]
        mem_est = estimate_memory_per_sample(
            pad_len=optimal_pad_len,
            n_goterms=1714,
            gc_dims=gc_dims,
            fc_dims=fc_dims,
            batch_size=1
        )
        total_gb = mem_est['total_estimated_mb'] / 1024
        print(f"第一次减小后（gc_dims={gc_dims}, fc_dims={fc_dims}）: {total_gb:.2f}GB")
        
        # 如果还是太大，进一步减小
        if total_gb > max_memory_gb * 0.7:
            print(f"继续减小模型参数...")
            gc_dims = [64, 64, 128]
            fc_dims = [128]
            mem_est = estimate_memory_per_sample(
                pad_len=optimal_pad_len,
                n_goterms=1714,
                gc_dims=gc_dims,
                fc_dims=fc_dims,
                batch_size=1
            )
            total_gb = mem_est['total_estimated_mb'] / 1024
            print(f"第二次减小后（gc_dims={gc_dims}, fc_dims={fc_dims}）: {total_gb:.2f}GB")
            
            # 如果还是太大，使用最小配置
            if total_gb > max_memory_gb * 0.7:
                print(f"使用最小模型配置...")
                gc_dims = [32, 32, 64]
                fc_dims = [64]
                mem_est = estimate_memory_per_sample(
                    pad_len=optimal_pad_len,
                    n_goterms=1714,
                    gc_dims=gc_dims,
                    fc_dims=fc_dims,
                    batch_size=1
                )
                total_gb = mem_est['total_estimated_mb'] / 1024
                print(f"最小配置（gc_dims={gc_dims}, fc_dims={fc_dims}）: {total_gb:.2f}GB")
    
    print(f"最终估算内存: {total_gb:.2f}GB")
    print(f"模型参数: gc_dims={gc_dims}, fc_dims={fc_dims}")
    
    # 3. 生成训练命令
    print("\n" + "=" * 70)
    print("步骤3: 生成训练命令")
    print("=" * 70)
    
    # 获取环境变量或使用默认值
    tfr_dir = os.environ.get('TFRECORD_DIR', './tfrecords')
    npz_dir_env = os.environ.get('NPZ_DIR', npz_dir)
    
    cmd = [
        'python', 'train_DeepFRI.py',
        '--train_tfrecord_fn', f'{tfr_dir}/galaxy_{ontology}_train',
        '--valid_tfrecord_fn', f'{tfr_dir}/galaxy_{ontology}_valid',
        '--annot_fn', annot_file,
        '--test_list', test_list,
        '--test_npz_dir', npz_dir_env,
        '--ontology', ontology,
        '--model_name', model_name,
        '--gc_layer', 'GraphConv',
        '--gc_dims'] + [str(d) for d in gc_dims] + [
        '--fc_dims'] + [str(d) for d in fc_dims] + [
        '--pad_len', str(optimal_pad_len),
        '--epochs', '5',
        '--batch_size', '1'
    ]
    
    return cmd, optimal_pad_len, gc_dims, fc_dims


def main():
    parser = argparse.ArgumentParser(
        description='自动优化训练参数以避免内存不足',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--prot_list', required=True,
                       help='训练集蛋白质列表文件（用于计算pad_len）')
    parser.add_argument('--npz_dir', required=True,
                       help='NPZ文件目录')
    parser.add_argument('--annot_file', required=True,
                       help='注释文件路径')
    parser.add_argument('--test_list', required=True,
                       help='测试集列表文件')
    parser.add_argument('--ontology', required=True,
                       choices=['mf', 'bp', 'cc', 'pf'],
                       help='Ontology类型')
    parser.add_argument('--model_name', required=True,
                       help='模型名称（不含扩展名）')
    parser.add_argument('--max_memory_gb', type=float, default=8.0,
                       help='最大可用内存（GB）')
    parser.add_argument('--execute', action='store_true',
                       help='直接执行训练命令（否则只打印）')
    
    args = parser.parse_args()
    
    # 生成优化后的训练命令
    result = generate_training_command(
        args.prot_list,
        args.npz_dir,
        args.annot_file,
        args.test_list,
        args.ontology,
        args.model_name,
        args.max_memory_gb
    )
    
    if result is None:
        sys.exit(1)
    
    cmd, pad_len, gc_dims, fc_dims = result
    
    print("\n" + "=" * 70)
    print("生成的训练命令:")
    print("=" * 70)
    print(' '.join(cmd))
    print("\n" + "=" * 70)
    print("参数摘要:")
    print(f"  pad_len: {pad_len}")
    print(f"  gc_dims: {gc_dims}")
    print(f"  fc_dims: {fc_dims}")
    print(f"  batch_size: 1")
    print("=" * 70)
    
    if args.execute:
        print("\n开始执行训练...")
        subprocess.run(cmd, check=True)
    else:
        print("\n提示: 使用 --execute 参数可以直接执行训练命令")


if __name__ == '__main__':
    main()

