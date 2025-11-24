#!/usr/bin/env python3
"""
检查TFRecord文件的内存占用情况，帮助诊断OOM问题
"""
import glob
import tensorflow as tf
import numpy as np
import argparse

def check_tfrecord_stats(tfrecord_pattern, cmap_type='ca', ont='mf'):
    """检查TFRecord文件的统计信息"""
    filenames = tf.io.gfile.glob(tfrecord_pattern)
    if not filenames:
        print(f"错误: 找不到匹配的文件: {tfrecord_pattern}")
        return
    
    print(f"找到 {len(filenames)} 个TFRecord文件")
    
    # 读取所有序列长度
    lengths = []
    max_length = 0
    total_samples = 0
    
    for filename in filenames:
        dataset = tf.data.TFRecordDataset(filename)
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            L = example.features.feature['L'].int64_list.value[0]
            lengths.append(L)
            max_length = max(max_length, L)
            total_samples += 1
    
    lengths = np.array(lengths)
    
    print(f"\n统计信息:")
    print(f"  总样本数: {total_samples}")
    print(f"  最大序列长度: {max_length}")
    print(f"  平均序列长度: {np.mean(lengths):.1f}")
    print(f"  中位数序列长度: {np.median(lengths):.1f}")
    print(f"  最小序列长度: {np.min(lengths)}")
    print(f"  95%分位数: {np.percentile(lengths, 95):.1f}")
    print(f"  99%分位数: {np.percentile(lengths, 99):.1f}")
    
    # 估算内存占用
    print(f"\n内存估算 (batch_size=1, pad_len={max_length}):")
    # Contact map: pad_len x pad_len x 4 bytes (float32)
    cmap_memory = max_length * max_length * 4 / (1024**2)  # MB
    # Sequence: pad_len x 26 x 4 bytes (float32)
    seq_memory = max_length * 26 * 4 / (1024**2)  # MB
    # 模型中间层（估算）
    print(f"  Contact map (单个样本): {cmap_memory:.2f} MB")
    print(f"  Sequence (单个样本): {seq_memory:.2f} MB")
    print(f"  单个样本总计: {cmap_memory + seq_memory:.2f} MB")
    
    # Shuffle buffer内存（默认2003）
    buffer_size = 2003
    buffer_memory = (cmap_memory + seq_memory) * buffer_size
    print(f"\nShuffle buffer内存 (buffer_size={buffer_size}):")
    print(f"  {buffer_memory:.2f} MB ({buffer_memory/1024:.2f} GB)")
    
    # 推荐的pad_len
    recommended_pad_len = int(np.percentile(lengths, 95))  # 使用95%分位数
    print(f"\n推荐设置:")
    print(f"  --pad_len {recommended_pad_len}  (基于95%分位数)")
    print(f"  --pad_len {max_length}  (使用最大长度，但可能浪费内存)")
    
    # 推荐的shuffle buffer
    if max_length > 2000:
        recommended_buffer = min(500, total_samples // 10)
    elif max_length > 1000:
        recommended_buffer = min(1000, total_samples // 10)
    else:
        recommended_buffer = 2003
    
    print(f"  建议减小shuffle buffer_size到: {recommended_buffer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查TFRecord文件的内存占用")
    parser.add_argument('--tfrecord_pattern', '-tf', type=str, required=True,
                       help='TFRecord文件模式，例如: ./tfrecords/cafa_mf_train*')
    parser.add_argument('--cmap_type', '-ct', type=str, default='ca', choices=['ca', 'cb'],
                       help='Contact map类型')
    parser.add_argument('--ont', type=str, default='mf', choices=['mf', 'bp', 'cc'],
                       help='Ontology类型')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TFRecord文件内存诊断")
    print("=" * 80)
    
    check_tfrecord_stats(args.tfrecord_pattern, args.cmap_type, args.ont)

