#!/usr/bin/env python
"""
检查TFRecord文件中的最大序列长度
"""
import argparse
import tensorflow as tf
import glob

def check_max_length(tfrecord_pattern, max_samples=10000):
    """检查TFRecord文件中的最大序列长度"""
    files = glob.glob(tfrecord_pattern)
    if not files:
        print(f"错误: 未找到匹配的文件: {tfrecord_pattern}")
        return None
    
    print(f"找到 {len(files)} 个TFRecord文件")
    if max_samples > 0:
        print(f"检查前 {max_samples} 条记录...")
    else:
        print(f"检查所有记录...")
    
    max_len = 0
    min_len = float('inf')
    total = 0
    lengths = []
    
    dataset = tf.data.TFRecordDataset(files)
    features = {
        "L": tf.io.FixedLenFeature([1], dtype=tf.int64)
    }
    
    for serialized in dataset.take(max_samples):
        parsed = tf.io.parse_single_example(serialized=serialized, features=features)
        seq_len = int(parsed['L'][0])
        max_len = max(max_len, seq_len)
        min_len = min(min_len, seq_len)
        lengths.append(seq_len)
        total += 1
    
    if total == 0:
        print("错误: 无法读取任何记录")
        return None
    
    lengths.sort()
    p95 = lengths[int(len(lengths) * 0.95)]
    p99 = lengths[int(len(lengths) * 0.99)]
    
    print(f"\n序列长度统计（基于 {total} 条记录）:")
    print(f"  最小长度: {min_len}")
    print(f"  最大长度: {max_len}")
    print(f"  95%分位数: {p95}")
    print(f"  99%分位数: {p99}")
    print(f"\n建议 pad_len: {max_len + 100} (最大长度 + 100)")
    
    return max_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="检查TFRecord文件中的最大序列长度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python 000_check_max_seq_len.py "./tfrecords/cafa_mf_train*"
  python 000_check_max_seq_len.py "./tfrecords/cafa_mf_train*" --max_samples 5000
  python 000_check_max_seq_len.py  # 使用默认路径
        """
    )
    parser.add_argument(
        "tfrecord_pattern", 
        nargs='?',
        default="./tfrecords/cafa_mf_train*",
        help="TFRecord文件路径模式（支持通配符，默认: ./tfrecords/cafa_mf_train*）"
    )
    parser.add_argument("--max_samples", type=int, default=10000, help="最大采样数量（默认: 10000）")
    
    args = parser.parse_args()
    check_max_length(args.tfrecord_pattern, args.max_samples)

