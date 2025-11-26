#!/usr/bin/env python3
"""
从TFRecord文件检查最大序列长度
用法: python 000_check_max_len_from_tfrecord.py <tfrecord_pattern> [--max_samples N]
"""
import sys
import tensorflow as tf
import glob

def find_max_len(tfrecord_pattern, max_samples=0):
    """从TFRecord文件找到最大序列长度"""
    files = glob.glob(tfrecord_pattern)
    if not files:
        print(f"错误: 找不到匹配的文件: {tfrecord_pattern}", file=sys.stderr)
        return None
    
    max_len = 0
    count = 0
    features = {
        "L": tf.io.FixedLenFeature([1], dtype=tf.int64)
    }
    
    print(f"检查 {len(files)} 个TFRecord文件...", file=sys.stderr)
    for tfrecord_file in files:
        try:
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            for serialized in dataset:
                if max_samples > 0 and count >= max_samples:
                    break
                try:
                    parsed = tf.io.parse_single_example(serialized=serialized, features=features)
                    seq_len = int(parsed['L'][0])
                    max_len = max(max_len, seq_len)
                    count += 1
                except Exception:
                    continue
            if max_samples > 0 and count >= max_samples:
                break
        except Exception as e:
            print(f"警告: 无法读取文件 {tfrecord_file}: {e}", file=sys.stderr)
            continue
    
    if count > 0:
        print(f"检查了 {count} 条记录，最大序列长度: {max_len}", file=sys.stderr)
        return max_len
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python 000_check_max_len_from_tfrecord.py <tfrecord_pattern> [--max_samples N]", file=sys.stderr)
        sys.exit(1)
    
    pattern = sys.argv[1]
    max_samples = 0
    
    if len(sys.argv) > 2 and sys.argv[2] == "--max_samples":
        try:
            max_samples = int(sys.argv[3])
        except (IndexError, ValueError):
            pass
    
    max_len = find_max_len(pattern, max_samples=max_samples)
    if max_len is not None:
        recommended = max_len + 200
        print(f"最大长度: {max_len}")
        print(f"推荐pad_len: {recommended}")
        sys.exit(0)
    else:
        print("错误: 无法确定最大长度", file=sys.stderr)
        sys.exit(1)

