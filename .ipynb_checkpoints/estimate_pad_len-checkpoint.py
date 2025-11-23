#!/usr/bin/env python
"""
estimate_pad_len.py

根据 protein 列表和 npz 目录，统计序列长度并给出推荐 pad_len。
"""
import argparse
import os
import numpy as np
from typing import List, Tuple


def load_protein_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def collect_lengths(proteins: List[str], npz_dir: str) -> Tuple[List[int], List[str], List[str]]:
    lengths = []
    missing = []
    failures = []

    for prot in proteins:
        npz_path = os.path.join(npz_dir, prot + '.npz')
        if not os.path.exists(npz_path):
            missing.append(prot)
            continue
        try:
            data = np.load(npz_path)
            seq = data['seqres']
            lengths.append(len(str(seq)))
        except Exception as exc:
            failures.append(f"{prot}: {exc}")
    return lengths, missing, failures


def main():
    parser = argparse.ArgumentParser(
        description="统计 npz 序列长度并给出推荐 pad_len",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--prot_list', required=True, help='蛋白质列表文件（*.txt）')
    parser.add_argument('--npz_dir', required=True, help='npz 目录')
    parser.add_argument('--percentiles', type=int, nargs='+', default=[90, 95, 99],
                        help='需要计算的分位数（整数，0-100）')
    args = parser.parse_args()

    proteins = load_protein_list(args.prot_list)
    lengths, missing, failures = collect_lengths(proteins, args.npz_dir)

    if not lengths:
        print("没有成功读取任何 npz 文件，请检查路径或文件格式。")
        return

    arr = np.array(lengths)
    print("=" * 70)
    print(f"总蛋白质数：{len(proteins)}")
    print(f"成功读取：{len(lengths)}")
    print(f"缺失文件：{len(missing)}")
    print(f"读取失败：{len(failures)}")
    print("-" * 70)
    print(f"最小长度：{arr.min()}")
    print(f"最大长度：{arr.max()}")
    print(f"平均长度：{arr.mean():.2f}")
    print(f"中位数：{np.median(arr):.2f}")

    for p in sorted(set(args.percentiles)):
        value = np.percentile(arr, p)
        print(f"{p:>3d}% 分位数：{value:.2f}")

    recommended = int(np.percentile(arr, max(min(95, max(args.percentiles)), 0)))
    print("-" * 70)
    print(f"推荐 pad_len（基于 {max(min(95, max(args.percentiles)), 0)}% 分位数）：{recommended}")
    print("如需更稳妥，可使用最大长度，但会占用更多显存/内存。")

    if missing:
        print("\n前 10 个缺失文件示例：")
        for prot in missing[:10]:
            print(f"  {prot}.npz")
    if failures:
        print("\n前 5 条读取失败示例：")
        for failure in failures[:5]:
            print(f"  {failure}")


if __name__ == '__main__':
    main()

