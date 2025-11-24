#!/usr/bin/env python
"""
修复和检查注释文件

检查训练/验证/测试集中的蛋白质是否都在注释文件中，
以及GO terms是否完整。
"""
import csv
import argparse
from collections import defaultdict

def check_annotation_file(annot_file, prot_list_file, ontology='mf'):
    """检查注释文件"""
    # 读取prot_list
    with open(prot_list_file, 'r', encoding='utf-8') as f:
        prot_list = [line.strip() for line in f if line.strip()]
    
    # 读取注释文件
    all_goterms = set()
    prot2goterms = {}
    
    with open(annot_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        # 跳过前6行（GO terms和names）
        for _ in range(6):
            next(reader)
        # 跳过表头
        next(reader)
        
        # 读取所有蛋白质的注释
        for row in reader:
            if len(row) < 4:
                continue
            prot_id = row[0]
            # 根据ontology选择对应的列
            if ontology.lower() == 'mf':
                terms = row[1].split(',') if row[1] else []
            elif ontology.lower() == 'bp':
                terms = row[2].split(',') if row[2] else []
            elif ontology.lower() == 'cc':
                terms = row[3].split(',') if row[3] else []
            else:
                terms = []
            
            prot2goterms[prot_id] = terms
            all_goterms.update(terms)
    
    # 检查prot_list中的蛋白质
    missing_prots = []
    missing_goterms = set()
    empty_annot_prots = []
    
    for prot in prot_list:
        if prot not in prot2goterms:
            missing_prots.append(prot)
        elif len(prot2goterms[prot]) == 0:
            empty_annot_prots.append(prot)
        else:
            for goterm in prot2goterms[prot]:
                if goterm and goterm not in all_goterms:
                    missing_goterms.add(goterm)
    
    print(f"\n{'='*60}")
    print(f"检查结果 ({ontology.upper()})")
    print(f"{'='*60}")
    print(f"prot_list中的蛋白质数量: {len(prot_list)}")
    print(f"注释文件中的蛋白质数量: {len(prot2goterms)}")
    print(f"缺失的蛋白质: {len(missing_prots)}")
    print(f"无注释的蛋白质: {len(empty_annot_prots)}")
    print(f"缺失的GO terms: {len(missing_goterms)}")
    
    if missing_prots:
        print(f"\n前10个缺失的蛋白质:")
        for p in missing_prots[:10]:
            print(f"  {p}")
    
    if empty_annot_prots:
        print(f"\n前10个无注释的蛋白质:")
        for p in empty_annot_prots[:10]:
            print(f"  {p}")
    
    if missing_goterms:
        print(f"\n缺失的GO terms: {sorted(missing_goterms)}")
    
    return {
        'missing_prots': missing_prots,
        'empty_annot_prots': empty_annot_prots,
        'missing_goterms': missing_goterms
    }


def main():
    parser = argparse.ArgumentParser(
        description='检查注释文件的完整性',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--annot_file', type=str, default='cafa_deepfri/cafa_mf_annot.tsv',
                       help='注释文件路径')
    parser.add_argument('--prot_list_file', type=str, default='cafa_deepfri/MF_train.txt',
                       help='蛋白质列表文件路径')
    parser.add_argument('--ontology', type=str, default='mf', choices=['mf', 'bp', 'cc', 'pf'],
                       help='Ontology类型')
    
    args = parser.parse_args()
    
    check_annotation_file(args.annot_file, args.prot_list_file, args.ontology)


if __name__ == "__main__":
    main()