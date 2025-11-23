#!/usr/bin/env python
"""
Galaxy 数据转换为 DeepFRI 格式（优化 OBO 加载）

特点：
1. 只加载一次 GO OBO 文件
2. 使用缓存避免重复查询
3. 保持 merge_all 逻辑和 DeepFRI 格式
"""

import os
import csv
import argparse
from collections import defaultdict
from pathlib import Path

# 全局 OBO 图缓存
GO_GRAPH_CACHE = None

def load_galaxy_annotations(data_file):
    """加载 Galaxy 格式的注释文件"""
    prot2goterms = {}
    all_goterms = set()
    
    with open(data_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # 跳过表头
        
        for row in reader:
            if len(row) < 2:
                continue
            prot_id = row[0].strip()
            go_labels = row[1].strip()
            
            if go_labels:
                goterms = [go.strip() for go in go_labels.split(';') if go.strip()]
                prot2goterms[prot_id] = goterms
                all_goterms.update(goterms)
            else:
                prot2goterms[prot_id] = []
    
    return prot2goterms, all_goterms

def load_protein_list(list_file):
    """加载蛋白质列表文件"""
    proteins = []
    with open(list_file, 'r') as f:
        for line in f:
            prot_id = line.strip()
            if prot_id:
                proteins.append(prot_id)
    return proteins

def load_go_obo(go_obo_file):
    """只加载一次 OBO 文件"""
    global GO_GRAPH_CACHE
    if GO_GRAPH_CACHE is None and go_obo_file and os.path.exists(go_obo_file):
        import obonet
        print(f"加载 GO OBO 文件: {go_obo_file}")
        GO_GRAPH_CACHE = obonet.read_obo(go_obo_file)
        print(f"  GO 图节点数量: {len(GO_GRAPH_CACHE.nodes)}")
    return GO_GRAPH_CACHE

def get_go_term_name(go_id, go_graph=None):
    """获取 GO term 名称，只使用预加载的 go_graph"""
    if go_graph and go_id in go_graph.nodes:
        return go_graph.nodes[go_id].get('name', go_id)
    return f"GO term {go_id}"

def create_deepfri_annotation_file(
    all_prot2annot,
    all_goterms_mf,
    all_goterms_bp=None,
    all_goterms_cc=None,
    output_file='galaxy_annot.tsv',
    go_graph=None
):
    """生成 DeepFRI 注释文件（符合原始格式）"""
    sorted_mf = sorted(all_goterms_mf) if all_goterms_mf else []
    sorted_bp = sorted(all_goterms_bp) if all_goterms_bp else []
    sorted_cc = sorted(all_goterms_cc) if all_goterms_cc else []

    mf_names = [get_go_term_name(go, go_graph) for go in sorted_mf]
    bp_names = [get_go_term_name(go, go_graph) for go in sorted_bp]
    cc_names = [get_go_term_name(go, go_graph) for go in sorted_cc]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')

        # 按照原始格式：每个ontology都有GO-terms和GO-names，即使为空
        writer.writerow(["### GO-terms (molecular_function)"])
        writer.writerow(sorted_mf)
        writer.writerow(["### GO-names (molecular_function)"])
        writer.writerow(mf_names)

        writer.writerow(["### GO-terms (biological_process)"])
        writer.writerow(sorted_bp)
        writer.writerow(["### GO-names (biological_process)"])
        writer.writerow(bp_names)

        writer.writerow(["### GO-terms (cellular_component)"])
        writer.writerow(sorted_cc)
        writer.writerow(["### GO-names (cellular_component)"])
        writer.writerow(cc_names)

        writer.writerow(["### PDB-chain", "GO-terms (molecular_function)", "GO-terms (biological_process)", "GO-terms (cellular_component)"])
        
        for prot_id in sorted(all_prot2annot.keys()):
            annot = all_prot2annot[prot_id]
            mf_terms = ','.join(sorted(annot.get('mf', [])))
            bp_terms = ','.join(sorted(annot.get('bp', [])))
            cc_terms = ','.join(sorted(annot.get('cc', [])))
            writer.writerow([prot_id, mf_terms, bp_terms, cc_terms])

    print(f"已生成注释文件: {output_file}")
    print(f"  MF GO terms: {len(sorted_mf)}")
    print(f"  BP GO terms: {len(sorted_bp)}")
    print(f"  CC GO terms: {len(sorted_cc)}")
    print(f"  蛋白质数量: {len(all_prot2annot)}")

def main():
    parser = argparse.ArgumentParser(
        description='将Galaxy格式的数据转换为DeepFRI训练格式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--galaxy_dir', type=str, default='org_galaxy', help='Galaxy数据目录')
    parser.add_argument('--output_dir', type=str, default='galaxy_deepfri', help='输出目录')
    parser.add_argument('--ontology', '-ont', type=str, default='mf', choices=['mf','bp','cc'], help='要转换的ontology')
    parser.add_argument('--go_obo', '-go', type=str, default="resources/go.obo", help='GO ontology文件路径（可选）')
    parser.add_argument('--merge_all', action='store_true', help='合并 MF/BP/CC 所有数据')
    args = parser.parse_args()

    galaxy_dir = Path(args.galaxy_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ontology = args.ontology.upper()

    print(f"输入目录: {galaxy_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Ontology: {ontology}")

    # 检查文件
    train_data_file = galaxy_dir / f"{ontology}_train_data.tsv"
    valid_data_file = galaxy_dir / f"{ontology}_validation_data.tsv"
    test_data_file = galaxy_dir / f"{ontology}_test_data.tsv"

    train_list_file = galaxy_dir / f"{ontology}_train_proteins.txt"
    valid_list_file = galaxy_dir / f"{ontology}_validation_proteins.txt"
    test_list_file = galaxy_dir / f"{ontology}_test_proteins.txt"

    if not train_data_file.exists():
        print(f"错误: 找不到文件 {train_data_file}")
        return

    # 加载 OBO 文件
    go_graph = load_go_obo(args.go_obo)

    # 加载注释
    all_prot2annot = defaultdict(lambda: {'mf': [], 'bp': [], 'cc': []})
    all_goterms_mf, all_goterms_bp, all_goterms_cc = set(), set(), set()

    for data_file in [train_data_file, valid_data_file, test_data_file]:
        if data_file.exists():
            prot2goterms, goterms = load_galaxy_annotations(data_file)
            ont_key = ontology.lower()
            for prot_id, go_list in prot2goterms.items():
                all_prot2annot[prot_id][ont_key] = go_list
                if ont_key == 'mf':
                    all_goterms_mf.update(go_list)
                elif ont_key == 'bp':
                    all_goterms_bp.update(go_list)
                elif ont_key == 'cc':
                    all_goterms_cc.update(go_list)

    if args.merge_all:
        for other_ont in ['MF','BP','CC']:
            if other_ont == ontology: continue
            for split in ['train','validation','test']:
                data_file = galaxy_dir / f"{other_ont}_{split}_data.tsv"
                if data_file.exists():
                    prot2goterms, goterms = load_galaxy_annotations(data_file)
                    ont_key = other_ont.lower()
                    for prot_id, go_list in prot2goterms.items():
                        all_prot2annot[prot_id][ont_key] = go_list
                        if ont_key == 'mf':
                            all_goterms_mf.update(go_list)
                        elif ont_key == 'bp':
                            all_goterms_bp.update(go_list)
                        elif ont_key == 'cc':
                            all_goterms_cc.update(go_list)

    # 生成注释文件
    annot_file = output_dir / f"galaxy_{ontology.lower()}_annot.tsv"
    if ontology=='MF':
        create_deepfri_annotation_file(all_prot2annot, all_goterms_mf, None, None, annot_file, go_graph)
    elif ontology=='BP':
        create_deepfri_annotation_file(all_prot2annot, set(), all_goterms_bp, None, annot_file, go_graph)
    else:
        create_deepfri_annotation_file(all_prot2annot, set(), set(), all_goterms_cc, annot_file, go_graph)

    # 生成训练/验证/测试集列表
    for split_name, list_file in [('train', train_list_file), ('valid', valid_list_file), ('test', test_list_file)]:
        if list_file.exists():
            proteins = load_protein_list(list_file)
            if split_name in ['train','valid']:
                output_file = output_dir / f"{ontology}_{split_name}.txt"
                with open(output_file,'w') as f:
                    for prot in proteins: f.write(f"{prot}\n")
            else:
                output_file = output_dir / f"{ontology}_{split_name}.csv"
                with open(output_file,'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['PDB-chain'])
                    for prot in proteins: writer.writerow([prot])

    print("转换完成！")

if __name__ == "__main__":
    main()
