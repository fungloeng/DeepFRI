#!/usr/bin/env python
"""
Cafa 数据转换为 DeepFRI 格式（优化 OBO 加载）

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

# 支持的 ontology 及其标签
ONTOLOGY_CONFIG = {
    'MF': {'key': 'mf', 'label': 'molecular_function'},
    'BP': {'key': 'bp', 'label': 'biological_process'},
    'CC': {'key': 'cc', 'label': 'cellular_component'},
    'PF': {'key': 'pf', 'label': 'primary_function'}
}
ONTOLOGY_ORDER = ['MF', 'BP', 'CC', 'PF']


def load_cafa_annotations(data_file):
    """加载 Cafa 格式的注释文件"""
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

def create_deepfri_annotation_file(all_prot2annot,
                                   goterms_by_ont,
                                   output_file='cafa_annot.tsv',
                                   go_graph=None):
    """生成 DeepFRI 注释文件（符合原始格式，支持 MF/BP/CC/PF）"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')

        # 输出每个 ontology 的 GO-terms / GO-names
        for ont in ONTOLOGY_ORDER:
            label = ONTOLOGY_CONFIG[ont]['label']
            ont_key = ONTOLOGY_CONFIG[ont]['key']
            sorted_terms = sorted(goterms_by_ont[ont_key]) if goterms_by_ont[ont_key] else []
            term_names = [get_go_term_name(go, go_graph) for go in sorted_terms]

            writer.writerow([f"### GO-terms ({label})"])
            writer.writerow(sorted_terms)
            writer.writerow([f"### GO-names ({label})"])
            writer.writerow(term_names)

        # 写入注释列表
        header = ["### PDB-chain"]
        header.extend([f"GO-terms ({ONTOLOGY_CONFIG[ont]['label']})" for ont in ONTOLOGY_ORDER])
        writer.writerow(header)

        for prot_id in sorted(all_prot2annot.keys()):
            row = [prot_id]
            for ont in ONTOLOGY_ORDER:
                ont_key = ONTOLOGY_CONFIG[ont]['key']
                terms = ','.join(sorted(all_prot2annot[prot_id].get(ont_key, [])))
                row.append(terms)
            writer.writerow(row)

    print(f"已生成注释文件: {output_file}")
    for ont in ONTOLOGY_ORDER:
        ont_key = ONTOLOGY_CONFIG[ont]['key']
        print(f"  {ont} GO terms: {len(goterms_by_ont[ont_key])}")
    print(f"  蛋白质数量: {len(all_prot2annot)}")

def main():
    parser = argparse.ArgumentParser(
        description='将Cafa格式的数据转换为DeepFRI训练格式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--cafa_dir', '-gd', type=str, default='cafa', help='Cafa数据目录')
    parser.add_argument('--output_dir', '-od', type=str, default='cafa_deepfri', help='输出目录')
    parser.add_argument('--ontology', '-ont', type=str, default='mf', choices=['mf','bp','cc','pf'], help='要转换的ontology')
    parser.add_argument('--go_obo', '-go', type=str, default="resources/go-basic.obo", help='GO ontology文件路径（可选）')
    args = parser.parse_args()

    cafa_dir = Path(args.cafa_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ontology = args.ontology.upper()

    print(f"输入目录: {cafa_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Ontology: {ontology}")

    # 检查文件
    train_data_file = cafa_dir / f"{ontology}_train_data.tsv"
    valid_data_file = cafa_dir / f"{ontology}_validation_data.tsv"
    test_data_file = cafa_dir / f"{ontology}_test_data.tsv"

    train_list_file = cafa_dir / f"{ontology}_train_proteins.txt"
    valid_list_file = cafa_dir / f"{ontology}_validation_proteins.txt"
    test_list_file = cafa_dir / f"{ontology}_test_proteins.txt"

    if not train_data_file.exists():
        print(f"错误: 找不到文件 {train_data_file}")
        return

    # 加载 OBO 文件
    go_graph = load_go_obo(args.go_obo)

    # 加载注释
    def empty_annotation():
        return {config['key']: [] for config in ONTOLOGY_CONFIG.values()}

    all_prot2annot = defaultdict(empty_annotation)
    goterms_by_ont = {config['key']: set() for config in ONTOLOGY_CONFIG.values()}

    ont_key = ONTOLOGY_CONFIG[ontology]['key']
    for data_file in [train_data_file, valid_data_file, test_data_file]:
        if data_file.exists():
            prot2goterms, _ = load_cafa_annotations(data_file)
            for prot_id, go_list in prot2goterms.items():
                all_prot2annot[prot_id][ont_key] = go_list
                goterms_by_ont[ont_key].update(go_list)

    # 生成注释文件
    annot_file = output_dir / f"cafa_{ontology.lower()}_annot.tsv"
    create_deepfri_annotation_file(all_prot2annot, goterms_by_ont, annot_file, go_graph)

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
