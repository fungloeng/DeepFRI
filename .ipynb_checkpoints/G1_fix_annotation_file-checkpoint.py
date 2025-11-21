# fix_annotation_file.py
import csv
from collections import defaultdict

# 读取注释文件
annot_file = 'galaxy_deepfri/galaxy_mf_annot.tsv'
prot_list_file = 'galaxy_deepfri/MF_train.txt'

# 读取prot_list
with open(prot_list_file, 'r') as f:
    prot_list = [line.strip() for line in f if line.strip()]

# 读取注释文件
all_goterms = set()
prot2goterms = {}

with open(annot_file, 'r') as f:
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
        mf_terms = row[1].split(',') if row[1] else []
        prot2goterms[prot_id] = mf_terms
        all_goterms.update(mf_terms)

# 检查prot_list中的蛋白质
missing_prots = []
missing_goterms = set()

for prot in prot_list:
    if prot not in prot2goterms:
        missing_prots.append(prot)
    else:
        for goterm in prot2goterms[prot]:
            if goterm not in all_goterms:
                missing_goterms.add(goterm)

print(f"prot_list中的蛋白质数量: {len(prot_list)}")
print(f"注释文件中的蛋白质数量: {len(prot2goterms)}")
print(f"缺失的蛋白质: {len(missing_prots)}")
print(f"缺失的GO terms: {len(missing_goterms)}")

if missing_goterms:
    print(f"\n缺失的GO terms: {sorted(missing_goterms)}")