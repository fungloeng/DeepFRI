#!/usr/bin/env python3
"""
检查注释文件格式，诊断 GO terms 是否为标准格式
"""
import csv
import sys

def check_annotation_file(annot_file):
    """检查注释文件的 GO terms 格式"""
    print(f"检查注释文件: {annot_file}\n")
    
    with open(annot_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        
        goterms_sections = {}
        in_goterms_section = False
        current_ont = None
        skip_next = False  # 跳过 GO-names 行
        
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                # 空行，如果是等待 GO terms，说明是空列表
                if in_goterms_section and current_ont:
                    goterms_sections[current_ont] = []
                    in_goterms_section = False
                    current_ont = None
                continue
            
            # 跳过 GO-names 行
            if skip_next and row[0].startswith("### GO-names"):
                skip_next = False
                continue
            
            # 检查 GO-terms 部分
            if row[0].startswith("### GO-terms"):
                in_goterms_section = True
                skip_next = True  # 下一行是 GO-names，需要跳过
                # 提取 ontology 名称
                label = row[0].replace("### GO-terms", "").strip()
                if label.startswith("(") and label.endswith(")"):
                    label = label[1:-1]
                current_ont = label.lower()
                continue
            
            if in_goterms_section and current_ont:
                # 这一行应该是 GO terms 列表
                if row[0].startswith("###"):
                    # 如果遇到新的标题行，说明 GO terms 列表是空的
                    goterms_sections[current_ont] = []
                    in_goterms_section = False
                    current_ont = None
                    # 继续处理当前行
                    if row[0].startswith("### GO-terms"):
                        in_goterms_section = True
                        skip_next = True
                        label = row[0].replace("### GO-terms", "").strip()
                        if label.startswith("(") and label.endswith(")"):
                            label = label[1:-1]
                        current_ont = label.lower()
                    continue
                elif len(row) > 0:
                    goterms = row
                    goterms_sections[current_ont] = goterms
                    in_goterms_section = False
                    current_ont = None
                    continue
            
            if row[0].startswith("### PDB-chain"):
                break
        
        # 分析 GO terms 格式
        print("=" * 70)
        print("GO Terms 格式分析")
        print("=" * 70)
        
        for ont, goterms in goterms_sections.items():
            print(f"\n{ont.upper()}:")
            print(f"  总数量: {len(goterms)}")
            
            # 检查前10个 GO terms
            sample_terms = goterms[:10]
            print(f"  前10个示例:")
            for i, term in enumerate(sample_terms, 1):
                print(f"    {i}. {term}")
            
            # 检查格式
            if len(goterms) == 0:
                print(f"\n  格式统计:")
                print(f"    空列表（该 ontology 没有数据）")
            else:
                # 检查格式：GO:XXXXXXX 或 PFXXXXX
                standard_go = sum(1 for term in goterms if term.startswith("GO:"))
                standard_pf = sum(1 for term in goterms if term.startswith("PF"))
                non_standard = len(goterms) - standard_go - standard_pf
                
                print(f"\n  格式统计:")
                if ont == 'primary_function':
                    # PF ontology 使用 PF 格式
                    print(f"    标准格式 (PFXXXXX): {standard_pf} ({standard_pf/len(goterms)*100:.1f}%)")
                    if standard_go > 0:
                        print(f"    GO格式 (GO:XXXXXXX): {standard_go} ({standard_go/len(goterms)*100:.1f}%)")
                    if non_standard > 0:
                        print(f"    非标准格式: {non_standard} ({non_standard/len(goterms)*100:.1f}%)")
                else:
                    # 其他 ontology 使用 GO 格式
                    print(f"    标准格式 (GO:XXXXXXX): {standard_go} ({standard_go/len(goterms)*100:.1f}%)")
                    if standard_pf > 0:
                        print(f"    PF格式 (PFXXXXX): {standard_pf} ({standard_pf/len(goterms)*100:.1f}%)")
                    if non_standard > 0:
                        print(f"    非标准格式: {non_standard} ({non_standard/len(goterms)*100:.1f}%)")
                
                if non_standard > 0:
                    print(f"\n  ⚠️  警告: 发现非标准格式的 terms!")
                    print(f"     这些可能是基因符号、别名或其他标识符")
                    if ont != 'primary_function':
                        print(f"     需要转换为标准 GO ID 格式才能正确使用")
        
        print("\n" + "=" * 70)
        print("建议:")
        print("=" * 70)
        
        # 检查是否有非标准格式（排除PF ontology的PF格式）
        has_non_standard = False
        for ont, goterms in goterms_sections.items():
            if ont == 'primary_function':
                # PF ontology 应该使用 PF 格式
                non_standard = sum(1 for term in goterms if not term.startswith("PF"))
            else:
                # 其他 ontology 应该使用 GO 格式
                non_standard = sum(1 for term in goterms if not term.startswith("GO:"))
            if non_standard > 0:
                has_non_standard = True
                break
        
        if has_non_standard:
            print("1. 注释文件包含非标准格式的 terms")
            if 'primary_function' in goterms_sections and len(goterms_sections['primary_function']) > 0:
                print("2. PF ontology 应该使用 PFXXXXX 格式")
            else:
                print("2. GO ontologies 应该使用 GO:XXXXXXX 格式")
            print("3. 需要创建映射文件，将非标准标识符映射到标准格式")
            print("4. 或者重新生成注释文件，确保使用标准格式")
            print("\n可能的解决方案:")
            print("  - 检查 CAFA 原始数据，确认 labels 的格式")
            if 'primary_function' not in goterms_sections or len(goterms_sections['primary_function']) == 0:
                print("  - 使用 GO OBO 文件查找对应的标准 GO ID")
            print("  - 重新运行 001_ConvertOrgData2Deepfri.py，确保正确转换")
        else:
            print("✓ 注释文件格式正确，所有 terms 都是标准格式")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python 001d3_check_annotation_format.py <annotation_file>")
        print("示例: python 001d3_check_annotation_format.py cafa_deepfri/cafa_mf_annot.tsv")
        sys.exit(1)
    
    annot_file = sys.argv[1]
    check_annotation_file(annot_file)

