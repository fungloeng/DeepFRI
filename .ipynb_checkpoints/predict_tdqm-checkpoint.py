import json
import argparse
from deepfrier.Predictor import Predictor
from tqdm import tqdm  # 导入进度条库


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seq', type=str,  help="Protein sequence to be annotated.")
    parser.add_argument('-cm', '--cmap', type=str,  help="Protein contact map to be annotated (in *npz file format).")
    parser.add_argument('-pdb', '--pdb_fn', type=str,  help="Protein PDB file to be annotated.")
    parser.add_argument('--cmap_csv', type=str,  help="Catalogue with chain to file path mapping.")
    parser.add_argument('--pdb_dir', type=str,  help="Directory with PDB files of predicted Rosetta/DMPFold structures.")
    parser.add_argument('--fasta_fn', type=str,  help="Fasta file with protein sequences.")
    parser.add_argument('--model_config', type=str, default='./trained_models/model_config.json', help="JSON file with model names.")
    parser.add_argument('-ont', '--ontology', type=str, default=['mf'], nargs='+', required=True, choices=['mf', 'bp', 'cc', 'ec'],
                        help="Gene Ontology/Enzyme Commission.")
    parser.add_argument('-o', '--output_fn_prefix', type=str, default='DeepFRI', help="Save predictions/saliency in file.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--use_guided_grads', help="Use guided grads to compute gradCAM.", action="store_true")
    parser.add_argument('--saliency', help="Compute saliency maps for every protein and every MF-GO term/EC number.", action="store_true")
    args = parser.parse_args()

    with open(args.model_config) as json_file:
        params = json.load(json_file)

    if args.seq is not None or args.fasta_fn is not None:
        params = params['cnn']
    elif args.cmap is not None or args.pdb_fn is not None or args.cmap_csv is not None or args.pdb_dir is not None:
        params = params['gcn']
    gcn = params['gcn']
    layer_name = params['layer_name']
    models = params['models']

    for ont in args.ontology:
        predictor = Predictor(models[ont], gcn=gcn)
        
        # 单个序列/文件处理：显示简单进度提示
        if args.seq is not None:
            print(f"Processing sequence (ontology: {ont})...")
            predictor.predict(args.seq)
        
        if args.cmap is not None:
            print(f"Processing contact map {args.cmap} (ontology: {ont})...")
            predictor.predict(args.cmap)
        
        if args.pdb_fn is not None:
            print(f"Processing PDB file {args.pdb_fn} (ontology: {ont})...")
            predictor.predict(args.pdb_fn)
        
        # 批量处理 FASTA 文件：添加进度条
        if args.fasta_fn is not None:
            # 先获取 FASTA 中的序列数量（用于进度条总长度）
            with open(args.fasta_fn, 'r') as f:
                total = sum(1 for line in f if line.startswith('>'))
            print(f"Processing {total} sequences from FASTA (ontology: {ont})...")
            # 使用 tqdm 包装迭代过程（假设 predict_from_fasta 内部可迭代）
            for _ in tqdm(predictor.predict_from_fasta(args.fasta_fn), total=total, desc=f"FASTA Progress ({ont})"):
                pass
        
        # 批量处理 CSV 列表：添加进度条
        if args.cmap_csv is not None:
            # 先获取 CSV 中的条目数量（假设每行一个条目）
            with open(args.cmap_csv, 'r') as f:
                total = sum(1 for line in f if line.strip())  # 排除空行
            print(f"Processing {total} entries from CSV (ontology: {ont})...")
            for _ in tqdm(predictor.predict_from_catalogue(args.cmap_csv), total=total, desc=f"CSV Progress ({ont})"):
                pass
        
        # 批量处理 PDB 目录：添加进度条
        if args.pdb_dir is not None:
            import os
            # 获取目录中 PDB 文件数量
            pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith(('.pdb', '.ent'))]
            total = len(pdb_files)
            print(f"Processing {total} PDB files (ontology: {ont})...")
            for _ in tqdm(predictor.predict_from_PDB_dir(args.pdb_dir), total=total, desc=f"PDB Dir Progress ({ont})"):
                pass

        # 保存预测结果
        predictor.export_csv(args.output_fn_prefix + "_" + ont.upper() + "_predictions.csv", args.verbose)
        predictor.save_predictions(args.output_fn_prefix + "_" + ont.upper() + "_pred_scores.json")

        # 保存显著性图
        if args.saliency and ont in ['mf', 'ec']:
            print(f"Computing saliency maps for {ont}...")
            predictor.compute_GradCAM(layer_name=layer_name, use_guided_grads=args.use_guided_grads)
            predictor.save_GradCAM(args.output_fn_prefix + "_" + ont.upper() + "_saliency_maps.json")