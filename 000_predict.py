#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
001_predict.py - correct per-PDB aggregation

Behavior changes:
- For each pdb in --pdb_dir we instantiate a Predictor, run predict(),
  export DeepFRI's per-protein CSV, parse it and append rows to a final
  TSV file with columns: protein_id<TAB>go_id<TAB>score
- This avoids the 'query_prot' problem and accidental overwrites.
- WARNING: instantiating Predictor per-file reloads model each time (slow).
  We can optimize later to reuse a single Predictor and clear its internal state
  if Predictor supports it.
"""

import json
import argparse
import warnings
from tqdm import tqdm
from Bio import BiopythonWarning
import os
import csv
import shutil
import tempfile

# ignore biopython warnings
warnings.filterwarnings("ignore", category=BiopythonWarning)

# try to import Predictor lazily (so script fails clearly if not present)
try:
    from deepfrier.Predictor import Predictor
except Exception as e:
    raise ImportError("Cannot import deepfrier.Predictor - make sure package is installed") from e


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def is_dir_like(path_str):
    # treat strings ending with os.sep or existing dir as directory
    if path_str.endswith(os.sep):
        return True
    return os.path.isdir(path_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seq', type=str, help="Protein sequence to be annotated.")
    parser.add_argument('-cm', '--cmap', type=str, help="Protein contact map to be annotated (in *npz file format).")
    parser.add_argument('-pdb', '--pdb_fn', type=str, help="Protein PDB file to be annotated.")
    parser.add_argument('--cmap_csv', type=str, help="Catalogue with chain to file path mapping.")
    parser.add_argument('--pdb_dir', type=str, help="Directory with PDB files of predicted Rosetta/DMPFold structures.")
    parser.add_argument('--fasta_fn', type=str, help="Fasta file with protein sequences.")
    parser.add_argument('--model_config', type=str, default='./trained_models/model_config.json', help="JSON file with model names.")
    parser.add_argument('-ont', '--ontology', type=str, default=['mf'], nargs='+', required=True, choices=['mf', 'bp', 'cc', 'ec'],
                        help="Gene Ontology/Enzyme Commission.")
    parser.add_argument('-o', '--output_fn_prefix', type=str, default='DeepFRI', help="Save predictions/saliency in file or directory.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--use_guided_grads', help="Use guided grads to compute gradCAM.", action="store_true")
    parser.add_argument('--saliency', help="Compute saliency maps for every protein and every MF-GO term/EC number.", action="store_true")
    args = parser.parse_args()

    # load model config
    with open(args.model_config) as json_file:
        params_all = json.load(json_file)

    # determine CNN/GCN params
    if args.seq is not None or args.fasta_fn is not None:
        params = params_all['cnn']
    else:
        params = params_all['gcn']
    gcn = params['gcn']
    layer_name = params['layer_name']
    models = params['models']

    # normalize output prefix: if it's directory-like, treat as directory
    out_prefix = args.output_fn_prefix
    if is_dir_like(out_prefix):
        out_dir = out_prefix if out_prefix.endswith(os.sep) else out_prefix + os.sep
        ensure_dir(out_dir)
    else:
        # parent dir
        out_dir = os.path.dirname(out_prefix)
        if out_dir == "":
            out_dir = "."
        ensure_dir(out_dir)

    # For each ontology, produce a final aggregated TSV
    for ont in args.ontology:
        print(f"\n=== Processing ontology: {ont} ===")
        aggregated_rows = []  # list of tuples (protein_id, go_id, score)

        # Case 1: single sequence / single pdb - keep original behaviour
        if args.seq is not None:
            predictor = Predictor(models[ont], gcn=gcn)
            predictor.predict(args.seq)
            # export as before (we won't aggregate seq mode)
            # Use smart path handling for outputs:
            if is_dir_like(out_prefix):
                out_base = os.path.join(out_dir, f"{ont.upper()}")
            else:
                out_base = out_prefix
            predictor.export_csv(out_base + "_predictions.csv", args.verbose)
            predictor.save_predictions(out_base + "_pred_scores.json")
            if args.saliency and ont in ['mf', 'ec']:
                predictor.compute_GradCAM(layer_name=layer_name, use_guided_grads=args.use_guided_grads)
                predictor.save_GradCAM(out_base + "_saliency_maps.json")
            continue

        if args.pdb_fn is not None:
            # single pdb file
            predictor = Predictor(models[ont], gcn=gcn)
            predictor.predict(args.pdb_fn)
            if is_dir_like(out_prefix):
                out_base = os.path.join(out_dir, f"{ont.upper()}")
            else:
                out_base = out_prefix
            predictor.export_csv(out_base + "_predictions.csv", args.verbose)
            predictor.save_predictions(out_base + "_pred_scores.json")
            if args.saliency and ont in ['mf', 'ec']:
                predictor.compute_GradCAM(layer_name=layer_name, use_guided_grads=args.use_guided_grads)
                predictor.save_GradCAM(out_base + "_saliency_maps.json")
            continue

        # Case 2: batch PDB directory
        if args.pdb_dir is not None:
            pdb_dir = args.pdb_dir
            pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
            if len(pdb_files) == 0:
                print("No .pdb files found in", pdb_dir)
            else:
                print(f"Processing {len(pdb_files)} PDB files from directory {pdb_dir} ...")
                # temporary directory to hold per-file CSV exports
                with tempfile.TemporaryDirectory() as tmpdir:
                    for fname in tqdm(pdb_files, desc=f"{ont.upper()} PDB files"):
                        pdb_path = os.path.join(pdb_dir, fname)
                        protein_id = os.path.splitext(fname)[0]  # use filename as protein id

                        # instantiate Predictor for each file (safe, albeit slower)
                        predictor = Predictor(models[ont], gcn=gcn)
                        predictor.predict(pdb_path)

                        # export single-protein csv to temp location
                        tmp_prefix = os.path.join(tmpdir, protein_id)
                        predictor.export_csv(tmp_prefix + "_" + ont.upper() + "_predictions.csv", False)
                        # load that csv and extract rows
                        csv_path = tmp_prefix + "_" + ont.upper() + "_predictions.csv"
                        if os.path.exists(csv_path):
                            with open(csv_path, newline='') as csvfile:
                                reader = csv.reader(csvfile)
                                # skip comment/header lines starting with '#'
                                for row in reader:
                                    if len(row) == 0:
                                        continue
                                    if row[0].startswith("#") or row[0].lower().startswith("protein"):
                                        # header or comment
                                        continue
                                    # expected format from DeepFRI: Protein,GO_term/EC_number,Score,GO_term/EC_number name
                                    # we take Protein (but override with our protein_id), GO_term, Score
                                    if len(row) >= 3:
                                        go_id = row[1].strip()
                                        score = row[2].strip()
                                        aggregated_rows.append((protein_id, go_id, score))
                        else:
                            # no csv produced -> warn
                            print(f"Warning: no csv produced for {pdb_path}")

                        # optionally compute and save per-file saliency and json if requested
                        if is_dir_like(out_prefix):
                            out_base = os.path.join(out_dir, protein_id + "_" + ont.upper())
                        else:
                            out_base = out_prefix + "_" + protein_id + "_" + ont.upper()

                        # save per-file JSON predictions
                        predictor.save_predictions(out_base + "_pred_scores.json")
                        if args.saliency and ont in ['mf', 'ec']:
                            predictor.compute_GradCAM(layer_name=layer_name, use_guided_grads=args.use_guided_grads)
                            predictor.save_GradCAM(out_base + "_saliency_maps.json")

        # Case 3: catalogue / fasta - keep existing behavior (not aggregated here)
        if args.fasta_fn is not None:
            predictor = Predictor(models[ont], gcn=gcn)
            predictor.predict_from_fasta(args.fasta_fn, progress_bar=True)
            # export
            if is_dir_like(out_prefix):
                out_base = os.path.join(out_dir, f"{ont.upper()}")
            else:
                out_base = out_prefix
            predictor.export_csv(out_base + "_predictions.csv", args.verbose)
            predictor.save_predictions(out_base + "_pred_scores.json")
            if args.saliency and ont in ['mf', 'ec']:
                predictor.compute_GradCAM(layer_name=layer_name, use_guided_grads=args.use_guided_grads)
                predictor.save_GradCAM(out_base + "_saliency_maps.json")

        if args.cmap_csv is not None:
            predictor = Predictor(models[ont], gcn=gcn)
            predictor.predict_from_catalogue(args.cmap_csv, progress_bar=True)
            if is_dir_like(out_prefix):
                out_base = os.path.join(out_dir, f"{ont.upper()}")
            else:
                out_base = out_prefix
            predictor.export_csv(out_base + "_predictions.csv", args.verbose)
            predictor.save_predictions(out_base + "_pred_scores.json")
            if args.saliency and ont in ['mf', 'ec']:
                predictor.compute_GradCAM(layer_name=layer_name, use_guided_grads=args.use_guided_grads)
                predictor.save_GradCAM(out_base + "_saliency_maps.json")

        # finally: write aggregated TSV for this ontology if any rows collected
        if len(aggregated_rows) > 0:
            if is_dir_like(out_prefix):
                final_tsv = os.path.join(out_dir, f"{ont.upper()}_protein_go_score.tsv")
            else:
                final_tsv = out_prefix + "_" + ont.upper() + "_protein_go_score.tsv"
            with open(final_tsv, "w", newline='') as outfh:
                outfh.write("protein_id\tgo_id\tscore\n")
                for pid, gid, sc in aggregated_rows:
                    outfh.write(f"{pid}\t{gid}\t{sc}\n")
            print(f"Aggregated results written to: {final_tsv}")
        else:
            print("No aggregated rows collected for ontology", ont)

    print("\n✔ All done.")
