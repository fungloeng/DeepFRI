# filename: batch_predict_cmap.py (CORRECTED VERSION)
# purpose:  Efficiently run predictions for multiple contact maps by loading the model only once.
# usage:    Designed to be called from a shell script.

import os
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# 从您现有的代码库中导入核心的 Predictor 类
from deepfrier.Predictor import Predictor

# --- 设置一个简单的日志，避免过多的 TensorFlow 警告 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 将 TensorFlow 的日志级别设置为 ERROR，以隐藏不必要的 INFO 和 WARNING 消息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def find_npz_path(acc, search_paths):
    """在给定的目录列表中顺序查找蛋白质的 NPZ 文件"""
    for base_path in search_paths:
        p = Path(base_path) / f"{acc}.npz"
        if p.is_file():
            return str(p)
    return None

def main():
    parser = argparse.ArgumentParser(description="DeepFRI Batch Prediction for Contact Maps.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--acc_list_file", required=True,
                        help="Path to a text file containing a list of protein ACCs, one per line.")
    #
    # THIS IS THE CORRECTED LINE:
    #
    parser.add_argument("--npz_search_path", required=True, nargs='+',
                        help="One or more directories to search for .npz contact map files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save prediction output files.")
    parser.add_argument("--model_config", default='./trained_models/model_config.json',
                        help="JSON file with model names and configurations.")
    parser.add_argument("-ont", "--ontology", required=True, choices=['mf', 'bp', 'cc', 'ec'],
                        help="Ontology to predict.")

    args = parser.parse_args()

    # 1. 加载模型配置文件
    logging.info(f"Loading model configuration from {args.model_config}")
    with open(args.model_config) as json_file:
        params = json.load(json_file)

    # 因为我们只处理 contact maps, 所以固定使用 'gcn' 的设置
    try:
        gcn_params = params['gcn']
        gcn_flag = gcn_params['gcn']
        model_path = gcn_params['models'][args.ontology]
    except KeyError as e:
        logging.error(f"Could not find required key {e} in model_config.json for 'gcn' and ontology '{args.ontology}'.")
        return

    # 2. 读取所有待处理的蛋白质 ACC 列表
    logging.info(f"Reading protein list from {args.acc_list_file}")
    with open(args.acc_list_file, "r") as f:
        acc_list = [line.strip() for line in f if line.strip()]

    if not acc_list:
        logging.warning("ACC list file is empty. Nothing to do.")
        return

    logging.info(f"Found {len(acc_list)} proteins to process for ontology '{args.ontology.upper()}'.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 3. 【核心优化】在这里只加载一次模型！
    logging.info(f"Loading {args.ontology.upper()} model from {model_path}...")
    try:
        predictor = Predictor(model_path, gcn=gcn_flag)
        logging.info("Model loaded successfully into memory.")
    except Exception as e:
        logging.error(f"Failed to load the model. Error: {e}")
        return

    # 4. 循环处理所有蛋白质
    skipped_count = 0
    no_npz_count = 0
    failed_count = 0
    
    # 使用 tqdm 创建一个漂亮的进度条
    pbar = tqdm(acc_list, desc=f"Predicting {args.ontology.upper()}")
    for acc in pbar:
        ont_upper = args.ontology.upper()
        output_prefix = str(Path(args.output_dir) / acc)
        output_csv = f"{output_prefix}_{ont_upper}_predictions.csv"

        # 如果输出已存在，则跳过
        if Path(output_csv).exists() and Path(output_csv).stat().st_size > 0:
            skipped_count += 1
            continue

        # 查找输入的 NPZ 文件
        npz_file_path = find_npz_path(acc, args.npz_search_path)
        if not npz_file_path:
            no_npz_count += 1
            continue

        # 5. 【飞快的一步】执行预测并保存结果
        try:
            # 这一步不再有模型加载的开销
            predictor.predict(npz_file_path)
            
            # 保存结果
            predictor.export_csv(output_csv, verbose=False)
            predictor.save_predictions(f"{output_prefix}_{ont_upper}_pred_scores.json")

        except Exception as e:
            logging.error(f"\nPrediction failed for {acc}. Reason: {e}")
            failed_count += 1
        
        # 更新进度条的后缀信息
        pbar.set_postfix_str(f"skipped={skipped_count}, no_npz={no_npz_count}, failed={failed_count}")

    print("\n" + "="*50)
    logging.info("Batch prediction complete.")
    logging.info(f"Summary for {args.ontology.upper()}:")
    logging.info(f"  - Total proteins in list: {len(acc_list)}")
    logging.info(f"  - Processed successfully: {len(acc_list) - skipped_count - no_npz_count - failed_count}")
    logging.info(f"  - Skipped (output existed): {skipped_count}")
    logging.info(f"  - Skipped (no .npz found): {no_npz_count}")
    logging.info(f"  - Failed during prediction: {failed_count}")
    print("="*50)


if __name__ == "__main__":
    main()