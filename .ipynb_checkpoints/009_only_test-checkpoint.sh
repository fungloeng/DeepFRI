#!/bin/bash

DATASET="galaxy"  # galaxy cafa

# 数据目录
DATA_DIR="${DATASET}_deepfri"

# 结果目录
RESULTS_DIR="results/${DATASET}"

# 模型名称前缀（首字母大写）
# 兼容不同 bash 版本的方法
if [[ "${DATASET}" == "galaxy" ]]; then
    MODEL_PREFIX="GalaxyModel"
elif [[ "${DATASET}" == "cafa" ]]; then
    MODEL_PREFIX="CafaModel"
fi

# NPZ目录
NPZ_DIR="npz_source"

# TFRecord目录（如果未设置，使用默认值）
TFRECORD_DIR="${TFRECORD_DIR:-./tfrecords}"


echo "=========================================="
echo "数据集: ${DATASET}"
echo "数据目录: ${DATA_DIR}"
echo "结果目录: ${RESULTS_DIR}"
echo "模型前缀: ${MODEL_PREFIX}"
echo "=========================================="
echo ""

# ============================================================================
# MF (Molecular Function)
# ============================================================================
# echo "=== Processing MF ==="

# python 000_predict_test_set.py \
#     --model_name trained_models/${MODEL_PREFIX}_MF \
#     --test_list ${DATA_DIR}/MF_test.csv \
#     --test_npz_dir ${NPZ_DIR}/ \
#     --annot_fn ${DATA_DIR}/${DATASET}_mf_annot.tsv \
#     --ontology mf \
#     --gc_layer GraphConv \
#     --output_file ${RESULTS_DIR}/mf_test_predictions.pckl

# # 导出结果（使用自适应阈值计算指标，TSV输出不进行过滤）
# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/mf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/mf_test_preds_${DATASET}_run1 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/mf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/mf_test_preds_${DATASET}_run2 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/mf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/mf_test_preds_${DATASET}_run3 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/mf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/mf_test_preds_${DATASET}_run4 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/mf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/mf_test_preds_${DATASET}_run5 \
#     --threshold 0.0




# echo "MF 完成"
# echo ""   

# ============================================================================
# CC (Cellular Component)
# ============================================================================
echo "=== Processing CC ==="

python 000_predict_test_set.py \
    --model_name trained_models/${MODEL_PREFIX}_CC \
    --test_list ${DATA_DIR}/CC_test.csv \
    --test_npz_dir ${NPZ_DIR}/ \
    --annot_fn ${DATA_DIR}/${DATASET}_cc_annot.tsv \
    --ontology cc \
    --gc_layer GraphConv \
    --output_file ${RESULTS_DIR}/cc_test_predictions.pckl

# 导出结果（使用自适应阈值计算指标，TSV输出不进行过滤）
python 000_export_results.py \
    --results_file ${RESULTS_DIR}/cc_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/cc_test_preds_${DATASET}_run1 \
    --threshold 0.0
    
python 000_export_results.py \
    --results_file ${RESULTS_DIR}/cc_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/cc_test_preds_${DATASET}_run2 \
    --threshold 0.0

python 000_export_results.py \
    --results_file ${RESULTS_DIR}/cc_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/cc_test_preds_${DATASET}_run3 \
    --threshold 0.0
    
python 000_export_results.py \
    --results_file ${RESULTS_DIR}/cc_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/cc_test_preds_${DATASET}_run4 \
    --threshold 0.0
    
python 000_export_results.py \
    --results_file ${RESULTS_DIR}/cc_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/cc_test_preds_${DATASET}_run5 \
    --threshold 0.0

echo "CC 完成"
echo ""   
# ============================================================================
# BP (Biological Process)
# ============================================================================
echo "=== Processing BP ==="

python 000_predict_test_set.py \
    --model_name trained_models/${MODEL_PREFIX}_BP \
    --test_list ${DATA_DIR}/BP_test.csv \
    --test_npz_dir ${NPZ_DIR}/ \
    --annot_fn ${DATA_DIR}/${DATASET}_bp_annot.tsv \
    --ontology bp \
    --gc_layer GraphConv \
    --output_file ${RESULTS_DIR}/bp_test_predictions.pckl

# 导出结果（使用自适应阈值计算指标，TSV输出不进行过滤）
python 000_export_results.py \
    --results_file ${RESULTS_DIR}/bp_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/bp_test_preds_${DATASET}_run1 \
    --threshold 0.0

python 000_export_results.py \
    --results_file ${RESULTS_DIR}/bp_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/bp_test_preds_${DATASET}_run2 \
    --threshold 0.0

python 000_export_results.py \
    --results_file ${RESULTS_DIR}/bp_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/bp_test_preds_${DATASET}_run3 \
    --threshold 0.0

python 000_export_results.py \
    --results_file ${RESULTS_DIR}/bp_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/bp_test_preds_${DATASET}_run4 \
    --threshold 0.0


python 000_export_results.py \
    --results_file ${RESULTS_DIR}/bp_test_predictions.pckl \
    --output_prefix ${RESULTS_DIR}/bp_test_preds_${DATASET}_run5 \
    --threshold 0.0


echo "BP 完成"
echo ""   
# ============================================================================
# PF (Primary Function)
# ============================================================================
# echo "=== Processing PF ==="


# python preprocessing/PDB2TFRecord.py \
#     -annot ${DATA_DIR}/${DATASET}_pf_annot.tsv \
#     -prot_list ${DATA_DIR}/PF_train.txt \
#     -npz_dir ${NPZ_DIR} \
#     -num_shards 20 \
#     -num_threads 20 \
#     -tfr_prefix ${TFRECORD_DIR}/${DATASET}_pf_train

# python preprocessing/PDB2TFRecord.py \
#     -annot ${DATA_DIR}/${DATASET}_pf_annot.tsv \
#     -prot_list ${DATA_DIR}/PF_valid.txt \
#     -npz_dir ${NPZ_DIR} \
#     -num_shards 3 \
#     -num_threads 3 \
#     -tfr_prefix ${TFRECORD_DIR}/${DATASET}_pf_valid

# python 002d1_auto_optimize_training.py \
#     --prot_list ${DATA_DIR}/PF_train.txt \
#     --npz_dir ${NPZ_DIR} \
#     --annot_file ${DATA_DIR}/${DATASET}_pf_annot.tsv \
#     --test_list ${DATA_DIR}/PF_test.csv \
#     --ontology pf \
#     --model_name trained_models/${MODEL_PREFIX}_PF \
#     --max_memory_gb 8

# python train_DeepFRI.py \
#     --train_tfrecord_fn ${TFRECORD_DIR}/${DATASET}_pf_train \
#     --valid_tfrecord_fn ${TFRECORD_DIR}/${DATASET}_pf_valid \
#     --annot_fn ${DATA_DIR}/${DATASET}_pf_annot.tsv \
#     --test_list ${DATA_DIR}/PF_test.csv \
#     --test_npz_dir ${NPZ_DIR} \
#     --ontology pf \
#     --model_name trained_models/${MODEL_PREFIX}_PF \
#     --gc_layer GraphConv \
#     --pad_len 2699 \
#     --gc_dims 32 32 64 \
#     --fc_dims 64 \
#     --epochs 2 \
#     --batch_size 1


# python 000_predict_test_set.py \
#     --model_name trained_models/${MODEL_PREFIX}_PF \
#     --test_list ${DATA_DIR}/PF_test.csv \
#     --test_npz_dir ${NPZ_DIR}/ \
#     --annot_fn ${DATA_DIR}/${DATASET}_pf_annot.tsv \
#     --ontology pf \
#     --gc_layer GraphConv \
#     --output_file ${RESULTS_DIR}/pf_test_predictions.pckl

# # 导出结果（使用自适应阈值计算指标，TSV输出不进行过滤）
# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/pf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/pf_test_preds_${DATASET}_run1 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/pf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/pf_test_preds_${DATASET}_run2 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/pf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/pf_test_preds_${DATASET}_run3 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/pf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/pf_test_preds_${DATASET}_run4 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file ${RESULTS_DIR}/pf_test_predictions.pckl \
#     --output_prefix ${RESULTS_DIR}/pf_test_preds_${DATASET}_run5 \
#     --threshold 0.0

echo "PF 完成"
echo ""

echo "=== All done! ==="