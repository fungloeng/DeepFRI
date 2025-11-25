#!/bin/bash
set -e  # 出现错误就停止

# NPZ目录
NPZ_DIR="cafa_npz_source"
# NPZ_DIR="npz_source"

# 输出TFRecord目录
TFRECORD_DIR="./tfrecords"

# 结果输出目录
RESULTS_DIR="results/cafa"
mkdir -p $RESULTS_DIR


# # ----------------------------
# # MF
# # ----------------------------
# echo "=== Processing MF ==="

# echo "[Step 1a] Generate TFRecords for MF - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_mf_annot.tsv \
#     -prot_list cafa_deepfri/MF_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 20 \
#     -num_threads 2 \
#     -tfr_prefix $TFRECORD_DIR/cafa_mf_train

# # 检查训练TFRecord文件是否生成成功
# TRAIN_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_mf_train_*.tfrecords 2>/dev/null | wc -l)
# if [ "$TRAIN_TFR_COUNT" -eq 0 ]; then
#     echo "错误: 训练TFRecord文件生成失败！"
#     exit 1
# fi
# echo "成功生成 $TRAIN_TFR_COUNT 个训练TFRecord文件"

# echo "[Step 1b] Generate TFRecords for MF - valid"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_mf_annot.tsv \
#     -prot_list cafa_deepfri/MF_valid.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 3 \
#     -num_threads 2 \
#     -tfr_prefix $TFRECORD_DIR/cafa_mf_valid

# # 检查验证TFRecord文件是否生成成功
# VALID_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_mf_valid_*.tfrecords 2>/dev/null | wc -l)
# if [ "$VALID_TFR_COUNT" -eq 0 ]; then
#     echo "错误: 验证TFRecord文件生成失败！"
#     exit 1
# fi
# echo "成功生成 $VALID_TFR_COUNT 个验证TFRecord文件"

# # 可选：自动优化训练参数（如果npz文件可用）
# echo "[Step 1c] 尝试自动优化训练参数（可选）"
# python 002d1_auto_optimize_training.py --prot_list cafa_deepfri/MF_train.txt --npz_dir $NPZ_DIR --annot_file cafa_deepfri/cafa_mf_annot.tsv --test_list cafa_deepfri/MF_test.csv --ontology mf --model_name trained_models/CafaModel_MF --max_memory_gb 8 || echo "警告: 自动优化失败，将使用默认参数"

# # 检查最大序列长度（用于验证pad_len）
# echo "[Step 1d] 检查最大序列长度（验证pad_len）"
# PAD_LEN=5000  # 默认值，如果检查发现需要更大，会自动调整
# if command -v python &> /dev/null; then
#     MAX_LEN_OUTPUT=$(python 000_check_max_seq_len.py "$TFRECORD_DIR/cafa_mf_train*" --max_samples 5000 2>&1)
#     MAX_LEN=$(echo "$MAX_LEN_OUTPUT" | grep "最大长度:" | grep -oE '[0-9]+' | head -1)
#     if [ ! -z "$MAX_LEN" ] && [ "$MAX_LEN" -gt 0 ]; then
#         PAD_LEN=$((MAX_LEN + 200))
#         echo "检测到最大序列长度: $MAX_LEN，自动设置 pad_len=$PAD_LEN"
#     else
#         echo "使用默认 pad_len=$PAD_LEN（如果训练时仍然报错，请手动增加此值）"
#     fi
# else
#     echo "无法运行检查脚本，使用默认 pad_len=$PAD_LEN"
# fi

# python train_DeepFRI.py \
#     --train_tfrecord_fn $TFRECORD_DIR/cafa_mf_train \
#     --valid_tfrecord_fn $TFRECORD_DIR/cafa_mf_valid \
#     --annot_fn cafa_deepfri/cafa_mf_annot.tsv \
#     --test_list cafa_deepfri/MF_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --ontology mf \
#     --model_name trained_models/CafaModel_MF \
#     --gc_layer GraphConv \
#     --pad_len $PAD_LEN \
#     --gc_dims 32 32 64 \
#     --fc_dims 64 \
#     --epochs 1 \
#     --batch_size 1


# # testset as test
# python 000_predict_test_set.py \
#     --model_name trained_models/CafaModel_MF \
#     --test_list cafa_deepfri/MF_test.csv \
#     --test_npz_dir cafa_npz_source/ \
#     --annot_fn cafa_deepfri/cafa_mf_annot.tsv \
#     --ontology mf \
#     --gc_layer GraphConv \
#     --output_file results/cafa/mf_test_predictions.pckl

# python 000_export_results.py \
#     --results_file results/cafa/mf_test_predictions.pckl \
#     --output_prefix results/cafa/mf_test_preds_deepfri_run1 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/mf_test_predictions.pckl \
#     --output_prefix results/cafa/mf_test_preds_deepfri_run2 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file results/cafa/mf_test_predictions.pckl \
#     --output_prefix results/cafa/mf_test_preds_deepfri_run3 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file results/cafa/mf_test_predictions.pckl \
#     --output_prefix results/cafa/mf_test_preds_deepfri_run4 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/mf_test_predictions.pckl \
#     --output_prefix results/cafa/mf_test_preds_deepfri_run5 \
#     --threshold 0.0


# ----------------------------
# CC
# ----------------------------
echo "=== Processing CC ==="

echo "[Step 1a] Generate TFRecords for CC - train"
python preprocessing/PDB2TFRecord.py \
    -annot cafa_deepfri/cafa_cc_annot.tsv \
    -prot_list cafa_deepfri/CC_train.txt \
    -npz_dir $NPZ_DIR \
    -num_shards 20 \
    -num_threads 2 \
    -tfr_prefix $TFRECORD_DIR/cafa_cc_train

# 检查训练TFRecord文件是否生成成功
TRAIN_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_cc_train_*.tfrecords 2>/dev/null | wc -l)
if [ "$TRAIN_TFR_COUNT" -eq 0 ]; then
    echo "错误: 训练TFRecord文件生成失败！"
    exit 1
fi
echo "成功生成 $TRAIN_TFR_COUNT 个训练TFRecord文件"

echo "[Step 1b] Generate TFRecords for CC - valid"
python preprocessing/PDB2TFRecord.py \
    -annot cafa_deepfri/cafa_cc_annot.tsv \
    -prot_list cafa_deepfri/CC_valid.txt \
    -npz_dir $NPZ_DIR \
    -num_shards 3 \
    -num_threads 2 \
    -tfr_prefix $TFRECORD_DIR/cafa_cc_valid

# 检查验证TFRecord文件是否生成成功
VALID_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_cc_valid_*.tfrecords 2>/dev/null | wc -l)
if [ "$VALID_TFR_COUNT" -eq 0 ]; then
    echo "错误: 验证TFRecord文件生成失败！"
    exit 1
fi
echo "成功生成 $VALID_TFR_COUNT 个验证TFRecord文件"

# 可选：自动优化训练参数（如果npz文件可用）
echo "[Step 1c] 尝试自动优化训练参数（可选）"
python 002d1_auto_optimize_training.py --prot_list cafa_deepfri/CC_train.txt --npz_dir $NPZ_DIR --annot_file cafa_deepfri/cafa_cc_annot.tsv --test_list cafa_deepfri/CC_test.csv --ontology cc --model_name trained_models/CafaModel_CC --max_memory_gb 8 || echo "警告: 自动优化失败，将使用默认参数"

# 检查最大序列长度（用于验证pad_len）
echo "[Step 1d] 检查最大序列长度（验证pad_len）"
PAD_LEN=5000  # 默认值，如果检查发现需要更大，会自动调整
if command -v python &> /dev/null; then
    # 检查所有记录以确保找到真正的最大长度
    echo "正在检查所有TFRecord文件以找到最大序列长度（这可能需要一些时间）..."
    MAX_LEN_OUTPUT=$(python 000_check_max_seq_len.py "$TFRECORD_DIR/cafa_cc_train*" --max_samples 0 2>&1)
    MAX_LEN=$(echo "$MAX_LEN_OUTPUT" | grep "最大长度:" | grep -oE '[0-9]+' | head -1)
    if [ ! -z "$MAX_LEN" ] && [ "$MAX_LEN" -gt 0 ]; then
        PAD_LEN=$((MAX_LEN + 200))
        echo "检测到最大序列长度: $MAX_LEN，自动设置 pad_len=$PAD_LEN"
    else
        echo "警告: 无法从输出中提取最大长度，使用默认 pad_len=$PAD_LEN"
        echo "检查脚本输出:"
        echo "$MAX_LEN_OUTPUT"
    fi
else
    echo "无法运行检查脚本，使用默认 pad_len=$PAD_LEN"
fi

python train_DeepFRI.py \
    --train_tfrecord_fn $TFRECORD_DIR/cafa_cc_train \
    --valid_tfrecord_fn $TFRECORD_DIR/cafa_cc_valid \
    --annot_fn cafa_deepfri/cafa_cc_annot.tsv \
    --test_list cafa_deepfri/CC_test.csv \
    --test_npz_dir $NPZ_DIR \
    --ontology cc \
    --model_name trained_models/CafaModel_CC \
    --gc_layer GraphConv \
    --pad_len $PAD_LEN \
    --gc_dims 32 32 64 \
    --fc_dims 64 \
    --epochs 1 \
    --batch_size 1


# testset as test
python 000_predict_test_set.py \
    --model_name trained_models/CafaModel_CC \
    --test_list cafa_deepfri/CC_test.csv \
    --test_npz_dir $NPZ_DIR \
    --annot_fn cafa_deepfri/cafa_cc_annot.tsv \
    --ontology cc \
    --gc_layer GraphConv \
    --output_file $RESULTS_DIR/cc_test_predictions.pckl

python 000_export_results.py \
    --results_file $RESULTS_DIR/cc_test_predictions.pckl \
    --output_prefix $RESULTS_DIR/cc_test_preds_deepfri_run1 \
    --threshold 0.0

# python 000_export_results.py \
#     --results_file $RESULTS_DIR/cc_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/cc_test_preds_deepfri_run2 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file $RESULTS_DIR/cc_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/cc_test_preds_deepfri_run3 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file $RESULTS_DIR/cc_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/cc_test_preds_deepfri_run4 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file $RESULTS_DIR/cc_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/cc_test_preds_deepfri_run5 \
#     --threshold 0.0

# # 清理CC的TFRecord文件，释放空间
# echo "[Step 1e] 清理CC的TFRecord文件..."
# rm -f $TFRECORD_DIR/cafa_cc_train_*.tfrecords
# rm -f $TFRECORD_DIR/cafa_cc_valid_*.tfrecords
# echo "CC的TFRecord文件已删除"

# ----------------------------
# BP
# ----------------------------
echo "=== Processing BP ==="

# echo "[Step 1a] Generate TFRecords for BP - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_bp_annot.tsv \
#     -prot_list cafa_deepfri/BP_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 20 \
#     -num_threads 2 \
#     -tfr_prefix $TFRECORD_DIR/cafa_bp_train

# # 检查训练TFRecord文件是否生成成功
# TRAIN_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_bp_train_*.tfrecords 2>/dev/null | wc -l)
# if [ "$TRAIN_TFR_COUNT" -eq 0 ]; then
#     echo "错误: 训练TFRecord文件生成失败！"
#     exit 1
# fi
# echo "成功生成 $TRAIN_TFR_COUNT 个训练TFRecord文件"

# echo "[Step 1b] Generate TFRecords for BP - valid"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_bp_annot.tsv \
#     -prot_list cafa_deepfri/BP_valid.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 3 \
#     -num_threads 2 \
#     -tfr_prefix $TFRECORD_DIR/cafa_bp_valid

# # 检查验证TFRecord文件是否生成成功
# VALID_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_bp_valid_*.tfrecords 2>/dev/null | wc -l)
# if [ "$VALID_TFR_COUNT" -eq 0 ]; then
#     echo "错误: 验证TFRecord文件生成失败！"
#     exit 1
# fi
# echo "成功生成 $VALID_TFR_COUNT 个验证TFRecord文件"

# # 可选：自动优化训练参数（如果npz文件可用）
# echo "[Step 1c] 尝试自动优化训练参数（可选）"
# python 002d1_auto_optimize_training.py --prot_list cafa_deepfri/BP_train.txt --npz_dir $NPZ_DIR --annot_file cafa_deepfri/cafa_bp_annot.tsv --test_list cafa_deepfri/BP_test.csv --ontology bp --model_name trained_models/CafaModel_BP --max_memory_gb 8 || echo "警告: 自动优化失败，将使用默认参数"

# # 检查最大序列长度（用于验证pad_len）
# echo "[Step 1d] 检查最大序列长度（验证pad_len）"
# PAD_LEN=5000  # 默认值，如果检查发现需要更大，会自动调整
# if command -v python &> /dev/null; then
#     MAX_LEN_OUTPUT=$(python 000_check_max_seq_len.py "$TFRECORD_DIR/cafa_bp_train*" --max_samples 5000 2>&1)
#     MAX_LEN=$(echo "$MAX_LEN_OUTPUT" | grep "最大长度:" | grep -oE '[0-9]+' | head -1)
#     if [ ! -z "$MAX_LEN" ] && [ "$MAX_LEN" -gt 0 ]; then
#         PAD_LEN=$((MAX_LEN + 200))
#         echo "检测到最大序列长度: $MAX_LEN，自动设置 pad_len=$PAD_LEN"
#     else
#         echo "使用默认 pad_len=$PAD_LEN（如果训练时仍然报错，请手动增加此值）"
#     fi
# else
#     echo "无法运行检查脚本，使用默认 pad_len=$PAD_LEN"
# fi

# python train_DeepFRI.py \
#     --train_tfrecord_fn $TFRECORD_DIR/cafa_bp_train \
#     --valid_tfrecord_fn $TFRECORD_DIR/cafa_bp_valid \
#     --annot_fn cafa_deepfri/cafa_bp_annot.tsv \
#     --test_list cafa_deepfri/BP_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --ontology bp \
#     --model_name trained_models/CafaModel_BP \
#     --gc_layer GraphConv \
#     --pad_len $PAD_LEN \
#     --gc_dims 32 32 64 \
#     --fc_dims 64 \
#     --epochs 1 \
#     --batch_size 1


# # testset as test
# python 000_predict_test_set.py \
#     --model_name trained_models/CafaModel_BP \
#     --test_list cafa_deepfri/BP_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --annot_fn cafa_deepfri/cafa_bp_annot.tsv \
#     --ontology bp \
#     --gc_layer GraphConv \
#     --output_file $RESULTS_DIR/bp_test_predictions.pckl

# python 000_export_results.py \
#     --results_file $RESULTS_DIR/bp_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/bp_test_preds_deepfri_run1 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file $RESULTS_DIR/bp_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/bp_test_preds_deepfri_run2 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file $RESULTS_DIR/bp_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/bp_test_preds_deepfri_run3 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file $RESULTS_DIR/bp_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/bp_test_preds_deepfri_run4 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file $RESULTS_DIR/bp_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/bp_test_preds_deepfri_run5 \
#     --threshold 0.0

# # 清理BP的TFRecord文件，释放空间
# echo "[Step 1e] 清理BP的TFRecord文件..."
# rm -f $TFRECORD_DIR/cafa_bp_train_*.tfrecords
# rm -f $TFRECORD_DIR/cafa_bp_valid_*.tfrecords
# echo "BP的TFRecord文件已删除"

# ----------------------------
# PF
# ----------------------------
echo "=== Processing PF ==="

# echo "[Step 1a] Generate TFRecords for PF - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_pf_annot.tsv \
#     -prot_list cafa_deepfri/PF_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 20 \
#     -num_threads 2 \
#     -tfr_prefix $TFRECORD_DIR/cafa_pf_train

# # 检查训练TFRecord文件是否生成成功
# TRAIN_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_pf_train_*.tfrecords 2>/dev/null | wc -l)
# if [ "$TRAIN_TFR_COUNT" -eq 0 ]; then
#     echo "错误: 训练TFRecord文件生成失败！"
#     exit 1
# fi
# echo "成功生成 $TRAIN_TFR_COUNT 个训练TFRecord文件"

# echo "[Step 1b] Generate TFRecords for PF - valid"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_pf_annot.tsv \
#     -prot_list cafa_deepfri/PF_valid.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 3 \
#     -num_threads 2 \
#     -tfr_prefix $TFRECORD_DIR/cafa_pf_valid

# # 检查验证TFRecord文件是否生成成功
# VALID_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_pf_valid_*.tfrecords 2>/dev/null | wc -l)
# if [ "$VALID_TFR_COUNT" -eq 0 ]; then
#     echo "错误: 验证TFRecord文件生成失败！"
#     exit 1
# fi
# echo "成功生成 $VALID_TFR_COUNT 个验证TFRecord文件"

# # 可选：自动优化训练参数（如果npz文件可用）
# echo "[Step 1c] 尝试自动优化训练参数（可选）"
# python 002d1_auto_optimize_training.py --prot_list cafa_deepfri/PF_train.txt --npz_dir $NPZ_DIR --annot_file cafa_deepfri/cafa_pf_annot.tsv --test_list cafa_deepfri/PF_test.csv --ontology pf --model_name trained_models/CafaModel_PF --max_memory_gb 8 || echo "警告: 自动优化失败，将使用默认参数"

# # 检查最大序列长度（用于验证pad_len）
# echo "[Step 1d] 检查最大序列长度（验证pad_len）"
# PAD_LEN=5000  # 默认值，如果检查发现需要更大，会自动调整
# if command -v python &> /dev/null; then
#     MAX_LEN_OUTPUT=$(python 000_check_max_seq_len.py "$TFRECORD_DIR/cafa_pf_train*" --max_samples 5000 2>&1)
#     MAX_LEN=$(echo "$MAX_LEN_OUTPUT" | grep "最大长度:" | grep -oE '[0-9]+' | head -1)
#     if [ ! -z "$MAX_LEN" ] && [ "$MAX_LEN" -gt 0 ]; then
#         PAD_LEN=$((MAX_LEN + 200))
#         echo "检测到最大序列长度: $MAX_LEN，自动设置 pad_len=$PAD_LEN"
#     else
#         echo "使用默认 pad_len=$PAD_LEN（如果训练时仍然报错，请手动增加此值）"
#     fi
# else
#     echo "无法运行检查脚本，使用默认 pad_len=$PAD_LEN"
# fi

# python train_DeepFRI.py \
#     --train_tfrecord_fn $TFRECORD_DIR/cafa_pf_train \
#     --valid_tfrecord_fn $TFRECORD_DIR/cafa_pf_valid \
#     --annot_fn cafa_deepfri/cafa_pf_annot.tsv \
#     --test_list cafa_deepfri/PF_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --ontology pf \
#     --model_name trained_models/CafaModel_PF \
#     --gc_layer GraphConv \
#     --pad_len $PAD_LEN \
#     --gc_dims 32 32 64 \
#     --fc_dims 64 \
#     --epochs 1 \
#     --batch_size 1


# testset as test
# python 000_predict_test_set.py \
#     --model_name trained_models/CafaModel_PF \
#     --test_list cafa_deepfri/PF_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --annot_fn cafa_deepfri/cafa_pf_annot.tsv \
#     --ontology pf \
#     --gc_layer GraphConv \
#     --output_file $RESULTS_DIR/pf_test_predictions.pckl

# python 000_export_results.py \
#     --results_file $RESULTS_DIR/pf_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/pf_test_preds_deepfri_run1 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file $RESULTS_DIR/pf_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/pf_test_preds_deepfri_run2 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file $RESULTS_DIR/pf_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/pf_test_preds_deepfri_run3 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file $RESULTS_DIR/pf_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/pf_test_preds_deepfri_run4 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file $RESULTS_DIR/pf_test_predictions.pckl \
#     --output_prefix $RESULTS_DIR/pf_test_preds_deepfri_run5 \
#     --threshold 0.0

# # 清理PF的TFRecord文件，释放空间
# echo "[Step 1e] 清理PF的TFRecord文件..."
# rm -f $TFRECORD_DIR/cafa_pf_train_*.tfrecords
# rm -f $TFRECORD_DIR/cafa_pf_valid_*.tfrecords
# echo "PF的TFRecord文件已删除"

echo "=== All done! ==="
