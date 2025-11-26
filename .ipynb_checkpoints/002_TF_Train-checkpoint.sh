#!/bin/bash
set -e  # 出现错误就停止

# NPZ目录
NPZ_DIR="cafa_npz_source"
# NPZ_DIR="npz_source"

# 输出TFRecord目录
TFRECORD_DIR="./tfrecords"


# # ----------------------------
# # 1. MF
# # ----------------------------
# echo "=== Processing MF ==="

# echo "[Step 1a] Generate TFRecords for MF - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_mf_annot.tsv \
#     -prot_list cafa_deepfri/MF_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 40 \
#     -num_threads 1 \
#     --ontology mf \
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
#     -num_shards 5 \
#     -num_threads 1 \
#     --ontology mf \
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
# if command -v python &> /dev/null && [ -f "000_check_max_seq_len.py" ]; then
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
#     # 如果之前训练过，使用之前的 pad_len
#     PAD_LEN=4499
# fi

# # 创建日志目录
# mkdir -p log

# echo "[Step 1e] 开始训练 MF 模型"
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
#     --batch_size 4 2>&1 | tee log/mf_train.log



# ----------------------------
# 1. CC
# ----------------------------
# echo "=== Processing CC ==="

# # 删除旧的TFRecord文件（如果存在）
# echo "[Step 0] 清理旧的TFRecord文件（如果存在）"
# rm -f $TFRECORD_DIR/cafa_cc_train*.tfrecords
# rm -f $TFRECORD_DIR/cafa_cc_valid*.tfrecords
# echo "已清理旧的TFRecord文件"

# echo "[Step 1a] Generate TFRecords for CC - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_cc_annot.tsv \
#     -prot_list cafa_deepfri/CC_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 40 \
#     -num_threads 1 \
#     --ontology cc \
#     -tfr_prefix $TFRECORD_DIR/cafa_cc_train

# # 检查训练TFRecord文件是否生成成功
# TRAIN_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_cc_train_*.tfrecords 2>/dev/null | wc -l)
# if [ "$TRAIN_TFR_COUNT" -eq 0 ]; then
#     echo "错误: 训练TFRecord文件生成失败！"
#     exit 1
# fi
# echo "成功生成 $TRAIN_TFR_COUNT 个训练TFRecord文件"

# echo "[Step 1b] Generate TFRecords for CC - valid"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_cc_annot.tsv \
#     -prot_list cafa_deepfri/CC_valid.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 5 \
#     -num_threads 1 \
#     --ontology cc \
#     -tfr_prefix $TFRECORD_DIR/cafa_cc_valid

# # 检查验证TFRecord文件是否生成成功
# VALID_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_cc_valid_*.tfrecords 2>/dev/null | wc -l)
# if [ "$VALID_TFR_COUNT" -eq 0 ]; then
#     echo "错误: 验证TFRecord文件生成失败！"
#     exit 1
# fi
# echo "成功生成 $VALID_TFR_COUNT 个验证TFRecord文件"

# # 检查最大序列长度（用于自动设置pad_len）
# echo "[Step 1c] 检查最大序列长度（自动设置pad_len）"
# PAD_LEN=5000  # 默认值
# if command -v python &> /dev/null && [ -f "000_check_max_len_from_tfrecord.py" ]; then
#     # 检查训练集和验证集，从标准输出中提取（stderr用于进度信息）
#     echo "  检查训练集..."
#     MAX_LEN_OUTPUT_TRAIN=$(python 000_check_max_len_from_tfrecord.py "$TFRECORD_DIR/cafa_cc_train*" 2>/dev/null)
#     echo "  检查验证集..."
#     MAX_LEN_OUTPUT_VALID=$(python 000_check_max_len_from_tfrecord.py "$TFRECORD_DIR/cafa_cc_valid*" 2>/dev/null)
    
#     # 提取最大长度值
#     MAX_LEN_TRAIN=$(echo "$MAX_LEN_OUTPUT_TRAIN" | grep "最大长度:" | grep -oE '[0-9]+' | head -1)
#     MAX_LEN_VALID=$(echo "$MAX_LEN_OUTPUT_VALID" | grep "最大长度:" | grep -oE '[0-9]+' | head -1)
    
#     # 找到真正的最大值
#     MAX_LEN=0
#     if [ ! -z "$MAX_LEN_TRAIN" ] && [ "$MAX_LEN_TRAIN" -gt 0 ]; then
#         MAX_LEN=$MAX_LEN_TRAIN
#         echo "  训练集最大长度: $MAX_LEN_TRAIN"
#     fi
#     if [ ! -z "$MAX_LEN_VALID" ] && [ "$MAX_LEN_VALID" -gt "$MAX_LEN" ]; then
#         MAX_LEN=$MAX_LEN_VALID
#         echo "  验证集最大长度: $MAX_LEN_VALID"
#     fi
    
#     if [ "$MAX_LEN" -gt 0 ]; then
#         # 根据最大长度智能设置 pad_len 和 batch_size
#         # 对于大序列，使用较小的安全余量以避免内存问题
#         if [ "$MAX_LEN" -gt 4000 ]; then
#             # 非常大的序列，使用最小的余量，并向上取整到100的倍数
#             PAD_LEN=$(( (MAX_LEN + 125) / 100 * 100 ))  # 向上取整到100的倍数，最小余量125
#             BATCH_SIZE=1  # 大序列使用 batch_size=1
#         elif [ "$MAX_LEN" -gt 3000 ]; then
#             # 大序列，使用中等余量
#             PAD_LEN=$(( (MAX_LEN + 150) / 100 * 100 ))  # 向上取整到100的倍数
#             BATCH_SIZE=2  # 中等序列使用 batch_size=2
#         else
#             # 正常序列，使用标准余量
#             PAD_LEN=$(( (MAX_LEN + 200) / 100 * 100 ))  # 向上取整到100的倍数
#             BATCH_SIZE=4  # 正常序列使用 batch_size=4
#         fi
#         echo "✓ 检测到最大序列长度: $MAX_LEN"
#         echo "✓ 自动设置 pad_len=$PAD_LEN (优化后的值，向上取整到100的倍数)"
#         echo "✓ 自动设置 batch_size=$BATCH_SIZE (根据序列长度和内存优化)"
#     else
#         echo "⚠ 无法检测最大长度，使用默认 pad_len=$PAD_LEN, batch_size=2"
#         BATCH_SIZE=2  # 默认使用较小的 batch_size
#     fi
# else
#     echo "⚠ 无法运行检查脚本，使用默认 pad_len=$PAD_LEN, batch_size=2"
#     BATCH_SIZE=2  # 默认使用较小的 batch_size
# fi

# # 可选：自动优化训练参数（如果npz文件可用）
# echo "[Step 1d] 尝试自动优化训练参数（可选）"
# python 002d1_auto_optimize_training.py --prot_list cafa_deepfri/CC_train.txt --npz_dir $NPZ_DIR --annot_file cafa_deepfri/cafa_cc_annot.tsv --test_list cafa_deepfri/CC_test.csv --ontology cc --model_name trained_models/CafaModel_CC --max_memory_gb 8 || echo "警告: 自动优化失败，将使用默认参数"

# # 创建日志目录
# mkdir -p log

# echo "[Step 1e] 开始训练 CC 模型"
# echo "  使用参数: pad_len=$PAD_LEN, batch_size=$BATCH_SIZE"
# python train_DeepFRI.py \
#     --train_tfrecord_fn $TFRECORD_DIR/cafa_cc_train \
#     --valid_tfrecord_fn $TFRECORD_DIR/cafa_cc_valid \
#     --annot_fn cafa_deepfri/cafa_cc_annot.tsv \
#     --test_list cafa_deepfri/CC_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --ontology cc \
#     --model_name trained_models/CafaModel_CC \
#     --gc_layer GraphConv \
#     --pad_len $PAD_LEN \
#     --gc_dims 32 32 64 \
#     --fc_dims 64 \
#     --epochs 1 \
#     --batch_size $BATCH_SIZE 2>&1 | tee log/cc_train.log


# # ----------------------------
# # 1. BP
# # ----------------------------
# echo "=== Processing BP ==="

# echo "[Step 1a] Generate TFRecords for BP - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_bp_annot.tsv \
#     -prot_list cafa_deepfri/BP_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 40 \
#     -num_threads 1 \
#     --ontology bp \
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
#     -num_shards 5 \
#     -num_threads 1 \
#     --ontology bp \
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
# if command -v python &> /dev/null && [ -f "000_check_max_seq_len.py" ]; then
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
#     # 如果之前训练过，使用之前的 pad_len
#     PAD_LEN=4499
# fi

# # 创建日志目录
# mkdir -p log

# echo "[Step 1e] 开始训练 BP 模型"
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
#     --batch_size 4 2>&1 | tee log/bp_train.log


# ----------------------------
# 4. PF
# ----------------------------
echo "=== Processing PF ==="

# # 删除旧的TFRecord文件
# rm -rf $TFRECORD_DIR/cafa_pf_*

# echo "[Step 1a] Generate TFRecords for PF - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot cafa_deepfri/cafa_pf_annot.tsv \
#     -prot_list cafa_deepfri/PF_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 40 \
#     -num_threads 1 \
#     --ontology pf \
#     --records_per_shard 400 \
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
#     -num_shards 5 \
#     -num_threads 1 \
#     --ontology pf \
#     --records_per_shard 400 \
#     -tfr_prefix $TFRECORD_DIR/cafa_pf_valid

# # 检查验证TFRecord文件是否生成成功
# VALID_TFR_COUNT=$(ls -1 $TFRECORD_DIR/cafa_pf_valid_*.tfrecords 2>/dev/null | wc -l)
# if [ "$VALID_TFR_COUNT" -eq 0 ]; then
#     echo "错误: 验证TFRecord文件生成失败！"
#     exit 1
# fi
# echo "成功生成 $VALID_TFR_COUNT 个验证TFRecord文件"

# 检查最大序列长度（用于自动设置pad_len）
# echo "[Step 1c] 检查最大序列长度（自动设置pad_len）"
# PAD_LEN=4700  # 默认值（保守设置，避免OOM，基于最大长度4575 + 125）
# BATCH_SIZE=1  # 默认使用 batch_size=1（保守设置，避免OOM）

# if command -v python &> /dev/null && [ -f "000_check_max_len_from_tfrecord.py" ]; then
#     # 检查训练集和验证集，从标准输出中提取（stderr用于进度信息）
#     echo "  检查训练集..."
#     MAX_LEN_OUTPUT_TRAIN=$(python 000_check_max_len_from_tfrecord.py "$TFRECORD_DIR/cafa_pf_train*" 2>/dev/null)
#     echo "  检查验证集..."
#     MAX_LEN_OUTPUT_VALID=$(python 000_check_max_len_from_tfrecord.py "$TFRECORD_DIR/cafa_pf_valid*" 2>/dev/null)
    
#     # 提取最大长度值
#     MAX_LEN_TRAIN=$(echo "$MAX_LEN_OUTPUT_TRAIN" | grep "最大长度:" | grep -oE '[0-9]+' | head -1)
#     MAX_LEN_VALID=$(echo "$MAX_LEN_OUTPUT_VALID" | grep "最大长度:" | grep -oE '[0-9]+' | head -1)
    
#     # 找到真正的最大值
#     MAX_LEN=0
#     if [ ! -z "$MAX_LEN_TRAIN" ] && [ "$MAX_LEN_TRAIN" -gt 0 ]; then
#         MAX_LEN=$MAX_LEN_TRAIN
#         echo "  训练集最大长度: $MAX_LEN_TRAIN"
#     fi
#     if [ ! -z "$MAX_LEN_VALID" ] && [ "$MAX_LEN_VALID" -gt "$MAX_LEN" ]; then
#         MAX_LEN=$MAX_LEN_VALID
#         echo "  验证集最大长度: $MAX_LEN_VALID"
#     fi
    
#     if [ "$MAX_LEN" -gt 0 ]; then
#         # 根据最大长度智能设置 pad_len 和 batch_size
#         # 对于GPU内存有限的情况（约1GB），使用非常保守的设置
#         if [ "$MAX_LEN" -gt 4000 ]; then
#             # 非常大的序列，使用最小的余量（仅100），并向上取整到100的倍数
#             # 强制使用 batch_size=1 以避免OOM
#             PAD_LEN=$(( (MAX_LEN + 100) / 100 * 100 ))  # 向上取整到100的倍数，最小余量100
#             BATCH_SIZE=1  # 强制使用 batch_size=1
#         elif [ "$MAX_LEN" -gt 3000 ]; then
#             # 大序列，使用中等余量
#             PAD_LEN=$(( (MAX_LEN + 150) / 100 * 100 ))  # 向上取整到100的倍数
#             BATCH_SIZE=1  # 保守使用 batch_size=1（GPU内存有限）
#         else
#             # 正常序列，使用标准余量
#             PAD_LEN=$(( (MAX_LEN + 200) / 100 * 100 ))  # 向上取整到100的倍数
#             BATCH_SIZE=2  # 正常序列可以使用 batch_size=2
#         fi
#         echo "✓ 检测到最大序列长度: $MAX_LEN"
#         echo "✓ 自动设置 pad_len=$PAD_LEN (优化后的值，向上取整到100的倍数)"
#         echo "✓ 自动设置 batch_size=$BATCH_SIZE (根据序列长度和内存优化)"
#     else
#         echo "⚠ 无法检测最大长度，使用保守默认值 pad_len=$PAD_LEN, batch_size=$BATCH_SIZE"
#     fi
# else
#     echo "⚠ 无法运行检查脚本，使用保守默认值 pad_len=$PAD_LEN, batch_size=$BATCH_SIZE"
# fi

# # 可选：自动优化训练参数（如果npz文件可用）
# echo "[Step 1d] 尝试自动优化训练参数（可选）"
# python 002d1_auto_optimize_training.py --prot_list cafa_deepfri/PF_train.txt --npz_dir $NPZ_DIR --annot_file cafa_deepfri/cafa_pf_annot.tsv --test_list cafa_deepfri/PF_test.csv --ontology pf --model_name trained_models/CafaModel_PF --max_memory_gb 8 || echo "警告: 自动优化失败，将使用默认参数"

# # 创建日志目录
# mkdir -p log

echo "[Step 1e] 开始训练 PF 模型"
echo "  使用参数: pad_len=$PAD_LEN, batch_size=$BATCH_SIZE"
python train_DeepFRI.py \
    --train_tfrecord_fn $TFRECORD_DIR/cafa_pf_train \
    --valid_tfrecord_fn $TFRECORD_DIR/cafa_pf_valid \
    --annot_fn cafa_deepfri/cafa_pf_annot.tsv \
    --test_list cafa_deepfri/PF_test.csv \
    --test_npz_dir $NPZ_DIR \
    --ontology pf \
    --model_name trained_models/CafaModel_PF \
    --gc_layer GraphConv \
    --pad_len 5000 \
    --gc_dims 32 32 64 \
    --fc_dims 64 \
    --epochs 1 \
    --batch_size 1 1>&1 | tee log/pf_train.log

echo "=== PF COMPLETE ==="

echo "=== All done! ==="
