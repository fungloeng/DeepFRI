#!/bin/bash
# 顺序执行 MF -> CC -> BP，训练完成后自动删除对应的 TFRecord，避免磁盘爆满

set -e  # 出错时停止

# NPZ 文件目录（所有ontology共用）
NPZ_DIR="npz_source"

# TFRecord 存放目录
TFR_DIR="./tfrecords"

# 注释文件目录
ANNOT_DIR="galaxy_deepfri"

# 测试集列表
TEST_LIST="${ANNOT_DIR}/test.csv"

# 模型参数
GC_LAYER="GraphConv"
EPOCHS=5
BATCH_SIZE=1

# 函数：生成 TFRecord + 训练 + 清理
run_ontology() {
    ONT=$1
    MODEL_NAME=$2

    echo "=================================================="
    echo "开始处理 ${ONT} ..."
    echo "=================================================="

    # 训练集 TFRecord
    python preprocessing/PDB2TFRecord.py \
        -annot ${ANNOT_DIR}/galaxy_${ONT}_annot.tsv \
        -prot_list ${ANNOT_DIR}/${ONT}_train.txt \
        -npz_dir ${NPZ_DIR} \
        -num_shards 20 \
        -num_threads 20 \
        -tfr_prefix ${TFR_DIR}/galaxy_${ONT}_train

    # 验证集 TFRecord
    python preprocessing/PDB2TFRecord.py \
        -annot ${ANNOT_DIR}/galaxy_${ONT}_annot.tsv \
        -prot_list ${ANNOT_DIR}/${ONT}_valid.txt \
        -npz_dir ${NPZ_DIR} \
        -num_shards 3 \
        -num_threads 3 \
        -tfr_prefix ${TFR_DIR}/galaxy_${ONT}_valid

    # 训练
    python train_DeepFRI.py \
        --train_tfrecord_fn ${TFR_DIR}/galaxy_${ONT}_train \
        --valid_tfrecord_fn ${TFR_DIR}/galaxy_${ONT}_valid \
        --annot_fn ${ANNOT_DIR}/galaxy_${ONT}_annot.tsv \
        --test_list ${TEST_LIST} \
        --test_npz_dir ${NPZ_DIR} \
        --ontology ${ONT} \
        --model_name trained_models/${MODEL_NAME} \
        --gc_layer ${GC_LAYER} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE}

    # 清理 TFRecord，释放空间
    echo "清理 ${ONT} TFRecord 文件..."
    rm -rf ${TFR_DIR}/galaxy_${ONT}_train_*.tfrecords
    rm -rf ${TFR_DIR}/galaxy_${ONT}_valid_*.tfrecords
    echo "${ONT} 处理完成。"
    echo ""
}

# 按顺序执行 MF -> CC -> BP
# run_ontology "mf" "GalaxyModel_MF"
run_ontology "cc" "GalaxyModel_CC"
run_ontology "bp" "GalaxyModel_BP"

echo "所有ontology训练完成！"
