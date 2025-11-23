#!/bin/bash
set -e  # 出现错误就停止

# NPZ目录
NPZ_DIR="npz_source"

# 输出TFRecord目录
TFRECORD_DIR="./tfrecords"

# ----------------------------
# 1. MF
# ----------------------------
echo "=== Processing MF ==="

echo "[Step 1a] Generate TFRecords for MF - train"
python preprocessing/PDB2TFRecord.py \
    -annot galaxy_deepfri/galaxy_mf_annot.tsv \
    -prot_list galaxy_deepfri/MF_train.txt \
    -npz_dir $NPZ_DIR \
    -num_shards 20 \
    -num_threads 20 \
    -tfr_prefix $TFRECORD_DIR/galaxy_mf_train

echo "[Step 1b] Generate TFRecords for MF - valid"
python preprocessing/PDB2TFRecord.py \
    -annot galaxy_deepfri/galaxy_mf_annot.tsv \
    -prot_list galaxy_deepfri/MF_valid.txt \
    -npz_dir $NPZ_DIR \
    -num_shards 3 \
    -num_threads 3 \
    -tfr_prefix $TFRECORD_DIR/galaxy_mf_valid

echo "[Step 1c] Train MF model"
python train_DeepFRI.py \
    --train_tfrecord_fn $TFRECORD_DIR/galaxy_mf_train \
    --valid_tfrecord_fn $TFRECORD_DIR/galaxy_mf_valid \
    --annot_fn galaxy_deepfri/galaxy_mf_annot.tsv \
    --test_list galaxy_deepfri/test.csv \
    --test_npz_dir $NPZ_DIR \
    --ontology mf \
    --model_name trained_models/GalaxyModel_MF \
    --gc_layer GraphConv \
    --epochs 5 \
    --batch_size 1

# ----------------------------
# 2. CC
# ----------------------------
echo "=== Processing CC ==="

python preprocessing/PDB2TFRecord.py \
    -annot galaxy_deepfri/galaxy_cc_annot.tsv \
    -prot_list galaxy_deepfri/CC_train.txt \
    -npz_dir $NPZ_DIR \
    -num_shards 20 \
    -num_threads 20 \
    -tfr_prefix $TFRECORD_DIR/galaxy_cc_train

python preprocessing/PDB2TFRecord.py \
    -annot galaxy_deepfri/galaxy_cc_annot.tsv \
    -prot_list galaxy_deepfri/CC_valid.txt \
    -npz_dir $NPZ_DIR \
    -num_shards 3 \
    -num_threads 3 \
    -tfr_prefix $TFRECORD_DIR/galaxy_cc_valid

python train_DeepFRI.py \
    --train_tfrecord_fn $TFRECORD_DIR/galaxy_cc_train \
    --valid_tfrecord_fn $TFRECORD_DIR/galaxy_cc_valid \
    --annot_fn galaxy_deepfri/galaxy_cc_annot.tsv \
    --test_list galaxy_deepfri/CC_test.csv \
    --test_npz_dir $NPZ_DIR \
    --ontology cc \
    --model_name trained_models/GalaxyModel_CC \
    --gc_layer GraphConv \
    --epochs 5 \
    --batch_size 1

# ----------------------------
# 3. BP
# ----------------------------
echo "=== Processing BP ==="

python preprocessing/PDB2TFRecord.py \
    -annot galaxy_deepfri/galaxy_bp_annot.tsv \
    -prot_list galaxy_deepfri/BP_train.txt \
    -npz_dir $NPZ_DIR \
    -num_shards 20 \
    -num_threads 20 \
    -tfr_prefix $TFRECORD_DIR/galaxy_bp_train

python preprocessing/PDB2TFRecord.py \
    -annot galaxy_deepfri/galaxy_bp_annot.tsv \
    -prot_list galaxy_deepfri/BP_valid.txt \
    -npz_dir $NPZ_DIR \
    -num_shards 3 \
    -num_threads 3 \
    -tfr_prefix $TFRECORD_DIR/galaxy_bp_valid

python train_DeepFRI.py \
    --train_tfrecord_fn $TFRECORD_DIR/galaxy_bp_train \
    --valid_tfrecord_fn $TFRECORD_DIR/galaxy_bp_valid \
    --annot_fn galaxy_deepfri/galaxy_bp_annot.tsv \
    --test_list galaxy_deepfri/BP_test.csv \
    --test_npz_dir $NPZ_DIR \
    --ontology bp \
    --model_name trained_models/GalaxyModel_BP \
    --gc_layer GraphConv \
    --epochs 5 \
    --batch_size 1

echo "=== All done! ==="
