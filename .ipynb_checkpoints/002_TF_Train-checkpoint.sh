#!/bin/bash
set -e  # 出现错误就停止

# NPZ目录
NPZ_DIR="galaxy_npz_source"

# 输出TFRecord目录
TFRECORD_DIR="./tfrecords"

# ----------------------------
# 1. MF
# ----------------------------
# echo "=== Processing MF ==="

# echo "[Step 1a] Generate TFRecords for MF - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot galaxy_deepfri/galaxy_mf_annot.tsv \
#     -prot_list galaxy_deepfri/MF_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 20 \
#     -num_threads 20 \
#     -tfr_prefix $TFRECORD_DIR/galaxy_mf_train

# echo "[Step 1b] Generate TFRecords for MF - valid"
# python preprocessing/PDB2TFRecord.py \
#     -annot galaxy_deepfri/galaxy_mf_annot.tsv \
#     -prot_list galaxy_deepfri/MF_valid.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 3 \
#     -num_threads 3 \
#     -tfr_prefix $TFRECORD_DIR/galaxy_mf_valid


# python 002d1_auto_optimize_training.py --prot_list galaxy_deepfri/MF_train.txt --npz_dir $NPZ_DIR --annot_file galaxy_deepfri/galaxy_mf_annot.tsv --test_list galaxy_deepfri/MF_test.csv --ontology mf --model_name trained_models/GalaxyModel_MF --max_memory_gb 8

# python train_DeepFRI.py \
#     --train_tfrecord_fn $TFRECORD_DIR/galaxy_mf_train \
#     --valid_tfrecord_fn $TFRECORD_DIR/galaxy_mf_valid \
#     --annot_fn galaxy_deepfri/galaxy_mf_annot.tsv \
#     --test_list galaxy_deepfri/MF_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --ontology mf \
#     --model_name trained_models/GalaxyModel_MF \
#     --gc_layer GraphConv \
#     --pad_len 2699 \
#     --gc_dims 32 32 64 \
#     --fc_dims 64 \
#     --epochs 5 \
#     --batch_size 1


# ----------------------------
# 2. CC
# ----------------------------
echo "=== Processing CC ==="

# echo "[Step 1a] Generate TFRecords for CC - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot galaxy_deepfri/galaxy_cc_annot.tsv \
#     -prot_list galaxy_deepfri/CC_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 20 \
#     -num_threads 20 \
#     -tfr_prefix $TFRECORD_DIR/galaxy_cc_train

# echo "[Step 1b] Generate TFRecords for CC - valid"
# python preprocessing/PDB2TFRecord.py \
#     -annot galaxy_deepfri/galaxy_cc_annot.tsv \
#     -prot_list galaxy_deepfri/CC_valid.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 3 \
#     -num_threads 3 \
#     -tfr_prefix $TFRECORD_DIR/galaxy_cc_valid


############################
# python 002d1_auto_optimize_training.py --prot_list galaxy_deepfri/CC_train.txt --npz_dir $NPZ_DIR --annot_file galaxy_deepfri/galaxy_cc_annot.tsv --test_list galaxy_deepfri/CC_test.csv --ontology cc --model_name trained_models/GalaxyModel_CC --max_memory_gb 8

# python train_DeepFRI.py \
#     --train_tfrecord_fn $TFRECORD_DIR/galaxy_cc_train \
#     --valid_tfrecord_fn $TFRECORD_DIR/galaxy_cc_valid \
#     --annot_fn galaxy_deepfri/galaxy_cc_annot.tsv \
#     --test_list galaxy_deepfri/CC_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --ontology cc \
#     --model_name trained_models/GalaxyModel_CC \
#     --gc_layer GraphConv \
#     --pad_len 2699 \
#     --gc_dims 32 32 64 \
#     --fc_dims 64 \
#     --epochs 1 \
#     --batch_size 1
############################




echo "=== CC COMPLETE ==="

# ----------------------------
# 3. BP
# ----------------------------
# echo "=== Processing BP ==="

# echo "[Step 1a] Generate TFRecords for BP - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot galaxy_deepfri/galaxy_mf_annot.tsv \
#     -prot_list galaxy_deepfri/BP_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 20 \
#     -num_threads 20 \
#     -tfr_prefix $TFRECORD_DIR/galaxy_mf_train

# echo "[Step 1b] Generate TFRecords for BP - valid"
# python preprocessing/PDB2TFRecord.py \
#     -annot galaxy_deepfri/galaxy_mf_annot.tsv \
#     -prot_list galaxy_deepfri/BP_valid.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 3 \
#     -num_threads 3 \
#     -tfr_prefix $TFRECORD_DIR/galaxy_mf_valid


# python 002d1_auto_optimize_training.py --prot_list galaxy_deepfri/BP_train.txt --npz_dir $NPZ_DIR --annot_file galaxy_deepfri/galaxy_mf_annot.tsv --test_list galaxy_deepfri/BP_test.csv --ontology mf --model_name trained_models/GalaxyModel_BP --max_memory_gb 8

# python train_DeepFRI.py \
#     --train_tfrecord_fn $TFRECORD_DIR/galaxy_mf_train \
#     --valid_tfrecord_fn $TFRECORD_DIR/galaxy_mf_valid \
#     --annot_fn galaxy_deepfri/galaxy_mf_annot.tsv \
#     --test_list galaxy_deepfri/BP_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --ontology mf \
#     --model_name trained_models/GalaxyModel_BP \
#     --gc_layer GraphConv \
#     --pad_len 2699 \
#     --gc_dims 32 32 64 \
#     --fc_dims 64 \
#     --epochs 5 \
#     --batch_size 1


# ----------------------------
# 4. PF
# ----------------------------
# echo "=== Processing PF ==="

# echo "[Step 1a] Generate TFRecords for PF - train"
# python preprocessing/PDB2TFRecord.py \
#     -annot galaxy_deepfri/galaxy_pf_annot.tsv \
#     -prot_list galaxy_deepfri/PF_train.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 20 \
#     -num_threads 20 \
#     -tfr_prefix $TFRECORD_DIR/galaxy_pf_train

# echo "[Step 1b] Generate TFRecords for PF - valid"
# python preprocessing/PDB2TFRecord.py \
#     -annot galaxy_deepfri/galaxy_pf_annot.tsv \
#     -prot_list galaxy_deepfri/PF_valid.txt \
#     -npz_dir $NPZ_DIR \
#     -num_shards 3 \
#     -num_threads 3 \
#     -tfr_prefix $TFRECORD_DIR/galaxy_pf_valid

# # python 002d1_auto_optimize_training.py --prot_list galaxy_deepfri/PF_train.txt --npz_dir $NPZ_DIR --annot_file galaxy_deepfri/galaxy_pf_annot.tsv --test_list galaxy_deepfri/PF_test.csv --ontology pf --model_name trained_models/GalaxyModel_PF --max_memory_gb 8

# echo "[Step 1c] Train PF model"
# python train_DeepFRI.py \
#     --train_tfrecord_fn $TFRECORD_DIR/galaxy_pf_train \
#     --valid_tfrecord_fn $TFRECORD_DIR/galaxy_pf_valid \
#     --annot_fn galaxy_deepfri/galaxy_pf_annot.tsv \
#     --test_list galaxy_deepfri/PF_test.csv \
#     --test_npz_dir $NPZ_DIR \
#     --ontology pf \
#     --model_name trained_models/GalaxyModel_PF \
#     --gc_layer GraphConv \
#     --pad_len 2699 \
#     --gc_dims 32 32 64 \
#     --fc_dims 64 \
#     --epochs 2 \
#     --batch_size 1

echo "=== All done! ==="
