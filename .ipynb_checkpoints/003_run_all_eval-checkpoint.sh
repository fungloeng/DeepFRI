#!/bin/bash
set -e  # 出现错误就停止


# ============================================================================
# MF (Molecular Function)
# ============================================================================

# 数据转换（如果需要）
# python preprocessing/PDB2TFRecord.py -annot cafa_deepfri/cafa_mf_annot.tsv -prot_list cafa_deepfri/MF_train.txt -npz_dir cafa_npz_source/ -num_shards 20 -num_threads 20 -tfr_prefix ./tfrecords/cafa_mf_train
# python preprocessing/PDB2TFRecord.py -annot cafa_deepfri/cafa_mf_annot.tsv -prot_list cafa_deepfri/MF_valid.txt -npz_dir cafa_npz_source/ -num_shards 3 -num_threads 3 -tfr_prefix ./tfrecords/cafa_mf_valid

# # 训练
# python train_DeepFRI.py \
#     --train_tfrecord_fn ./tfrecords/cafa_mf_train \
#     --valid_tfrecord_fn ./tfrecords/cafa_mf_valid \
#     --annot_fn cafa_deepfri/cafa_mf_annot.tsv \
#     --test_list cafa_deepfri/MF_test.csv \
#     --test_npz_dir cafa_npz_source/ \
#     --ontology mf \
#     --model_name trained_models/CafaModel_MF \
#     --gc_layer GraphConv \
#     --epochs 10 \
#     --batch_size 1

# testset as test
python 000_predict_test_set.py \
    --model_name trained_models/CafaModel_MF \
    --test_list cafa_deepfri/MF_test.csv \
    --test_npz_dir cafa_npz_source/ \
    --annot_fn cafa_deepfri/cafa_mf_annot.tsv \
    --ontology mf \
    --gc_layer GraphConv \
    --output_file results/cafa/mf_test_predictions.pckl

python 000_export_results.py \
    --results_file results/cafa/mf_test_predictions.pckl \
    --output_prefix results/cafa/mf_test_preds_deepfri_run1 \
    --threshold 0.0

# validset as test
python 000_test_valid.py \
    --model_name trained_models/CafaModel_MF \
    --ontology mf \
    --dataset_type valid \
    --test_npz_dir cafa_npz_source/

# # trainset as test
# python 000_test_valid.py \
#     --model_name trained_models/CafaModel_MF \
#     --ontology mf \
#     --dataset_type train \
#     --test_npz_dir cafa_npz_source/


# ============================================================================
# CC (Cellular Component)
# ============================================================================

# 数据转换（如果需要）
# python preprocessing/PDB2TFRecord.py -annot cafa_deepfri/cafa_cc_annot.tsv -prot_list cafa_deepfri/CC_train.txt -npz_dir cafa_npz_source/ -num_shards 20 -num_threads 20 -tfr_prefix ./tfrecords/cafa_cc_train
# python preprocessing/PDB2TFRecord.py -annot cafa_deepfri/cafa_cc_annot.tsv -prot_list cafa_deepfri/CC_valid.txt -npz_dir cafa_npz_source/ -num_shards 3 -num_threads 3 -tfr_prefix ./tfrecords/cafa_cc_valid

# # 训练
# python train_DeepFRI.py \
#     --train_tfrecord_fn ./tfrecords/cafa_cc_train \
#     --valid_tfrecord_fn ./tfrecords/cafa_cc_valid \
#     --annot_fn cafa_deepfri/cafa_cc_annot.tsv \
#     --test_list cafa_deepfri/CC_test.csv \
#     --test_npz_dir cafa_npz_source/ \
#     --ontology cc \
#     --model_name trained_models/CafaModel_CC \
#     --gc_layer GraphConv \
#     --epochs 10 \
#     --batch_size 1

# testset as test
# python 000_predict_test_set.py \
#     --model_name trained_models/CafaModel_CC \
#     --test_list cafa_deepfri/CC_test.csv \
#     --test_npz_dir cafa_npz_source/ \
#     --annot_fn cafa_deepfri/cafa_cc_annot.tsv \
#     --ontology cc \
#     --gc_layer GraphConv \
#     --output_file results/cafa/cc_test_predictions.pckl

# python 000_export_results.py \
#     --results_file results/cafa/cc_test_predictions.pckl \
#     --output_prefix results/cafa/cc_test_preds_deepfri_run1 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file results/cafa/cc_test_predictions.pckl \
#     --output_prefix results/cafa/cc_test_preds_deepfri_run2 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file results/cafa/cc_test_predictions.pckl \
#     --output_prefix results/cafa/cc_test_preds_deepfri_run3 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/cc_test_predictions.pckl \
#     --output_prefix results/cafa/cc_test_preds_deepfri_run4 \
#     --threshold 0.0
    
# python 000_export_results.py \
#     --results_file results/cafa/cc_test_predictions.pckl \
#     --output_prefix results/cafa/cc_test_preds_deepfri_run5 \
#     --threshold 0.0


# # validset as test
# python 000_test_valid.py \
#     --model_name trained_models/CafaModel_CC \
#     --ontology cc \
#     --dataset_type valid \
#     --test_npz_dir cafa_npz_source/

# # trainset as test
# python 000_test_valid.py \
#     --model_name trained_models/CafaModel_CC \
#     --ontology cc \
#     --dataset_type train \
#     --test_npz_dir cafa_npz_source/



# ============================================================================
# BP (Biological Process)
# ============================================================================

# 数据转换（如果需要）
# python preprocessing/PDB2TFRecord.py -annot cafa_deepfri/cafa_bp_annot.tsv -prot_list cafa_deepfri/BP_train.txt -npz_dir cafa_npz_source/ -num_shards 20 -num_threads 20 -tfr_prefix ./tfrecords/cafa_bp_train
# python preprocessing/PDB2TFRecord.py -annot cafa_deepfri/cafa_bp_annot.tsv -prot_list cafa_deepfri/BP_valid.txt -npz_dir cafa_npz_source/ -num_shards 3 -num_threads 3 -tfr_prefix ./tfrecords/cafa_bp_valid

# # 训练
# python train_DeepFRI.py \
#     --train_tfrecord_fn ./tfrecords/cafa_bp_train \
#     --valid_tfrecord_fn ./tfrecords/cafa_bp_valid \
#     --annot_fn cafa_deepfri/cafa_bp_annot.tsv \
#     --test_list cafa_deepfri/BP_test.csv \
#     --test_npz_dir cafa_npz_source/ \
#     --ontology bp \
#     --model_name trained_models/CafaModel_BP \
#     --gc_layer GraphConv \
#     --epochs 10 \
#     --batch_size 1

# testset as test
# python 000_predict_test_set.py \
#     --model_name trained_models/CafaModel_BP \
#     --test_list cafa_deepfri/BP_test.csv \
#     --test_npz_dir cafa_npz_source/ \
#     --annot_fn cafa_deepfri/cafa_bp_annot.tsv \
#     --ontology bp \
#     --gc_layer GraphConv \
#     --output_file results/cafa/bp_test_predictions.pckl

# python 000_export_results.py \
#     --results_file results/cafa/bp_test_predictions.pckl \
#     --output_prefix results/cafa/bp_test_preds_deepfri_run1 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/bp_test_predictions.pckl \
#     --output_prefix results/cafa/bp_test_preds_deepfri_run2 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/bp_test_predictions.pckl \
#     --output_prefix results/cafa/bp_test_preds_deepfri_run3 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/bp_test_predictions.pckl \
#     --output_prefix results/cafa/bp_test_preds_deepfri_run4 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/bp_test_predictions.pckl \
#     --output_prefix results/cafa/bp_test_preds_deepfri_run5 \
#     --threshold 0.0

# # validset as test
# python 000_test_valid.py \
#     --model_name trained_models/CafaModel_BP \
#     --ontology bp \
#     --dataset_type valid \
#     --test_npz_dir cafa_npz_source/

# # trainset as test
# python 000_test_valid.py \
#     --model_name trained_models/CafaModel_BP \
#     --ontology bp \
#     --dataset_type train \
#     --test_npz_dir cafa_npz_source/




# ============================================================================
# PF (Biological Process)
# ============================================================================

# 数据转换（如果需要）
# python preprocessing/PDB2TFRecord.py -annot cafa_deepfri/cafa_pf_annot.tsv -prot_list cafa_deepfri/PF_train.txt -npz_dir cafa_npz_source/ -num_shards 20 -num_threads 20 -tfr_prefix ./tfrecords/cafa_pf_train
# python preprocessing/PDB2TFRecord.py -annot cafa_deepfri/cafa_pf_annot.tsv -prot_list cafa_deepfri/PF_valid.txt -npz_dir cafa_npz_source/ -num_shards 3 -num_threads 3 -tfr_prefix ./tfrecords/cafa_pf_valid

# # 训练
# python train_DeepFRI.py \
#     --train_tfrecord_fn ./tfrecords/cafa_pf_train \
#     --valid_tfrecord_fn ./tfrecords/cafa_pf_valid \
#     --annot_fn cafa_deepfri/cafa_pf_annot.tsv \
#     --test_list cafa_deepfri/PF_test.csv \
#     --test_npz_dir cafa_npz_source/ \
#     --ontology pf \
#     --model_name trained_models/CafaModel_PF \
#     --gc_layer GraphConv \
#     --epochs 10 \
#     --batch_size 1

# testset as test
# python 000_predict_test_set.py \
#     --model_name trained_models/CafaModel_PF \
#     --test_list cafa_deepfri/PF_test.csv \
#     --test_npz_dir cafa_npz_source/ \
#     --annot_fn cafa_deepfri/cafa_pf_annot.tsv \
#     --ontology pf \
#     --gc_layer GraphConv \
#     --output_file results/cafa/pf_test_predictions.pckl

# python 000_export_results.py \
#     --results_file results/cafa/pf_test_predictions.pckl \
#     --output_prefix results/cafa/pf_test_preds_deepfri_run1 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/pf_test_predictions.pckl \
#     --output_prefix results/cafa/pf_test_preds_deepfri_run2 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/pf_test_predictions.pckl \
#     --output_prefix results/cafa/pf_test_preds_deepfri_run3 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/pf_test_predictions.pckl \
#     --output_prefix results/cafa/pf_test_preds_deepfri_run4 \
#     --threshold 0.0

# python 000_export_results.py \
#     --results_file results/cafa/pf_test_predictions.pckl \
#     --output_prefix results/cafa/pf_test_preds_deepfri_run5 \
#     --threshold 0.0

# # validset as test
# python 000_test_valid.py \
#     --model_name trained_models/CafaModel_PF \
#     --ontology pf \
#     --dataset_type valid \
#     --test_npz_dir cafa_npz_source/

# # trainset as test
# python 000_test_valid.py \
#     --model_name trained_models/CafaModel_PF \
#     --ontology pf \
#     --dataset_type train \
#     --test_npz_dir cafa_npz_source/

