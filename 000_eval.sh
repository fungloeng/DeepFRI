#!/bin/bash
set -e 

echo "START EVAL"

echo "MF TEST"
# python 001_predict.py --pdb_dir pdb_splits/MF/test/ -ont mf -o results/MF/test/  2>&1 | tee log/MF_test_pred.log

echo "MF VAL"
python 001_predict.py --pdb_dir pdb_splits/MF/validation/ -ont mf -o results/MF/validation/ 2>&1 | tee log/MF_validation_pred.log

echo "CC TEST"
python 001_predict.py --pdb_dir pdb_splits/CC/test/ -ont cc -o results/CC/test/ 2>&1 | tee log/CC_test_pred.log

echo "CC VAL"
python 001_predict.py --pdb_dir pdb_splits/CC/validation/ -ont cc -o results/CC/validation/ 2>&1 | tee log/CC_validation_pred.log

echo "BP TEST"
python 001_predict.py --pdb_dir pdb_splits/BP/test/ -ont bp -o results/BP/test/ 2>&1 | tee log/BP_tesst_pred.log

echo "BP VAL"
python 001_predict.py --pdb_dir pdb_splits/BP/validation/ -ont bp -o results/BP/validation/ 2>&1 | tee log/BP_validation_pred.log
