#!/bin/bash
#PBS -N retrain_fom1_xgb
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Retrain XGBoost+Descriptors FOM1 models
# ============================================================
#
# Runs data_preparation.py (computes descriptors, creates folds)
# then train_xgboost_descriptors.py (trains 5-fold XGBoost) for
# both FOM1_40 and FOM1_100.
#
# The trained models embed descriptor_columns in the joblib so
# they are self-contained and immune to descriptor_columns.json
# being regenerated with a different RDKit version.
#
# Usage:
#   qsub retrain_xgb_descriptors.sh        # HPC
#   bash retrain_xgb_descriptors.sh         # local
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"

if [ -z "$DIRECTORY" ]; then
    DIRECTORY=$(cd "$(dirname "$0")" && pwd)
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

log_file="${LOG_DIR}/retrain_xgb_descriptors_$(date +%Y%m%d_%H%M%S).log"

{
echo "=========================================="
echo "Retrain XGBoost+Descriptors FOM1 Models"
echo "=========================================="
echo "Started at: $(date)"
echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "RDKit version: $(python3 -c 'from rdkit import rdBase; print(rdBase.rdkitVersion)')"
echo "=========================================="

# Step 1: Data preparation for FOM1_40
echo ""
echo "=== Step 1a: Data preparation (FOM1_40) ==="
python3 data_preparation.py --target FOM1_40 --output_dir ./results
if [ $? -ne 0 ]; then
    echo "ERROR: data_preparation.py failed for FOM1_40"
    exit 1
fi

# Step 2: Data preparation for FOM1_100
echo ""
echo "=== Step 1b: Data preparation (FOM1_100) ==="
python3 data_preparation.py --target FOM1_100 --output_dir ./results
if [ $? -ne 0 ]; then
    echo "ERROR: data_preparation.py failed for FOM1_100"
    exit 1
fi

# Step 3: Train XGBoost for FOM1_40 (all 5 folds)
echo ""
echo "=== Step 2a: Train XGBoost Descriptors (FOM1_40, all folds) ==="
python3 train_xgboost_descriptors.py \
    --data_dir ./results/fom1_40 \
    --target FOM1_40 \
    --n_iter 200 \
    --seed 42
if [ $? -ne 0 ]; then
    echo "ERROR: train_xgboost_descriptors.py failed for FOM1_40"
    exit 1
fi

# Step 4: Train XGBoost for FOM1_100 (all 5 folds)
echo ""
echo "=== Step 2b: Train XGBoost Descriptors (FOM1_100, all folds) ==="
python3 train_xgboost_descriptors.py \
    --data_dir ./results/fom1_100 \
    --target FOM1_100 \
    --n_iter 200 \
    --seed 42
if [ $? -ne 0 ]; then
    echo "ERROR: train_xgboost_descriptors.py failed for FOM1_100"
    exit 1
fi

echo ""
echo "=========================================="
echo "All training complete!"
echo "Finished at: $(date)"
echo ""
echo "Model files:"
ls -la ./results/fom1_40/xgboost_descriptors_fold*_model.joblib
ls -la ./results/fom1_100/xgboost_descriptors_fold*_model.joblib
echo ""
echo "Descriptor columns:"
python3 -c "
import json
for t in ['fom1_40', 'fom1_100']:
    with open(f'results/{t}/descriptor_columns.json') as f:
        cols = json.load(f)
    print(f'  {t}: {len(cols)} descriptors')
"
echo ""
echo "Metrics summary:"
python3 -c "
import json
for t in ['fom1_40', 'fom1_100']:
    r2s = []
    for fold in range(5):
        with open(f'results/{t}/xgboost_descriptors_fold{fold}_metrics.json') as f:
            m = json.load(f)
        r2s.append(m['r2'])
    import numpy as np
    print(f'  {t}: R² = {np.mean(r2s):.4f} ± {np.std(r2s):.4f} (folds: {[\"{:.4f}\".format(r) for r in r2s]})')
"
echo "=========================================="

} 2>&1 | tee "$log_file"

echo ""
echo "Log saved to: $log_file"
