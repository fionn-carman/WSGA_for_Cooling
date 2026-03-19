#!/bin/bash
#PBS -N xgb_biodeg
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# XGBoost Biodegradability Classification (Mordred + Optuna)
# ============================================================
#
# Trains biodegradability binary classifier with:
#   - Mordred + RDKit descriptors
#   - Optuna nested CV (100 trials/fold, 5 outer folds, ROC-AUC)
#   - SHAP analysis + publication-quality plots (ROC, PR, confusion)
#
# 1888 molecules (31.8% positive)
#
# Usage:
#   qsub train_biodeg_mordred.sh
# ============================================================

PROJ_DIR="/rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling"
TRAIN_DIR="${PROJ_DIR}/training/XGBoost/constraints"
LOG_DIR="${TRAIN_DIR}/logs"

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/biodegradability_${PBS_JOBID}.log"

exec > "$LOGFILE" 2>&1
echo "=== XGBoost biodegradability classification ==="
echo "Job ID: ${PBS_JOBID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$TRAIN_DIR"

python train_classification.py \
    --properties biodegradability \
    --n_trials 100 \
    --timeout 600

echo ""
echo "End: $(date)"
echo "=== Done ==="
