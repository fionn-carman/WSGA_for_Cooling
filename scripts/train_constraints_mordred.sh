#!/bin/bash
#PBS -N xgb_constraints
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 0-3
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# XGBoost Constraint Training (Mordred + Optuna)
# ============================================================
#
# Trains 4 constraint property models with:
#   - Mordred + RDKit descriptors
#   - Optuna nested CV (100 trials/fold, 5 outer folds)
#   - SHAP analysis + publication-quality plots
#
# Array indices:
#   0 = dc  (dielectric constant, 529 molecules)
#   1 = mp  (melting point, 2823 molecules)
#   2 = bp  (boiling point, 2504 molecules)
#   3 = fp  (flash point, 3302 molecules)
#
# Usage:
#   qsub train_constraints_mordred.sh          # all 4 in parallel
#   qsub -J 0-0 train_constraints_mordred.sh   # single test (dc)
# ============================================================

PROPERTIES=(dc mp bp fp)
PROP=${PROPERTIES[$PBS_ARRAY_INDEX]}

PROJ_DIR="/rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling"
TRAIN_DIR="${PROJ_DIR}/training/XGBoost/constraints"
LOG_DIR="${TRAIN_DIR}/logs"

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/${PROP}_${PBS_JOBID}.log"

exec > "$LOGFILE" 2>&1
echo "=== XGBoost constraint training: ${PROP} ==="
echo "Job ID: ${PBS_JOBID}"
echo "Array index: ${PBS_ARRAY_INDEX}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$TRAIN_DIR"

python train_single_temp.py \
    --properties "$PROP" \
    --n_trials 100 \
    --timeout 600

echo ""
echo "End: $(date)"
echo "=== Done ==="
