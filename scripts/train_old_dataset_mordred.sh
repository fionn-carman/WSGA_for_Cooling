#!/bin/bash
#PBS -N xgb_old_dataset
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 0-5
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# XGBoost Old Dataset Training (Mordred + Optuna)
# ============================================================
#
# Trains 6 property models on the original ~833 molecule dataset
# for direct comparison with NIST 8100 results. Identical methodology.
#
# Array indices:
#   0 = density   (833 molecules)
#   1 = viscosity  (831 molecules)
#   2 = tc         (843 molecules)
#   3 = cpsat      (661 molecules)
#   4 = beta       (832 molecules, linear approx from density 40C/100C)
#   5 = fom1       (638 molecules)
#
# Usage:
#   qsub train_old_dataset_mordred.sh          # all 6 in parallel
#   qsub -J 0-0 train_old_dataset_mordred.sh   # single test (density)
# ============================================================

PROPERTIES=(density viscosity tc cpsat beta fom1)
PROP=${PROPERTIES[$PBS_ARRAY_INDEX]}

PROJ_DIR="/rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling"
TRAIN_DIR="${PROJ_DIR}/training/XGBoost/old_dataset"
LOG_DIR="${TRAIN_DIR}/logs"

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/${PROP}_40C_${PBS_JOBID}.log"

exec > "$LOGFILE" 2>&1
echo "=== XGBoost old dataset training: ${PROP} ==="
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
