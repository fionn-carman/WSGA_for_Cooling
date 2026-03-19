#!/bin/bash
#PBS -N mlp_single_temp
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 0-5
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# MLP Single-Temp Training (Mordred + Optuna)
# ============================================================
#
# Trains 6 property models at 40C with:
#   - Mordred + RDKit descriptors (1403 features)
#   - Optuna nested CV (100 trials/fold, 5 outer folds)
#   - Permutation importance + publication-quality plots
#
# Array indices:
#   0 = density
#   1 = viscosity
#   2 = tc
#   3 = cpsat
#   4 = beta
#   5 = fom1
#
# Usage:
#   qsub train_mlp_single_temp.sh          # all 6 in parallel
#   qsub -J 0-0 train_mlp_single_temp.sh   # single test (density)
# ============================================================

PROPERTIES=(density viscosity tc cpsat beta fom1)
PROP=${PROPERTIES[$PBS_ARRAY_INDEX]}

PROJ_DIR="/rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling"
TRAIN_DIR="${PROJ_DIR}/training/MLP/nist8100"
LOG_DIR="${TRAIN_DIR}/logs"

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/${PROP}_40C_${PBS_JOBID}.log"

exec > "$LOGFILE" 2>&1
echo "=== MLP single-temp training: ${PROP} ==="
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
