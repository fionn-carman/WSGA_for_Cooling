#!/bin/bash
#PBS -N chemprop_single_temp
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -J 0-5
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Chemprop D-MPNN Single-Temp Training
# ============================================================
#
# Trains 6 property models at 40C with:
#   - D-MPNN via chemprop v1 CLI (hidden_size=300, depth=3)
#   - Hash-based 5-fold CV (same folds as XGBoost/MLP)
#   - 100 epochs, early stopping on val RMSE
#   - Publication-quality parity + diagnostics plots
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
#   qsub train_chemprop.sh          # all 6 in parallel
#   qsub -J 0-0 train_chemprop.sh   # single test (density)
# ============================================================

PROPERTIES=(density viscosity tc cpsat beta fom1)
PROP=${PROPERTIES[$PBS_ARRAY_INDEX]}

PROJ_DIR="/rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling"
TRAIN_DIR="${PROJ_DIR}/training/Chemprop/nist8100"
LOG_DIR="${TRAIN_DIR}/logs"

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/${PROP}_40C_${PBS_JOBID}.log"

exec > "$LOGFILE" 2>&1
echo "=== Chemprop D-MPNN single-temp training: ${PROP} ==="
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
    --epochs 100 \
    --patience 20 \
    --batch_size 64 \
    --lr 1e-4

echo ""
echo "End: $(date)"
echo "=== Done ==="
