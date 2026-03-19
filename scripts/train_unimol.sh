#!/bin/bash
#PBS -N unimol_single_temp
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -J 0-5
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Uni-Mol v1 Single-Temp Training
# ============================================================
#
# Trains 6 property models at 40C with:
#   - Uni-Mol v1 (3D molecular representation)
#   - Hash-based 5-fold CV (same folds as XGBoost/MLP/Chemprop)
#   - 150 epochs, early stopping patience 30
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
#   qsub train_unimol.sh          # all 6 in parallel
#   qsub -J 0-0 train_unimol.sh   # single test (density)
# ============================================================

PROPERTIES=(density viscosity tc cpsat beta fom1)
PROP=${PROPERTIES[$PBS_ARRAY_INDEX]}

PROJ_DIR="/rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling"
TRAIN_DIR="${PROJ_DIR}/training/UniMol/nist8100"
LOG_DIR="${TRAIN_DIR}/logs"

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/${PROP}_40C_${PBS_JOBID}.log"

exec > "$LOGFILE" 2>&1
echo "=== Uni-Mol v1 single-temp training: ${PROP} ==="
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
    --epochs 150 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --early_stopping 30 \
    --warmup_ratio 0.06

echo ""
echo "End: $(date)"
echo "=== Done ==="
