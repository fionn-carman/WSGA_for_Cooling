#!/bin/bash
#PBS -N chno_train
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -J 0-9
#PBS -o /dev/null
#PBS -e /dev/null

# CHNO XGBoost model training (10 properties)
# Array indices:
#   0-5: thermophysical (density, viscosity, tc, cpsat, beta, fom1)
#   6-9: constraints (fp, bp, mp, dc)
#
# Usage:
#   qsub train_chno_models.sh          # all 10 jobs
#   qsub -J 0-0 train_chno_models.sh   # single test job (density)
#   bash train_chno_models.sh 0         # local test

# --- Conda activation ---
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

# --- Array index ---
if [ -n "$PBS_ARRAY_INDEX" ]; then
    IDX=$PBS_ARRAY_INDEX
else
    IDX=${1:-0}
fi

# --- Map index to property ---
PROPERTIES=("density" "viscosity" "tc" "cpsat" "beta" "fom1" "fp" "bp" "mp" "dc")
PROP=${PROPERTIES[$IDX]}

if [ -z "$PROP" ]; then
    echo "ERROR: Invalid array index $IDX (max 9)"
    exit 1
fi

echo "=== CHNO Training: $PROP (index $IDX) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# --- Navigate to training directory ---
cd ~/mol-rl/training/CHNO/ || { echo "ERROR: Cannot cd to training/CHNO/"; exit 1; }

# --- Log file ---
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${PROP}_$(date +%Y%m%d_%H%M%S).log"

# --- Train ---
python train_single_temp.py \
    --properties "$PROP" \
    --n_trials 100 \
    --timeout 600 \
    2>&1 | tee "$LOGFILE"

echo "=== Done: $PROP ($(date)) ==="
