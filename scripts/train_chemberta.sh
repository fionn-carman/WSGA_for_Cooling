#!/bin/bash
#PBS -N chemberta_single_temp
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -J 0-5
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# ChemBERTa-2 Single-Temp Training (Optuna-tuned)
# ============================================================
#
# Fine-tunes DeepChem/ChemBERTa-77M-MTR for 6 thermophysical
# properties at 40C with:
#   - Optuna HP tuning (30 trials/fold, 15% inner holdout)
#   - Hash-based 5-fold CV (same folds as XGBoost/MLP/Chemprop/UniMol)
#   - Heteroscedastic Gaussian NLL loss
#   - StandardScaler on y, ReduceLROnPlateau scheduler
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
#   qsub train_chemberta.sh          # all 6 in parallel
#   qsub -J 0-0 train_chemberta.sh   # single test (density)
# ============================================================

PROPERTIES=(density viscosity tc cpsat beta fom1)
PROP=${PROPERTIES[$PBS_ARRAY_INDEX]}

PROJ_DIR="/rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling"
TRAIN_DIR="${PROJ_DIR}/training/ChemBERTa/nist8100"
LOG_DIR="${TRAIN_DIR}/logs"

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/${PROP}_40C_${PBS_JOBID}.log"

exec > "$LOGFILE" 2>&1
echo "=== ChemBERTa-2 single-temp training: ${PROP} ==="
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
    --n_trials 30 \
    --timeout 3600 \
    --epochs 100 \
    --batch_size 16 \
    --patience 15

echo ""
echo "End: $(date)"
echo "=== Done ==="
