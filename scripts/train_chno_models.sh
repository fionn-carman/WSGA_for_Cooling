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

PROPERTIES=(density viscosity tc cpsat beta fom1 fp bp mp dc)
PROP=${PROPERTIES[$PBS_ARRAY_INDEX]}

PROJ_DIR="/rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling"
TRAIN_DIR="${PROJ_DIR}/training/CHNO"
LOG_DIR="${TRAIN_DIR}/logs"

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/${PROP}_${PBS_JOBID}.log"

exec > "$LOGFILE" 2>&1
echo "=== CHNO Training: ${PROP} ==="
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
echo "=== Done: ${PROP} ($(date)) ==="
