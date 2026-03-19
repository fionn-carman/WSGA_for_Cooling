#!/bin/bash
#PBS -N compute_desc_old
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Precompute Mordred + RDKit descriptors for old dataset
# ============================================================
#
# Single job — computes descriptors for all unique SMILES across
# the original ~833 molecule dataset (density, viscosity, tc, cpsat, fom1).
#
# Usage:
#   qsub compute_old_dataset_descriptors.sh
# ============================================================

PROJ_DIR="/rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling"
TRAIN_DIR="${PROJ_DIR}/training/XGBoost/old_dataset"
LOG_DIR="${TRAIN_DIR}/logs"

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/compute_descriptors_${PBS_JOBID}.log"

exec > "$LOGFILE" 2>&1
echo "=== Compute old dataset descriptors (Mordred + RDKit) ==="
echo "Job ID: ${PBS_JOBID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$TRAIN_DIR"

python compute_descriptors.py --force

echo ""
echo "End: $(date)"
echo "=== Done ==="
