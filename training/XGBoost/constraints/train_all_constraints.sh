#!/bin/bash
#PBS -N train_constraints
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=16:mem=64gb
#PBS -J 0-3

# ============================================================
# Training Script for Constraint Regression Models
# ============================================================
#
# Trains XGBoost regression models for safety/constraint properties:
#   0: BP-Measured     (boiling point)
#   1: MP-Measured     (melting point)
#   2: flashpoint      (flash point)
#   3: DC_exp          (dielectric constant)
#
# For biodegradability classification, use train_all_classification.sh
#
# Usage:
#   qsub train_all_constraints.sh              # Submit all 4 jobs
#   qsub -J 0-0 train_all_constraints.sh       # Submit only BP
#   bash train_all_constraints.sh 0            # Run BP locally
#
# ============================================================

if [ $# -eq 1 ]; then
    N=$1
    DIRECTORY=$(pwd)
    JOB_ID="local"
else
    DIRECTORY="$PBS_O_WORKDIR"
    N=${PBS_ARRAY_INDEX}
    JOB_ID=${PBS_JOBID}
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$DIRECTORY"

# ==============================
# Define constraint targets
# ==============================
declare -a targets
declare -a log_flags

targets[0]="BP-Measured"
targets[1]="MP-Measured"
targets[2]="flashpoint"
targets[3]="DC_exp"

log_flags[0]=0  # BP
log_flags[1]=0  # MP
log_flags[2]=0  # FP
log_flags[3]=0  # DC

# ==============================
# Training settings
# ==============================
N_ITER=10000
RFE_REPEATS=3
STEP_SIZE=2

# =================================
# Get current target configuration
# =================================
target_name=${targets[$N]}
use_log=${log_flags[$N]}

cmd_args="--data_dir ../data/constraints --target $target_name --outdir ./output --n_random_iter $N_ITER --n_rfe_repeats $RFE_REPEATS --rfe_step_size $STEP_SIZE"

if [ $use_log -eq 1 ]; then
    cmd_args="$cmd_args --log_target"
fi

mkdir -p "./output"
log_file="./output/${target_name}_training.log"

echo "========================================" | tee "$log_file"
echo "Starting CONSTRAINT training for: ${target_name}" | tee -a "$log_file"
echo "Job started at: $(date)" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

python3 train_regression.py $cmd_args 2>&1 | tee -a "$log_file"

exit_status=$?

echo "========================================" | tee -a "$log_file"
echo "Training completed at: $(date)" | tee -a "$log_file"
echo "Exit status: ${exit_status}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

if [ "$JOB_ID" != "local" ]; then
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'[' -f1 | cut -d'.' -f1)
    rm -f "${DIRECTORY}/train_constraints.e${BASE_JOB_ID}.${N}" 2>/dev/null
    rm -f "${DIRECTORY}/train_constraints.o${BASE_JOB_ID}.${N}" 2>/dev/null
fi

exit $exit_status
