#!/bin/bash
#PBS -N train_5fold_wtt
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=16:mem=64gb
#PBS -J 0-9

# ============================================================
# 5-Fold Ensemble Training on NIST WTT Dataset
# ============================================================
#
# Uses train_5fold_rfe.py (5-fold ensemble with RFE + StandardScaler)
# on WTT-only training data in training_expanded/data/:
#
#   0-1:  Density 40/100C
#   2-3:  Kinematic Viscosity 40/100C (log)
#   4-5:  Thermal Conductivity 40/100C
#   6-7:  Cpsat 40/100C (log)
#   8-9:  FOM1_sat 40/100C
#
# Usage:
#   cd training_expanded && qsub train_5fold_wtt.sh
#   qsub -J 0-0 train_5fold_wtt.sh      # Single test job
#
# Local test:
#   cd training_expanded && bash train_5fold_wtt.sh 0
#
# ============================================================

# Check if running locally or on cluster
if [ $# -eq 1 ]; then
    N=$1
    DIRECTORY=$(pwd)
    JOB_ID="local"
else
    DIRECTORY="$PBS_O_WORKDIR"
    N=${PBS_ARRAY_INDEX}
    JOB_ID=${PBS_JOBID}
fi

# Activate Conda Environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$DIRECTORY"

# ==============================
# Define all regression targets
# ==============================
declare -a targets
declare -a log_flags

targets[0]="Density_40C_g_cm^3"
targets[1]="Density_100C_g_cm^3"
targets[2]="Kinematic_Viscosity_40C"
targets[3]="Kinematic_Viscosity_100C"
targets[4]="Thermal_Conductivity_40C"
targets[5]="Thermal_Conductivity_100C"
targets[6]="Cpsat_40C_J_K_Mol"
targets[7]="Cpsat_100C_J_K_Mol"
targets[8]="FOM1_sat_40"
targets[9]="FOM1_sat_100"

log_flags[0]=""
log_flags[1]=""
log_flags[2]="--log_target"
log_flags[3]="--log_target"
log_flags[4]=""
log_flags[5]=""
log_flags[6]="--log_target"
log_flags[7]="--log_target"
log_flags[8]=""
log_flags[9]=""

# ==============================
# Training settings
# ==============================
N_RANDOM_ITER=1000
N_RFE_REPEATS=3
RFE_STEP_SIZE=2
DATA_DIR="./data"
OUTDIR="./output_5fold"

# =================================
# Get current target configuration
# =================================
target_name=${targets[$N]}
log_flag=${log_flags[$N]}

# Build command — use train_5fold_rfe.py from ../training/
cmd_args="--target $target_name --data_dir $DATA_DIR --outdir $OUTDIR --n_random_iter $N_RANDOM_ITER --n_rfe_repeats $N_RFE_REPEATS --rfe_step_size $RFE_STEP_SIZE $log_flag"

# Log file
mkdir -p "${OUTDIR}/${target_name}"
log_file="${OUTDIR}/${target_name}/training.log"

echo "========================================" | tee "$log_file"
echo "5-Fold RFE (WTT Data): ${target_name}" | tee -a "$log_file"
echo "Log transform: ${log_flag:-none}" | tee -a "$log_file"
echo "RandomizedSearchCV: ${N_RANDOM_ITER} iter" | tee -a "$log_file"
echo "RFE repeats: ${N_RFE_REPEATS}" | tee -a "$log_file"
echo "Step size: ${RFE_STEP_SIZE}" | tee -a "$log_file"
echo "Job started at: $(date)" | tee -a "$log_file"
echo "Working directory: $(pwd)" | tee -a "$log_file"
echo "Data directory: ${DATA_DIR}" | tee -a "$log_file"
echo "Output directory: ${OUTDIR}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# ===================
# Run training script
# ===================
python3 ../training/train_5fold_rfe.py $cmd_args 2>&1 | tee -a "$log_file"

exit_status=$?

echo "========================================" | tee -a "$log_file"
echo "Training completed at: $(date)" | tee -a "$log_file"
echo "Exit status: ${exit_status}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# Clean up PBS output files
if [ "$JOB_ID" != "local" ]; then
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'[' -f1 | cut -d'.' -f1)
    rm -f "${DIRECTORY}/train_5fold_wtt.e${BASE_JOB_ID}.${N}" 2>/dev/null
    rm -f "${DIRECTORY}/train_5fold_wtt.o${BASE_JOB_ID}.${N}" 2>/dev/null
fi

exit $exit_status
