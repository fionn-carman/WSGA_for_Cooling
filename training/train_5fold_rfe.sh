#!/bin/bash
#PBS -N train_5fold_rfe
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=16:mem=64gb
#PBS -J 0-11

# ============================================================
# 5-Fold XGBoost Ensemble with RFE - All Regression Targets
# ============================================================
#
# Trains 12 regression models (one per PBS array index):
#   0  BP-Measured
#   1  DC_exp
#   2  Density_40C_g_cm^3
#   3  Density_100C_g_cm^3
#   4  flashpoint
#   5  Heat_Capacity_Constant_Pressure_40C_J_K_Mol
#   6  Heat_Capacity_Constant_Pressure_100C_J_K_Mol
#   7  Kinematic_Viscosity_40C            (--log_target)
#   8  Kinematic_Viscosity_100C           (--log_target)
#   9  MP-Measured
#   10 Thermal_Conductivity_40C
#   11 Thermal_Conductivity_100C
#
# Usage:
#   qsub train_5fold_rfe.sh              # Submit all 12 jobs
#   qsub -J 0-0 train_5fold_rfe.sh       # Single test job (BP-Measured)
#   bash train_5fold_rfe.sh 2             # Local test (Density_40C)
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

# Activate conda
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env

cd "$DIRECTORY"

# ==============================
# Target configuration
# ==============================
declare -a targets
declare -a log_flags

targets[0]="BP-Measured"
targets[1]="DC_exp"
targets[2]="Density_40C_g_cm^3"
targets[3]="Density_100C_g_cm^3"
targets[4]="flashpoint"
targets[5]="Heat_Capacity_Constant_Pressure_40C_J_K_Mol"
targets[6]="Heat_Capacity_Constant_Pressure_100C_J_K_Mol"
targets[7]="Kinematic_Viscosity_40C"
targets[8]="Kinematic_Viscosity_100C"
targets[9]="MP-Measured"
targets[10]="Thermal_Conductivity_40C"
targets[11]="Thermal_Conductivity_100C"

# Log-transform flags (viscosity benefits from log transform)
log_flags[0]=""
log_flags[1]=""
log_flags[2]=""
log_flags[3]=""
log_flags[4]=""
log_flags[5]=""
log_flags[6]=""
log_flags[7]="--log_target"
log_flags[8]="--log_target"
log_flags[9]=""
log_flags[10]=""
log_flags[11]=""

# ==============================
# Training settings
# ==============================
N_RANDOM_ITER=1000     # RandomizedSearchCV iterations
N_RFE_REPEATS=3        # RFE repeats per fold
RFE_STEP_SIZE=2        # Features dropped per RFE step

target_name=${targets[$N]}
log_flag=${log_flags[$N]}

# Build command
cmd_args="--target $target_name --data_dir ./data --outdir ./output_5fold --n_random_iter $N_RANDOM_ITER --n_rfe_repeats $N_RFE_REPEATS --rfe_step_size $RFE_STEP_SIZE $log_flag"

# Log file
mkdir -p "./output_5fold/${target_name}"
log_file="./output_5fold/${target_name}/training.log"

echo "========================================" | tee "$log_file"
echo "5-Fold RFE Training: ${target_name}" | tee -a "$log_file"
echo "Log transform: ${log_flag:-none}" | tee -a "$log_file"
echo "RandomizedSearchCV: ${N_RANDOM_ITER} iter" | tee -a "$log_file"
echo "RFE repeats: ${N_RFE_REPEATS}" | tee -a "$log_file"
echo "Started: $(date)" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

python3 train_5fold_rfe.py $cmd_args 2>&1 | tee -a "$log_file"

exit_status=$?

echo "========================================" | tee -a "$log_file"
echo "Completed: $(date)" | tee -a "$log_file"
echo "Exit status: ${exit_status}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# Clean up PBS output files
if [ "$JOB_ID" != "local" ]; then
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'[' -f1 | cut -d'.' -f1)
    rm -f "${DIRECTORY}/train_5fold_rfe.e${BASE_JOB_ID}.${N}" 2>/dev/null
    rm -f "${DIRECTORY}/train_5fold_rfe.o${BASE_JOB_ID}.${N}" 2>/dev/null
fi

exit $exit_status
