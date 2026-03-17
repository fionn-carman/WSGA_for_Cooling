#!/bin/bash
#PBS -N retrain_expanded
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=16:mem=64gb
#PBS -J 0-11

# ============================================================
# Retrain All Property Models (5-Fold Ensemble) with Expanded Data
# ============================================================
#
# Uses train_5fold_rfe.py (5-fold ensemble with RFE + StandardScaler)
# on expanded/cleaned training data:
#   0-1:  Density 40/100C — WTT expansion + corrections
#   2-3:  Kinematic Viscosity 40/100C — WTT expansion (log)
#   4-5:  Thermal Conductivity 40/100C — WTT expansion
#   6-7:  Heat Capacity 40/100C — WTT expansion (log)
#   8:    Flash point — canonicalized SMILES + median dedup
#   9:    DC — canonicalized SMILES + dedup
#   10-11: FOM1 40/100 — WTT expansion, new descriptors computed
#
# Usage:
#   qsub retrain_expanded.sh              # Submit all 12 jobs
#   qsub -J 0-0 retrain_expanded.sh       # Submit only Density_40C
#
# Or run locally:
#   bash retrain_expanded.sh 0            # Run Density_40C
#
# ============================================================

# Check if running locally (with argument) or on cluster
if [ $# -eq 1 ]; then
    # Running locally with target index as argument
    N=$1
    DIRECTORY=$(pwd)
    JOB_ID="local"
else
    # Running on cluster via PBS
    DIRECTORY="$PBS_O_WORKDIR"
    N=${PBS_ARRAY_INDEX}
    JOB_ID=${PBS_JOBID}
fi

# Activate Conda Environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env

cd "$DIRECTORY"

# ==============================
# Define all regression targets
# ==============================
declare -a targets
declare -a log_flags

# Target names (12 total)
targets[0]="Density_40C_g_cm^3"
targets[1]="Density_100C_g_cm^3"
targets[2]="Kinematic_Viscosity_40C"
targets[3]="Kinematic_Viscosity_100C"
targets[4]="Thermal_Conductivity_40C"
targets[5]="Thermal_Conductivity_100C"
targets[6]="Heat_Capacity_Constant_Pressure_40C_J_K_Mol"
targets[7]="Heat_Capacity_Constant_Pressure_100C_J_K_Mol"
targets[8]="flashpoint"
targets[9]="DC_exp"
targets[10]="FOM1_exp_40"
targets[11]="FOM1_exp_100"

# Log-transform flags
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
log_flags[10]=""
log_flags[11]=""

# ==============================
# Training settings
# ==============================
N_RANDOM_ITER=1000     # RandomizedSearchCV iterations
N_RFE_REPEATS=3        # RFE repeats per fold
RFE_STEP_SIZE=2        # Features dropped per RFE step
OUTDIR="./output_5fold_retrain"

# =================================
# Get current target configuration
# =================================
target_name=${targets[$N]}
log_flag=${log_flags[$N]}

# Build command
cmd_args="--target $target_name --data_dir ./data --outdir $OUTDIR --n_random_iter $N_RANDOM_ITER --n_rfe_repeats $N_RFE_REPEATS --rfe_step_size $RFE_STEP_SIZE $log_flag"

# Log file
mkdir -p "${OUTDIR}/${target_name}"
log_file="${OUTDIR}/${target_name}/training.log"

echo "========================================" | tee "$log_file"
echo "5-Fold RFE Retrain (Expanded Data): ${target_name}" | tee -a "$log_file"
echo "Log transform: ${log_flag:-none}" | tee -a "$log_file"
echo "RandomizedSearchCV: ${N_RANDOM_ITER} iter" | tee -a "$log_file"
echo "RFE repeats: ${N_RFE_REPEATS}" | tee -a "$log_file"
echo "Step size: ${RFE_STEP_SIZE}" | tee -a "$log_file"
echo "Job started at: $(date)" | tee -a "$log_file"
echo "Working directory: $(pwd)" | tee -a "$log_file"
echo "Data directory: ./data" | tee -a "$log_file"
echo "Output directory: ${OUTDIR}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# ===================
# Run training script
# ===================
python3 train_5fold_rfe.py $cmd_args 2>&1 | tee -a "$log_file"

# Capture exit status
exit_status=$?

echo "========================================" | tee -a "$log_file"
echo "Training completed at: $(date)" | tee -a "$log_file"
echo "Exit status: ${exit_status}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# ===================
# Clean up PBS output files
# ===================
if [ "$JOB_ID" != "local" ]; then
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'[' -f1 | cut -d'.' -f1)
    rm -f "${DIRECTORY}/retrain_expanded.e${BASE_JOB_ID}.${N}" 2>/dev/null
    rm -f "${DIRECTORY}/retrain_expanded.o${BASE_JOB_ID}.${N}" 2>/dev/null
    echo "Cleaned up PBS output files" | tee -a "$log_file"
fi

exit $exit_status
