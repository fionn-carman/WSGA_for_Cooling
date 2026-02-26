#!/bin/bash
#PBS -N train_shared
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=16:mem=64gb
#PBS -J 0-9

# ============================================================
# Shared Test Set Training — All 10 Targets
# ============================================================
#
# Trains XGBoost models for 8 thermophysical properties + 2
# direct FOM1 targets using a common train/test split so that
# predictions on the same held-out molecules can be compared.
#
# Phase 1 — Array job (this script):
#   qsub train_shared_test.sh              # Submit all 10 jobs
#   qsub -J 0-0 train_shared_test.sh       # Submit only Density_40C
#
# Phase 2 — FOM1 comparison (run after all 10 complete):
#   python3 train_shared_test.py \
#       --data_dir ./data \
#       --outdir ./output_shared \
#       --compare_only
#
# Or run locally:
#   bash train_shared_test.sh 0            # Run Density_40C
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
# Target configuration (10 total)
# ==============================
declare -a targets
declare -a log_flags
declare -a aux_features

targets[0]="Density_40C_g_cm^3"
targets[1]="Density_100C_g_cm^3"
targets[2]="Kinematic_Viscosity_40C"
targets[3]="Kinematic_Viscosity_100C"
targets[4]="Thermal_Conductivity_40C"
targets[5]="Thermal_Conductivity_100C"
targets[6]="Heat_Capacity_Constant_Pressure_40C_J_K_Mol"
targets[7]="Heat_Capacity_Constant_Pressure_100C_J_K_Mol"
targets[8]="FOM1_exp_40"
targets[9]="FOM1_exp_100"

# Log transform flags (1 = use log, 0 = don't)
log_flags[0]=0  # Density_40C
log_flags[1]=0  # Density_100C
log_flags[2]=1  # Viscosity_40C (log)
log_flags[3]=1  # Viscosity_100C (log)
log_flags[4]=0  # Thermal_Conductivity_40C
log_flags[5]=0  # Thermal_Conductivity_100C
log_flags[6]=1  # Heat_Capacity_40C (log)
log_flags[7]=1  # Heat_Capacity_100C (log)
log_flags[8]=0  # FOM1_exp_40
log_flags[9]=0  # FOM1_exp_100

# Auxiliary features
aux_features[0]=""                      # Density_40C - no auxiliary
aux_features[1]="Density_40C_g_cm^3"   # Density_100C - use Density_40C
aux_features[2]=""                      # Viscosity_40C - no auxiliary
aux_features[3]=""                      # Viscosity_100C - no auxiliary
aux_features[4]=""                      # Thermal_Conductivity_40C - no auxiliary
aux_features[5]=""                      # Thermal_Conductivity_100C - no auxiliary
aux_features[6]=""                      # Heat_Capacity_40C - no auxiliary
aux_features[7]=""                      # Heat_Capacity_100C - no auxiliary
aux_features[8]=""                      # FOM1_exp_40 - no auxiliary
aux_features[9]=""                      # FOM1_exp_100 - no auxiliary

# ==============================
# Training settings
# ==============================
N_ITER=10000        # RandomizedSearchCV iterations
RFE_REPEATS=3       # RFE repeats for stability
STEP_SIZE=2         # Feature removal step size
OUTPUT_DIR="./output_shared"

# =================================
# Get current target configuration
# =================================
target_name=${targets[$N]}
use_log=${log_flags[$N]}
aux_feature=${aux_features[$N]}

# Build command arguments
cmd_args="--data_dir ./data --outdir $OUTPUT_DIR --target_index $N --n_random_iter $N_ITER --n_rfe_repeats $RFE_REPEATS --rfe_step_size $STEP_SIZE"

# Log file for this job
mkdir -p "$OUTPUT_DIR"
log_file="${OUTPUT_DIR}/${target_name}_shared_training.log"

echo "========================================" | tee "$log_file"
echo "Starting SHARED TEST SET training for: ${target_name}" | tee -a "$log_file"
echo "Target index: ${N}" | tee -a "$log_file"
echo "RandomizedSearchCV iterations: ${N_ITER}" | tee -a "$log_file"
echo "RFE repeats: ${RFE_REPEATS}" | tee -a "$log_file"
echo "Step size: ${STEP_SIZE}" | tee -a "$log_file"
echo "Log transform: ${use_log}" | tee -a "$log_file"
if [ -n "$aux_feature" ]; then
    echo "Auxiliary feature: ${aux_feature}" | tee -a "$log_file"
fi
echo "Job started at: $(date)" | tee -a "$log_file"
echo "Working directory: $(pwd)" | tee -a "$log_file"
echo "Data directory: ./data" | tee -a "$log_file"
echo "Output directory: ${OUTPUT_DIR}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# ===================
# Run training script
# ===================
python3 train_shared_test.py $cmd_args 2>&1 | tee -a "$log_file"

# Capture exit status
exit_status=$?

echo "========================================" | tee -a "$log_file"
echo "Training completed at: $(date)" | tee -a "$log_file"
echo "Exit status: ${exit_status}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# ===================
# Summary of outputs
# ===================
echo "" | tee -a "$log_file"
echo "Output locations:" | tee -a "$log_file"
echo "  Model:       ${OUTPUT_DIR}/${target_name}/model/xgb_model.joblib" | tee -a "$log_file"
echo "  Summary:     ${OUTPUT_DIR}/${target_name}/model/model_summary.txt" | tee -a "$log_file"
echo "  Features:    ${OUTPUT_DIR}/${target_name}/RFE/selected_features.txt" | tee -a "$log_file"
echo "  Plots:       ${OUTPUT_DIR}/${target_name}/plots/" | tee -a "$log_file"
echo "  Test preds:  ${OUTPUT_DIR}/${target_name}/test_predictions.csv" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# ===================
# Clean up PBS output files
# ===================
if [ "$JOB_ID" != "local" ]; then
    # Extract the base job ID (without array index)
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'[' -f1 | cut -d'.' -f1)

    # Remove the .e and .o files for this array job
    rm -f "${DIRECTORY}/train_shared.e${BASE_JOB_ID}.${N}" 2>/dev/null
    rm -f "${DIRECTORY}/train_shared.o${BASE_JOB_ID}.${N}" 2>/dev/null

    echo "Cleaned up PBS output files" | tee -a "$log_file"
fi

exit $exit_status
