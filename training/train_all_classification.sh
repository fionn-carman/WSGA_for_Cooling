#!/bin/bash
#PBS -N train_classification
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=16:mem=64gb
#PBS -J 0-12

# ============================================================
# Hybrid Training Script for All Classification Targets
# ============================================================
#
# Trains 13 classification models:
#   - 12 Tox21 targets (indices 0-11)
#   - 1 Biodegradability target (index 12)
#
# Combines proven methods (RFE with repeats, RandomizedSearchCV 10K)
# with modern improvements (SHAP, comprehensive plots)
#
# Usage:
#   qsub train_all_classification.sh              # Submit all 13 jobs
#   qsub -J 0-11 train_all_classification.sh      # Submit only Tox21 jobs
#   qsub -J 12-12 train_all_classification.sh     # Submit only Biodegradability
#
# Or run locally:
#   bash train_all_classification.sh 0            # Run NR-AR
#   bash train_all_classification.sh 12           # Run Biodegradability
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
# Define all classification targets
# ==============================
declare -a targets
declare -a data_files

# Tox21 targets (indices 0-11)
targets[0]="NR-AR"
targets[1]="NR-AR-LBD"
targets[2]="NR-AhR"
targets[3]="NR-Aromatase"
targets[4]="NR-ER"
targets[5]="NR-ER-LBD"
targets[6]="NR-PPAR-gamma"
targets[7]="SR-ARE"
targets[8]="SR-ATAD5"
targets[9]="SR-HSE"
targets[10]="SR-MMP"
targets[11]="SR-p53"

# Biodegradability target (index 12)
targets[12]="Activity"

# Data files for each target
data_files[0]="tox21_cleaned.csv"
data_files[1]="tox21_cleaned.csv"
data_files[2]="tox21_cleaned.csv"
data_files[3]="tox21_cleaned.csv"
data_files[4]="tox21_cleaned.csv"
data_files[5]="tox21_cleaned.csv"
data_files[6]="tox21_cleaned.csv"
data_files[7]="tox21_cleaned.csv"
data_files[8]="tox21_cleaned.csv"
data_files[9]="tox21_cleaned.csv"
data_files[10]="tox21_cleaned.csv"
data_files[11]="tox21_cleaned.csv"
data_files[12]="biodegradability_cleaned.csv"

# ==============================
# Hybrid method settings
# ==============================
N_ITER=10000        # RandomizedSearchCV iterations (proven method)
RFE_REPEATS=3       # RFE repeats for stability (proven method)
STEP_SIZE=2         # Feature removal step size (proven method)

# =================================
# Get current target configuration
# =================================
target_name=${targets[$N]}
data_file=${data_files[$N]}

# Determine output subdirectory based on dataset
if [ "$data_file" == "tox21_cleaned.csv" ]; then
    output_subdir="tox21"
else
    output_subdir="biodegradability"
fi

# Build command arguments
cmd_args="--data_dir ./data --data_file $data_file --target $target_name --outdir ./output/$output_subdir --n_random_iter $N_ITER --n_rfe_repeats $RFE_REPEATS --rfe_step_size $STEP_SIZE"

# Log file for this job
mkdir -p "./output/${output_subdir}"
log_file="./output/${output_subdir}/${target_name}_hybrid_training.log"

echo "========================================" | tee "$log_file"
echo "Starting HYBRID CLASSIFICATION training for: ${target_name}" | tee -a "$log_file"
echo "Dataset: ${data_file}" | tee -a "$log_file"
echo "RandomizedSearchCV iterations: ${N_ITER}" | tee -a "$log_file"
echo "RFE repeats: ${RFE_REPEATS}" | tee -a "$log_file"
echo "Step size: ${STEP_SIZE}" | tee -a "$log_file"
echo "Job started at: $(date)" | tee -a "$log_file"
echo "Working directory: $(pwd)" | tee -a "$log_file"
echo "Data directory: ./data" | tee -a "$log_file"
echo "Output directory: ./output/${output_subdir}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# ===================
# Run training script
# ===================
python3 train_classification.py $cmd_args 2>&1 | tee -a "$log_file"

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
echo "  Model:    ./output/${output_subdir}/${target_name}/model/xgb_model.joblib" | tee -a "$log_file"
echo "  Summary:  ./output/${output_subdir}/${target_name}/model/model_summary.txt" | tee -a "$log_file"
echo "  Features: ./output/${output_subdir}/${target_name}/RFE/selected_features.txt" | tee -a "$log_file"
echo "  Plots:    ./output/${output_subdir}/${target_name}/plots/" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# ===================
# Clean up PBS output files
# ===================
if [ "$JOB_ID" != "local" ]; then
    # Extract the base job ID (without array index)
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'[' -f1 | cut -d'.' -f1)
    
    # Remove the .e and .o files for this array job
    rm -f "${DIRECTORY}/train_classification.e${BASE_JOB_ID}.${N}" 2>/dev/null
    rm -f "${DIRECTORY}/train_classification.o${BASE_JOB_ID}.${N}" 2>/dev/null
    
    echo "Cleaned up PBS output files" | tee -a "$log_file"
fi

exit $exit_status
