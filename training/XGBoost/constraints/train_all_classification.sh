#!/bin/bash
#PBS -N train_classification
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=16:mem=64gb

# ============================================================
# Hybrid Training Script for Biodegradability Classification
# ============================================================
#
# Trains the biodegradability classification model.
# (Tox21 toxicity prediction now uses pretrained PaccMann MCA from GT4SD.)
#
# Usage:
#   qsub train_all_classification.sh
#
# Or run locally:
#   bash train_all_classification.sh
#
# ============================================================

# Check if running locally or on cluster
if [ -n "$PBS_O_WORKDIR" ]; then
    DIRECTORY="$PBS_O_WORKDIR"
    JOB_ID=${PBS_JOBID}
else
    DIRECTORY=$(pwd)
    JOB_ID="local"
fi

# Activate Conda Environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$DIRECTORY"

# ==============================
# Hybrid method settings
# ==============================
N_ITER=10000        # RandomizedSearchCV iterations
RFE_REPEATS=3       # RFE repeats for stability
STEP_SIZE=2         # Feature removal step size

target_name="Activity"
data_file="biodegradability_cleaned.csv"
output_subdir="biodegradability"

# Build command arguments
cmd_args="--data_dir ../data/constraints --data_file $data_file --target $target_name --outdir ./output/$output_subdir --n_random_iter $N_ITER --n_rfe_repeats $RFE_REPEATS --rfe_step_size $STEP_SIZE"

# Log file
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
echo "========================================" | tee -a "$log_file"

# ===================
# Run training script
# ===================
python3 train_classification.py $cmd_args 2>&1 | tee -a "$log_file"

exit_status=$?

echo "========================================" | tee -a "$log_file"
echo "Training completed at: $(date)" | tee -a "$log_file"
echo "Exit status: ${exit_status}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

echo "" | tee -a "$log_file"
echo "Output locations:" | tee -a "$log_file"
echo "  Model:    ./output/${output_subdir}/${target_name}/model/xgb_model.joblib" | tee -a "$log_file"
echo "  Summary:  ./output/${output_subdir}/${target_name}/model/model_summary.txt" | tee -a "$log_file"
echo "  Features: ./output/${output_subdir}/${target_name}/RFE/selected_features.txt" | tee -a "$log_file"
echo "  Plots:    ./output/${output_subdir}/${target_name}/plots/" | tee -a "$log_file"

# Clean up PBS output files
if [ "$JOB_ID" != "local" ]; then
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'[' -f1 | cut -d'.' -f1)
    rm -f "${DIRECTORY}/train_classification.e${BASE_JOB_ID}" 2>/dev/null
    rm -f "${DIRECTORY}/train_classification.o${BASE_JOB_ID}" 2>/dev/null
fi

exit $exit_status
