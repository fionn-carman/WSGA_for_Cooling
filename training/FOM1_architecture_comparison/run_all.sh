#!/bin/bash
#PBS -N fom1_arch
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=16:mem=64gb
#PBS -J 0-79

# ============================================================
# FOM1 Architecture Comparison — All Temperatures + Folds
# ============================================================
#
# 80 jobs total: 8 architectures × 2 temperatures × 5 folds
#
# Index decoding:
#   arch_idx  = N / 10       (0..7)
#   temp_idx  = (N % 10) / 5 (0 or 1)
#   fold_idx  = N % 5        (0..4)
#
# Submit all 80 jobs:
#   qsub run_all.sh
#
# Submit a subset (e.g. XGBoost only, both temps):
#   qsub -J 0-9 run_all.sh
#
# Run locally (e.g. XGBoost / FOM1_40 / fold 0):
#   bash run_all.sh 0
#
# After all jobs complete, run comparison:
#   python compare_all.py --target FOM1_40
#   python compare_all.py --target FOM1_100
#   python compare_all.py  # combined cross-temperature view
#
# ============================================================

# Check if running locally (with argument) or on cluster
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
# Architecture configuration
# ==============================
ARCHITECTURES=(
    xgboost_baseline
    mlp_descriptors
    mlp_fingerprints
    mlp_combined
    chemberta
    ridge_embeddings
    bilstm
    ensemble_mlp
)

TARGETS=(FOM1_40 FOM1_100)
TARGET_KEYS=(fom1_40 fom1_100)

# ==============================
# Decode job index
# ==============================
arch_idx=$((N / 10))
remainder=$((N % 10))
temp_idx=$((remainder / 5))
fold_idx=$((remainder % 5))

arch_name=${ARCHITECTURES[$arch_idx]}
target=${TARGETS[$temp_idx]}
target_key=${TARGET_KEYS[$temp_idx]}

script="train_${arch_name}.py"

# ==============================
# Validate
# ==============================
if [ ! -f "$script" ]; then
    echo "ERROR: Script not found: $script"
    exit 1
fi

data_dir="./results/${target_key}"
if [ ! -d "$data_dir" ]; then
    echo "ERROR: Data directory not found: $data_dir"
    echo "Run data_preparation.py first:"
    echo "  python data_preparation.py --target $target --output_dir ./results"
    exit 1
fi

# ==============================
# Log setup
# ==============================
log_dir="./logs"
mkdir -p "$log_dir"
log_file="${log_dir}/${arch_name}_${target_key}_fold${fold_idx}.log"

echo "========================================" | tee "$log_file"
echo "FOM1 Architecture Comparison" | tee -a "$log_file"
echo "Architecture: ${arch_name}" | tee -a "$log_file"
echo "Target:       ${target}" | tee -a "$log_file"
echo "Fold:         ${fold_idx}" | tee -a "$log_file"
echo "Job index:    ${N}" | tee -a "$log_file"
echo "Job started:  $(date)" | tee -a "$log_file"
echo "Working dir:  $(pwd)" | tee -a "$log_file"
echo "Data dir:     ${data_dir}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# ==============================
# Run training
# ==============================
python3 "$script" \
    --data_dir "$data_dir" \
    --target "$target" \
    --fold "$fold_idx" \
    --seed 42 \
    2>&1 | tee -a "$log_file"

exit_status=$?

echo "========================================" | tee -a "$log_file"
echo "Training completed at: $(date)" | tee -a "$log_file"
echo "Exit status: ${exit_status}" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# ==============================
# Clean up PBS output files
# ==============================
if [ "$JOB_ID" != "local" ]; then
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'[' -f1 | cut -d'.' -f1)
    rm -f "${DIRECTORY}/fom1_arch.e${BASE_JOB_ID}.${N}" 2>/dev/null
    rm -f "${DIRECTORY}/fom1_arch.o${BASE_JOB_ID}.${N}" 2>/dev/null
fi

exit $exit_status
