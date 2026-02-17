#!/bin/bash
#PBS -N wsga_chem_space
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mem=64gb
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Chemical Space Comparison — Two-step pipeline
# ============================================================
#
# Step 1: Runs compute_chemical_space.py to generate reference
#         molecules, load evaluated molecules from a selection
#         of extreme hyperparameter configs, fit a single UMAP
#         embedding, and save everything to a .npz file.
#
# Step 2: Runs compare_chemical_space.py to generate all
#         comparison figures from the saved embedding.
#
# This is a single job (not an array) because UMAP must be fit
# on all molecules simultaneously for coordinates to be comparable.
#
# Memory: 64 GB requested. With ~8 configs × 3 seeds × ~100k
#         molecules each, plus 5k reference, the fingerprint
#         matrix can be large. Adjust if needed.
#
# Usage:
#   qsub chemical_space_sweep.sh
#   bash chemical_space_sweep.sh          # local run
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"

if [ -z "$DIRECTORY" ]; then
    DIRECTORY=$(pwd)
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env
cd "$DIRECTORY"

# ==============================
# Configuration
# ==============================

# Sweep directory containing all hyperparam runs
SWEEP_DIR="../outputs/hyperparam_sweep"

# Data directory for reference molecule generation
DATA_DIR="../data"

# Output paths
OUTPUT_DIR="../outputs/analysis/chemical_space_comparison"
NPZ_FILE="${OUTPUT_DIR}/chemical_space_data.npz"

# Parameters
N_REF=5000                 # Reference molecules to generate
MAX_CONFIGS=8              # Number of extreme configs to select
SAMPLE_PER_SEED=0          # 0 = use all molecules (no sampling)

mkdir -p "$OUTPUT_DIR"

# ==============================
# Logging
# ==============================
log_file="${OUTPUT_DIR}/chemical_space.log"
echo "========================================" > "$log_file"
echo "Chemical Space Comparison Analysis" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Sweep dir:       $SWEEP_DIR" >> "$log_file"
echo "Data dir:        $DATA_DIR" >> "$log_file"
echo "N reference:     $N_REF" >> "$log_file"
echo "Max configs:     $MAX_CONFIGS" >> "$log_file"
echo "Sample per seed: $SAMPLE_PER_SEED" >> "$log_file"
echo "Started at:      $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Step 1: Compute UMAP embedding
# ==============================
echo "" >> "$log_file"
echo "=== Step 1: Computing UMAP embedding ===" >> "$log_file"
echo "" >> "$log_file"

python3 compute_chemical_space.py \
    --sweep_dir "$SWEEP_DIR" \
    --data_dir "$DATA_DIR" \
    --output "$NPZ_FILE" \
    --n_ref $N_REF \
    --max_configs $MAX_CONFIGS \
    --sample_per_seed $SAMPLE_PER_SEED \
    2>&1 | tee -a "$log_file"

step1_status=$?

if [ $step1_status -ne 0 ]; then
    echo "Step 1 FAILED with exit status $step1_status" >> "$log_file"
    echo "Finished at: $(date)" >> "$log_file"
    exit $step1_status
fi

# ==============================
# Step 2: Generate comparison figures
# ==============================
echo "" >> "$log_file"
echo "=== Step 2: Generating comparison figures ===" >> "$log_file"
echo "" >> "$log_file"

python3 compare_chemical_space.py \
    --input "$NPZ_FILE" \
    --out_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$log_file"

step2_status=$?

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Step 1 exit: $step1_status" >> "$log_file"
echo "Step 2 exit: $step2_status" >> "$log_file"

exit $step2_status
