#!/bin/bash
#PBS -N wsga_chem_space
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=1:mem=64gb
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Chemical Space Analysis — Single Top Configuration
# ============================================================
#
# Generates publication-quality chemical space figures for the
# top hyperparameter configuration identified by
# analyse_hyperparam_sweep.py.
#
# Step 1: Generate 10k n-gram reference molecules (skips if
#         data/ngram_reference_10k.csv already exists).
#
# Step 2: Run plot_chemical_space.py to compute shared UMAP,
#         produce 6 figures (4 panels + combined grid + FG
#         over generations), and print coverage statistics.
#
# Prerequisites:
#   - analyse_hyperparam_sweep.py must have been run first
#     (produces configs_aggregated.csv)
#
# Usage:
#   qsub chemical_space_sweep.sh       # submit to PBS
#   bash chemical_space_sweep.sh       # run locally
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"

if [ -z "$DIRECTORY" ]; then
    DIRECTORY=$(pwd)
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env
cd "$DIRECTORY"

# Ensure pipe failures are caught
set -o pipefail

# ==============================
# Configuration
# ==============================

SWEEP_DIR="../outputs/hyperparam_sweep"
DATA_DIR="../data"
REF_CSV="../data/ngram_reference_10k.csv"
OUTPUT_DIR="../outputs/analysis/chemical_space"

mkdir -p "$OUTPUT_DIR"

# ==============================
# Logging
# ==============================
log_file="${OUTPUT_DIR}/chemical_space.log"
echo "========================================" > "$log_file"
echo "Chemical Space Analysis (top config)" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Sweep dir:   $SWEEP_DIR" >> "$log_file"
echo "Data dir:    $DATA_DIR" >> "$log_file"
echo "Ref CSV:     $REF_CSV" >> "$log_file"
echo "Output dir:  $OUTPUT_DIR" >> "$log_file"
echo "Started at:  $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Step 1: Generate reference set
# ==============================
echo "" >> "$log_file"
echo "=== Step 1: Reference set ===" >> "$log_file"
echo "" >> "$log_file"

python3 generate_reference_set.py \
    --data_dir "$DATA_DIR" \
    --output "$REF_CSV" \
    --n_mol 10000 \
    --seed 42 \
    2>&1 | tee -a "$log_file"

step1_status=$?

if [ $step1_status -ne 0 ]; then
    echo "Step 1 FAILED with exit status $step1_status" >> "$log_file"
    echo "Finished at: $(date)" >> "$log_file"
    exit $step1_status
fi

# ==============================
# Step 2: Chemical space analysis
# ==============================
echo "" >> "$log_file"
echo "=== Step 2: Chemical space analysis ===" >> "$log_file"
echo "" >> "$log_file"

python3 plot_chemical_space.py \
    --sweep_dir "$SWEEP_DIR" \
    --data_dir "$DATA_DIR" \
    --ref_csv "$REF_CSV" \
    --out_dir "$OUTPUT_DIR" \
    --n_top 50 \
    --sample_ga 50000 \
    2>&1 | tee -a "$log_file"

step2_status=$?

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Step 1 exit: $step1_status" >> "$log_file"
echo "Step 2 exit: $step2_status" >> "$log_file"

exit $step2_status
