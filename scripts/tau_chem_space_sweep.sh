#!/bin/bash
#PBS -N wsga_tau_chem_space
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=1:mem=64gb
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Tau Sweep — Chemical Space Analysis (Shared UMAP)
# ============================================================
#
# Analyses the tau sweep results and generates chemical space
# figures with a single shared UMAP across all tau values.
#
# Steps:
#   1. Run analyse_tau_sweep.py (summary stats + tau figures)
#   2. Generate 10k n-gram reference set (skips if cached)
#   3. Run plot_tau_chemical_space.py (shared UMAP + 2x4 grid)
#
# Prerequisites:
#   - tau_sweep.sh must have completed (all 40 jobs)
#
# Usage:
#   qsub tau_chem_space_sweep.sh       # submit to PBS
#   bash tau_chem_space_sweep.sh       # run locally
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

SWEEP_DIR="../outputs/tau_sweep"
DATA_DIR="../data"
REF_CSV="../data/ngram_reference_10k.csv"
OUTPUT_DIR="../outputs/tau_sweep/analysis/chemical_space"

mkdir -p "$OUTPUT_DIR"

# ==============================
# Logging
# ==============================
log_file="${OUTPUT_DIR}/tau_chem_space.log"
echo "========================================" > "$log_file"
echo "Tau Sweep — Chemical Space Analysis" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Sweep dir:   $SWEEP_DIR" >> "$log_file"
echo "Data dir:    $DATA_DIR" >> "$log_file"
echo "Ref CSV:     $REF_CSV" >> "$log_file"
echo "Output dir:  $OUTPUT_DIR" >> "$log_file"
echo "Started at:  $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Step 1: Analyse tau sweep
# ==============================
echo "" >> "$log_file"
echo "=== Step 1: Analyse tau sweep ===" >> "$log_file"
echo "" >> "$log_file"

python3 analyse_tau_sweep.py \
    --sweep_dir "$SWEEP_DIR" \
    2>&1 | tee -a "$log_file"

step1_status=$?

if [ $step1_status -ne 0 ]; then
    echo "Step 1 FAILED with exit status $step1_status" >> "$log_file"
    echo "Finished at: $(date)" >> "$log_file"
    exit $step1_status
fi

# ==============================
# Step 2: Generate reference set
# ==============================
echo "" >> "$log_file"
echo "=== Step 2: Reference set ===" >> "$log_file"
echo "" >> "$log_file"

python3 generate_reference_set.py \
    --data_dir "$DATA_DIR" \
    --output "$REF_CSV" \
    --n_mol 10000 \
    --seed 42 \
    2>&1 | tee -a "$log_file"

step2_status=$?

if [ $step2_status -ne 0 ]; then
    echo "Step 2 FAILED with exit status $step2_status" >> "$log_file"
    echo "Finished at: $(date)" >> "$log_file"
    exit $step2_status
fi

# ==============================
# Step 3: Chemical space analysis
# ==============================
echo "" >> "$log_file"
echo "=== Step 3: Chemical space analysis (shared UMAP) ===" >> "$log_file"
echo "" >> "$log_file"

python3 plot_tau_chemical_space.py \
    --sweep_dir "$SWEEP_DIR" \
    --data_dir "$DATA_DIR" \
    --ref_csv "$REF_CSV" \
    --out_dir "$OUTPUT_DIR" \
    --n_top 50 \
    --sample_ga 50000 \
    2>&1 | tee -a "$log_file"

step3_status=$?

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Step 1 exit: $step1_status" >> "$log_file"
echo "Step 2 exit: $step2_status" >> "$log_file"
echo "Step 3 exit: $step3_status" >> "$log_file"

exit $step3_status
