#!/bin/bash
#PBS -N wsga_mahal_plot
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=1:mem=64gb
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Mahalanobis Distance — Tau Comparison Plotting
# ============================================================
#
# Combines the per-run Mahalanobis CSVs and reruns the tau
# chemical space plots with OOD panels.
#
# Steps:
#   1. Concatenate all 40 _mahal.csv files into one combined CSV
#   2. Replot tau chemical space with --mahal_csv (OOD panels)
#
# Prerequisites:
#   - mahal_tau_comparison.sh array job must have completed
#   - tau_umap_cache.npz must already exist (from a prior
#     plot_tau_chemical_space.py run). If not, uses --sweep_dir
#     to compute UMAP from scratch.
#
# Usage:
#   qsub mahal_tau_plot.sh
#   qsub -W depend=afterokarray:<MAHAL_JOB_ID> mahal_tau_plot.sh
#
# Or chain both:
#   MAHAL_ID=$(qsub mahal_tau_comparison.sh)
#   qsub -W depend=afterokarray:$MAHAL_ID mahal_tau_plot.sh
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"

if [ -z "$DIRECTORY" ]; then
    DIRECTORY=$(pwd)
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env
cd "$DIRECTORY"

set -o pipefail

# ==============================
# Configuration
# ==============================
SWEEP_DIR="../outputs/tau_sweep"
OUTPUT_DIR="${SWEEP_DIR}/analysis/chemical_space"
COMBINED_CSV="${SWEEP_DIR}/all_mahal_combined.csv"
NPZ_CACHE="${OUTPUT_DIR}/tau_umap_cache.npz"
REF_CSV="../data/training_reference.csv"

tau_values=(0.0 0.05 0.10 0.15 0.20 0.30 0.40 0.50)
seeds=(42 123 7 256 999)

# Build directory list from tau × seed grid
dirs=()
for tau in "${tau_values[@]}"; do
    for seed in "${seeds[@]}"; do
        dirs+=("tau${tau}_seed${seed}")
    done
done

mkdir -p "$OUTPUT_DIR"

# ==============================
# Logging
# ==============================
log_file="${OUTPUT_DIR}/mahal_plot.log"
echo "========================================" > "$log_file"
echo "Mahalanobis Tau Comparison — Plotting" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Sweep dir:      $SWEEP_DIR" >> "$log_file"
echo "Output dir:     $OUTPUT_DIR" >> "$log_file"
echo "Combined CSV:   $COMBINED_CSV" >> "$log_file"
echo "NPZ cache:      $NPZ_CACHE" >> "$log_file"
echo "Started at:     $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Step 1: Combine Mahalanobis CSVs
# ==============================
echo "" >> "$log_file"
echo "=== Step 1: Combine Mahalanobis CSVs ===" >> "$log_file"
echo "" >> "$log_file"

# Validate all mahal CSVs exist
all_present=true
for d in "${dirs[@]}"; do
    mahal_csv="${SWEEP_DIR}/${d}/all_evaluated_molecules_mahal.csv"
    if [ ! -f "$mahal_csv" ]; then
        echo "ERROR: Missing mahal CSV: $mahal_csv" >> "$log_file"
        echo "ERROR: Missing mahal CSV: $mahal_csv"
        all_present=false
    fi
done

if [ "$all_present" = false ]; then
    echo "ERROR: Not all Mahalanobis CSVs found. Did the array job complete?" >> "$log_file"
    exit 1
fi

# Take header from first file
first_csv="${SWEEP_DIR}/${dirs[0]}/all_evaluated_molecules_mahal.csv"
head -1 "$first_csv" > "$COMBINED_CSV"

# Append data rows from all files
for d in "${dirs[@]}"; do
    mahal_csv="${SWEEP_DIR}/${d}/all_evaluated_molecules_mahal.csv"
    n_rows=$(tail -n +2 "$mahal_csv" | wc -l)
    echo "  ${d}: ${n_rows} rows" >> "$log_file"
    tail -n +2 "$mahal_csv" >> "$COMBINED_CSV"
done

total_rows=$(tail -n +2 "$COMBINED_CSV" | wc -l)
echo "  Combined: ${total_rows} total rows" >> "$log_file"
echo "  Saved to: $COMBINED_CSV" >> "$log_file"

# ==============================
# Step 2: Plot with OOD panels
# ==============================
echo "" >> "$log_file"
echo "=== Step 2: Chemical space + OOD plots ===" >> "$log_file"
echo "" >> "$log_file"

if [ -f "$NPZ_CACHE" ]; then
    echo "  Using existing NPZ cache (fast replot mode)" >> "$log_file"
    python3 plot_tau_chemical_space.py \
        --npz "$NPZ_CACHE" \
        --mahal_csv "$COMBINED_CSV" \
        --out_dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$log_file"
else
    echo "  No NPZ cache found — computing UMAP from scratch" >> "$log_file"
    python3 plot_tau_chemical_space.py \
        --sweep_dir "$SWEEP_DIR" \
        --ref_csv "$REF_CSV" \
        --mahal_csv "$COMBINED_CSV" \
        --out_dir "$OUTPUT_DIR" \
        --n_top 50 \
        --sample_ga 50000 \
        2>&1 | tee -a "$log_file"
fi

step2_status=$?

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Step 2 exit: $step2_status" >> "$log_file"

exit $step2_status
