#!/bin/bash
#PBS -N wsga_mahal_compute
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-5
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Mahalanobis Distance — Tau Comparison (Parallel)
# ============================================================
#
# PBS array job: runs compute_mahalanobis.py in parallel for
# each of the 6 tau/seed configurations.
#
# Array index mapping:
#   0 → tau0.05_seed42      3 → tau0.20_seed42
#   1 → tau0.05_seed123     4 → tau0.20_seed123
#   2 → tau0.05_seed7       5 → tau0.20_seed7
#
# After all 6 complete, submit the plotting job:
#   qsub -W depend=afterokarray:<ARRAY_JOB_ID> mahal_tau_plot.sh
#
# Or submit both at once:
#   MAHAL_ID=$(qsub mahal_tau_comparison.sh)
#   qsub -W depend=afterokarray:$MAHAL_ID mahal_tau_plot.sh
#
# Usage:
#   qsub mahal_tau_comparison.sh
#   qsub -J 0-0 mahal_tau_comparison.sh   # single test job
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash mahal_tau_comparison.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env
cd "$DIRECTORY"

# ==============================
# Configuration
# ==============================
COMPARISON_DIR="../outputs/tau_comparison"

dirs=(
    "tau0.05_seed42"
    "tau0.05_seed123"
    "tau0.05_seed7"
    "tau0.20_seed42"
    "tau0.20_seed123"
    "tau0.20_seed7"
)

RUN_DIR="${dirs[$N]}"
CSV_PATH="${COMPARISON_DIR}/${RUN_DIR}/all_evaluated_molecules.csv"
OUT_PATH="${COMPARISON_DIR}/${RUN_DIR}/all_evaluated_molecules_mahal.csv"

# ==============================
# Logging
# ==============================
LOG_DIR="${COMPARISON_DIR}/${RUN_DIR}"
mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/mahalanobis.log"

echo "========================================" > "$log_file"
echo "Mahalanobis Distance Computation" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:  $N" >> "$log_file"
echo "Run dir:      $RUN_DIR" >> "$log_file"
echo "Input CSV:    $CSV_PATH" >> "$log_file"
echo "Output CSV:   $OUT_PATH" >> "$log_file"
echo "Started at:   $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Validate input
# ==============================
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: Input CSV not found: $CSV_PATH" >> "$log_file"
    echo "ERROR: Input CSV not found: $CSV_PATH"
    exit 1
fi

# ==============================
# Run Mahalanobis computation
# ==============================
python3 compute_mahalanobis.py \
    --csv "$CSV_PATH" \
    --models_dir ../models \
    --training_dir ../training/data \
    --out "$OUT_PATH" \
    2>&1 | tee -a "$log_file"

exit_status=$?

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: $exit_status" >> "$log_file"

exit $exit_status
