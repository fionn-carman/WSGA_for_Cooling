#!/bin/bash
#PBS -N wsga_mahal_compute
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-39
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Mahalanobis Distance — Tau Sweep (Parallel)
# ============================================================
#
# PBS array job: runs compute_mahalanobis.py in parallel for
# each of the 40 tau/seed configurations from the tau sweep.
#
# Grid: 8 tau values × 5 seeds = 40 jobs
#   tau: [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
#   seeds: [42, 123, 7, 256, 999]
#
# After all 40 complete, submit the plotting job:
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
SWEEP_DIR="../outputs/tau_sweep"

tau_values=(0.0 0.05 0.10 0.15 0.20 0.30 0.40 0.50)
seeds=(42 123 7 256 999)

n_seed=${#seeds[@]}
seed_idx=$((N % n_seed))
tau_idx=$((N / n_seed))

TAU=${tau_values[$tau_idx]}
SEED=${seeds[$seed_idx]}

RUN_DIR="tau${TAU}_seed${SEED}"

CSV_PATH="${SWEEP_DIR}/${RUN_DIR}/all_evaluated_molecules.csv"
OUT_PATH="${SWEEP_DIR}/${RUN_DIR}/all_evaluated_molecules_mahal.csv"

# ==============================
# Logging
# ==============================
LOG_DIR="${SWEEP_DIR}/${RUN_DIR}"
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
