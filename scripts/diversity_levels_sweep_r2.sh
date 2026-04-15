#!/bin/bash
#PBS -N div_r2
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-159
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Diversity Levels at Morgan radius 2 (matching WSGA niching production)
# ============================================================
#
# Applies to BOTH niching sweeps (FP100 and FP125) — 80 runs each = 160 jobs.
# Writes to diversity_levels_r2.csv inside each run dir, leaving the existing
# diversity_levels.csv (r=8) untouched.
#
# Aggregate afterwards:
#   for base in niching_sweep niching_sweep_fp125; do
#     head -1 $(ls outputs/$base/*/diversity_levels_r2.csv | head -1) \
#       > outputs/$base/analysis/diversity_levels_r2_all.csv
#     for f in outputs/$base/*/diversity_levels_r2.csv; do tail -1 "$f"; done \
#       >> outputs/$base/analysis/diversity_levels_r2_all.csv
#   done
#
# Usage:
#   qsub diversity_levels_sweep_r2.sh
#   bash diversity_levels_sweep_r2.sh <index>
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash diversity_levels_sweep_r2.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

# Build a combined list: first FP100 runs (0-79), then FP125 runs (80-159)
mapfile -t RUN_DIRS < <(
    find "../outputs/niching_sweep"       -maxdepth 2 -name "all_evaluated_molecules.csv" -exec dirname {} \; | sort
    find "../outputs/niching_sweep_fp125" -maxdepth 2 -name "all_evaluated_molecules.csv" -exec dirname {} \; | sort
)

if [ "$N" -ge "${#RUN_DIRS[@]}" ] || [ "$N" -lt 0 ]; then
    echo "ERROR: Array index $N out of range [0, $((${#RUN_DIRS[@]} - 1))]"
    exit 1
fi

RUN_DIR="${RUN_DIRS[$N]}"
echo "Job $N: Processing $RUN_DIR"

python3 compute_diversity_levels.py "$RUN_DIR" \
    --fp_radius 2 \
    --output_name diversity_levels_r2.csv

exit $?
