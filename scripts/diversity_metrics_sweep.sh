#!/bin/bash
#PBS -N div_metrics
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -J 0-70
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Diversity Metrics — parallel computation for niching sweep
# ============================================================
#
# Computes BRICS fragments, mean pairwise Tanimoto (ECFP4),
# and functional group breakdown for each niching sweep run.
#
# Array index 0-70 maps to sorted run directories.
# Each job writes <run_dir>/diversity_metrics.csv.
# Aggregate afterwards with:
#   cd outputs/niching_sweep
#   head -1 $(ls */diversity_metrics.csv | head -1) > analysis/diversity_all.csv
#   for f in */diversity_metrics.csv; do tail -1 "$f"; done >> analysis/diversity_all.csv
#
# Usage:
#   qsub diversity_metrics_sweep.sh
#   qsub -J 0-0 diversity_metrics_sweep.sh   # single test job
#   bash diversity_metrics_sweep.sh <index>   # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash diversity_metrics_sweep.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

SWEEP_DIR="../outputs/niching_sweep"

# Build sorted list of run directories that have all_evaluated_molecules.csv
mapfile -t RUN_DIRS < <(find "$SWEEP_DIR" -maxdepth 2 -name "all_evaluated_molecules.csv" -exec dirname {} \; | sort)

if [ "$N" -ge "${#RUN_DIRS[@]}" ] || [ "$N" -lt 0 ]; then
    echo "ERROR: Array index $N out of range [0, $((${#RUN_DIRS[@]} - 1))]"
    exit 1
fi

RUN_DIR="${RUN_DIRS[$N]}"
echo "Job $N: Processing $RUN_DIR"

python3 compute_diversity_metrics.py "$RUN_DIR"

exit $?
