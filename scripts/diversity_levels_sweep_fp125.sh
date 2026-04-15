#!/bin/bash
#PBS -N div_lev125
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-79
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Diversity Levels — BRICS + Tanimoto (FP >= 125 C sweep)
# ============================================================
#
# Aggregate afterwards:
#   cd outputs/niching_sweep_fp125
#   head -1 $(ls */diversity_levels.csv | head -1) > analysis/diversity_levels_all.csv
#   for f in */diversity_levels.csv; do tail -1 "$f"; done >> analysis/diversity_levels_all.csv
#
# Usage:
#   qsub diversity_levels_sweep_fp125.sh
#   bash diversity_levels_sweep_fp125.sh <index>
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash diversity_levels_sweep_fp125.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

SWEEP_DIR="../outputs/niching_sweep_fp125"

mapfile -t RUN_DIRS < <(find "$SWEEP_DIR" -maxdepth 2 -name "all_evaluated_molecules.csv" -exec dirname {} \; | sort)

if [ "$N" -ge "${#RUN_DIRS[@]}" ] || [ "$N" -lt 0 ]; then
    echo "ERROR: Array index $N out of range [0, $((${#RUN_DIRS[@]} - 1))]"
    exit 1
fi

RUN_DIR="${RUN_DIRS[$N]}"
echo "Job $N: Processing $RUN_DIR"

python3 compute_diversity_levels.py "$RUN_DIR"

exit $?
