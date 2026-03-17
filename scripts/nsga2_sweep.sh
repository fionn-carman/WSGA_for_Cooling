#!/bin/bash
#PBS -N wsga_nsga2_sweep
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-19
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# NSGA-II Multi-Objective WSGA Sweep
# ============================================================
#
# Runs NSGA-II (FOM1 vs MolPrice) across 4 categories × 5 seeds = 20 jobs.
# Replaces the 120-job parametric MolPrice threshold sweep with direct
# multi-objective optimisation.
#
# Categories:
#   0  bio_stable       biodeg + stability_mode strict
#   1  bio_unstable     biodeg only
#   2  nonbio_stable    no_biodeg + stability_mode strict
#   3  nonbio_unstable  no_biodeg only
#
# Index decoding:
#   category_idx = N / n_seed    # 0-3
#   seed_idx     = N % n_seed    # 0-4
#
# Usage:
#   qsub nsga2_sweep.sh
#   qsub -J 0-0 nsga2_sweep.sh    # single test job
#   bash nsga2_sweep.sh 0           # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash nsga2_sweep.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

# ==============================
# GA parameters (matched to threshold sweep)
# ==============================
POP=3000
MR=0.8
K=2
TAU=0.15
NUM_GENERATIONS=200

# ==============================
# Sweep grid
# ==============================
seeds=(42 123 7 256 999)
category_labels=(bio_stable bio_unstable nonbio_stable nonbio_unstable)

n_seed=${#seeds[@]}
n_category=${#category_labels[@]}

category_idx=$((N / n_seed))
seed_idx=$((N % n_seed))

CATEGORY=${category_labels[$category_idx]}
SEED=${seeds[$seed_idx]}

# Fixed constraint thresholds
TARGET="FOM1_direct"
MP_THRESHOLD=-30
FP_THRESHOLD=373
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/nsga2_sweep/${CATEGORY}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "NSGA-II Multi-Objective WSGA"            >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Category:          $CATEGORY"             >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "Population size:   $POP"                  >> "$log_file"
echo "Mutation rate:     $MR"                   >> "$log_file"
echo "Tournament k:      $K"                    >> "$log_file"
echo "Tau (niching):     $TAU"                  >> "$log_file"
echo "Generations:       $NUM_GENERATIONS"      >> "$log_file"
echo "Seed label:        $SEED"                 >> "$log_file"
echo "MP threshold:      $MP_THRESHOLD"         >> "$log_file"
echo "FP threshold:      $FP_THRESHOLD"         >> "$log_file"
echo "BP threshold:      $BP_THRESHOLD"         >> "$log_file"
echo "DC threshold:      $DC_THRESHOLD"         >> "$log_file"
echo "Started at:        $(date)"               >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Run NSGA-II WSGA
# ==============================
PYTHONHASHSEED=$SEED python3 ../src/nsga2_wsga.py \
    --target $TARGET \
    --population_size $POP \
    --mutation_rate $MR \
    --tournament_k $K \
    --Tau $TAU \
    --num_generations $NUM_GENERATIONS \
    --category $CATEGORY \
    --mp_threshold $MP_THRESHOLD \
    --fp_threshold $FP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --fom1_model_dir ../training/FOM1_architecture_comparison/results \
    --data_dir ../data \
    --model_dir ../models \
    --output_dir $output_dir \
    2>&1 | tee -a "$log_file"

exit_status=${PIPESTATUS[0]}

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: $exit_status" >> "$log_file"

echo "" >> "$log_file"
echo "Output files:" >> "$log_file"
ls -la "$output_dir"/ >> "$log_file" 2>&1

exit $exit_status
