#!/bin/bash
#PBS -N wsga_fp_sweep
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-59
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# WSGA Flash Point Threshold Sweep (NIST 8100 models)
# ============================================================
#
# Investigates GA performance under EV-relevant flash point
# constraints. Higher FP thresholds reflect thermal runaway
# safety requirements for battery cooling fluids.
#
# All runs are nocost (no MolPrice penalty) to isolate the
# effect of the FP constraint on FOM1.
#
# Sweep grid: 3 FP thresholds x 4 categories x 5 seeds = 60 jobs
#
# FP thresholds (Celsius -> Kelvin):
#   0  125C  (398.15 K)  — moderate EV safety
#   1  150C  (423.15 K)  — thermal runaway onset (NMC)
#   2  175C  (448.15 K)  — conservative margin
#
# Categories:
#   0  bio_stable       biodeg + stability_mode strict
#   1  bio_unstable     biodeg only
#   2  nonbio_stable    no_biodeg + stability_mode strict
#   3  nonbio_unstable  no_biodeg only
#
# Index decoding:
#   fp_idx       = N / (n_category * n_seed)    # 0-2
#   category_idx = (N / n_seed) % n_category    # 0-3
#   seed_idx     = N % n_seed                   # 0-4
#
# Usage:
#   qsub fp_sweep_nist8100.sh
#   qsub -J 0-0 fp_sweep_nist8100.sh   # single test job
#   bash fp_sweep_nist8100.sh 0          # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash fp_sweep_nist8100.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

# ==============================
# Best GA parameters (from sweeps)
# ==============================
POP=3000
MR=0.8
ER=0.25
BER=0.5
K=3
TAU=0.15
NUM_GENERATIONS=200

# ==============================
# Sweep grid
# ==============================
seeds=(42 123 7 256 999)

# FP thresholds in Kelvin
fp_labels=(fp125 fp150 fp175)
fp_values=(398.15 423.15 448.15)

category_labels=(bio_stable bio_unstable nonbio_stable nonbio_unstable)

# ==============================
# Decode array index
# ==============================
n_seed=${#seeds[@]}
n_category=${#category_labels[@]}

fp_idx=$((N / (n_category * n_seed)))
category_idx=$(( (N / n_seed) % n_category ))
seed_idx=$((N % n_seed))

FP_LABEL=${fp_labels[$fp_idx]}
FP_THRESHOLD=${fp_values[$fp_idx]}
SEED=${seeds[$seed_idx]}
CATEGORY=${category_labels[$category_idx]}

# ==============================
# Category flags
# ==============================
# Biodeg flag
case "$CATEGORY" in
    bio_stable|bio_unstable)
        BIODEG_FLAG=""
        ;;
    nonbio_stable|nonbio_unstable)
        BIODEG_FLAG="--no_biodeg"
        ;;
esac

# Stability flag
case "$CATEGORY" in
    bio_stable|nonbio_stable)
        STABILITY_FLAG="--stability_mode strict"
        ;;
    bio_unstable|nonbio_unstable)
        STABILITY_FLAG=""
        ;;
esac

# No MolPrice (nocost) — all runs
MOLPRICE_SOFT=0.0
MOLPRICE_HARD=0.0

# Fixed constraint thresholds
TARGET="FOM1"
MP_THRESHOLD=-30
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/fp_sweep_nist8100/${FP_LABEL}_${CATEGORY}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA FP Threshold Sweep (NIST 8100)"     >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "FP label:          $FP_LABEL"             >> "$log_file"
echo "FP threshold (K):  $FP_THRESHOLD"         >> "$log_file"
echo "Category:          $CATEGORY"             >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "Population size:   $POP"                  >> "$log_file"
echo "Mutation rate:     $MR"                   >> "$log_file"
echo "Elitism rate:      $ER"                   >> "$log_file"
echo "Best elite ratio:  $BER"                  >> "$log_file"
echo "Tournament k:      $K"                    >> "$log_file"
echo "Tau (niching):     $TAU"                  >> "$log_file"
echo "Generations:       $NUM_GENERATIONS"      >> "$log_file"
echo "Seed label:        $SEED"                 >> "$log_file"
echo "MP threshold:      $MP_THRESHOLD"         >> "$log_file"
echo "BP threshold:      $BP_THRESHOLD"         >> "$log_file"
echo "DC threshold:      $DC_THRESHOLD"         >> "$log_file"
echo "Biodeg flag:       ${BIODEG_FLAG:-(enabled)}" >> "$log_file"
echo "Stability flag:    ${STABILITY_FLAG:-(disabled)}" >> "$log_file"
echo "MolPrice:          nocost"                >> "$log_file"
echo "Started at:        $(date)"               >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Run WSGA
# ==============================
python3 ../src/wsga.py \
    --target $TARGET \
    --population_size $POP \
    --elitism_rate $ER \
    --best_elite_ratio $BER \
    --mutation_rate $MR \
    --tournament_k $K \
    --Tau $TAU \
    --num_generations $NUM_GENERATIONS \
    --mp_threshold $MP_THRESHOLD \
    --fp_threshold $FP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --molprice_soft $MOLPRICE_SOFT \
    --molprice_hard $MOLPRICE_HARD \
    --data_dir ../data \
    --model_dir ../models \
    --training_data_dir ../training/data \
    --output_dir $output_dir \
    $BIODEG_FLAG \
    $STABILITY_FLAG \
    2>&1 | tee -a "$log_file"

exit_status=${PIPESTATUS[0]}

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: $exit_status" >> "$log_file"

# List output files for verification
echo "" >> "$log_file"
echo "Output files:" >> "$log_file"
ls -la "$output_dir"/ >> "$log_file" 2>&1

exit $exit_status
