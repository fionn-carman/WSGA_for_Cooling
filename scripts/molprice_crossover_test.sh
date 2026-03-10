#!/bin/bash
#PBS -N wsga_molprice_xover
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-19
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# WSGA MolPrice + Bond-Level Crossover Test
# ============================================================
#
# Tests the new MolPrice cost penalty and bond-level crossover.
# Compares four constraint variants across 5 seeds:
#
#   - bio_cost:        biodeg ON,  MolPrice ON   (indices 0-4)
#   - bio_nocost:      biodeg ON,  MolPrice OFF  (indices 5-9)
#   - nonbio_cost:     biodeg OFF, MolPrice ON   (indices 10-14)
#   - nonbio_nocost:   biodeg OFF, MolPrice OFF  (indices 15-19)
#
# Seeds: [42, 123, 7, 256, 999]  (5 per variant)
#
# Total: 4 * 5 = 20 jobs
#
# All runs use bond-level crossover (default).
# Uses best GA hyperparameters from previous sweeps.
#
# Usage:
#   qsub molprice_crossover_test.sh
#   qsub -J 0-0 molprice_crossover_test.sh   # single test job
#   bash molprice_crossover_test.sh 0          # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash molprice_crossover_test.sh <array_index>"
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

# ==============================
# Decode array index
# ==============================
n_seed=${#seeds[@]}

# 4 variants: 0=bio_cost, 1=bio_nocost, 2=nonbio_cost, 3=nonbio_nocost
variant_idx=$((N / n_seed))
seed_idx=$((N % n_seed))
SEED=${seeds[$seed_idx]}

case $variant_idx in
    0)
        VARIANT="bio_cost"
        BIODEG_FLAG=""
        MOLPRICE_FLAG="--molprice_model ../models/MolPrice/MP_Morgan_hybrid.pkl"
        ;;
    1)
        VARIANT="bio_nocost"
        BIODEG_FLAG=""
        MOLPRICE_FLAG=""
        ;;
    2)
        VARIANT="nonbio_cost"
        BIODEG_FLAG="--no_biodeg"
        MOLPRICE_FLAG="--molprice_model ../models/MolPrice/MP_Morgan_hybrid.pkl"
        ;;
    3)
        VARIANT="nonbio_nocost"
        BIODEG_FLAG="--no_biodeg"
        MOLPRICE_FLAG=""
        ;;
esac

# Fixed parameters
TARGET="FOM1_direct"
MP_SOFT=-30
MP_HARD=-10
FP_THRESHOLD=373
BP_THRESHOLD=70
DC_THRESHOLD=8
SC_THRESHOLD=3
TOX_THRESHOLD=3

# MolPrice thresholds (only used when MOLPRICE_FLAG is set)
MOLPRICE_SOFT=3.0
MOLPRICE_HARD=6.0

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/molprice_crossover_test/${VARIANT}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA MolPrice + Bond Crossover Test"     >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Variant:           $VARIANT"              >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "Population size:   $POP"                  >> "$log_file"
echo "Mutation rate:     $MR"                   >> "$log_file"
echo "Elitism rate:      $ER"                   >> "$log_file"
echo "Best elite ratio:  $BER"                  >> "$log_file"
echo "Tournament k:      $K"                    >> "$log_file"
echo "Tau (niching):     $TAU"                  >> "$log_file"
echo "Generations:       $NUM_GENERATIONS"      >> "$log_file"
echo "Seed label:        $SEED"                 >> "$log_file"
echo "MP soft/hard:      ${MP_SOFT}/${MP_HARD}" >> "$log_file"
echo "FP threshold:      $FP_THRESHOLD"         >> "$log_file"
echo "Biodeg flag:       ${BIODEG_FLAG:-(enabled)}" >> "$log_file"
echo "MolPrice model:    ${MOLPRICE_FLAG:-(disabled)}" >> "$log_file"
echo "MolPrice soft/hard: ${MOLPRICE_SOFT}/${MOLPRICE_HARD}" >> "$log_file"
echo "Crossover:         bond-level (default)"  >> "$log_file"
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
    --mp_soft $MP_SOFT \
    --mp_hard $MP_HARD \
    --fp_threshold $FP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --molprice_soft $MOLPRICE_SOFT \
    --molprice_hard $MOLPRICE_HARD \
    --fom1_model_dir ../training/FOM1_architecture_comparison/results \
    --data_dir ../data \
    --model_dir ../models \
    --output_dir $output_dir \
    $BIODEG_FLAG \
    $MOLPRICE_FLAG \
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
