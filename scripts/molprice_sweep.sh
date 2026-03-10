#!/bin/bash
#PBS -N wsga_molprice_sweep
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-59
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# WSGA MolPrice Threshold Sweep
# ============================================================
#
# Sweeps MolPrice cost-penalty thresholds to characterise the
# FOM1 vs cost trade-off.
#
# Sweep grid: 6 threshold levels × 2 biodeg settings × 5 seeds = 60 jobs
#
# Threshold levels (soft, hard):
#   0  nocost      —    —     No MolPrice model (baseline)
#   1  gentle      3.5  6.0   Very mild pressure
#   2  moderate    3.0  5.0   Current-ish defaults
#   3  firm        2.5  4.5   Noticeable pressure
#   4  tight       2.5  4.0   Strong pressure
#   5  aggressive  2.0  3.5   Very strong pressure
#
# Biodeg: bio (default), nonbio (--no_biodeg)
# Seeds: [42, 123, 7, 256, 999]
#
# Index decoding:
#   level_idx  = N / (n_biodeg * n_seed)
#   biodeg_idx = (N / n_seed) % n_biodeg
#   seed_idx   = N % n_seed
#
# Usage:
#   qsub molprice_sweep.sh
#   qsub -J 0-0 molprice_sweep.sh   # single test job
#   bash molprice_sweep.sh 0          # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash molprice_sweep.sh <array_index>"
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

# Threshold levels: label, soft, hard
level_labels=(nocost gentle moderate firm tight aggressive)
level_soft=(0.0 3.5 3.0 2.5 2.5 2.0)
level_hard=(0.0 6.0 5.0 4.5 4.0 3.5)

biodeg_labels=(bio nonbio)

# ==============================
# Decode array index
# ==============================
n_seed=${#seeds[@]}
n_biodeg=${#biodeg_labels[@]}

level_idx=$((N / (n_biodeg * n_seed)))
biodeg_idx=$(( (N / n_seed) % n_biodeg ))
seed_idx=$((N % n_seed))

LEVEL=${level_labels[$level_idx]}
SEED=${seeds[$seed_idx]}
BIODEG_LABEL=${biodeg_labels[$biodeg_idx]}

# Biodeg flag
if [ "$BIODEG_LABEL" == "nonbio" ]; then
    BIODEG_FLAG="--no_biodeg"
else
    BIODEG_FLAG=""
fi

# MolPrice flags
if [ "$LEVEL" == "nocost" ]; then
    MOLPRICE_FLAG=""
    MOLPRICE_SOFT=0.0
    MOLPRICE_HARD=0.0
else
    MOLPRICE_FLAG="--molprice_model ../models/MolPrice/MP_Morgan_hybrid.pkl"
    MOLPRICE_SOFT=${level_soft[$level_idx]}
    MOLPRICE_HARD=${level_hard[$level_idx]}
fi

# Fixed parameters
TARGET="FOM1_direct"
MP_SOFT=-30
MP_HARD=-10
FP_THRESHOLD=373
BP_THRESHOLD=70
DC_THRESHOLD=8
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/molprice_sweep/${LEVEL}_${BIODEG_LABEL}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA MolPrice Threshold Sweep"           >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Level:             $LEVEL"                >> "$log_file"
echo "Biodeg:            $BIODEG_LABEL"         >> "$log_file"
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
