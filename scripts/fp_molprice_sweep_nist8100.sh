#!/bin/bash
#PBS -N wsga_fp_mp
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-59
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# WSGA FP x MolPrice Production Sweep (NIST 8100 models)
# ============================================================
#
# Combined sweep crossing flash point safety levels with
# MolPrice cost regimes. Biodeg always enabled.
#
# GA parameters: manuscript production settings
#   POP=3000, ER=0.25, BER=0.5, MR=0.3, K=5
#   Niching: r=2, tau=0.25, 150 generations
#
# Sweep grid: 4 FP levels x 3 MolPrice levels x 5 seeds = 60 jobs
#
# FP levels (Celsius -> Kelvin):
#   0  fp100  373.15 K  — Default / general purpose
#   1  fp125  398.15 K  — Moderate EV safety
#   2  fp150  423.15 K  — Thermal runaway onset (NMC)
#   3  fp175  448.15 K  — Conservative margin
#
# MolPrice levels (soft, hard) -- fixed width of 2.0:
#   0  nocost       --    --     No MolPrice model (baseline)
#   1  gentle      4.0   6.0    Mild cost pressure
#   2  aggressive  2.0   4.0    Strong cost pressure
#
# Index decoding:
#   fp_idx    = N / (n_level * n_seed)    # 0-3
#   level_idx = (N / n_seed) % n_level    # 0-2
#   seed_idx  = N % n_seed                # 0-4
#
# Spot checks:
#   N=0  -> fp100/nocost/seed42
#   N=14 -> fp100/aggressive/seed999
#   N=15 -> fp125/nocost/seed42
#   N=59 -> fp175/aggressive/seed999
#
# Usage:
#   qsub fp_molprice_sweep_nist8100.sh
#   qsub -J 0-0 fp_molprice_sweep_nist8100.sh   # single test job
#   bash fp_molprice_sweep_nist8100.sh 0          # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash fp_molprice_sweep_nist8100.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

# ==============================
# GA parameters (manuscript production)
# ==============================
POP=3000
MR=0.3
ER=0.25
BER=0.5
K=5
TAU=0.25
NICHING_RADIUS=2
NUM_GENERATIONS=150

# ==============================
# Sweep grid
# ==============================
seeds=(42 123 7 256 999)

# FP thresholds in Kelvin
fp_labels=(fp100 fp125 fp150 fp175)
fp_values=(373.15 398.15 423.15 448.15)

# MolPrice levels: label, soft, hard
level_labels=(nocost gentle aggressive)
level_soft=(0.0 4.0 2.0)
level_hard=(0.0 6.0 4.0)

# ==============================
# Decode array index
# ==============================
n_seed=${#seeds[@]}
n_level=${#level_labels[@]}

fp_idx=$((N / (n_level * n_seed)))
level_idx=$(( (N / n_seed) % n_level ))
seed_idx=$((N % n_seed))

FP_LABEL=${fp_labels[$fp_idx]}
FP_THRESHOLD=${fp_values[$fp_idx]}
LEVEL=${level_labels[$level_idx]}
SEED=${seeds[$seed_idx]}

# ==============================
# MolPrice flags
# ==============================
if [ "$LEVEL" == "nocost" ]; then
    MOLPRICE_FLAG=""
    MOLPRICE_SOFT=0.0
    MOLPRICE_HARD=0.0
else
    MOLPRICE_FLAG="--molprice_model ../models/MolPrice/MP_Morgan_hybrid.pkl"
    MOLPRICE_SOFT=${level_soft[$level_idx]}
    MOLPRICE_HARD=${level_hard[$level_idx]}
fi

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
output_dir="../outputs/fp_molprice_sweep_nist8100/${FP_LABEL}_${LEVEL}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA FP x MolPrice Sweep (NIST 8100)"   >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "FP label:          $FP_LABEL"             >> "$log_file"
echo "FP threshold (K):  $FP_THRESHOLD"         >> "$log_file"
echo "MolPrice level:    $LEVEL"                >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "Population size:   $POP"                  >> "$log_file"
echo "Mutation rate:     $MR"                   >> "$log_file"
echo "Elitism rate:      $ER"                   >> "$log_file"
echo "Best elite ratio:  $BER"                  >> "$log_file"
echo "Tournament k:      $K"                    >> "$log_file"
echo "Tau (niching):     $TAU"                  >> "$log_file"
echo "Niching radius:    $NICHING_RADIUS"       >> "$log_file"
echo "Generations:       $NUM_GENERATIONS"      >> "$log_file"
echo "Seed:              $SEED"                 >> "$log_file"
echo "MP threshold:      $MP_THRESHOLD"         >> "$log_file"
echo "BP threshold:      $BP_THRESHOLD"         >> "$log_file"
echo "DC threshold:      $DC_THRESHOLD"         >> "$log_file"
echo "Biodeg:            enabled"               >> "$log_file"
echo "MolPrice model:    ${MOLPRICE_FLAG:-(disabled)}" >> "$log_file"
echo "MolPrice soft/hard: ${MOLPRICE_SOFT}/${MOLPRICE_HARD}" >> "$log_file"
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
    --niching_radius $NICHING_RADIUS \
    --num_generations $NUM_GENERATIONS \
    --seed $SEED \
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
