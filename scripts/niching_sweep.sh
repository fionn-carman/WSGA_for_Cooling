#!/bin/bash
#PBS -N wsga_niche
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-79
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Niching Sweep — Radius x Tau Calibrated Study
# ============================================================
#
# Sweeps Morgan fingerprint radius and niching threshold tau
# together, with radius-specific tau grids calibrated to
# comparable niche strictness.
#
# Conditions (16):
#   0-4:   radius=2, tau=0.25, 0.30, 0.35, 0.40, 0.45
#   5-9:   radius=4, tau=0.15, 0.18, 0.21, 0.24, 0.27
#   10-14: radius=8, tau=0.11, 0.13, 0.15, 0.17, 0.19
#   15:    no-niching control
#
# Seeds: 42, 123, 7, 256, 999  (5)
#
# Fixed (from manuscript production settings):
#   Pop = 3000, ER = 0.25, BER = 0.5, MR = 0.3, k = 5
#   Generations = 150, Init = gru, Biodeg = OFF
#   FP = 373 K, BP = 70 C, MP = -30 C, DC = 8, SC = 3, Tox = 3
#
# Total: 16 * 5 = 80 jobs
#
# Usage:
#   qsub niching_sweep.sh
#   qsub -J 0-0 niching_sweep.sh       # single test job
#   bash niching_sweep.sh <array_index> # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash niching_sweep.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

# ==============================
# Condition lookup table (16 conditions)
# ==============================
# Flat arrays: index 0-4 = radius 2, 5-9 = radius 4, 10-14 = radius 8, 15 = no niching
radii=(  2    2    2    2    2    4    4    4    4    4    8    8    8    8    8    0)
taus=(0.25 0.30 0.35 0.40 0.45 0.15 0.18 0.21 0.24 0.27 0.11 0.13 0.15 0.17 0.19 0.00)
seeds=(42 123 7 256 999)

n_conditions=16
n_seeds=5
n_total=$((n_conditions * n_seeds))

if [ "$N" -ge "$n_total" ] || [ "$N" -lt 0 ]; then
    echo "ERROR: Array index $N out of range [0, $((n_total - 1))]"
    exit 1
fi

# ==============================
# Decode array index
# ==============================
seed_idx=$((N % n_seeds))
cond_idx=$((N / n_seeds))

RADIUS=${radii[$cond_idx]}
TAU=${taus[$cond_idx]}
SEED=${seeds[$seed_idx]}

# ==============================
# Fixed GA parameters (manuscript production)
# ==============================
POP=3000
MR=0.3
ER=0.25
BER=0.5
K=5
NUM_GENERATIONS=150
TARGET="FOM1"

# ==============================
# Build output dir and niching flags
# ==============================
if [ "$RADIUS" -eq 0 ]; then
    NICHING_FLAG="--no_niching"
    RADIUS_FLAG=""
    output_dir="../outputs/niching_sweep/no_niching_seed${SEED}"
else
    NICHING_FLAG=""
    RADIUS_FLAG="--niching_radius $RADIUS"
    output_dir="../outputs/niching_sweep/r${RADIUS}_tau${TAU}_seed${SEED}"
fi

mkdir -p "$output_dir"

# ==============================
# Config log
# ==============================
log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA Niching Sweep" >> "$log_file"
echo "  Radius x Tau calibrated study" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N" >> "$log_file"
echo "Condition index:   $cond_idx" >> "$log_file"
echo "Target:            $TARGET" >> "$log_file"
echo "Population size:   $POP" >> "$log_file"
echo "Mutation rate:     $MR" >> "$log_file"
echo "Elitism rate:      $ER" >> "$log_file"
echo "Best elite ratio:  $BER" >> "$log_file"
echo "Tournament k:      $K" >> "$log_file"
echo "Niching radius:    $RADIUS" >> "$log_file"
echo "Tau (niching):     $TAU" >> "$log_file"
echo "No niching:        $([ -n "$NICHING_FLAG" ] && echo 'true' || echo 'false')" >> "$log_file"
echo "Generations:       $NUM_GENERATIONS" >> "$log_file"
echo "Init method:       gru" >> "$log_file"
echo "Seed:              $SEED" >> "$log_file"
echo "Started at:        $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Run WSGA
# ==============================
python3 ../src/wsga.py \
    --target $TARGET \
    --population_size $POP \
    --mutation_rate $MR \
    --elitism_rate $ER \
    --best_elite_ratio $BER \
    --tournament_k $K \
    --Tau $TAU \
    $RADIUS_FLAG \
    $NICHING_FLAG \
    --num_generations $NUM_GENERATIONS \
    --no_biodeg \
    --fp_threshold 373 \
    --bp_threshold 70 \
    --dc_threshold 8 \
    --mp_threshold -30 \
    --sc_threshold 3 \
    --tox_threshold 3 \
    --init_method gru \
    --seed $SEED \
    --data_dir ../data \
    --model_dir ../models \
    --training_data_dir ../training/data \
    --output_dir $output_dir \
    2>&1 | tee -a "$log_file"

exit_status=$?

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: $exit_status" >> "$log_file"

exit $exit_status
