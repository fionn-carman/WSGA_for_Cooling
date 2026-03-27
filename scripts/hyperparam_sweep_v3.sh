#!/bin/bash
#PBS -N wsga_hp_v3
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-1499
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Hyperparameter Sweep v3 — Production-Ready with GRU Init
# ============================================================
#
# Final paper-ready sweep with:
#   - GRU prior for initial population (replaces 8-gram)
#   - Fixed mutation rate (adaptive mutation removed)
#   - Stagnation restarts still active (population injection)
#   - NIST 8100 Mordred models
#
# Parameters swept:
#   - Population size:     [500, 1000, 2000, 3000]    (4)
#   - Mutation rate:       [0.3, 0.5, 0.7, 0.8, 0.9]  (5)
#   - Elitism rate:        [0.1, 0.25, 0.5, 0.75, 0.9] (5)
#   - Tournament k:        [2, 3, 5]                    (3)
#   - Seeds:               [42, 123, 7, 256, 999]       (5)
#
# Fixed:
#   - Tau               = 0.15
#   - Best elite ratio  = 0.5
#   - Generations       = 150
#   - Init method       = gru
#   - Biodeg filter     = ON
#   - Target            = FOM1
#
# Total: 4 * 5 * 5 * 3 * 5 = 1,500 jobs
#
# Usage:
#   qsub hyperparam_sweep_v3.sh
#   qsub -J 0-0 hyperparam_sweep_v3.sh   # single test job
#   bash hyperparam_sweep_v3.sh <array_index>
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash hyperparam_sweep_v3.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

# ==============================
# Parameter grid
# ==============================
pop_values=(500 1000 2000 3000)
mr_values=(0.3 0.5 0.7 0.8 0.9)
er_values=(0.1 0.25 0.5 0.75 0.9)
k_values=(2 3 5)
seeds=(42 123 7 256 999)

# ==============================
# Decode array index
# ==============================
n_pop=${#pop_values[@]}
n_mr=${#mr_values[@]}
n_er=${#er_values[@]}
n_k=${#k_values[@]}
n_seed=${#seeds[@]}

seed_idx=$((N % n_seed))
remaining=$((N / n_seed))

k_idx=$((remaining % n_k))
remaining=$((remaining / n_k))

er_idx=$((remaining % n_er))
remaining=$((remaining / n_er))

mr_idx=$((remaining % n_mr))
remaining=$((remaining / n_mr))

pop_idx=$((remaining % n_pop))

POP=${pop_values[$pop_idx]}
MR=${mr_values[$mr_idx]}
ER=${er_values[$er_idx]}
K=${k_values[$k_idx]}
SEED=${seeds[$seed_idx]}

# Fixed parameters
TARGET="FOM1"
TAU=0.15
BER=0.5
NUM_GENERATIONS=150

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/hyperparam_sweep_v3/pop${POP}_mr${MR}_er${ER}_k${K}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA Hyperparameter Sweep v3" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N" >> "$log_file"
echo "Target:            $TARGET" >> "$log_file"
echo "Population size:   $POP" >> "$log_file"
echo "Mutation rate:     $MR (fixed)" >> "$log_file"
echo "Elitism rate:      $ER" >> "$log_file"
echo "Best elite ratio:  $BER (fixed)" >> "$log_file"
echo "Tournament k:      $K" >> "$log_file"
echo "Tau (niching):     $TAU (fixed)" >> "$log_file"
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
    --num_generations $NUM_GENERATIONS \
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
