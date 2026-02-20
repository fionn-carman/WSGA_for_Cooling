#!/bin/bash
#PBS -N wsga_stage2
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-959
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Hyperparameter Sweep — Stage 2: Full GA Parameter Sweep
# ============================================================
#
# Comprehensive sweep assuming no prior knowledge, covering all
# key GA hyperparameters including the newly exposed
# BEST_ELITE_RATIO (exploitation vs diversity in elite selection).
#
# Parameters swept:
#   - Population size:     [500, 1000, 2000, 3000]              (4)
#   - Elitism rate:        [0.1, 0.25, 0.5, 0.75]              (4)
#   - Best elite ratio:   [0.0, 0.25, 0.5, 0.75, 1.0]         (5)
#   - Tournament k:        [2, 3, 5, 7]                         (4)
#   - Seeds:              [42, 123, 7]                           (3)
#
# Fixed:
#   - Mutation rate  = 0.8 (adaptive mutation overrides this)
#   - Tau            = 0.15
#   - Generations    = 150
#   - Biodegradability filter ON
#
# Total: 4 * 4 * 5 * 4 * 3 = 960 jobs
#
# Usage:
#   qsub hyperparam_sweep_stage2.sh
#   qsub -J 0-0 hyperparam_sweep_stage2.sh   # single test job
#   bash hyperparam_sweep_stage2.sh <array_index>
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash hyperparam_sweep_stage2.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env
cd "$DIRECTORY"

# ==============================
# Parameter grid
# ==============================
pop_values=(500 1000 2000 3000)
er_values=(0.1 0.25 0.5 0.75)
ber_values=(0.0 0.25 0.5 0.75 1.0)
k_values=(2 3 5 7)
seeds=(42 123 7)

# ==============================
# Decode array index
# ==============================
n_pop=${#pop_values[@]}
n_er=${#er_values[@]}
n_ber=${#ber_values[@]}
n_k=${#k_values[@]}
n_seed=${#seeds[@]}

seed_idx=$((N % n_seed))
remaining=$((N / n_seed))

k_idx=$((remaining % n_k))
remaining=$((remaining / n_k))

ber_idx=$((remaining % n_ber))
remaining=$((remaining / n_ber))

er_idx=$((remaining % n_er))
remaining=$((remaining / n_er))

pop_idx=$((remaining % n_pop))

POP=${pop_values[$pop_idx]}
ER=${er_values[$er_idx]}
BER=${ber_values[$ber_idx]}
K=${k_values[$k_idx]}
SEED=${seeds[$seed_idx]}
NUM_GENERATIONS=150

# Fixed parameters
TARGET="FOM1"
MR=0.8
TAU=0.15
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
output_dir="../outputs/hyperparam_sweep_stage2/pop${POP}_er${ER}_ber${BER}_k${K}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA Hyperparameter Sweep — Stage 2" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N" >> "$log_file"
echo "Target:            $TARGET" >> "$log_file"
echo "Population size:   $POP" >> "$log_file"
echo "Elitism rate:      $ER" >> "$log_file"
echo "Best elite ratio:  $BER" >> "$log_file"
echo "Tournament k:      $K" >> "$log_file"
echo "Mutation rate:     $MR (fixed — adaptive)" >> "$log_file"
echo "Tau (niching):     $TAU (fixed)" >> "$log_file"
echo "Generations:       $NUM_GENERATIONS" >> "$log_file"
echo "Seed label:        $SEED" >> "$log_file"
echo "MP soft/hard:      ${MP_SOFT}/${MP_HARD}" >> "$log_file"
echo "FP threshold:      $FP_THRESHOLD" >> "$log_file"
echo "Started at:        $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Run WSGA
# ==============================
python3 ../src/wsga.py \
    --target $TARGET \
    --population_size $POP \
    --elitism_rate $ER \
    --mutation_rate $MR \
    --tournament_k $K \
    --best_elite_ratio $BER \
    --Tau $TAU \
    --num_generations $NUM_GENERATIONS \
    --mp_soft $MP_SOFT \
    --mp_hard $MP_HARD \
    --fp_threshold $FP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --data_dir ../data \
    --model_dir ../models \
    --output_dir $output_dir \
    2>&1 | tee -a "$log_file"

exit_status=$?

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: $exit_status" >> "$log_file"

exit $exit_status
