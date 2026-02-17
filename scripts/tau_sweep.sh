#!/bin/bash
#PBS -N wsga_tau_sweep
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-39
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Tau / Niching Sweep — Stage 2
# ============================================================
#
# Uses the best GA hyperparameters from Stage 1 (update below)
# and sweeps tau to characterise the niching penalty.
#
# Parameters swept:
#   - Tau (niching):  [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]  (8)
#   - Seeds:          [42, 123, 7, 256, 999]                              (5)
#
# Total: 8 * 5 = 40 jobs
#
# IMPORTANT: Update the "Best GA parameters from Stage 1" section
#            below with the results of analyse_hyperparam_sweep.py
#            before submitting this sweep.
#
# Usage:
#   qsub tau_sweep.sh
#   qsub -J 0-0 tau_sweep.sh   # single test job
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash tau_sweep.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env
cd "$DIRECTORY"

# ==============================
# Best GA parameters from Stage 1
# (UPDATE THESE after running analyse_hyperparam_sweep.py)
# ==============================
POP=2000
MR=0.8
ER=0.3
K=3
NUM_GENERATIONS=150

# ==============================
# Tau sweep grid
# ==============================
tau_values=(0.0 0.05 0.10 0.15 0.20 0.30 0.40 0.50)
seeds=(42 123 7 256 999)

# ==============================
# Decode array index
# ==============================
n_tau=${#tau_values[@]}
n_seed=${#seeds[@]}

seed_idx=$((N % n_seed))
tau_idx=$((N / n_seed))

TAU=${tau_values[$tau_idx]}
SEED=${seeds[$seed_idx]}

# Fixed parameters
TARGET="FOM1"
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
output_dir="../outputs/tau_sweep/tau${TAU}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA Tau Sweep — Stage 2" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N" >> "$log_file"
echo "Target:            $TARGET" >> "$log_file"
echo "Population size:   $POP (from Stage 1)" >> "$log_file"
echo "Mutation rate:     $MR (from Stage 1)" >> "$log_file"
echo "Elitism rate:      $ER (from Stage 1)" >> "$log_file"
echo "Tournament k:      $K (from Stage 1)" >> "$log_file"
echo "Tau (niching):     $TAU" >> "$log_file"
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
