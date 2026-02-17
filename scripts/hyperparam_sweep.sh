#!/bin/bash
#PBS -N wsga_hyperparam
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-959
#PBS -o /dev/null
#PBS -e /dev/null

# ============================================================
# Hyperparameter Sweep for WSGA
# ============================================================
#
# Parameters swept:
#   - Population size:  [500, 750, 1000, 1500, 2000]  (5)
#   - Mutation rate:    [0.5, 0.7, 0.8, 0.9]          (4)
#   - Elitism rate:     [0.1, 0.2, 0.3, 0.4]          (4)
#   - Tau (niching):    [0.05, 0.10, 0.15, 0.20]      (4)
#   - Seeds:            [42, 123, 7]                   (3)
#
# Total: 5 * 4 * 4 * 4 * 3 = 960 jobs
# Biodegradability filter is ON for all runs.
#
# Usage:
#   qsub hyperparam_sweep.sh
#   qsub -J 0-0 hyperparam_sweep.sh   # single test job
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash hyperparam_sweep.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env
cd "$DIRECTORY"

# ==============================
# Parameter grid
# ==============================
pop_values=(500 750 1000 1500 2000)
mr_values=(0.5 0.7 0.8 0.9)
er_values=(0.1 0.2 0.3 0.4)
tau_values=(0.05 0.10 0.15 0.20)
seeds=(42 123 7)

# ==============================
# Decode array index
# ==============================
n_pop=${#pop_values[@]}
n_mr=${#mr_values[@]}
n_er=${#er_values[@]}
n_tau=${#tau_values[@]}
n_seed=${#seeds[@]}

seed_idx=$((N % n_seed))
remaining=$((N / n_seed))

tau_idx=$((remaining % n_tau))
remaining=$((remaining / n_tau))

er_idx=$((remaining % n_er))
remaining=$((remaining / n_er))

mr_idx=$((remaining % n_mr))
remaining=$((remaining / n_mr))

pop_idx=$((remaining % n_pop))

POP=${pop_values[$pop_idx]}
MR=${mr_values[$mr_idx]}
ER=${er_values[$er_idx]}
TAU=${tau_values[$tau_idx]}
SEED=${seeds[$seed_idx]}
NUM_GENERATIONS=200

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
output_dir="../outputs/hyperparam_sweep/pop${POP}_mr${MR}_er${ER}_tau${TAU}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA Hyperparameter Sweep" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N" >> "$log_file"
echo "Target:            $TARGET" >> "$log_file"
echo "Population size:   $POP" >> "$log_file"
echo "Mutation rate:     $MR" >> "$log_file"
echo "Elitism rate:      $ER" >> "$log_file"
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
