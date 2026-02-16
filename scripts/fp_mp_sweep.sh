#!/bin/bash
#PBS -N wsga_fp_mp_sweep
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-389
#PBS -o /dev/null
#PBS -e /dev/null

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env
cd "$DIRECTORY"

# Parameter Arrays
fp_values=(50 60 70 80 90 100 110 120 130 140 150 160 170)
mp_values=(-50 -45 -40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15 20)
biodeg_options=("yes" "no")

# Decode Array Index
n_fp=${#fp_values[@]}
n_mp=${#mp_values[@]}
n_per_biodeg=$((n_fp * n_mp))

biodeg_idx=$((N / n_per_biodeg))
remaining=$((N % n_per_biodeg))
mp_idx=$((remaining / n_fp))
fp_idx=$((remaining % n_fp))

FP_C=${fp_values[$fp_idx]}
MP_C=${mp_values[$mp_idx]}
BIODEG=${biodeg_options[$biodeg_idx]}

FP_K=$(awk "BEGIN {printf \"%.2f\", $FP_C + 273.15}")
MP_HARD=$MP_C
MP_SOFT=$((MP_C - 20))

# Fixed Hyperparameters
TARGET="FOM1"
POP=1000
ER=0.1
MR=0.9
TAU=0.015
NUM_GENERATIONS=100

output_dir="../outputs/fp_mp_sweep/fp${FP_C}_mp${MP_C}_bio${BIODEG}"
mkdir -p "$output_dir"

CMD="python3 ../src/wsga.py \
    --target $TARGET \
    --population_size $POP \
    --elitism_rate $ER \
    --mutation_rate $MR \
    --Tau $TAU \
    --num_generations $NUM_GENERATIONS \
    --mp_soft $MP_SOFT \
    --mp_hard $MP_HARD \
    --fp_threshold $FP_K \
    --data_dir ../data \
    --model_dir ../models \
    --output_dir $output_dir"

if [ "$BIODEG" == "no" ]; then
    CMD="$CMD --no_biodeg"
fi

eval $CMD
