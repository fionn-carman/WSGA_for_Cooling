#!/bin/bash
#PBS -N wsga_fom1_direct
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-9
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# WSGA FOM1 Direct Prediction Run
# ============================================================
#
# Uses XGBoost+Descriptor FOM1 models to predict FOM1 directly
# at 40C and 100C, and maximises their average.
#
# Two constraint variants:
#   - bio:     biodegradability filter ON  (indices 0-4)
#   - non_bio: biodegradability filter OFF (indices 5-9)
#
# Seeds: [42, 123, 7, 256, 999]  (5 per variant)
#
# Total: 2 * 5 = 10 jobs
#
# Uses best GA hyperparameters from previous sweeps.
#
# Usage:
#   qsub fom1_direct_run.sh
#   qsub -J 0-0 fom1_direct_run.sh   # single test job
#   bash fom1_direct_run.sh 0          # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash fom1_direct_run.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rdkit_env
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

# bio variant: indices 0-4, non_bio variant: indices 5-9
variant_idx=$((N / n_seed))
seed_idx=$((N % n_seed))
SEED=${seeds[$seed_idx]}

if [ $variant_idx -eq 0 ]; then
    VARIANT="bio"
    BIODEG_FLAG=""
else
    VARIANT="non_bio"
    BIODEG_FLAG="--no_biodeg"
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
output_dir="../outputs/fom1_direct/${VARIANT}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA FOM1 Direct Prediction Run" >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N" >> "$log_file"
echo "Variant:           $VARIANT" >> "$log_file"
echo "Target:            $TARGET" >> "$log_file"
echo "Population size:   $POP" >> "$log_file"
echo "Mutation rate:     $MR" >> "$log_file"
echo "Elitism rate:      $ER" >> "$log_file"
echo "Best elite ratio:  $BER" >> "$log_file"
echo "Tournament k:      $K" >> "$log_file"
echo "Tau (niching):     $TAU" >> "$log_file"
echo "Generations:       $NUM_GENERATIONS" >> "$log_file"
echo "Seed label:        $SEED" >> "$log_file"
echo "MP soft/hard:      ${MP_SOFT}/${MP_HARD}" >> "$log_file"
echo "FP threshold:      $FP_THRESHOLD" >> "$log_file"
echo "Biodeg flag:       ${BIODEG_FLAG:-(enabled)}" >> "$log_file"
echo "Started at:        $(date)" >> "$log_file"
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
    --fom1_model_dir ../training/FOM1_architecture_comparison/results \
    --data_dir ../data \
    --model_dir ../models \
    --output_dir $output_dir \
    $BIODEG_FLAG \
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
