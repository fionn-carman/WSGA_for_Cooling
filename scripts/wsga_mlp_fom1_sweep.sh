#!/bin/bash
#PBS -N wsga_mlp_fom1
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-19
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# WSGA MLP FOM1 Sweep (nocost, 4 categories x 5 seeds)
# ============================================================
#
# Runs WSGA with MLP FOM1 predictions instead of XGBoost,
# to test whether the MLP's reduced regression-to-mean bias
# leads to different/better candidate discovery.
#
# All other property models (BP, MP, FP, DC, density, viscosity,
# tc, cpsat, beta) remain XGBoost.
#
# Sweep grid: 4 categories x 5 seeds = 20 jobs
#
# Categories:
#   0  bio_stable     biodeg + stability_mode strict
#   1  bio_unstable   biodeg only
#   2  nonbio_stable  no_biodeg + stability_mode strict
#   3  nonbio_unstable no_biodeg only
#
# Index decoding:
#   category_idx = N / n_seed    # 0-3
#   seed_idx     = N % n_seed    # 0-4
#
# Usage:
#   qsub wsga_mlp_fom1_sweep.sh
#   qsub -J 0-0 wsga_mlp_fom1_sweep.sh   # single test job
#   bash wsga_mlp_fom1_sweep.sh 0          # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash wsga_mlp_fom1_sweep.sh <array_index>"
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
category_labels=(bio_stable bio_unstable nonbio_stable nonbio_unstable)

# ==============================
# Decode array index
# ==============================
n_seed=${#seeds[@]}

category_idx=$((N / n_seed))
seed_idx=$((N % n_seed))

SEED=${seeds[$seed_idx]}
CATEGORY=${category_labels[$category_idx]}

# ==============================
# Category flags
# ==============================
case "$CATEGORY" in
    bio_stable|bio_unstable)
        BIODEG_FLAG=""
        ;;
    nonbio_stable|nonbio_unstable)
        BIODEG_FLAG="--no_biodeg"
        ;;
esac

case "$CATEGORY" in
    bio_stable|nonbio_stable)
        STABILITY_FLAG="--stability_mode strict"
        ;;
    bio_unstable|nonbio_unstable)
        STABILITY_FLAG=""
        ;;
esac

# Fixed constraint thresholds
TARGET="FOM1"
MP_THRESHOLD=-30
FP_THRESHOLD=373
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# MLP FOM1 model
FOM1_MLP_MODEL="../models/fom1_40C_mlp/model.joblib"

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/wsga_mlp_fom1_sweep/${CATEGORY}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA MLP FOM1 Sweep"                     >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Category:          $CATEGORY"             >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "FOM1 model:        MLP ($FOM1_MLP_MODEL)" >> "$log_file"
echo "Population size:   $POP"                  >> "$log_file"
echo "Mutation rate:     $MR"                   >> "$log_file"
echo "Elitism rate:      $ER"                   >> "$log_file"
echo "Best elite ratio:  $BER"                  >> "$log_file"
echo "Tournament k:      $K"                    >> "$log_file"
echo "Tau (niching):     $TAU"                  >> "$log_file"
echo "Generations:       $NUM_GENERATIONS"      >> "$log_file"
echo "Seed:              $SEED"                 >> "$log_file"
echo "MP threshold:      $MP_THRESHOLD"         >> "$log_file"
echo "FP threshold:      $FP_THRESHOLD"         >> "$log_file"
echo "BP threshold:      $BP_THRESHOLD"         >> "$log_file"
echo "DC threshold:      $DC_THRESHOLD"         >> "$log_file"
echo "Biodeg flag:       ${BIODEG_FLAG:-(enabled)}" >> "$log_file"
echo "Stability flag:    ${STABILITY_FLAG:-(disabled)}" >> "$log_file"
echo "MolPrice:          nocost (disabled)"     >> "$log_file"
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
    --mp_threshold $MP_THRESHOLD \
    --fp_threshold $FP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --data_dir ../data \
    --model_dir ../models \
    --output_dir $output_dir \
    --fom1_mlp_model $FOM1_MLP_MODEL \
    --seed $SEED \
    $BIODEG_FLAG \
    $STABILITY_FLAG \
    2>&1 | tee -a "$log_file"

exit_status=${PIPESTATUS[0]}

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: $exit_status" >> "$log_file"

echo "" >> "$log_file"
echo "Output files:" >> "$log_file"
ls -la "$output_dir"/ >> "$log_file" 2>&1

exit $exit_status
