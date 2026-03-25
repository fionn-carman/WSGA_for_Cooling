#!/bin/bash
#PBS -N wsga_pubchem_init
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-19
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# WSGA PubChem Initial Population Sweep
# ============================================================
#
# Tests whether sampling the initial population from the 500K
# PubChem CHO corpus (instead of n-gram) changes GA convergence.
#
# Sweep grid: 4 categories x 5 seeds = 20 jobs
# Categories match the REINVENT PubChem sweep for direct comparison.
#
# Usage:
#   qsub wsga_pubchem_sweep.sh
#   qsub -J 0-0 wsga_pubchem_sweep.sh   # single test job
#   bash wsga_pubchem_sweep.sh 0          # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash wsga_pubchem_sweep.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

# Prevent OMP/MKL segfault when PyTorch + XGBoost coexist
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# ==============================
# Fixed WSGA hyperparameters
# ==============================
POP=2000
NUM_GENERATIONS=150
MR=0.8
ER=0.4
BER=0.3
K=3
TAU=0.15

# ==============================
# Sweep grid
# ==============================
seeds=(42 123 7 256 999)
category_labels=(bio_stable bio_unstable nonbio_stable nonbio_unstable)

# ==============================
# Decode array index
# ==============================
n_seed=${#seeds[@]}
n_category=${#category_labels[@]}

category_idx=$((N / n_seed))
seed_idx=$((N % n_seed))

CATEGORY=${category_labels[$category_idx]}
SEED=${seeds[$seed_idx]}

# ==============================
# Category flags
# ==============================
# Biodeg flag
case "$CATEGORY" in
    bio_stable|bio_unstable)
        BIODEG_FLAG=""
        ;;
    nonbio_stable|nonbio_unstable)
        BIODEG_FLAG="--no_biodeg"
        ;;
esac

# Stability flag
case "$CATEGORY" in
    bio_stable|nonbio_stable)
        STABILITY_FLAG="--stability_mode strict"
        ;;
    bio_unstable|nonbio_unstable)
        STABILITY_FLAG=""
        ;;
esac

# Constraint thresholds (matching REINVENT PubChem sweep)
TARGET="FOM1"
MP_THRESHOLD=-30
FP_THRESHOLD=373
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/wsga_pubchem_init/${CATEGORY}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "WSGA PubChem Init Sweep"                  >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Category:          $CATEGORY"             >> "$log_file"
echo "Seed:              $SEED"                 >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "Init method:       pubchem"               >> "$log_file"
echo "Population size:   $POP"                  >> "$log_file"
echo "Generations:       $NUM_GENERATIONS"       >> "$log_file"
echo "Mutation rate:     $MR"                   >> "$log_file"
echo "Elitism rate:      $ER"                   >> "$log_file"
echo "Best elite ratio:  $BER"                  >> "$log_file"
echo "Tournament k:      $K"                    >> "$log_file"
echo "Tau:               $TAU"                  >> "$log_file"
echo "MP threshold:      $MP_THRESHOLD"         >> "$log_file"
echo "FP threshold:      $FP_THRESHOLD"         >> "$log_file"
echo "BP threshold:      $BP_THRESHOLD"         >> "$log_file"
echo "DC threshold:      $DC_THRESHOLD"         >> "$log_file"
echo "Biodeg flag:       ${BIODEG_FLAG:-(enabled)}" >> "$log_file"
echo "Stability flag:    ${STABILITY_FLAG:-(disabled)}" >> "$log_file"
echo "Started at:        $(date)"               >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Run WSGA
# ==============================
python3 ../src/wsga.py \
    --target $TARGET \
    --init_method pubchem \
    --pubchem_path ../data/pubchem_cho_5_30ha.csv \
    --seed $SEED \
    --population_size $POP \
    --num_generations $NUM_GENERATIONS \
    --elitism_rate $ER \
    --best_elite_ratio $BER \
    --mutation_rate $MR \
    --tournament_k $K \
    --Tau $TAU \
    --mp_threshold $MP_THRESHOLD \
    --fp_threshold $FP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --data_dir ../data \
    --model_dir ../models \
    --training_data_dir ../training/data \
    --output_dir $output_dir \
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
