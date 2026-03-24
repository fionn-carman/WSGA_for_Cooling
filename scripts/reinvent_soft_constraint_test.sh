#!/bin/bash
#PBS -N reinvent_soft_test
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-3
#PBS -o /dev/null
#PBS -e /dev/null

# 4-job soft constraint test: nocost x 4 categories x seed 42
# Diversity ON (scaffold filter + experience replay)
# Soft constraints ON (sigmoid penalties instead of hard gates)
#
# Maps array index 0-3 to main sweep indices 0,5,10,15 (nocost level)

set -o pipefail

DIRECTORY="$PBS_O_WORKDIR"
LOCAL_IDX=${PBS_ARRAY_INDEX}

if [ -z "$LOCAL_IDX" ]; then
    if [ $# -eq 1 ]; then
        LOCAL_IDX=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_soft_constraint_test.sh <0-3>"
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
# REINVENT hyperparameters
# ==============================
PRETRAIN_EPOCHS=30
RL_STEPS=5000
BATCH_SIZE=128
SIGMA=0.5
LR_RL=5e-5
CONVERGENCE_PATIENCE=200
SEED=42

# ==============================
# Fixed constraint thresholds
# ==============================
TARGET="FOM1"
MP_THRESHOLD=-30
FP_THRESHOLD=373
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# nocost (no MolPrice penalty)
MOLPRICE_SOFT=0.0
MOLPRICE_HARD=0.0

# ==============================
# Category decoding (0-3)
# ==============================
category_labels=(bio_stable bio_unstable nonbio_stable nonbio_unstable)
CATEGORY=${category_labels[$LOCAL_IDX]}

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

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/reinvent_soft_constraint_test/nocost_${CATEGORY}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "REINVENT Soft Constraint Test"            >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $LOCAL_IDX"            >> "$log_file"
echo "Category:          $CATEGORY"             >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "Soft constraints:  ON"                    >> "$log_file"
echo "Diversity filter:  ON"                    >> "$log_file"
echo "Pretrain epochs:   $PRETRAIN_EPOCHS"      >> "$log_file"
echo "RL steps (max):    $RL_STEPS"             >> "$log_file"
echo "Batch size:        $BATCH_SIZE"           >> "$log_file"
echo "Sigma:             $SIGMA"                >> "$log_file"
echo "LR (RL):           $LR_RL"               >> "$log_file"
echo "Patience:          $CONVERGENCE_PATIENCE" >> "$log_file"
echo "Seed:              $SEED"                 >> "$log_file"
echo "MP threshold:      $MP_THRESHOLD"         >> "$log_file"
echo "FP threshold:      $FP_THRESHOLD"         >> "$log_file"
echo "BP threshold:      $BP_THRESHOLD"         >> "$log_file"
echo "DC threshold:      $DC_THRESHOLD"         >> "$log_file"
echo "Biodeg flag:       ${BIODEG_FLAG:-(enabled)}" >> "$log_file"
echo "Stability flag:    ${STABILITY_FLAG:-(disabled)}" >> "$log_file"
echo "MolPrice soft/hard: ${MOLPRICE_SOFT}/${MOLPRICE_HARD}" >> "$log_file"
echo "Started at:        $(date)"               >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Run REINVENT with soft constraints
# ==============================
python3 ../src/reinvent/run_reinvent.py \
    --target $TARGET \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --rl_steps $RL_STEPS \
    --batch_size $BATCH_SIZE \
    --sigma $SIGMA \
    --lr_rl $LR_RL \
    --convergence_patience $CONVERGENCE_PATIENCE \
    --seed $SEED \
    --mp_threshold $MP_THRESHOLD \
    --fp_threshold $FP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --molprice_soft $MOLPRICE_SOFT \
    --molprice_hard $MOLPRICE_HARD \
    --soft_constraints \
    --diversity_filter \
    --max_per_scaffold 25 \
    --replay_buffer_size 100 \
    --replay_fraction 0.5 \
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
