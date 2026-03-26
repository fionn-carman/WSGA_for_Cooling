#!/bin/bash
#PBS -N reinvent_sigma_nist8100
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-359
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT Sigma Sweep (NIST 8100 + Diversity)
# ============================================================
#
# Reuses the same 6x4x5 grid as reinvent_sweep_nist8100.sh,
# but sweeps sigma = {1.0, 2.0, 4.0} to increase reward
# influence relative to the prior anchor.
#
# Sweep grid: 3 sigmas x 6 levels x 4 categories x 5 seeds = 360 jobs
#
# Sigma values:
#   0  1.0   Moderate increase (2x baseline)
#   1  2.0   Strong increase (4x baseline)
#   2  4.0   Very strong (8x baseline, reward-dominated)
#
# Threshold levels (soft, hard) -- fixed width of 2.0:
#   0  nocost      --    --     No MolPrice penalty (baseline)
#   1  gentle      4.0  6.0   Mild pressure
#   2  moderate    3.5  5.5   Moderate
#   3  firm        3.0  5.0   Noticeable
#   4  tight       2.5  4.5   Strong
#   5  aggressive  2.0  4.0   Very strong
#
# Categories:
#   0  bio_stable     biodeg + stability_mode strict
#   1  bio_unstable   biodeg only
#   2  nonbio_stable  no_biodeg + stability_mode strict
#   3  nonbio_unstable no_biodeg only
#
# Index decoding (360 jobs):
#   sigma_idx    = N / (n_level * n_category * n_seed)   # 0-2
#   level_idx    = (N / (n_category * n_seed)) % n_level # 0-5
#   category_idx = (N / n_seed) % n_category             # 0-3
#   seed_idx     = N % n_seed                            # 0-4
#
# Prior reuse: Phase 1 pretraining is sigma-independent, so we
# reuse prior.pt + vocabulary.json from the sigma=0.5 sweep if
# available. Falls back to pretraining from scratch if not found.
#
# Usage:
#   qsub reinvent_sigma_sweep_nist8100.sh
#   qsub -J 0-0 reinvent_sigma_sweep_nist8100.sh   # single test job
#   bash reinvent_sigma_sweep_nist8100.sh 0          # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_sigma_sweep_nist8100.sh <array_index>"
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
LR_RL=5e-5
CONVERGENCE_PATIENCE=200

# ==============================
# Sweep grid
# ==============================
sigmas=(1.0 2.0 4.0)
seeds=(42 123 7 256 999)

# Threshold levels: label, soft, hard (width = 2.0)
level_labels=(nocost gentle moderate firm tight aggressive)
level_soft=(0.0 4.0 3.5 3.0 2.5 2.0)
level_hard=(0.0 6.0 5.5 5.0 4.5 4.0)

category_labels=(bio_stable bio_unstable nonbio_stable nonbio_unstable)

# ==============================
# Decode array index
# ==============================
n_seed=${#seeds[@]}
n_category=${#category_labels[@]}
n_level=${#level_labels[@]}

sigma_idx=$((N / (n_level * n_category * n_seed)))
level_idx=$(( (N / (n_category * n_seed)) % n_level ))
category_idx=$(( (N / n_seed) % n_category ))
seed_idx=$((N % n_seed))

SIGMA=${sigmas[$sigma_idx]}
LEVEL=${level_labels[$level_idx]}
SEED=${seeds[$seed_idx]}
CATEGORY=${category_labels[$category_idx]}

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

# MolPrice thresholds
MOLPRICE_SOFT=${level_soft[$level_idx]}
MOLPRICE_HARD=${level_hard[$level_idx]}

# Fixed constraint thresholds (same as WSGA sweep)
TARGET="FOM1"
MP_THRESHOLD=-30
FP_THRESHOLD=373
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ==============================
# Prior reuse (Phase 1 is sigma-independent)
# ==============================
# Look for prior from the sigma=0.5 diversity sweep
PRIOR_DIR="../outputs/reinvent_diversity_sweep_nist8100/${LEVEL}_${CATEGORY}_seed${SEED}"
PRIOR_PT="${PRIOR_DIR}/prior.pt"
VOCAB_JSON="${PRIOR_DIR}/vocabulary.json"

PRIOR_FLAGS=""
if [ -f "$PRIOR_PT" ] && [ -f "$VOCAB_JSON" ]; then
    PRIOR_FLAGS="--prior_path $PRIOR_PT --vocab_path $VOCAB_JSON"
    PRIOR_STATUS="reused from $PRIOR_DIR"
else
    PRIOR_STATUS="training from scratch (prior not found at $PRIOR_DIR)"
fi

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/reinvent_sigma_sweep_nist8100/sigma${SIGMA}_${LEVEL}_${CATEGORY}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "REINVENT Sigma Sweep (NIST 8100)"        >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Sigma:             $SIGMA"                >> "$log_file"
echo "Level:             $LEVEL"                >> "$log_file"
echo "Category:          $CATEGORY"             >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "Pretrain epochs:   $PRETRAIN_EPOCHS"      >> "$log_file"
echo "RL steps (max):    $RL_STEPS"             >> "$log_file"
echo "Batch size:        $BATCH_SIZE"           >> "$log_file"
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
echo "Prior:             $PRIOR_STATUS"         >> "$log_file"
echo "Started at:        $(date)"               >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Run REINVENT
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
    --diversity_filter \
    --max_per_scaffold 25 \
    --replay_buffer_size 100 \
    --replay_fraction 0.5 \
    --model_dir ../models \
    --training_data_dir ../training/data \
    --output_dir $output_dir \
    $BIODEG_FLAG \
    $STABILITY_FLAG \
    $PRIOR_FLAGS \
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
