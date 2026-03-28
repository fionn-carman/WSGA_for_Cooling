#!/bin/bash
#PBS -N reinvent_div_nist8100
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-119
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT NIST 8100 Diversity Sweep
# ============================================================
#
# Matches molprice_sweep_nist8100.sh exactly: same 120-job grid,
# same constraint thresholds, same model paths, same stability modes.
#
# Sweep grid: 6 threshold levels x 4 categories x 5 seeds = 120 jobs
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
# Index decoding:
#   level_idx    = N / (n_category * n_seed)    # 0-5
#   category_idx = (N / n_seed) % n_category    # 0-3
#   seed_idx     = N % n_seed                   # 0-4
#
# Usage:
#   qsub reinvent_sweep_nist8100.sh
#   qsub -J 0-0 reinvent_sweep_nist8100.sh   # single test job
#   bash reinvent_sweep_nist8100.sh 0          # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_sweep_nist8100.sh <array_index>"
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

# ==============================
# Sweep grid
# ==============================
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

level_idx=$((N / (n_category * n_seed)))
category_idx=$(( (N / n_seed) % n_category ))
seed_idx=$((N % n_seed))

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

# MolPrice thresholds (no separate model flag -- always loaded from default)
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
# Setup output directory
# ==============================
output_dir="../outputs/reinvent_diversity_sweep_nist8100/${LEVEL}_${CATEGORY}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "REINVENT NIST 8100 Sweep"                 >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Level:             $LEVEL"                >> "$log_file"
echo "Category:          $CATEGORY"             >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
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
# Run REINVENT
# ==============================
python3 ../src/reinvent/run_reinvent.py \
    --target $TARGET \
    --prior_path ../models/init_corpus/gru_prior.pt \
    --vocab_path ../models/init_corpus/vocabulary.json \
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
