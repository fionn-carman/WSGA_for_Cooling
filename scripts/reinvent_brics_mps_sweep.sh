#!/bin/bash
#PBS -N reinvent_brics_mps
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-39
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT BRICS max_per_scaffold Sweep
# ============================================================
#
# 8 mps settings x 5 seeds = 40 jobs:
#   0-4:   BRICS max_per_scaffold=5
#   5-9:   BRICS max_per_scaffold=10
#  10-14:  BRICS max_per_scaffold=25
#  15-19:  BRICS max_per_scaffold=50
#  20-24:  BRICS max_per_scaffold=100
#  25-29:  BRICS max_per_scaffold=250
#  30-34:  BRICS max_per_scaffold=500
#  35-39:  No diversity filter (mps_disabled baseline)
#
# Uses manuscript-optimal HPs (sigma=1.0, lr=5e-5, bs=128, replay=100).
# Prior: gru_prior.pt (3x SMILES augmentation, ~95% validity, ~91% unique).
# Isolates the BRICS filter effect — no Tanimoto niching.
#
# Usage:
#   qsub reinvent_brics_mps_sweep.sh
#   bash reinvent_brics_mps_sweep.sh <array_index>
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_brics_mps_sweep.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

export PYTHONUNBUFFERED=1
cd "$DIRECTORY"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

seeds=(42 123 7 256 999)
SEED=${seeds[$((N % 5))]}
CONDITION=$((N / 5))

# Manuscript-optimal HPs
TARGET="FOM1"
SIGMA=1.0
LR=5e-5
BATCH_SIZE=128
REPLAY=100
REPLAY_FRACTION=0.5
RL_STEPS=5000
CONVERGENCE_PATIENCE=200

# Constraints (FP >= 100 C)
FP_THRESHOLD=373
MP_THRESHOLD=-30
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ==============================
# Decode condition
# ==============================
case $CONDITION in
    0) LABEL="mps5";         DIVERSITY_FLAGS="--diversity_filter --diversity_mode brics --max_per_scaffold 5"   ;;
    1) LABEL="mps10";        DIVERSITY_FLAGS="--diversity_filter --diversity_mode brics --max_per_scaffold 10"  ;;
    2) LABEL="mps25";        DIVERSITY_FLAGS="--diversity_filter --diversity_mode brics --max_per_scaffold 25"  ;;
    3) LABEL="mps50";        DIVERSITY_FLAGS="--diversity_filter --diversity_mode brics --max_per_scaffold 50"  ;;
    4) LABEL="mps100";       DIVERSITY_FLAGS="--diversity_filter --diversity_mode brics --max_per_scaffold 100" ;;
    5) LABEL="mps250";       DIVERSITY_FLAGS="--diversity_filter --diversity_mode brics --max_per_scaffold 250" ;;
    6) LABEL="mps500";       DIVERSITY_FLAGS="--diversity_filter --diversity_mode brics --max_per_scaffold 500" ;;
    7) LABEL="mps_disabled"; DIVERSITY_FLAGS="--no_diversity_filter"                                            ;;
    *) echo "ERROR: Unknown condition $CONDITION"; exit 1 ;;
esac

output_dir="../outputs/reinvent_brics_mps_sweep/${LABEL}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "REINVENT BRICS max_per_scaffold Sweep"    >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Condition:         $LABEL"                >> "$log_file"
echo "Seed:              $SEED"                 >> "$log_file"
echo "Sigma:             $SIGMA"                >> "$log_file"
echo "Diversity flags:   $DIVERSITY_FLAGS"      >> "$log_file"
echo "Prior:             gru_prior.pt (3x aug)" >> "$log_file"
echo "Started at:        $(date)"               >> "$log_file"
echo "========================================" >> "$log_file"

python3 ../src/reinvent/run_reinvent.py \
    --target $TARGET \
    --prior_path ../models/init_corpus/gru_prior.pt \
    --vocab_path ../models/init_corpus/vocabulary.json \
    --rl_steps $RL_STEPS \
    --batch_size $BATCH_SIZE \
    --sigma $SIGMA \
    --lr_rl $LR \
    --convergence_patience $CONVERGENCE_PATIENCE \
    --seed $SEED \
    --fp_threshold $FP_THRESHOLD \
    --mp_threshold $MP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --no_biodeg \
    --molprice_soft 0.0 \
    --molprice_hard 0.0 \
    $DIVERSITY_FLAGS \
    --replay_buffer_size $REPLAY \
    --replay_fraction $REPLAY_FRACTION \
    --model_dir ../models \
    --training_data_dir ../training/data \
    --output_dir $output_dir \
    2>&1 | tee -a "$log_file"

exit_status=${PIPESTATUS[0]}

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: $exit_status" >> "$log_file"

exit $exit_status
