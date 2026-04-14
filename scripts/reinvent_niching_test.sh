#!/bin/bash
#PBS -N reinvent_niche
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-4
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT Tanimoto Niching Test
# ============================================================
#
# Quick test: scaffold filter OFF, Tanimoto niching ON (tau=0.15).
# 5 seeds at FP >= 100 C. Compare with scaffold sweep disabled
# condition to see if fingerprint-based diversity helps.
#
# Usage:
#   qsub reinvent_niching_test.sh
#   bash reinvent_niching_test.sh <array_index>  # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_niching_test.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

seeds=(42 123 7 256 999)
SEED=${seeds[$N]}

# Fixed settings (best HP sweep config)
TARGET="FOM1"
SIGMA=0.5
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

output_dir="../outputs/reinvent_niching_test/niching_tau0.15_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "REINVENT Tanimoto Niching Test"           >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Scaffold filter:   OFF"                   >> "$log_file"
echo "Tanimoto niching:  ON (tau=0.15, r=8)"    >> "$log_file"
echo "Seed:              $SEED"                 >> "$log_file"
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
    --no_diversity_filter \
    --tanimoto_niching \
    --niching_tau 0.15 \
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
