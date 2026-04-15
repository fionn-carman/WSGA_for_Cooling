#!/bin/bash
#PBS -N reinvent_sphere
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-29
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT Sphere Exclusion max_per_sphere Sweep
# ============================================================
#
# 6 max_per_sphere values × 5 seeds = 30 jobs:
#   0-4:   mps=5   (very aggressive)
#   5-9:   mps=10
#  10-14:  mps=25
#  15-19:  mps=50
#  20-24:  mps=100
#  25-29:  no filter (baseline)
#
# Fingerprint: Morgan radius 2 (ECFP4), matching WSGA niching production.
# Sphere threshold: 0.45 Tanimoto — empirically tuned for family-level
# clustering on the REINVENT output distribution (mean cluster size ~7).
#
# All other HPs are identical to the BRICS mps sweep (reinvent_brics_mps_sweep.sh):
#   sigma=1.0, lr=5e-5, batch=128, replay=100, rl_steps=5000, patience=200,
#   constraints fp>=100C, mp<=-30C, bp>=100C, dc<=7, sc<=3, tox<=3, no_biodeg.
#
# Usage:
#   qsub reinvent_sphere_mps_sweep.sh
#   bash reinvent_sphere_mps_sweep.sh <array_index>
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_sphere_mps_sweep.sh <array_index>"
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

# Manuscript-optimal REINVENT HPs
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

# Sphere-mode defaults (both fixed for this sweep)
SPHERE_THRESHOLD=0.45
SPHERE_FP_RADIUS=2

case $CONDITION in
    0) LABEL="mps5";         DIVERSITY_FLAGS="--diversity_filter --diversity_mode sphere --max_per_scaffold 5"   ;;
    1) LABEL="mps10";        DIVERSITY_FLAGS="--diversity_filter --diversity_mode sphere --max_per_scaffold 10"  ;;
    2) LABEL="mps25";        DIVERSITY_FLAGS="--diversity_filter --diversity_mode sphere --max_per_scaffold 25"  ;;
    3) LABEL="mps50";        DIVERSITY_FLAGS="--diversity_filter --diversity_mode sphere --max_per_scaffold 50"  ;;
    4) LABEL="mps100";       DIVERSITY_FLAGS="--diversity_filter --diversity_mode sphere --max_per_scaffold 100" ;;
    5) LABEL="mps_disabled"; DIVERSITY_FLAGS="--no_diversity_filter"                                             ;;
    *) echo "ERROR: Unknown condition $CONDITION"; exit 1 ;;
esac

output_dir="../outputs/reinvent_sphere_mps_sweep/${LABEL}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
{
    echo "========================================"
    echo "REINVENT Sphere Exclusion mps Sweep"
    echo "========================================"
    echo "Array index:       $N"
    echo "Condition:         $LABEL"
    echo "Seed:              $SEED"
    echo "Sigma:             $SIGMA"
    echo "Diversity flags:   $DIVERSITY_FLAGS"
    echo "Sphere threshold:  $SPHERE_THRESHOLD"
    echo "Sphere fp_radius:  $SPHERE_FP_RADIUS (ECFP4)"
    echo "Prior:             gru_prior.pt (3x aug, 95% valid)"
    echo "Started at:        $(date)"
    echo "========================================"
} > "$log_file"

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
    --tanimoto_threshold $SPHERE_THRESHOLD \
    --tanimoto_fp_radius $SPHERE_FP_RADIUS \
    $DIVERSITY_FLAGS \
    --replay_buffer_size $REPLAY \
    --replay_fraction $REPLAY_FRACTION \
    --model_dir ../models \
    --training_data_dir ../training/data \
    --output_dir $output_dir \
    2>&1 | tee -a "$log_file"

exit_status=${PIPESTATUS[0]}

{
    echo ""
    echo "Finished at: $(date)"
    echo "Exit status: $exit_status"
} >> "$log_file"

exit $exit_status
