#!/bin/bash
#PBS -N reinvent_prod
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-59
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT Production FP × MolPrice Sweep
# ============================================================
#
# Identical grid to the WSGA production sweep
# (scripts/fp_molprice_sweep_nist8100.sh):
#
#   4 FP levels × 3 MolPrice levels × 5 seeds = 60 jobs
#
#   FP:       100C (373K), 125C (398K), 150C (423K), 175C (448K)
#   MolPrice: nocost (0/0), gentle (4/6), aggressive (2/4)
#   Seeds:    42, 123, 7, 256, 999
#
# REINVENT best HPs (from manuscript HP sweep):
#   sigma=1.0, lr=5e-5, batch=128, replay=100, patience=200
#
# Sphere diversity filter (from calibration):
#   mode=sphere, threshold=0.35, fp_radius=2, max_per_scaffold=100
#
# Usage:
#   qsub reinvent_fp_molprice_production.sh
#   bash reinvent_fp_molprice_production.sh <array_index>
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_fp_molprice_production.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

export PYTHONUNBUFFERED=1
cd "$DIRECTORY"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# ── Grid encoding (matches WSGA sweep exactly) ──
FP_LABELS=(fp100 fp125 fp150 fp175)
FP_VALUES=(373.15 398.15 423.15 448.15)
LEVEL_LABELS=(nocost gentle aggressive)
LEVEL_SOFT=(0.0 4.0 2.0)
LEVEL_HARD=(0.0 6.0 4.0)
seeds=(42 123 7 256 999)

n_level=${#LEVEL_LABELS[@]}
n_seed=${#seeds[@]}

fp_idx=$((N / (n_level * n_seed)))
level_idx=$(((N / n_seed) % n_level))
seed_idx=$((N % n_seed))

FP_LABEL=${FP_LABELS[$fp_idx]}
FP_THRESHOLD=${FP_VALUES[$fp_idx]}
LEVEL=${LEVEL_LABELS[$level_idx]}
MOLPRICE_SOFT=${LEVEL_SOFT[$level_idx]}
MOLPRICE_HARD=${LEVEL_HARD[$level_idx]}
SEED=${seeds[$seed_idx]}

# ── REINVENT best HPs ──
TARGET="FOM1"
SIGMA=1.0
LR=5e-5
BATCH_SIZE=128
REPLAY=100
REPLAY_FRACTION=0.5
RL_STEPS=5000
CONVERGENCE_PATIENCE=200

# ── Fixed constraints (non-FP, non-MolPrice) ──
MP_THRESHOLD=-30
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ── Sphere diversity filter ──
SPHERE_THRESHOLD=0.35
SPHERE_FP_RADIUS=2
MAX_PER_SCAFFOLD=100

output_dir="../outputs/reinvent_fp_molprice_production/${FP_LABEL}_${LEVEL}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
{
    echo "========================================"
    echo "REINVENT Production FP × MolPrice Sweep"
    echo "========================================"
    echo "Array index:       $N"
    echo "FP:                $FP_LABEL ($FP_THRESHOLD K)"
    echo "MolPrice:          $LEVEL (soft=$MOLPRICE_SOFT, hard=$MOLPRICE_HARD)"
    echo "Seed:              $SEED"
    echo "Sigma:             $SIGMA"
    echo "Diversity:         sphere (threshold=$SPHERE_THRESHOLD, r=$SPHERE_FP_RADIUS, max=$MAX_PER_SCAFFOLD)"
    echo "Biodeg:            on"
    echo "Prior:             gru_prior.pt (3x aug)"
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
    --molprice_soft $MOLPRICE_SOFT \
    --molprice_hard $MOLPRICE_HARD \
    --diversity_filter \
    --diversity_mode sphere \
    --tanimoto_threshold $SPHERE_THRESHOLD \
    --tanimoto_fp_radius $SPHERE_FP_RADIUS \
    --max_per_scaffold $MAX_PER_SCAFFOLD \
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
