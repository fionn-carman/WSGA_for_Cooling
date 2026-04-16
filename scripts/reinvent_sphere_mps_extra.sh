#!/bin/bash
#PBS -N reinvent_sph_ex
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-9
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# Extra sphere-exclusion runs at N_max = 150 and 200
# to check if the FOM1 ceiling can be reached with a looser cap
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_sphere_mps_extra.sh <array_index>"
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

TARGET="FOM1"
SIGMA=1.0
LR=5e-5
BATCH_SIZE=128
REPLAY=100
REPLAY_FRACTION=0.5
RL_STEPS=5000
CONVERGENCE_PATIENCE=200

FP_THRESHOLD=373
MP_THRESHOLD=-30
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

SPHERE_THRESHOLD=0.35
SPHERE_FP_RADIUS=2

case $CONDITION in
    0) LABEL="mps150"; MPS=150 ;;
    1) LABEL="mps200"; MPS=200 ;;
    *) echo "ERROR: Unknown condition $CONDITION"; exit 1 ;;
esac

DIVERSITY_FLAGS="--diversity_filter --diversity_mode sphere --max_per_scaffold $MPS"

output_dir="../outputs/reinvent_sphere_mps_sweep/${LABEL}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
{
    echo "========================================"
    echo "REINVENT Sphere mps Extra (150/200)"
    echo "========================================"
    echo "Array index:       $N"
    echo "Condition:         $LABEL"
    echo "Seed:              $SEED"
    echo "Sigma:             $SIGMA"
    echo "max_per_sphere:    $MPS"
    echo "Sphere threshold:  $SPHERE_THRESHOLD"
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
