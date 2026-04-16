#!/bin/bash
#PBS -N reinvent_murcko
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-4
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT with standard Murcko scaffold filter (5 seeds)
# Demonstrates mode collapse on acyclic chemical space
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_murcko_baseline.sh <array_index>"
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
SEED=${seeds[$N]}

output_dir="../outputs/reinvent_murcko_baseline/murcko_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
{
    echo "========================================"
    echo "REINVENT Murcko Scaffold Baseline"
    echo "========================================"
    echo "Seed:              $SEED"
    echo "Diversity mode:    scaffold (Murcko generic)"
    echo "max_per_scaffold:  25"
    echo "Sigma:             1.0"
    echo "Started at:        $(date)"
    echo "========================================"
} > "$log_file"

python3 ../src/reinvent/run_reinvent.py \
    --target FOM1 \
    --prior_path ../models/init_corpus/gru_prior.pt \
    --vocab_path ../models/init_corpus/vocabulary.json \
    --rl_steps 5000 \
    --batch_size 128 \
    --sigma 1.0 \
    --lr_rl 5e-5 \
    --convergence_patience 200 \
    --seed $SEED \
    --fp_threshold 373 \
    --mp_threshold -30 \
    --bp_threshold 100 \
    --dc_threshold 7 \
    --sc_threshold 3 \
    --tox_threshold 3 \
    --no_biodeg \
    --molprice_soft 0.0 \
    --molprice_hard 0.0 \
    --diversity_filter \
    --diversity_mode scaffold \
    --max_per_scaffold 25 \
    --replay_buffer_size 100 \
    --replay_fraction 0.5 \
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
