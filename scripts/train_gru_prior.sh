#!/bin/bash
#PBS -N train_gru_prior
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# Train GRU prior for WSGA initial population
# ============================================================
#
# One-time job: curates unified CHO corpus, augments 5x,
# trains GRU (30 epochs), saves to models/init_corpus/.
#
# Usage:
#   qsub train_gru_prior.sh
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
if [ -z "$DIRECTORY" ]; then
    DIRECTORY=$(pwd)
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

output_dir="../models/init_corpus"
mkdir -p "$output_dir"

log_file="${output_dir}/training.log"

python3 ../src/train_gru_prior.py \
    --training_data_dir ../training/data \
    --output_dir "$output_dir" \
    --n_augmentations 5 \
    --epochs 30 \
    --batch_size 512 \
    --lr 1e-3 \
    --device cpu \
    2>&1 | tee "$log_file"

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: ${PIPESTATUS[0]}" >> "$log_file"
