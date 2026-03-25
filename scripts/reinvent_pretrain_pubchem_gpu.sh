#!/bin/bash
#PBS -N reinvent_pretrain_gpu
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT Prior Pretraining — GPU (L40S)
# ============================================================
#
# Trains the GRU prior once on the PubChem+NIST corpus (504K SMILES,
# 50 epochs) on GPU. Saves prior.pt + vocabulary.json for reuse.
#
# Racing against the CPU version to see which finishes first.
#
# Usage:
#   qsub reinvent_pretrain_pubchem_gpu.sh
#   bash reinvent_pretrain_pubchem_gpu.sh    # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
if [ -z "$PBS_O_WORKDIR" ]; then
    DIRECTORY=$(pwd)
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

output_dir="../outputs/pretrained_priors/pubchem_nist_gpu"
mkdir -p "$output_dir"

log_file="${output_dir}/pretrain.log"
echo "========================================" > "$log_file"
echo "REINVENT Prior Pretraining (GPU)"        >> "$log_file"
echo "========================================" >> "$log_file"
echo "Corpus:    pubchem+nist (504K SMILES)"   >> "$log_file"
echo "Epochs:    50"                            >> "$log_file"
echo "Device:    cuda (L40S)"                   >> "$log_file"
echo "Started:   $(date)"                       >> "$log_file"
echo "========================================" >> "$log_file"

python3 ../src/reinvent/run_reinvent.py \
    --pretrain_only \
    --prior_corpus pubchem+nist \
    --pubchem_path ../data/pubchem_cho_5_30ha.csv \
    --pretrain_epochs 50 \
    --batch_size 512 \
    --lr_pretrain 1e-3 \
    --device cuda \
    --seed 42 \
    --model_dir ../models \
    --training_data_dir ../training/data \
    --output_dir $output_dir \
    2>&1 | tee -a "$log_file"

exit_status=${PIPESTATUS[0]}

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: $exit_status" >> "$log_file"
ls -la "$output_dir"/ >> "$log_file" 2>&1

exit $exit_status
