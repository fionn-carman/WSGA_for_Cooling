#!/bin/bash
#PBS -N train_dc_nolog
#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=16:mem=64gb

# Train DC model on CHO-667 WITHOUT log transform
# Hypothesis: log_transform=True is hurting DC performance

if [ $# -eq 0 ] && [ -z "$PBS_JOBID" ]; then
    DIRECTORY=$(pwd)
    JOB_ID="local"
elif [ $# -eq 1 ]; then
    DIRECTORY=$(pwd)
    JOB_ID="local"
else
    DIRECTORY="$PBS_O_WORKDIR"
    JOB_ID=${PBS_JOBID}
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$DIRECTORY"

mkdir -p "./output/single_temp"
log_file="./output/single_temp/dc_cho667_nolog_training.log"

echo "========================================" | tee "$log_file"
echo "Training DC model (no log) on CHO-667"   | tee -a "$log_file"
echo "Job started at: $(date)"                  | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

python3 train_single_temp.py --properties dc_cho667_nolog --n_trials 100 --timeout 600 2>&1 | tee -a "$log_file"

exit_status=$?

echo "========================================" | tee -a "$log_file"
echo "Training completed at: $(date)"          | tee -a "$log_file"
echo "Exit status: ${exit_status}"             | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

exit $exit_status
