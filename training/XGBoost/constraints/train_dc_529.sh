#!/bin/bash
#PBS -N train_dc_529
#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=16:mem=64gb

# Retrain DC model on original 529 CHO dataset (CHO descriptor cache)
# Baseline: R2=0.826 from previous run

if [ $# -eq 0 ] && [ -z "$PBS_JOBID" ]; then
    DIRECTORY=$(pwd)
else
    DIRECTORY="$PBS_O_WORKDIR"
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$DIRECTORY"

python3 train_single_temp.py --properties dc --n_trials 100 --timeout 600 2>&1 | tee "./output/single_temp/dc_retrain.log"

exit $?
