#!/bin/bash
#PBS -N reinvent_div_test
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-3
#PBS -o /dev/null
#PBS -e /dev/null

# 4-job diversity test: nocost x 4 categories x seed 42
# Maps array index 0-3 to main sweep indices 0,5,10,15

set -o pipefail

DIRECTORY="$PBS_O_WORKDIR"
LOCAL_IDX=${PBS_ARRAY_INDEX}

if [ -z "$LOCAL_IDX" ]; then
    if [ $# -eq 1 ]; then
        LOCAL_IDX=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_diversity_test.sh <0-3>"
        exit 1
    fi
fi

# Map 0-3 to actual sweep indices (nocost, 4 categories, seed 42)
SWEEP_INDICES=(0 5 10 15)
export PBS_ARRAY_INDEX=${SWEEP_INDICES[$LOCAL_IDX]}

cd "$DIRECTORY"
exec bash reinvent_sweep_nist8100.sh
