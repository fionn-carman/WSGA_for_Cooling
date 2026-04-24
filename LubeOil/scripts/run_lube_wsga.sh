#!/bin/bash
# WSGA for lubricant base-oil design.
# Same GA (and production hyperparameters) as our cooling-fluid paper; only the
# fitness function (FOM_LUBE) is different.
#
# Production HPs from ML-for-Immersion-Cooling manuscript:
#   population_size=3000, num_generations=150, mutation_rate=0.3,
#   elitism_rate=0.25, tournament_k=5, Tau=0.25, niching_radius=2.
#
# Usage:
#   bash LubeOil/scripts/run_lube_wsga.sh [profile|idx]   # local, one profile
#   qsub -J 0-5 LubeOil/scripts/run_lube_wsga.sh          # HPC, all six

set -euo pipefail

PROFILES=(visc tc hc dvi tox even)

if [[ -n "${PBS_ARRAY_INDEX:-}" ]]; then
    IDX=${PBS_ARRAY_INDEX}
elif [[ $# -ge 1 && "$1" =~ ^[0-5]$ ]]; then
    IDX=$1
elif [[ $# -ge 1 ]]; then
    PROFILE=$1
    for i in "${!PROFILES[@]}"; do
        [[ "${PROFILES[$i]}" == "$PROFILE" ]] && IDX=$i
    done
    IDX=${IDX:-5}
else
    IDX=5
fi

PROFILE=${PROFILES[$IDX]}
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="LubeOil/outputs/wsga_${PROFILE}_${TS}"
mkdir -p "$OUT_DIR"

if [[ -f ~/miniforge3/etc/profile.d/conda.sh ]]; then
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate mol-rl
fi

cd LubeOil/src

python lube_wsga.py \
    --weight_profile "$PROFILE" \
    --population_size 3000 \
    --num_generations 150 \
    --mutation_rate 0.3 \
    --elitism_rate 0.25 \
    --tournament_k 5 \
    --Tau 0.25 \
    --niching_radius 2 \
    --no_biodeg \
    --output_dir "../../$OUT_DIR" \
    --model_dir "../../models" \
    --data_dir "../../data" \
    --training_data_dir "../../training/data"

echo "WSGA profile $PROFILE complete: $OUT_DIR"
