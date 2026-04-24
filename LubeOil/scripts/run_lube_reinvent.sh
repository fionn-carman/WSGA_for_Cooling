#!/bin/bash
# REINVENT for lubricant base-oil design.
# Production HPs from ML-for-Immersion-Cooling manuscript:
#   lr_rl=5e-5, sigma=1.0, batch_size=128, convergence_patience=200.
# Shared pre-trained GRU prior (models/init_corpus/gru_prior.pt).
#
# Usage:
#   bash LubeOil/scripts/run_lube_reinvent.sh [profile|idx]
#   qsub -J 0-4 LubeOil/scripts/run_lube_reinvent.sh

set -euo pipefail

PROFILES=(visc tc hc dvi even)

if [[ -n "${PBS_ARRAY_INDEX:-}" ]]; then
    IDX=${PBS_ARRAY_INDEX}
elif [[ $# -ge 1 && "$1" =~ ^[0-4]$ ]]; then
    IDX=$1
elif [[ $# -ge 1 ]]; then
    PROFILE=$1
    for i in "${!PROFILES[@]}"; do
        [[ "${PROFILES[$i]}" == "$PROFILE" ]] && IDX=$i
    done
    IDX=${IDX:-4}
else
    IDX=4
fi

PROFILE=${PROFILES[$IDX]}
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="LubeOil/outputs/reinvent_${PROFILE}_${TS}"
mkdir -p "$OUT_DIR"

if [[ -f ~/miniforge3/etc/profile.d/conda.sh ]]; then
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate mol-rl
fi

cd LubeOil/src

python reinvent/run_lube_reinvent.py \
    --weight_profile "$PROFILE" \
    --rl_steps 5000 \
    --batch_size 128 \
    --sigma 1.0 \
    --lr_rl 5e-5 \
    --convergence_patience 200 \
    --molprice_soft 3.0 \
    --molprice_hard 6.0 \
    --no_biodeg \
    --output_dir "../../$OUT_DIR" \
    --model_dir "../../models" \
    --training_data_dir "../../training/data"

echo "REINVENT profile $PROFILE complete: $OUT_DIR"
