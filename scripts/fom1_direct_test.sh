#!/bin/bash
# ============================================================
# Local test of WSGA FOM1 Direct Prediction (mol-rl env)
# ============================================================
#
# Quick single-run test to verify the WSGA works with the
# mol-rl conda environment and the FOM1 direct prediction models.
#
# Uses small population and few generations for fast iteration.
#
# Usage:
#   bash fom1_direct_test.sh
# ============================================================

DIRECTORY=$(cd "$(dirname "$0")" && pwd)
cd "$DIRECTORY"

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

# ==============================
# Small test parameters
# ==============================
POP=100
MR=0.8
ER=0.25
BER=0.5
K=3
TAU=0.15
NUM_GENERATIONS=3

TARGET="FOM1_direct"
MP_SOFT=-30
MP_HARD=-10
FP_THRESHOLD=373
BP_THRESHOLD=70
DC_THRESHOLD=8
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/fom1_direct_test"
mkdir -p "$output_dir"

echo "========================================"
echo "WSGA FOM1 Direct — Local Test (mol-rl)"
echo "========================================"
echo "Target:            $TARGET"
echo "Population size:   $POP"
echo "Generations:       $NUM_GENERATIONS"
echo "Output:            $output_dir"
echo "========================================"

# ==============================
# Run WSGA
# ==============================
python3 ../src/wsga.py \
    --target $TARGET \
    --population_size $POP \
    --elitism_rate $ER \
    --best_elite_ratio $BER \
    --mutation_rate $MR \
    --tournament_k $K \
    --Tau $TAU \
    --num_generations $NUM_GENERATIONS \
    --mp_soft $MP_SOFT \
    --mp_hard $MP_HARD \
    --fp_threshold $FP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --fom1_model_dir ../training/FOM1_architecture_comparison/results \
    --data_dir ../data \
    --model_dir ../models \
    --output_dir $output_dir

exit_status=$?

echo ""
echo "Exit status: $exit_status"

if [ $exit_status -eq 0 ]; then
    echo "Test PASSED - WSGA ran successfully with mol-rl env"
else
    echo "Test FAILED - check output above for errors"
fi

exit $exit_status
