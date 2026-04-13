#!/bin/bash
#PBS -N r2_ood_pct
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=06:00:00
#PBS -J 0-9
#PBS -o /dev/null
#PBS -e /dev/null

# R² vs OOD metric percentile — multi-property array job
# Each array task runs one property.
# Set METRICS env var to restrict which OOD metrics to run (default: all).
# Example: qsub -v METRICS="ts" scripts/r2_vs_ood_pct.sh

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$PBS_O_WORKDIR" || exit 1

PROPERTIES=(fom1_40C cpsat_40C beta_40C density_40C viscosity_40C tc_40C bp mp fp dc)
PROP=${PROPERTIES[$PBS_ARRAY_INDEX]}

OUT_DIR="OOD_testing/r2_vs_ood_percentile/${PROP}"
mkdir -p "$OUT_DIR"

# Default to all metrics if METRICS not set
METRIC_ARGS=""
if [ -n "$METRICS" ]; then
    METRIC_ARGS="--metrics $METRICS"
fi

echo "=== Job $PBS_ARRAY_INDEX: $PROP (metrics: ${METRICS:-all}) ===" | tee "${OUT_DIR}/run_log.txt"
echo "Start: $(date)" | tee -a "${OUT_DIR}/run_log.txt"

python OOD_testing/r2_vs_ood_percentile.py --property "$PROP" $METRIC_ARGS \
    2>&1 | tee -a "${OUT_DIR}/run_log.txt"

echo "End: $(date)" | tee -a "${OUT_DIR}/run_log.txt"
