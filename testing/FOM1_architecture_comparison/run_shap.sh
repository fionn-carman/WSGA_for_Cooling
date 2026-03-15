#!/bin/bash
#PBS -N fom1_shap
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=4:mem=16gb

# ============================================================
# SHAP Analysis for XGBoost Descriptor Models
# ============================================================
#
# Runs shap_analysis.py for both temperatures + combined plot.
#
# Submit:
#   qsub run_shap.sh
#
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
JOB_ID=${PBS_JOBID}

# Activate Conda Environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl

cd "$DIRECTORY"

echo "========================================"
echo "SHAP Analysis"
echo "Job started:  $(date)"
echo "Working dir:  $(pwd)"
echo "========================================"

python3 shap_analysis.py 2>&1

echo "========================================"
echo "SHAP analysis completed at: $(date)"
echo "========================================"

# Clean up PBS output files
if [ -n "$JOB_ID" ]; then
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'.' -f1)
    rm -f "${DIRECTORY}/fom1_shap.e${BASE_JOB_ID}" 2>/dev/null
    rm -f "${DIRECTORY}/fom1_shap.o${BASE_JOB_ID}" 2>/dev/null
fi
