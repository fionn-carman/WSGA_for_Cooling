#!/bin/bash
#PBS -N reinvent_scaff125
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-29
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT Scaffold Diversity Sweep — FP >= 125 C (398 K)
# ============================================================
#
# Identical to reinvent_scaffold_sweep.sh but with tighter
# flash point constraint (398 K / 125 C). Tests whether
# scaffold diversity matters more on a harder landscape.
#
# Sweep grid:  6 scaffold settings x 5 seeds = 30 jobs
#
# Parameters:
#   --max_per_scaffold  [5, 10, 25, 50, 100, disabled]  (6)
#   --seed              [42, 123, 7, 256, 999]           (5)
#
# Index decoding:
#   seed_idx     = N % 5
#   scaffold_idx = N / 5
#
# Fixed (best HP sweep config):
#   target            = FOM1
#   sigma             = 0.5
#   lr_rl             = 5e-5
#   batch_size        = 128
#   replay_buffer     = 100
#   replay_fraction   = 0.5
#   rl_steps (max)    = 5000
#   convergence_pat.  = 200
#   biodeg filter     = OFF (--no_biodeg)
#   FP threshold      = 398 K (125 C)
#   BP / DC / MP / SC / Tox = 100 C / 7 / -30 C / 3 / 3
#
# Usage:
#   qsub reinvent_scaffold_sweep_fp125.sh
#   qsub -J 0-0 reinvent_scaffold_sweep_fp125.sh        # single test job
#   bash reinvent_scaffold_sweep_fp125.sh <array_index>  # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_scaffold_sweep_fp125.sh <array_index>"
        exit 1
    fi
fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mol-rl
cd "$DIRECTORY"

# Prevent OMP/MKL segfault when PyTorch + XGBoost coexist
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# ==============================
# Sweep grid
# ==============================
# Index 0-4 = mps 5, 5-9 = mps 10, ..., 25-29 = disabled
scaffold_values=(5 10 25 50 100 0)   # 0 = disabled
seeds=(42 123 7 256 999)

n_scaffold=${#scaffold_values[@]}
n_seed=${#seeds[@]}
n_total=$((n_scaffold * n_seed))

if [ "$N" -ge "$n_total" ] || [ "$N" -lt 0 ]; then
    echo "ERROR: Array index $N out of range [0, $((n_total - 1))]"
    exit 1
fi

# ==============================
# Decode array index
# ==============================
seed_idx=$((N % n_seed))
scaffold_idx=$((N / n_seed))

MPS=${scaffold_values[$scaffold_idx]}
SEED=${seeds[$seed_idx]}

# ==============================
# Fixed REINVENT settings (best HP sweep config)
# ==============================
TARGET="FOM1"
SIGMA=0.5
LR=5e-5
BATCH_SIZE=128
REPLAY=100
REPLAY_FRACTION=0.5
RL_STEPS=5000
CONVERGENCE_PATIENCE=200

# Constraint thresholds — FP tightened to 398 K (125 C)
FP_THRESHOLD=398
MP_THRESHOLD=-30
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ==============================
# Build diversity filter flags
# ==============================
if [ "$MPS" -eq 0 ]; then
    DIVERSITY_FLAGS="--no_diversity_filter"
    MPS_LABEL="disabled"
else
    DIVERSITY_FLAGS="--diversity_filter --max_per_scaffold $MPS"
    MPS_LABEL="$MPS"
fi

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/reinvent_scaffold_sweep_fp125/mps${MPS_LABEL}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "REINVENT Scaffold Sweep (FP >= 125 C)"   >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "max_per_scaffold:  $MPS_LABEL"            >> "$log_file"
echo "sigma:             $SIGMA"                >> "$log_file"
echo "lr_rl:             $LR"                   >> "$log_file"
echo "batch_size:        $BATCH_SIZE"           >> "$log_file"
echo "replay_buffer:     $REPLAY"               >> "$log_file"
echo "replay_fraction:   $REPLAY_FRACTION"      >> "$log_file"
echo "RL steps (max):    $RL_STEPS"             >> "$log_file"
echo "Patience:          $CONVERGENCE_PATIENCE" >> "$log_file"
echo "Seed:              $SEED"                 >> "$log_file"
echo "FP threshold:      $FP_THRESHOLD K"       >> "$log_file"
echo "MP threshold:      $MP_THRESHOLD C"       >> "$log_file"
echo "BP threshold:      $BP_THRESHOLD C"       >> "$log_file"
echo "DC threshold:      $DC_THRESHOLD"         >> "$log_file"
echo "SC threshold:      $SC_THRESHOLD"         >> "$log_file"
echo "Tox threshold:     $TOX_THRESHOLD"        >> "$log_file"
echo "Biodeg filter:     OFF (--no_biodeg)"     >> "$log_file"
echo "Started at:        $(date)"               >> "$log_file"
echo "========================================" >> "$log_file"

# ==============================
# Run REINVENT
# ==============================
python3 ../src/reinvent/run_reinvent.py \
    --target $TARGET \
    --prior_path ../models/init_corpus/gru_prior.pt \
    --vocab_path ../models/init_corpus/vocabulary.json \
    --rl_steps $RL_STEPS \
    --batch_size $BATCH_SIZE \
    --sigma $SIGMA \
    --lr_rl $LR \
    --convergence_patience $CONVERGENCE_PATIENCE \
    --seed $SEED \
    --fp_threshold $FP_THRESHOLD \
    --mp_threshold $MP_THRESHOLD \
    --bp_threshold $BP_THRESHOLD \
    --dc_threshold $DC_THRESHOLD \
    --sc_threshold $SC_THRESHOLD \
    --tox_threshold $TOX_THRESHOLD \
    --no_biodeg \
    --molprice_soft 0.0 \
    --molprice_hard 0.0 \
    $DIVERSITY_FLAGS \
    --replay_buffer_size $REPLAY \
    --replay_fraction $REPLAY_FRACTION \
    --model_dir ../models \
    --training_data_dir ../training/data \
    --output_dir $output_dir \
    2>&1 | tee -a "$log_file"

exit_status=${PIPESTATUS[0]}

echo "" >> "$log_file"
echo "Finished at: $(date)" >> "$log_file"
echo "Exit status: $exit_status" >> "$log_file"

exit $exit_status
