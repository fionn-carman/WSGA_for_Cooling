#!/bin/bash
#PBS -N reinvent_hp
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -J 0-719
#PBS -o /dev/null
#PBS -e /dev/null

set -o pipefail

# ============================================================
# REINVENT HP Sweep — FP >= 100 C (373 K)
# ============================================================
#
# Mirror of WSGA hyperparam_sweep_v4 (4D HP grid x 5 seeds).
# Uses the shared pretrained GRU prior to skip Phase 1.
#
# Sweep grid:  4 x 4 x 3 x 3 x 5 = 720 jobs
#
# Parameters:
#   --lr_rl              [1e-5, 5e-5, 1e-4, 5e-4]   (4)
#   --sigma              [0.1, 0.25, 0.5, 1.0]      (4)
#   --batch_size         [64, 128, 256]             (3)
#   --replay_buffer_size [0, 100, 200]              (3)
#   --seed               [42, 123, 7, 256, 999]     (5)
#
# Index decoding (nested modular arithmetic):
#   seed_idx   = N % 5
#   rem        = N / 5
#   replay_idx = rem % 3
#   rem        = rem / 3
#   bs_idx     = rem % 3
#   rem        = rem / 3
#   sigma_idx  = rem % 4
#   lr_idx     = rem / 4
#
# Fixed:
#   target            = FOM1
#   rl_steps (max)    = 5000
#   convergence_pat.  = 200
#   pretrain_epochs   = 30  (skipped via --prior_path)
#   replay_fraction   = 0.5
#   max_per_scaffold  = 25
#   diversity_filter  = ON
#   biodeg filter     = OFF (--no_biodeg)
#   FP threshold      = 373 K (100 C)
#   BP / DC / MP / SC / Tox = 100 C / 7 / -30 C / 3 / 3
#
# Constraint parity with WSGA v4 is automatic — REINVENT imports
# has_invalid_fragments and assign_validity from wsga_helper.
#
# Usage:
#   qsub reinvent_hp_sweep.sh
#   qsub -J 0-0 reinvent_hp_sweep.sh        # single test job
#   bash reinvent_hp_sweep.sh <array_index>  # local test
# ============================================================

DIRECTORY="$PBS_O_WORKDIR"
N=${PBS_ARRAY_INDEX}

if [ -z "$N" ]; then
    if [ $# -eq 1 ]; then
        N=$1
        DIRECTORY=$(pwd)
    else
        echo "Usage: bash reinvent_hp_sweep.sh <array_index>"
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
lr_values=(1e-5 5e-5 1e-4 5e-4)
sigma_values=(0.1 0.25 0.5 1.0)
bs_values=(64 128 256)
replay_values=(0 100 200)
seeds=(42 123 7 256 999)

n_lr=${#lr_values[@]}
n_sigma=${#sigma_values[@]}
n_bs=${#bs_values[@]}
n_replay=${#replay_values[@]}
n_seed=${#seeds[@]}
n_total=$((n_lr * n_sigma * n_bs * n_replay * n_seed))

if [ "$N" -ge "$n_total" ] || [ "$N" -lt 0 ]; then
    echo "ERROR: Array index $N out of range [0, $((n_total - 1))]"
    exit 1
fi

# ==============================
# Decode array index
# ==============================
seed_idx=$((N % n_seed))
remaining=$((N / n_seed))

replay_idx=$((remaining % n_replay))
remaining=$((remaining / n_replay))

bs_idx=$((remaining % n_bs))
remaining=$((remaining / n_bs))

sigma_idx=$((remaining % n_sigma))
lr_idx=$((remaining / n_sigma))

LR=${lr_values[$lr_idx]}
SIGMA=${sigma_values[$sigma_idx]}
BATCH_SIZE=${bs_values[$bs_idx]}
REPLAY=${replay_values[$replay_idx]}
SEED=${seeds[$seed_idx]}

# ==============================
# Fixed REINVENT settings
# ==============================
TARGET="FOM1"
RL_STEPS=5000
CONVERGENCE_PATIENCE=200
REPLAY_FRACTION=0.5
MAX_PER_SCAFFOLD=25

# Constraint thresholds (match WSGA v4 sweep)
FP_THRESHOLD=373
MP_THRESHOLD=-30
BP_THRESHOLD=100
DC_THRESHOLD=7
SC_THRESHOLD=3
TOX_THRESHOLD=3

# ==============================
# Setup output directory
# ==============================
output_dir="../outputs/reinvent_hp_sweep/lr${LR}_sigma${SIGMA}_bs${BATCH_SIZE}_replay${REPLAY}_seed${SEED}"
mkdir -p "$output_dir"

log_file="${output_dir}/config.log"
echo "========================================" > "$log_file"
echo "REINVENT HP Sweep (FP >= 100 C)"          >> "$log_file"
echo "========================================" >> "$log_file"
echo "Array index:       $N"                    >> "$log_file"
echo "Target:            $TARGET"               >> "$log_file"
echo "lr_rl:             $LR"                   >> "$log_file"
echo "sigma:             $SIGMA"                >> "$log_file"
echo "batch_size:        $BATCH_SIZE"           >> "$log_file"
echo "replay_buffer:     $REPLAY"               >> "$log_file"
echo "replay_fraction:   $REPLAY_FRACTION"      >> "$log_file"
echo "max_per_scaffold:  $MAX_PER_SCAFFOLD"     >> "$log_file"
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
    --diversity_filter \
    --max_per_scaffold $MAX_PER_SCAFFOLD \
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
