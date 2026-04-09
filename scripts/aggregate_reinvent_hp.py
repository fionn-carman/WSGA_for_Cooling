"""
Aggregate valid molecules across all runs of a REINVENT HP sweep.

Adapted from aggregate_v4_valid.py — same pattern, different directory
naming convention. REINVENT runs have directories of the form:

    lr{LR}_sigma{S}_bs{BS}_replay{R}_seed{SEED}

Outputs:
  - all_runs.csv               (one row per run, summary stats)
  - valid_molecules_summary.csv (all unique valid molecules pooled
                                 across the sweep, deduplicated)
  - pareto_front.csv           (FP vs FOM1 Pareto-optimal subset)

Reads from each run's `top_n_tracking.csv` (REINVENT's per-step output,
schema-equivalent to WSGA's top_n_tracking).

Usage:
    python aggregate_reinvent_hp.py <sweep_dir> <fp_threshold_K>
"""

import os
import re
import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python aggregate_reinvent_hp.py <sweep_dir> <fp_threshold_K>")
    sys.exit(1)

SWEEP_DIR = sys.argv[1]
FP_MIN = float(sys.argv[2])

DIR_RE = re.compile(
    r"lr([\d.eE+-]+)_sigma([\d.]+)_bs(\d+)_replay(\d+)_seed(\d+)"
)


def parse_dir(name):
    m = DIR_RE.match(name)
    if not m:
        return None
    return {
        "lr_rl": float(m.group(1)),
        "sigma": float(m.group(2)),
        "batch_size": int(m.group(3)),
        "replay": int(m.group(4)),
        "seed": int(m.group(5)),
    }


def pareto(fp_c, fom1):
    """Pareto front maximising both FP and FOM1."""
    order = np.argsort(fp_c)
    keep = []
    max_f = -np.inf
    for i in reversed(order):
        if fom1[i] > max_f:
            keep.append(i)
            max_f = fom1[i]
    return sorted(keep)


all_valid_chunks = []
run_summary = []
n_scanned = 0
n_complete = 0

for d in sorted(os.listdir(SWEEP_DIR)):
    run = os.path.join(SWEEP_DIR, d)
    if not os.path.isdir(run) or d == "analysis":
        continue
    n_scanned += 1
    params = parse_dir(d)
    if params is None:
        continue

    track = os.path.join(run, "top_n_tracking.csv")
    if not os.path.exists(track) or os.path.getsize(track) < 100:
        continue

    try:
        df = pd.read_csv(track, low_memory=False)
    except Exception:
        continue

    req = {"fp", "fom1_40C", "is_valid", "FitnessScore", "mp"}
    if df.empty or not req.issubset(df.columns):
        continue
    n_complete += 1

    valid = df[(df["is_valid"] == 1)
               & (df["FitnessScore"] > 0)
               & (df["mp"] < -30)
               & (df["fp"] >= FP_MIN)]

    if not valid.empty:
        valid = valid.sort_values("fom1_40C", ascending=False)
        valid_unique = valid.drop_duplicates(subset=["SMILES"], keep="first")
        all_valid_chunks.append(valid_unique)

        best_row = valid_unique.iloc[0]
        run_summary.append({
            **params,
            "dir": d,
            "best_fom1": best_row["fom1_40C"],
            "best_smiles": best_row["SMILES"],
            "best_fp": best_row["fp"],
            "n_valid": len(valid_unique),
        })
    else:
        run_summary.append({
            **params,
            "dir": d,
            "best_fom1": float("nan"),
            "best_smiles": "",
            "best_fp": float("nan"),
            "n_valid": 0,
        })

    if n_complete % 100 == 0:
        sys.stdout.write(f"{n_complete} / {n_scanned}\n")
        sys.stdout.flush()

print(f"\nScanned {n_scanned} dirs, {n_complete} complete")

out_dir = os.path.join(SWEEP_DIR, "analysis")
os.makedirs(out_dir, exist_ok=True)

# All runs CSV
out_runs = os.path.join(out_dir, "all_runs.csv")
pd.DataFrame(run_summary).to_csv(out_runs, index=False)
print(f"Wrote: {out_runs}  ({len(run_summary)} rows)")

# Pooled unique valid molecules
if all_valid_chunks:
    combined = pd.concat(all_valid_chunks, ignore_index=True)
    print(f"Total valid rows before global dedup: {len(combined)}")
    combined = combined.sort_values("fom1_40C", ascending=False)
    combined = combined.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    print(f"Unique valid SMILES after dedup: {len(combined)}")

    out_valid = os.path.join(out_dir, "valid_molecules_summary.csv")
    combined.to_csv(out_valid, index=False)
    print(f"Wrote: {out_valid}  ({len(combined)} rows)")

    # Pareto front
    fp_c = combined["fp"].values - 273.15
    fom1 = combined["fom1_40C"].values
    pidx = pareto(fp_c, fom1)
    front = combined.iloc[pidx].copy()
    front["fp_C"] = front["fp"] - 273.15
    front = front.sort_values("fp_C").reset_index(drop=True)

    out_pareto = os.path.join(out_dir, "pareto_front.csv")
    front.to_csv(out_pareto, index=False)
    print(f"Wrote: {out_pareto}  ({len(front)} Pareto-optimal molecules)")
    print(f"  FP range: {front['fp_C'].min():.0f} - {front['fp_C'].max():.0f} C")
    print(f"  FOM1 range: {front['fom1_40C'].min():.2f} - {front['fom1_40C'].max():.2f}")
else:
    print("No valid molecules found across the sweep.")
