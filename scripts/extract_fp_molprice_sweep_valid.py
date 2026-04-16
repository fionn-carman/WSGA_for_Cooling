#!/usr/bin/env python
"""
Extract valid molecules from FP x MolPrice sweep all_evaluated_molecules.csv files.

Designed to run on HPC where the full CSVs (~246 MB each) live. Produces:
  1. valid_molecules_summary.csv   — deduplicated valid molecules per run
  2. generation_stats_combined.csv — generation_stats.csv stacked across all runs

Directory format: {fp_label}_{level}_seed{SEED}
  e.g. fp100_nocost_seed42, fp150_aggressive_seed256

Usage (on HPC):
    cd /rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling
    python scripts/extract_fp_molprice_sweep_valid.py

Usage (local):
    python scripts/extract_fp_molprice_sweep_valid.py \
        --sweep_dir outputs/fp_molprice_sweep_nist8100
"""

import os
import re
import argparse
import pandas as pd
import numpy as np

FP_LABELS = ["fp100", "fp125", "fp150", "fp175"]
LEVEL_ORDER = ["nocost", "gentle", "aggressive"]
SEEDS = [42, 123, 7, 256, 999]

KEEP_COLUMNS = [
    "SMILES",
    # Core properties
    "fom1_40C", "fp", "bp", "mp", "dc",
    "density_40C", "viscosity_40C", "tc_40C", "cpsat_40C", "beta_40C",
    # Cost and fitness
    "MolPrice", "FitnessScore", "MolPrice_Penalty",
    "SCScore", "Tox21_Score", "Biodegradable",
    # Per-model OOD flags
    "OOD_bp", "OOD_mp", "OOD_fp", "OOD_dc",
    "OOD_density_40C", "OOD_viscosity_40C", "OOD_tc_40C",
    "OOD_cpsat_40C", "OOD_beta_40C", "OOD_fom1_40C",
    "OOD_any", "OOD_count",
    # Per-model Mahalanobis distances
    "Mahal_bp", "Mahal_mp", "Mahal_fp", "Mahal_dc",
    "Mahal_density_40C", "Mahal_viscosity_40C", "Mahal_tc_40C",
    "Mahal_cpsat_40C", "Mahal_beta_40C", "Mahal_fom1_40C",
    # Fold ensemble uncertainty
    "fom1_40C_fold_mean", "fom1_40C_fold_std",
    "fom1_40C_ci_lower", "fom1_40C_ci_upper",
    "fp_fold_std", "mp_fold_std", "bp_fold_std", "dc_fold_std",
    # Generation
    "generation",
]


def parse_sweep_dir(dirname):
    """Extract fp_label, cost_level, seed from directory name."""
    m = re.match(r"(fp\d+)_(nocost|gentle|aggressive)_seed(\d+)", dirname)
    if not m:
        return None
    fp_label = m.group(1)
    if fp_label not in FP_LABELS:
        return None
    return {
        "fp_label": fp_label,
        "cost_level": m.group(2),
        "seed": int(m.group(3)),
    }


def process_run(run_path, fp_label, cost_level, seed):
    """Read all_evaluated_molecules.csv (or top_n_tracking.csv fallback),
    filter valid, deduplicate, return compact df."""
    csv_path = os.path.join(run_path, "all_evaluated_molecules.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(run_path, "top_n_tracking.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"  ERROR reading {csv_path}: {e}")
        return None

    if df.empty:
        return None

    total_evaluated = len(df)

    if "is_valid" in df.columns:
        df = df[df["is_valid"] == 1].copy()
    if df.empty:
        return None

    smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in df.columns else "SMILES"

    subset = pd.DataFrame()
    for col in KEEP_COLUMNS:
        if col == "SMILES":
            subset["SMILES"] = df[smiles_col].values
        elif col in df.columns:
            subset[col] = df[col].values
        else:
            subset[col] = np.nan

    subset = subset.sort_values("fom1_40C", ascending=False)
    subset = subset.drop_duplicates(subset=["SMILES"], keep="first")

    subset["fp_label"] = fp_label
    subset["cost_level"] = cost_level
    subset["seed"] = seed
    subset["total_evaluated"] = total_evaluated

    return subset


def process_gen_stats(run_path, fp_label, cost_level, seed):
    """Read generation_stats.csv and add metadata columns."""
    csv_path = os.path.join(run_path, "generation_stats.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None

    df["fp_label"] = fp_label
    df["cost_level"] = cost_level
    df["seed"] = seed
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract valid molecules from FP x MolPrice sweep runs")
    parser.add_argument(
        "--sweep_dir", type=str,
        default="outputs/fp_molprice_sweep_nist8100",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    mol_output = args.output or os.path.join(sweep_dir, "valid_molecules_summary.csv")
    stats_output = os.path.join(sweep_dir, "generation_stats_combined.csv")

    print(f"Scanning: {sweep_dir}")

    expected = set()
    for fp in FP_LABELS:
        for lv in LEVEL_ORDER:
            for s in SEEDS:
                expected.add(f"{fp}_{lv}_seed{s}")

    all_mol_dfs = []
    all_stats_dfs = []
    processed = 0
    skipped = 0
    missing = []

    dirs = sorted(os.listdir(sweep_dir))
    found_dirs = set()

    for dirname in dirs:
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue

        params = parse_sweep_dir(dirname)
        if params is None:
            continue

        found_dirs.add(dirname)

        mol_df = process_run(
            run_path, params["fp_label"], params["cost_level"], params["seed"])
        stats_df = process_gen_stats(
            run_path, params["fp_label"], params["cost_level"], params["seed"])

        if mol_df is None:
            skipped += 1
            print(f"  SKIP {dirname} (no data)")
            continue

        processed += 1
        all_mol_dfs.append(mol_df)
        if stats_df is not None:
            all_stats_dfs.append(stats_df)
        print(f"  [{processed + skipped}/{len(expected)}] {dirname}: "
              f"{len(mol_df)} valid unique molecules")

    missing = sorted(expected - found_dirs)
    if missing:
        print(f"\nMISSING RUNS ({len(missing)}):")
        for m in missing:
            print(f"  {m}")

    if not all_mol_dfs:
        print("No data found.")
        return

    combined = pd.concat(all_mol_dfs, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"Total valid molecule rows: {len(combined)}")
    print(f"Processed: {processed} runs, skipped: {skipped}, missing: {len(missing)}")
    print(f"\nPer FP level:")
    for fp in FP_LABELS:
        n = len(combined[combined["fp_label"] == fp])
        print(f"  {fp:<8} {n:>8} molecules")
    print(f"\nPer cost level:")
    for lv in LEVEL_ORDER:
        n = len(combined[combined["cost_level"] == lv])
        print(f"  {lv:<12} {n:>8} molecules")

    os.makedirs(os.path.dirname(os.path.abspath(mol_output)), exist_ok=True)
    combined.to_csv(mol_output, index=False)
    print(f"\nSaved molecules: {mol_output} ({os.path.getsize(mol_output) / 1e6:.1f} MB)")

    if all_stats_dfs:
        stats_combined = pd.concat(all_stats_dfs, ignore_index=True)
        stats_combined.to_csv(stats_output, index=False)
        print(f"Saved gen stats: {stats_output} ({os.path.getsize(stats_output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
