#!/usr/bin/env python
"""
Extract valid molecules from MolPrice sweep all_evaluated_molecules.csv files.

Designed to run on HPC where the full CSVs (~246 MB each) live. Produces a
single compact summary CSV with only the columns needed for post-hoc FP vs
FOM1 analysis.

Directory format: {level}_{category}_seed{SEED}

Usage (on HPC):
    cd /rds/general/user/fc4018/projects/fionn2023/live/WSGA_for_Cooling
    python scripts/extract_molprice_sweep_valid.py

Usage (local, if CSVs are available):
    python scripts/extract_molprice_sweep_valid.py \
        --sweep_dir outputs/molprice_sweep_nist8100
"""

import os
import re
import argparse
import pandas as pd
import numpy as np

LEVEL_ORDER = ["nocost", "gentle", "moderate", "firm", "tight", "aggressive"]
CATEGORY_ORDER = ["bio_stable", "bio_unstable", "nonbio_stable", "nonbio_unstable"]

KEEP_COLUMNS = [
    "SMILES", "fom1_40C", "fp", "MolPrice", "FitnessScore", "MolPrice_Penalty",
    "OOD_any", "OOD_count",
]


def parse_sweep_dir(dirname):
    """Extract level, category, seed from directory name."""
    m = re.match(
        r"([a-z]+)_(bio_stable|bio_unstable|nonbio_stable|nonbio_unstable)_seed(\d+)",
        dirname,
    )
    if not m:
        return None
    level = m.group(1)
    if level not in LEVEL_ORDER:
        return None
    return {"level": level, "category": m.group(2), "seed": int(m.group(3))}


def process_run(run_path, level, category, seed):
    """Read all_evaluated_molecules.csv (or top_n_tracking.csv fallback),
    filter, deduplicate, return compact df."""
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

    # Filter to valid molecules
    if "is_valid" in df.columns:
        df = df[df["is_valid"] == 1].copy()
    if df.empty:
        return None

    # Resolve SMILES column
    smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in df.columns else "SMILES"

    # Select and rename columns
    available = [c for c in KEEP_COLUMNS if c in df.columns or
                 (c == "SMILES" and smiles_col in df.columns)]

    subset = pd.DataFrame()
    for col in KEEP_COLUMNS:
        if col == "SMILES":
            subset["SMILES"] = df[smiles_col].values
        elif col in df.columns:
            subset[col] = df[col].values
        else:
            subset[col] = np.nan

    # Deduplicate by SMILES, keeping highest fom1_40C
    subset = subset.sort_values("fom1_40C", ascending=False)
    subset = subset.drop_duplicates(subset=["SMILES"], keep="first")

    # Add metadata
    subset["cost_level"] = level
    subset["category"] = category
    subset["seed"] = seed

    return subset


def main():
    parser = argparse.ArgumentParser(
        description="Extract valid molecules from MolPrice sweep runs")
    parser.add_argument(
        "--sweep_dir", type=str,
        default="outputs/molprice_sweep_nist8100",
        help="Root directory containing sweep run subdirectories",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: {sweep_dir}/valid_molecules_summary.csv)",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    output_path = args.output or os.path.join(sweep_dir, "valid_molecules_summary.csv")

    print(f"Scanning: {sweep_dir}")
    print(f"Output:   {output_path}")

    all_dfs = []
    processed = 0
    skipped = 0

    dirs = sorted(os.listdir(sweep_dir))
    total_dirs = sum(1 for d in dirs if os.path.isdir(os.path.join(sweep_dir, d))
                     and parse_sweep_dir(d) is not None)

    for dirname in dirs:
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue

        params = parse_sweep_dir(dirname)
        if params is None:
            continue

        df = process_run(run_path, params["level"], params["category"], params["seed"])
        if df is None:
            skipped += 1
            print(f"  [{processed + skipped}/{total_dirs}] SKIP {dirname}")
            continue

        processed += 1
        all_dfs.append(df)
        print(f"  [{processed + skipped}/{total_dirs}] {dirname}: "
              f"{len(df)} valid unique molecules")

    if not all_dfs:
        print("No data found.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"Total rows: {len(combined)}")
    print(f"Processed: {processed} runs, skipped: {skipped}")
    print(f"\nPer cost level:")
    for level in LEVEL_ORDER:
        n = len(combined[combined["cost_level"] == level])
        if n > 0:
            print(f"  {level:<12} {n:>8} molecules")
    print(f"\nPer category:")
    for cat in CATEGORY_ORDER:
        n = len(combined[combined["category"] == cat])
        if n > 0:
            print(f"  {cat:<22} {n:>8} molecules")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
