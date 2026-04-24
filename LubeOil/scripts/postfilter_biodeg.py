#!/usr/bin/env python3
"""Post-hoc biodegradability filter for WSGA/REINVENT output CSVs.

Reads all_evaluated_molecules.csv (or any CSV with a SMILES column) and adds a
biodegradable prediction column, then emits top-N biodegradable candidates
sorted by FOM_LUBE. Does not re-score fitness.

Usage:
    python LubeOil/scripts/postfilter_biodeg.py \\
        --input LubeOil/outputs/even_20260424_1500/all_evaluated_molecules.csv \\
        --top_n 30
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LubeOil" / "src"))

from lube_wsga_helper import load_biodeg_model, is_biodegradable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model_dir", default=str(REPO / "models"))
    ap.add_argument("--sort_col", default="FOM_LUBE")
    ap.add_argument("--top_n", type=int, default=30)
    ap.add_argument("--output", default=None,
                    help="Output CSV path; default: <input>_biodegradable_top{N}.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "SMILES" not in df.columns:
        raise ValueError("Input CSV must have a SMILES column")

    biodeg = load_biodeg_model(os.path.join(args.model_dir, "biodeg"))
    df["Biodegradable_postfilter"] = df["SMILES"].apply(lambda s: is_biodegradable(s, biodeg))

    if "Biodegradable" in df.columns and df["Biodegradable"].notna().any():
        print(f"Original Biodegradable==True: {int(df['Biodegradable'].sum())}/{len(df)}")
    print(f"Post-filter biodeg: {int(df['Biodegradable_postfilter'].sum())}/{len(df)}")

    bio = df[df["Biodegradable_postfilter"] == True].copy()
    if args.sort_col in bio.columns:
        bio = bio.sort_values(args.sort_col, ascending=False)
    else:
        print(f"(sort column {args.sort_col} not present, keeping original order)")
    top = bio.head(args.top_n)

    out = args.output or args.input.replace(".csv", f"_biodegradable_top{args.top_n}.csv")
    top.to_csv(out, index=False)
    print(f"Saved {len(top)} biodegradable top-{args.top_n} to {out}")


if __name__ == "__main__":
    main()
