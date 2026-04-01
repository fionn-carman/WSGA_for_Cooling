"""
Categorise FOM1 dataset molecules into 2 categories + excluded.

Takes the baseline evaluation output (with pre-computed constraint properties),
predicts SCScore and MolPrice, applies constraints matching the GA's filters
(including always-on reactive group bans), and classifies each molecule.

Categories:
  - bio:      passes base constraints AND biodegradable
  - nonbio:   passes base constraints AND NOT biodegradable
  - excluded: fails base constraints

Output: output/fom1_dataset_categories.csv

Usage:
    cd BaselineFOM1Eval/
    python categorise_fom1_dataset.py
    python categorise_fom1_dataset.py --baseline_csv output/baseline_fom1_results.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path for model imports
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.error')

from wsga_helper import has_invalid_fragments
from evaluation import get_scscore_cached
from SCScorer import SCScorer


def predict_scscore(smiles_list, model_dir):
    """Predict SCScore for a list of SMILES."""
    sc_model = SCScorer()
    sc_model.restore(os.path.join(
        model_dir, "SCScorer/scscore/models/full_reaxys_model_1024bool/"
        "model.ckpt-10654.as_numpy.json.gz"))
    return [get_scscore_cached(sc_model, smi) for smi in smiles_list]


def predict_molprice(smiles_list, molprice_path):
    """Predict MolPrice for a list of SMILES."""
    from molprice import MolPriceModel
    mp_model = MolPriceModel(molprice_path)
    return mp_model.predict_batch(smiles_list)


def main():
    parser = argparse.ArgumentParser(
        description="Categorise FOM1 dataset into bio/nonbio/excluded categories")
    parser.add_argument("--baseline_csv", type=str,
                        default="output/baseline_fom1_results.csv")
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--molprice_model", type=str,
                        default="../models/MolPrice/MP_Morgan_hybrid.pkl")
    parser.add_argument("--output", type=str,
                        default="output/fom1_dataset_categories.csv")
    # Constraint thresholds (matching sweep defaults)
    parser.add_argument("--mp_threshold", type=float, default=-30)
    parser.add_argument("--bp_threshold", type=float, default=100)
    parser.add_argument("--fp_threshold", type=float, default=398.15)
    parser.add_argument("--dc_threshold", type=float, default=7)
    parser.add_argument("--tox_threshold", type=float, default=3)
    parser.add_argument("--sc_threshold", type=float, default=3)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load baseline data
    # ------------------------------------------------------------------
    print(f"Loading baseline: {args.baseline_csv}")
    df = pd.read_csv(args.baseline_csv)
    n_total = len(df)
    print(f"  {n_total} molecules loaded")

    # ------------------------------------------------------------------
    # 2. Predict SCScore (if not already present)
    # ------------------------------------------------------------------
    if "SCScore" not in df.columns:
        print("Predicting SCScore...")
        df["SCScore"] = predict_scscore(df["SMILES"].tolist(), args.model_dir)
    else:
        print("SCScore already present")

    # ------------------------------------------------------------------
    # 3. Predict MolPrice
    # ------------------------------------------------------------------
    if os.path.exists(args.molprice_model):
        print(f"Predicting MolPrice from {args.molprice_model}...")
        df["MolPrice"] = predict_molprice(df["SMILES"].tolist(),
                                           args.molprice_model)
    else:
        print(f"WARNING: MolPrice model not found at {args.molprice_model}")
        df["MolPrice"] = np.nan

    # ------------------------------------------------------------------
    # 4. Apply base constraints (matching GA defaults)
    # ------------------------------------------------------------------
    print("\nApplying base constraints...")

    # Structural filter (banned fragments)
    df["has_banned_fragments"] = df["SMILES"].apply(has_invalid_fragments)
    n_banned = df["has_banned_fragments"].sum()
    print(f"  Banned fragments: {n_banned} molecules")

    # Build constraint mask with available columns
    constraint_parts = [~df["has_banned_fragments"]]
    if "mp" in df.columns:
        constraint_parts.append(df["mp"] < args.mp_threshold)
    if "bp" in df.columns:
        constraint_parts.append(df["bp"] >= args.bp_threshold)
    if "fp" in df.columns:
        constraint_parts.append(df["fp"] >= args.fp_threshold)
    if "dc" in df.columns:
        constraint_parts.append(df["dc"] <= args.dc_threshold)
    if "Tox21_Score" in df.columns:
        constraint_parts.append(df["Tox21_Score"] <= args.tox_threshold)
    if "SCScore" in df.columns:
        constraint_parts.append(df["SCScore"] <= args.sc_threshold)

    base_mask = constraint_parts[0]
    for part in constraint_parts[1:]:
        base_mask = base_mask & part

    df["passes_base_constraints"] = base_mask
    n_pass = base_mask.sum()
    print(f"  Pass all base constraints: {n_pass}/{n_total}")

    # Per-constraint breakdown
    if "mp" in df.columns:
        print(f"    MP < {args.mp_threshold}: "
              f"{(df['mp'] < args.mp_threshold).sum()}")
    if "bp" in df.columns:
        print(f"    BP >= {args.bp_threshold}: "
              f"{(df['bp'] >= args.bp_threshold).sum()}")
    if "fp" in df.columns:
        print(f"    FP >= {args.fp_threshold}: "
              f"{(df['fp'] >= args.fp_threshold).sum()}")
    if "dc" in df.columns:
        print(f"    DC <= {args.dc_threshold}: "
              f"{(df['dc'] <= args.dc_threshold).sum()}")
    if "Tox21_Score" in df.columns:
        print(f"    Tox21 <= {args.tox_threshold}: "
              f"{(df['Tox21_Score'] <= args.tox_threshold).sum()}")
    if "SCScore" in df.columns:
        print(f"    SCScore <= {args.sc_threshold}: "
              f"{(df['SCScore'] <= args.sc_threshold).sum()}")
    print(f"    No banned fragments: {(~df['has_banned_fragments']).sum()}")

    # ------------------------------------------------------------------
    # 5. Classify biodegradability
    # ------------------------------------------------------------------
    if "Biodegradable" in df.columns:
        df["is_biodegradable"] = df["Biodegradable"].astype(bool)
    else:
        df["is_biodegradable"] = False
    n_biodeg = df["is_biodegradable"].sum()
    print(f"\n  Biodegradable: {n_biodeg}/{n_total}")

    # ------------------------------------------------------------------
    # 6. Assign categories (for molecules passing base constraints)
    # ------------------------------------------------------------------
    def _assign_category(row):
        if not row["passes_base_constraints"]:
            return "excluded"
        if row["is_biodegradable"]:
            return "bio"
        return "nonbio"

    df["category"] = df.apply(_assign_category, axis=1)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  CATEGORY SUMMARY (molecules passing base constraints)")
    print(f"{'='*60}")

    for cat in ["bio", "nonbio"]:
        sub = df[df["category"] == cat]
        n = len(sub)
        fom1_str = ""
        if n > 0 and "FOM1_exp_avg" in sub.columns:
            fvals = sub["FOM1_exp_avg"].dropna()
            if len(fvals) > 0:
                fom1_str = (f"  FOM1: {fvals.mean():.1f} "
                            f"(max {fvals.max():.1f})")
        print(f"  {cat:<20} {n:>4} molecules{fom1_str}")

    excluded = len(df[df["category"] == "excluded"])
    print(f"  {'excluded':<20} {excluded:>4} molecules")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
