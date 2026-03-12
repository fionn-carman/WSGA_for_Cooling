#!/usr/bin/env python
"""
Draw skeletal structures for Pareto front molecules (WSGA and baseline FOM1 dataset).
Produces one image per category (bio/nonbio), each showing WSGA and baseline fronts.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# Allow importing from ../src
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def _pareto_front(x, y):
    """Return indices of 1st Pareto front maximising both x and y."""
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dom = (x[i] >= x) & (y[i] >= y) & ((x[i] > x) | (y[i] > y))
        dom[i] = False
        is_pareto[dom] = False
    return np.where(is_pareto)[0]


def _get_baseline_pareto(baseline_csv, molprice_model_path, biodeg):
    """Compute baseline Pareto front for a given biodeg category."""
    from molprice import MolPriceModel
    from wsga_helper import has_invalid_fragments

    df = pd.read_csv(baseline_csv)

    # Apply constraints (same as analyse_molprice_sweep.py)
    df = df.dropna(subset=["FOM1_exp_avg"])
    if "MP-Measured" in df.columns:
        df = df[df["MP-Measured"] < -30]
    if "BP-Measured" in df.columns:
        df = df[df["BP-Measured"] >= 100]
    if "DC_exp" in df.columns:
        df = df[df["DC_exp"] <= 7]
    if "flashpoint" in df.columns:
        df = df[df["flashpoint"] >= 373.15]
    if "Tox21_Score" in df.columns:
        df = df[df["Tox21_Score"] <= 3]

    # Banned fragments
    mask = df["SMILES"].apply(lambda s: not has_invalid_fragments(s))
    df = df[mask]

    # Biodeg split
    if biodeg == "bio":
        df = df[df["Biodegradable"] == True]
    # nonbio = all molecules (no biodeg constraint)

    if len(df) == 0:
        return pd.DataFrame()

    # Predict MolPrice
    mp = MolPriceModel(molprice_model_path)
    df["MolPrice"] = mp.predict_batch(df["SMILES"].tolist())
    df = df.dropna(subset=["MolPrice"])

    # Pareto front
    afford = -df["MolPrice"].values
    fom1 = df["FOM1_exp_avg"].values
    front_idx = _pareto_front(afford, fom1)

    result = df.iloc[front_idx].sort_values("FOM1_exp_avg", ascending=False)
    return result


def draw_grid(mols, labels, subtitle_lines, out_path, mols_per_row=4):
    """Draw a grid of molecules with labels underneath."""
    n = len(mols)
    if n == 0:
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(350, 350),
        legends=labels,
    )
    img.save(out_path)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw Pareto front molecules")
    parser.add_argument("--wsga_bio",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "outputs", "molprice_reference",
                            "pareto_front_molecules_bio.csv"))
    parser.add_argument("--wsga_nonbio",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "outputs", "molprice_reference",
                            "pareto_front_molecules_nonbio.csv"))
    parser.add_argument("--baseline_csv",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "BaselineFOM1Eval", "output",
                            "baseline_fom1_results.csv"))
    parser.add_argument("--molprice_model",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "models", "MolPrice",
                            "MP_Morgan_hybrid.pkl"))
    parser.add_argument("--out_dir",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "outputs", "molprice_reference"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Process each category
    for biodeg, wsga_path in [("bio", args.wsga_bio),
                               ("nonbio", args.wsga_nonbio)]:
        tag = "Biodegradable" if biodeg == "bio" else "Non-biodegradable"
        print(f"\n=== {tag} ===")

        # --- WSGA Pareto front ---
        wsga_df = pd.read_csv(wsga_path)
        wsga_df = wsga_df[wsga_df["front"] == 1].sort_values(
            "FOM1_avg", ascending=False)
        print(f"  WSGA Pareto: {len(wsga_df)} molecules")

        wsga_mols = []
        wsga_labels = []
        for _, row in wsga_df.iterrows():
            mol = Chem.MolFromSmiles(row["CanonicalSMILES"])
            if mol is None:
                continue
            wsga_mols.append(mol)
            wsga_labels.append(
                f"FOM1={row['FOM1_avg']:.1f}  MP={row['MolPrice']:.2f}")

        out = os.path.join(args.out_dir, f"pareto_wsga_{biodeg}.png")
        draw_grid(wsga_mols, wsga_labels, [], out)

        # --- Baseline Pareto front ---
        if os.path.exists(args.baseline_csv) and \
           os.path.exists(args.molprice_model):
            base_df = _get_baseline_pareto(
                args.baseline_csv, args.molprice_model, biodeg)
            print(f"  Baseline Pareto: {len(base_df)} molecules")

            if len(base_df) > 0:
                base_mols = []
                base_labels = []
                for _, row in base_df.iterrows():
                    mol = Chem.MolFromSmiles(row["SMILES"])
                    if mol is None:
                        continue
                    base_mols.append(mol)
                    base_labels.append(
                        f"FOM1={row['FOM1_exp_avg']:.1f}  "
                        f"MP={row['MolPrice']:.2f}")

                out = os.path.join(args.out_dir,
                                   f"pareto_baseline_{biodeg}.png")
                draw_grid(base_mols, base_labels, [], out)
        else:
            print("  Baseline CSV or MolPrice model not found, skipping")


if __name__ == "__main__":
    main()
