#!/usr/bin/env python
"""
Generate a reference figure showing skeletal structures of common molecules
alongside their MolPrice-predicted cost (log USD/mmol).

Includes 3 well-known CHO reference compounds and 6 from the FOM1 dataset
spanning MolPrice 3–6.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw

# Allow importing from ../src
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
from molprice import MolPriceModel


# ------------------------------------------------------------------
# Known reference molecules (CHO only)
# ------------------------------------------------------------------
REFERENCE = [
    ("Cyclohexane", "C1CCCCC1"),
    ("Ethanol",     "CCO"),
    ("Toluene",     "Cc1ccccc1"),
    ("Triglyme",    "COCCOCCOCCOC"),
]


def _is_cho(smi):
    """Return True if molecule contains only C, H, O."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return all(a.GetAtomicNum() in {6, 1, 8} for a in mol.GetAtoms())


def _pick_from_dataset(baseline_csv, mp_model, n=6, lo=3.0, hi=6.0):
    """Pick n CHO molecules from the FOM1 dataset evenly spread in [lo, hi]."""
    df = pd.read_csv(baseline_csv)
    df["MolPrice"] = mp_model.predict_batch(df["SMILES"].tolist())
    df = df[df["SMILES"].apply(_is_cho)].copy()
    df = df[(df["MolPrice"] >= lo) & (df["MolPrice"] <= hi)]
    df = df.sort_values("MolPrice")

    targets = np.linspace(lo, hi, n)
    picked = []
    used = set()
    for t in targets:
        best_idx, best_dist = None, 999
        for i, row in df.iterrows():
            if i in used:
                continue
            d = abs(row["MolPrice"] - t)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None:
            used.add(best_idx)
            picked.append(df.loc[best_idx])

    return [(row["SMILES"], row["MolPrice"]) for _, row in
            pd.DataFrame(picked).iterrows()]


def smi_to_image(smi, size=(400, 400)):
    """Render SMILES as a PIL image."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return Image.new("RGB", size, "white")
    return Draw.MolToImage(mol, size=size)


def main():
    parser = argparse.ArgumentParser(
        description="Reference figure: molecule structures + MolPrice")
    parser.add_argument("--molprice_model",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "models", "MolPrice",
                            "MP_Morgan_hybrid.pkl"))
    parser.add_argument("--baseline_csv",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "BaselineFOM1Eval", "output",
                            "baseline_fom1_results.csv"))
    parser.add_argument("--out_dir",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "outputs", "molprice_reference"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    mp = MolPriceModel(args.molprice_model)

    # Reference compounds
    entries = []
    for name, smi in REFERENCE:
        cost = mp.predict(smi)
        entries.append((name, smi, cost))

    # FOM1 dataset compounds
    dataset_mols = _pick_from_dataset(args.baseline_csv, mp, n=5, lo=3.5)
    for smi, cost in dataset_mols:
        entries.append(("FOM1 dataset", smi, cost))

    # Sort cheap → expensive
    entries.sort(key=lambda x: x[2])

    for name, smi, cost in entries:
        print(f"  {cost:+.2f}  {name:20s}  {smi}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    n = len(entries)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for idx in range(len(axes)):
        ax = axes[idx]
        ax.set_axis_off()
        if idx >= n:
            continue

        name, smi, cost = entries[idx]
        img = smi_to_image(smi, size=(400, 400))
        ax.imshow(img, aspect="equal")

        label = name if name != "FOM1 dataset" else "FOM1 dataset molecule"
        cost_str = f"MolPrice = {cost:+.2f}"
        ax.set_title(f"{label}\n{cost_str}",
                     fontsize=10, fontweight="bold", pad=4)

    fig.suptitle("MolPrice predictions: log(USD / mmol)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "molprice_reference.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
