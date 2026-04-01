#!/usr/bin/env python
"""
Plot baseline FOM1 category figures:
  1. Molecule grid with category headers (top molecules per category)
  2. FOM1 vs -MolPrice scatter with Pareto fronts

Categories: bio (biodegradable) and nonbio (not biodegradable).

Usage:
    cd BaselineFOM1Eval/
    python plot_categories.py
    python plot_categories.py --categories_csv output/fom1_dataset_categories.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from io import BytesIO
from PIL import Image

RDLogger.DisableLog("rdApp.*")


# ======================================================================
# Category styling
# ======================================================================

CATEGORY_ORDER = ["bio", "nonbio"]
CATEGORY_LABELS = {
    "bio": "Biodegradable",
    "nonbio": "Non-biodegradable",
}
CATEGORY_COLORS = {
    "bio": "#3A907C",
    "nonbio": "#2E5050",
}
CATEGORY_MARKERS = {
    "bio": "^",
    "nonbio": "o",
}


# ======================================================================
# Figure 1: Molecule grid with category headers
# ======================================================================

def plot_molecule_grid(df, out_path, n_rows=6):
    """Plot top molecules per category in a grid with coloured headers."""
    n_cols = len(CATEGORY_ORDER)

    # Filter to passing molecules and get top per category
    passing = df[df["passes_base_constraints"] == True].copy()
    cat_dfs = {}
    for cat in CATEGORY_ORDER:
        sub = passing[passing["category"] == cat].sort_values(
            "FOM1_exp_40", ascending=False).head(n_rows)
        cat_dfs[cat] = sub

    # Layout
    cell_w, cell_h = 260, 280
    header_h = 80
    pad = 10
    img_w = n_cols * cell_w + (n_cols + 1) * pad
    img_h = header_h + n_rows * cell_h + (n_rows + 1) * pad + pad

    fig, ax = plt.subplots(figsize=(img_w / 100, img_h / 100), dpi=100)
    ax.set_xlim(0, img_w)
    ax.set_ylim(0, img_h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Draw category headers
    for col_idx, cat in enumerate(CATEGORY_ORDER):
        count = len(passing[passing["category"] == cat])
        x = pad + col_idx * (cell_w + pad)
        y = img_h - header_h

        rect = FancyBboxPatch(
            (x, y), cell_w, header_h - pad,
            boxstyle="round,pad=6",
            facecolor=CATEGORY_COLORS[cat], edgecolor="none",
        )
        ax.add_patch(rect)

        ax.text(x + cell_w / 2, y + (header_h - pad) * 0.65, str(count),
                ha="center", va="center", fontsize=24, fontweight="bold",
                color="white")
        ax.text(x + cell_w / 2, y + (header_h - pad) * 0.25,
                CATEGORY_LABELS[cat],
                ha="center", va="center", fontsize=9, color="white")

    # Draw molecules
    for col_idx, cat in enumerate(CATEGORY_ORDER):
        sub = cat_dfs[cat]
        for row_idx, (_, mol_row) in enumerate(sub.iterrows()):
            if row_idx >= n_rows:
                break

            smi = mol_row["SMILES"]
            fom1 = mol_row.get("FOM1_exp_40", np.nan)
            molprice = mol_row.get("MolPrice", np.nan)

            mol = Chem.MolFromSmiles(smi)
            x = pad + col_idx * (cell_w + pad)
            y = img_h - header_h - pad - (row_idx + 1) * cell_h

            # Render molecule image
            if mol is not None:
                img = Draw.MolToImage(mol, size=(cell_w, cell_h - 40))
                img_array = np.array(img)
                ax.imshow(img_array, extent=[x, x + cell_w, y + 35, y + cell_h - 5],
                          aspect="auto", zorder=2)

            # Label
            fom1_str = f"FOM1 = {fom1:.0f}" if not np.isnan(fom1) else "FOM1 = ?"
            mp_str = f"MolPrice = {molprice:.1f}" if not np.isnan(molprice) else ""
            label = f"{fom1_str} | {mp_str}" if mp_str else fom1_str
            ax.text(x + cell_w / 2, y + 15, label,
                    ha="center", va="center", fontsize=6.5, color="#555555")

    ax.set_ylim(-pad, img_h + pad)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ======================================================================
# Figure 2: FOM1 vs -MolPrice scatter with Pareto fronts
# ======================================================================

def _pareto_front(x, y):
    """Compute 1st Pareto front for maximising both x and y."""
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dom = ((x[i] >= x) & (y[i] >= y) & ((x[i] > x) | (y[i] > y)))
        dom[i] = False
        is_pareto[dom] = False
    return np.where(is_pareto)[0]


def plot_fom1_vs_molprice(df, out_path):
    """Scatter plot of FOM1 vs -MolPrice with category markers and Pareto fronts."""
    passing = df[df["passes_base_constraints"] == True].copy()
    passing = passing.dropna(subset=["FOM1_exp_40", "MolPrice"])

    fig, ax = plt.subplots(figsize=(8, 6))

    for cat in CATEGORY_ORDER:
        sub = passing[passing["category"] == cat]
        if sub.empty:
            continue

        neg_mp = -sub["MolPrice"].values
        fom1 = sub["FOM1_exp_40"].values

        ax.scatter(neg_mp, fom1,
                   c=CATEGORY_COLORS[cat],
                   marker=CATEGORY_MARKERS[cat],
                   s=40, alpha=0.7, edgecolors="none", zorder=3,
                   label=f"{CATEGORY_LABELS[cat].replace(chr(10), ' ')} (n={len(sub)})")

        # Pareto front
        if len(sub) >= 2:
            pidx = _pareto_front(neg_mp, fom1)
            if len(pidx) >= 2:
                px, py = neg_mp[pidx], fom1[pidx]
                order = np.argsort(px)
                ax.plot(px[order], py[order], "-",
                        color=CATEGORY_COLORS[cat], linewidth=1.5,
                        alpha=0.6, zorder=2)

    ax.set_xlabel(u"\u2212MolPrice / log USD mmol\u207b\u00b9", fontsize=11)
    ax.set_ylabel(u"FOM1 (40 \u00b0C) / W m\u207b\u00b9 K\u207b\u00b9", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot baseline category figures")
    parser.add_argument("--categories_csv", type=str,
                        default="output/fom1_dataset_categories.csv")
    parser.add_argument("--out_dir", type=str, default="output")
    args = parser.parse_args()

    df = pd.read_csv(args.categories_csv)
    passing = df[df["passes_base_constraints"] == True]
    print(f"Loaded {len(df)} molecules, {len(passing)} passing constraints")

    for cat in CATEGORY_ORDER:
        n = len(passing[passing["category"] == cat])
        print(f"  {cat}: {n}")

    os.makedirs(args.out_dir, exist_ok=True)

    plot_molecule_grid(df, os.path.join(args.out_dir, "baseline_molecule_grid.png"))
    plot_fom1_vs_molprice(df, os.path.join(args.out_dir, "baseline_fom1_vs_molprice.png"))


if __name__ == "__main__":
    main()
