"""
Plot WSGA Pareto front molecules by category in a grid layout.

Reads per-category Pareto front CSVs from the stability sweep analysis,
draws 2D molecular structures with FOM1/MolPrice annotations.

Usage:
    cd scripts/
    python plot_pareto_molecules.py
    python plot_pareto_molecules.py --analysis_dir ../outputs/stability_molprice_sweep/analysis
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from io import BytesIO
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw


# Category ordering: most relaxed (left) to most constrained (right)
CATEGORIES = [
    {
        "key": "nonbio_unstable",
        "label": "Biodegradable = False\nStable = False",
        "color": "#6B9F78",
    },
    {
        "key": "nonbio_stable",
        "label": "Biodegradable = False\nStable = True",
        "color": "#3D5A45",
    },
    {
        "key": "bio_unstable",
        "label": "Biodegradable = True\nStable = False",
        "color": "#C0504D",
    },
    {
        "key": "bio_stable",
        "label": "Biodegradable = True\nStable = True",
        "color": "#D4A843",
    },
]


def smiles_to_image(smiles, size=(300, 200)):
    """Convert SMILES to a PIL Image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        img = Image.new("RGB", size, "white")
        return img
    return Draw.MolToImage(mol, size=size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=str,
                        default="../outputs/stability_molprice_sweep/analysis")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load Pareto front CSVs
    cat_data = {}
    for cat in CATEGORIES:
        csv_path = os.path.join(args.analysis_dir,
                                f"pareto_front_{cat['key']}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Sort by FOM1 descending
            df = df.sort_values("FOM1_avg", ascending=False).reset_index(drop=True)
            smiles_col = ("CanonicalSMILES" if "CanonicalSMILES" in df.columns
                          else "SMILES")
            cat_data[cat["key"]] = {
                "df": df,
                "smiles_col": smiles_col,
            }
        else:
            print(f"  WARNING: {csv_path} not found")
            cat_data[cat["key"]] = {"df": pd.DataFrame(), "smiles_col": "SMILES"}

    n_cols = len(CATEGORIES)
    max_rows = max(len(cat_data[c["key"]]["df"]) for c in CATEGORIES)

    if max_rows == 0:
        print("No Pareto front data found.")
        return

    # Layout parameters
    mol_img_size = (300, 200)
    cell_w = 3.5       # inches per column
    header_h = 0.9     # header row height
    mol_h = 2.8        # height per molecule cell
    fig_w = cell_w * n_cols
    fig_h = header_h + mol_h * max_rows + 0.3

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.invert_yaxis()

    # Draw header boxes
    for col_idx, cat in enumerate(CATEGORIES):
        n_mols = len(cat_data[cat["key"]]["df"])
        x = col_idx * cell_w + 0.15
        w = cell_w - 0.3

        box = FancyBboxPatch(
            (x, 0.1), w, header_h - 0.2,
            boxstyle="round,pad=0.1",
            facecolor=cat["color"], edgecolor="none",
            transform=ax.transData, zorder=2,
        )
        ax.add_patch(box)

        # Count number (large, bold)
        ax.text(x + w / 2, 0.28, str(n_mols),
                fontsize=26, fontweight="bold", color="white",
                ha="center", va="center", zorder=3)
        # Label (smaller)
        ax.text(x + w / 2, 0.65, cat["label"],
                fontsize=7.5, color="white", fontweight="bold",
                ha="center", va="center", zorder=3,
                linespacing=1.3)

    # Draw molecules
    for col_idx, cat in enumerate(CATEGORIES):
        data = cat_data[cat["key"]]
        df = data["df"]
        smiles_col = data["smiles_col"]

        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            smi = row[smiles_col]
            fom1 = row["FOM1_avg"]
            mp = row.get("MolPrice", np.nan)

            # Cell position
            x_center = col_idx * cell_w + cell_w / 2
            y_top = header_h + row_idx * mol_h + 0.15

            # Draw molecule image
            img = smiles_to_image(smi, size=mol_img_size)
            img_array = np.array(img)

            # Position for image
            img_x = col_idx * cell_w + 0.35
            img_y = y_top + 0.1
            img_w = cell_w - 0.7
            img_h = mol_h - 0.9

            ax.imshow(img_array,
                      extent=[img_x, img_x + img_w,
                              img_y + img_h, img_y],
                      aspect="auto", zorder=1,
                      interpolation="bilinear")

            # Caption
            mp_str = f"{mp:.1f}" if not np.isnan(mp) else "n/a"
            caption = f"FOM1 = {fom1:.0f}  |  MolPrice = {mp_str}"
            ax.text(x_center, img_y + img_h + 0.25, caption,
                    fontsize=8, color="#333333", ha="center", va="top",
                    zorder=3)

    fig.tight_layout(pad=0.2)

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(args.analysis_dir,
                                "pareto_molecules_by_category.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
