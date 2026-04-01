#!/usr/bin/env python
"""
Plot flash point vs FOM1 scatter for the full NIST baseline dataset.

Colour encodes predicted MolPrice (cost). Vertical dashed lines mark 4
flash-point thresholds (100/125/150/175 C) to show the constraint cliff.

Usage:
    cd BaselineFOM1Eval/
    python plot_fp_fom1_tradeoff.py
    python plot_fp_fom1_tradeoff.py --categories_csv output/fom1_dataset_categories.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


FP_LINES_K = [373.15, 398.15, 423.15, 448.15]
FP_LABELS = ["100 \u00b0C", "125 \u00b0C", "150 \u00b0C", "175 \u00b0C"]


def plot_fp_fom1(df, out_path):
    """Scatter: flash point (x) vs FOM1_40 (y), coloured by MolPrice."""
    # Drop rows with missing FP or FOM1
    mask = df["fp"].notna() & df["FOM1_exp_40"].notna()
    d = df[mask].copy()
    print(f"Plotting {len(d)} molecules with valid FP and FOM1")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colour by MolPrice (clipped for visual range)
    has_mp = d["MolPrice"].notna()
    vmin = d.loc[has_mp, "MolPrice"].quantile(0.02) if has_mp.any() else 0
    vmax = d.loc[has_mp, "MolPrice"].quantile(0.98) if has_mp.any() else 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd

    # Plot molecules without MolPrice in grey
    if (~has_mp).any():
        ax.scatter(d.loc[~has_mp, "fp"], d.loc[~has_mp, "FOM1_exp_40"],
                   c="#BBBBBB", s=12, alpha=0.3, edgecolors="none",
                   zorder=2, label="No cost data")

    # Plot molecules with MolPrice
    sc = ax.scatter(d.loc[has_mp, "fp"], d.loc[has_mp, "FOM1_exp_40"],
                    c=d.loc[has_mp, "MolPrice"], cmap=cmap, norm=norm,
                    s=14, alpha=0.5, edgecolors="none", zorder=3)

    # Colourbar
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        pad=0.02, fraction=0.04)
    cbar.set_label("Predicted Cost / MolPrice", fontsize=11)
    cbar.outline.set_visible(False)

    # FP threshold lines
    line_colors = ["#555555", "#555555", "#555555", "#555555"]
    for fp_k, label, lc in zip(FP_LINES_K, FP_LABELS, line_colors):
        ax.axvline(fp_k, color=lc, linestyle="--", linewidth=1.0,
                   alpha=0.7, zorder=1)
        ax.text(fp_k + 1.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 150,
                label, fontsize=9, color="#555555", va="top", ha="left",
                rotation=90)

    # Re-apply text positions after data is plotted (ylim may have changed)
    ymax = d["FOM1_exp_40"].quantile(0.995) * 1.05
    ymin = d["FOM1_exp_40"].quantile(0.005) * 0.95
    ax.set_ylim(ymin, ymax)

    # Redraw labels at correct positions
    for txt in ax.texts:
        txt.remove()
    for fp_k, label in zip(FP_LINES_K, FP_LABELS):
        ax.text(fp_k + 1.5, ymax * 0.98, label,
                fontsize=9, color="#555555", va="top", ha="left",
                rotation=90)

    ax.set_xlabel("Flash Point / K", fontsize=12)
    ax.set_ylabel("Cooling Performance / FOM1$_{40}$", fontsize=12)

    # Only show legend if there are labelled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot FP vs FOM1 scatter (full baseline dataset)")
    parser.add_argument("--categories_csv", type=str,
                        default="output/fom1_dataset_categories.csv")
    parser.add_argument("--output", type=str,
                        default="output/baseline_fp_fom1_tradeoff.png")
    args = parser.parse_args()

    df = pd.read_csv(args.categories_csv)
    print(f"Loaded {len(df)} molecules from {args.categories_csv}")

    plot_fp_fom1(df, args.output)


if __name__ == "__main__":
    main()
