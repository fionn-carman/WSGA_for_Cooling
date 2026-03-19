#!/usr/bin/env python
"""
Plot the baseline FOM1 filter funnel figure.

Reads the categorised dataset from categorise_fom1_dataset.py and renders
a trapezoid funnel with the cumulative filter cascade, plus a 4-category
breakdown of survivors (biodeg x stability).

Usage:
    cd BaselineFOM1Eval/
    python plot_filter_funnel.py
    python plot_filter_funnel.py --categories_csv output/fom1_dataset_categories.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from wsga_helper import has_invalid_fragments


def compute_funnel_data(df):
    """Compute cumulative filter counts from the categories CSV."""
    n_total = len(df)

    mask_synth = (
        (~df["has_banned_fragments"]) &
        (df["SCScore"] <= 3) &
        (df["Tox21_Score"] <= 3) &
        (df["BP-Measured"] >= 100)
    )
    mask_dc = mask_synth & (df["DC_exp"] <= 7)
    mask_fp = mask_dc & (df["flashpoint"] >= 373.15)
    mask_mp = mask_fp & (df["MP-Measured"] < -30)

    funnel = [
        ("NIST 8,100 dataset\n(FOM1 subset)", n_total),
        ("Synthesisability, safety,\nand structural filters", int(mask_synth.sum())),
        ("Dielectric constant \u2264 7", int(mask_dc.sum())),
        ("Flash point \u2265 100 \u00b0C", int(mask_fp.sum())),
        ("Melting point < \u221230 \u00b0C", int(mask_mp.sum())),
    ]

    survivors = df[mask_mp]
    cats = {
        "nonbio_unstable": int(((~survivors["is_biodegradable"]) & (~survivors["is_stable"])).sum()),
        "nonbio_stable": int(((~survivors["is_biodegradable"]) & (survivors["is_stable"])).sum()),
        "bio_unstable": int(((survivors["is_biodegradable"]) & (~survivors["is_stable"])).sum()),
        "bio_stable": int(((survivors["is_biodegradable"]) & (survivors["is_stable"])).sum()),
    }

    return funnel, cats


def plot_funnel(funnel, cats, out_path):
    """Render the trapezoid funnel figure with drop shadows and polished styling."""
    fig, ax = plt.subplots(figsize=(9.5, 14.5))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-6, 13)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#F5F5F2")

    # Funnel geometry
    cx = 4.2
    y_top = 12.0
    layer_h = 1.20
    gap = 0.10
    w_top = 3.8
    w_bot = 1.4
    n = len(funnel)
    shadow_dx = 0.08
    shadow_dy = -0.08

    funnel_colors = ["#A8AA90", "#728262", "#5A7050", "#466040", "#335030"]
    shadow_color = "#C0BFB8"

    badge_right_x = 8.8

    for i, (label, count) in enumerate(funnel):
        t0 = i / n
        t1 = (i + 1) / n

        top_y = y_top - i * (layer_h + gap)
        bot_y = top_y - layer_h

        hw_t = w_top + (w_bot - w_top) * t0
        hw_b = w_top + (w_bot - w_top) * t1

        verts = [
            (cx - hw_t, top_y),
            (cx + hw_t, top_y),
            (cx + hw_b, bot_y),
            (cx - hw_b, bot_y),
        ]

        shadow_verts = [(x + shadow_dx, y + shadow_dy) for x, y in verts]
        ax.add_patch(Polygon(shadow_verts, closed=True,
                             facecolor=shadow_color, edgecolor="none",
                             alpha=0.45, zorder=1))

        ax.add_patch(Polygon(verts, closed=True,
                             facecolor=funnel_colors[i], edgecolor="none",
                             zorder=2))

        mid_y = (top_y + bot_y) / 2

        fs = 14 if i == 0 else 12.5
        style = "italic" if i == 0 else "normal"
        ax.text(cx, mid_y, label, ha="center", va="center",
                fontsize=fs, color="white", fontstyle=style,
                fontweight="normal", zorder=3)

        badge_w, badge_h = 1.35, 0.75
        badge = FancyBboxPatch(
            (badge_right_x - badge_w / 2, mid_y - badge_h / 2),
            badge_w, badge_h,
            boxstyle="round,pad=0.15",
            facecolor="#E6E8DA", edgecolor="#C5C7B8", linewidth=1.2,
            zorder=4,
        )
        ax.add_patch(badge)
        ax.text(badge_right_x, mid_y, f"{count:,}",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color="#505050", zorder=5)

    # Two-level split: Stable / Unstable -> Bio / Non-bio
    last_bot_y = y_top - n * (layer_h + gap) + gap

    n_stable = cats["nonbio_stable"] + cats["bio_stable"]
    n_unstable = cats["nonbio_unstable"] + cats["bio_unstable"]

    # Row 1: Stable vs Unstable
    row1_h = 1.35
    row1_w = 3.3
    row1_gap = 0.40
    row1_y_bot = last_bot_y - 2.2
    row1_left_x = cx - row1_w - row1_gap / 2
    row1_right_x = cx + row1_gap / 2

    ax.plot([cx - w_bot * 0.15, row1_left_x + row1_w / 2],
            [last_bot_y - 0.05, row1_y_bot + row1_h + 0.1],
            color="#B5B5AD", linewidth=1.2, zorder=1)
    ax.plot([cx + w_bot * 0.15, row1_right_x + row1_w / 2],
            [last_bot_y - 0.05, row1_y_bot + row1_h + 0.1],
            color="#B5B5AD", linewidth=1.2, zorder=1)

    row1_data = [
        (row1_left_x,  n_unstable, "Unstable", "#8B7355"),
        (row1_right_x, n_stable,   "Stable",   "#5A7A5A"),
    ]

    for bx, count, label, color in row1_data:
        ax.add_patch(FancyBboxPatch(
            (bx + 0.08, row1_y_bot - 0.08), row1_w, row1_h,
            boxstyle="round,pad=0.16",
            facecolor="#BBBBB5", edgecolor="none", alpha=0.5, zorder=3))
        ax.add_patch(FancyBboxPatch(
            (bx, row1_y_bot), row1_w, row1_h,
            boxstyle="round,pad=0.16",
            facecolor=color, edgecolor="none", zorder=4))
        ax.text(bx + row1_w / 2, row1_y_bot + row1_h * 0.6, str(count),
                ha="center", va="center", fontsize=28, fontweight="bold",
                color="white", zorder=5)
        ax.text(bx + row1_w / 2, row1_y_bot + row1_h * 0.2, label,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white", zorder=5)

    # Row 2: Bio / Non-bio under each
    row2_h = 1.25
    row2_inner_gap = 0.30
    row2_w = (row1_w - row2_inner_gap) / 2
    row2_y_bot = row1_y_bot - 1.9

    for parent_x, parent_w in [(row1_left_x, row1_w), (row1_right_x, row1_w)]:
        pcx = parent_x + parent_w / 2
        child_l = parent_x + row2_w / 2
        child_r = parent_x + row2_w + row2_inner_gap + row2_w / 2
        ax.plot([pcx, child_l], [row1_y_bot - 0.02, row2_y_bot + row2_h + 0.08],
                color="#B5B5AD", linewidth=1.0, zorder=1)
        ax.plot([pcx, child_r], [row1_y_bot - 0.02, row2_y_bot + row2_h + 0.08],
                color="#B5B5AD", linewidth=1.0, zorder=1)

    row2_data = [
        (row1_left_x,                                cats["nonbio_unstable"], "Non-bio", "#C4A535"),
        (row1_left_x + row2_w + row2_inner_gap,      cats["bio_unstable"],   "Bio",     "#B85540"),
        (row1_right_x,                               cats["nonbio_stable"],  "Non-bio", "#2E5050"),
        (row1_right_x + row2_w + row2_inner_gap,     cats["bio_stable"],     "Bio",     "#3A907C"),
    ]

    for bx, count, label, color in row2_data:
        ax.add_patch(FancyBboxPatch(
            (bx + 0.06, row2_y_bot - 0.06), row2_w, row2_h,
            boxstyle="round,pad=0.14",
            facecolor="#BBBBB5", edgecolor="none", alpha=0.5, zorder=3))
        ax.add_patch(FancyBboxPatch(
            (bx, row2_y_bot), row2_w, row2_h,
            boxstyle="round,pad=0.14",
            facecolor=color, edgecolor="none", zorder=4))
        ax.text(bx + row2_w / 2, row2_y_bot + row2_h * 0.6, str(count),
                ha="center", va="center", fontsize=22, fontweight="bold",
                color="white", zorder=5)
        ax.text(bx + row2_w / 2, row2_y_bot + row2_h * 0.2, label,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white", zorder=5)

    ax.set_ylim(row2_y_bot - 0.8, 13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#F5F5F2")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot baseline filter funnel")
    parser.add_argument("--categories_csv", type=str,
                        default="output/fom1_dataset_categories.csv")
    parser.add_argument("--output", type=str,
                        default="output/baseline_filter_funnel.png")
    args = parser.parse_args()

    df = pd.read_csv(args.categories_csv)
    print(f"Loaded {len(df)} molecules from {args.categories_csv}")

    funnel, cats = compute_funnel_data(df)

    print("\nFunnel steps:")
    for label, count in funnel:
        print(f"  {label.replace(chr(10), ' ')}: {count}")

    print(f"\nSurvivor categories:")
    for k, v in cats.items():
        print(f"  {k}: {v}")

    plot_funnel(funnel, cats, args.output)


if __name__ == "__main__":
    main()
