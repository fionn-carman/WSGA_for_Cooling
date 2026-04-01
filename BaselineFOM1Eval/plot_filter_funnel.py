#!/usr/bin/env python
"""
Plot the baseline FOM1 filter funnel figure.

Reads the categorised dataset and renders a trapezoid funnel with the
cumulative filter cascade (including reactive-group bans and MP < -30 C
as its own stage), branching into 4 flash-point thresholds (100/125/150
/175 C) at the bottom, each with bio/non-bio sub-counts.

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


# Flash-point thresholds (K) and display labels
FP_THRESHOLDS = [
    (373.15, "100 \u00b0C"),
    (398.15, "125 \u00b0C"),
    (423.15, "150 \u00b0C"),
    (448.15, "175 \u00b0C"),
]


def compute_funnel_data(df):
    """Compute cumulative filter counts and 4-branch FP breakdown."""
    n_total = len(df)

    # Common cascade
    mask_synth = (
        (~df["has_banned_fragments"]) &
        (df["SCScore"] <= 3) &
        (df["Tox21_Score"] <= 3) &
        (df["bp"] >= 100)
    )
    mask_dc = mask_synth & (df["dc"] <= 7)
    mask_mp = mask_dc & (df["mp"] < -30)

    funnel = [
        ("NIST experimental dataset", n_total),
        ("Synthesisability, safety,\nand structural filters", int(mask_synth.sum())),
        ("Dielectric constant \u2264 7", int(mask_dc.sum())),
        ("Melting point < \u221230 \u00b0C", int(mask_mp.sum())),
    ]

    # 4 FP branches (MP already applied above)
    branches = []
    for fp_k, fp_label in FP_THRESHOLDS:
        mask_branch = mask_mp & (df["fp"] >= fp_k)
        survivors = df[mask_branch]
        n_bio = int(survivors["is_biodegradable"].sum())
        n_nonbio = int((~survivors["is_biodegradable"]).sum())
        n_tot = n_bio + n_nonbio
        branches.append({
            "fp_label": fp_label,
            "total": n_tot,
            "bio": n_bio,
            "nonbio": n_nonbio,
        })

    return funnel, branches


def plot_funnel(funnel, branches, out_path):
    """Render the trapezoid funnel with 4 FP-threshold boxes at the bottom."""
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-8, 14)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # -- Funnel geometry --
    cx = 5.0
    y_top = 12.5
    layer_h = 1.20
    gap = 0.10
    w_top = 3.8
    w_bot = 1.6
    n = len(funnel)
    shadow_dx = 0.08
    shadow_dy = -0.08

    # Blue scale: light to dark
    funnel_colors = ["#9DB8D2", "#6A96BE", "#4275A8", "#2B5C8A"]
    shadow_color = "#C8CDD3"
    badge_right_x = 9.2

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
            facecolor="#E0E6ED", edgecolor="#B0BCC8", linewidth=1.2,
            zorder=4,
        )
        ax.add_patch(badge)
        ax.text(badge_right_x, mid_y, f"{count:,}",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color="#3A3A3A", zorder=5)

    # -- "Flash point threshold" label row --
    last_bot_y = y_top - n * (layer_h + gap) + gap
    label_y = last_bot_y - 0.9
    ax.text(cx, label_y,
            "Flash point threshold",
            ha="center", va="center", fontsize=12, color="#3A3A3A",
            fontstyle="italic", zorder=5)

    # Connector from funnel bottom to label
    ax.plot([cx, cx], [last_bot_y - 0.05, label_y + 0.25],
            color="#A0AAB5", linewidth=1.5, zorder=1)

    # -- 4 FP-threshold boxes --
    n_boxes = len(branches)
    box_w = 2.0
    box_h_main = 1.6
    box_gap = 0.30
    total_w = n_boxes * box_w + (n_boxes - 1) * box_gap
    x_start = cx - total_w / 2
    box_top_y = label_y - 0.7

    # Blue scale for boxes (progressively darker)
    box_colors = ["#3B6FA0", "#2E5A87", "#224870", "#183A5A"]

    for i, branch in enumerate(branches):
        bx = x_start + i * (box_w + box_gap)
        by = box_top_y - box_h_main

        # Connector from label to box
        box_cx = bx + box_w / 2
        ax.plot([box_cx, box_cx], [label_y - 0.25, box_top_y + 0.05],
                color="#A0AAB5", linewidth=1.0, zorder=1)

        # Shadow
        ax.add_patch(FancyBboxPatch(
            (bx + shadow_dx, by + shadow_dy), box_w, box_h_main,
            boxstyle="round,pad=0.14",
            facecolor=shadow_color, edgecolor="none", alpha=0.45, zorder=1))

        # Main box
        ax.add_patch(FancyBboxPatch(
            (bx, by), box_w, box_h_main,
            boxstyle="round,pad=0.14",
            facecolor=box_colors[i], edgecolor="none", zorder=4))

        # FP label at top of box
        ax.text(box_cx, by + box_h_main * 0.82,
                f"FP \u2265 {branch['fp_label']}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white", zorder=5)

        # Total count (large)
        ax.text(box_cx, by + box_h_main * 0.48,
                str(branch["total"]),
                ha="center", va="center", fontsize=26, fontweight="bold",
                color="white", zorder=5)

        # Bio / Non-bio sub-counts
        ax.text(box_cx, by + box_h_main * 0.15,
                f"{branch['bio']}B / {branch['nonbio']}NB",
                ha="center", va="center", fontsize=9, fontweight="normal",
                color="#B0C4DE", zorder=5)

    # -- Bio/non-bio sub-bars beneath each box --
    bar_h = 0.50
    bar_gap = 0.08
    bar_top_y = box_top_y - box_h_main - 0.45

    bio_color = "#2980B9"
    nonbio_color = "#7F8C8D"

    for i, branch in enumerate(branches):
        bx = x_start + i * (box_w + box_gap)
        box_cx = bx + box_w / 2

        # Connector line
        ax.plot([box_cx, box_cx],
                [box_top_y - box_h_main - 0.05, bar_top_y + bar_h + 0.05],
                color="#A0AAB5", linewidth=0.8, zorder=1)

        # Two small bars side by side
        half_w = (box_w - bar_gap) / 2

        # Bio bar (left)
        bio_x = bx
        ax.add_patch(FancyBboxPatch(
            (bio_x, bar_top_y), half_w, bar_h,
            boxstyle="round,pad=0.08",
            facecolor=bio_color, edgecolor="none", zorder=4))
        ax.text(bio_x + half_w / 2, bar_top_y + bar_h * 0.55,
                str(branch["bio"]),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white", zorder=5)
        ax.text(bio_x + half_w / 2, bar_top_y + bar_h * 0.15,
                "Bio", ha="center", va="center", fontsize=8,
                color="white", zorder=5)

        # Non-bio bar (right)
        nb_x = bx + half_w + bar_gap
        nb_color = nonbio_color if branch["nonbio"] > 0 else "#B0B0B0"
        ax.add_patch(FancyBboxPatch(
            (nb_x, bar_top_y), half_w, bar_h,
            boxstyle="round,pad=0.08",
            facecolor=nb_color, edgecolor="none", zorder=4))
        ax.text(nb_x + half_w / 2, bar_top_y + bar_h * 0.55,
                str(branch["nonbio"]),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white", zorder=5)
        ax.text(nb_x + half_w / 2, bar_top_y + bar_h * 0.15,
                "Non-bio", ha="center", va="center", fontsize=8,
                color="white", zorder=5)

    ax.set_ylim(bar_top_y - 0.6, 14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
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

    funnel, branches = compute_funnel_data(df)

    print("\nFunnel steps:")
    for label, count in funnel:
        print(f"  {label.replace(chr(10), ' ')}: {count}")

    print(f"\nFP branches:")
    for b in branches:
        print(f"  FP >= {b['fp_label']}: {b['total']} "
              f"({b['bio']} bio, {b['nonbio']} non-bio)")

    plot_funnel(funnel, branches, args.output)


if __name__ == "__main__":
    main()
