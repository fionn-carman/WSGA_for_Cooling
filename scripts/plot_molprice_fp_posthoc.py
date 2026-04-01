#!/usr/bin/env python
"""
Post-hoc analysis: does MolPrice cost pressure change where the WSGA finds
molecules on the FP vs FOM1 frontier?

Reads the compact valid_molecules_summary.csv produced by
extract_molprice_sweep_valid.py, computes Pareto fronts per cost level,
overlays with the baseline survivors, and prints summary statistics.

Usage:
    cd scripts/
    python plot_molprice_fp_posthoc.py
    python plot_molprice_fp_posthoc.py --summary ../outputs/molprice_sweep_nist8100/valid_molecules_summary.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
LEVEL_ORDER = ["nocost", "gentle", "moderate", "firm", "tight", "aggressive"]
LEVEL_LABELS = {
    "nocost": "No cost",
    "gentle": "Gentle",
    "moderate": "Moderate",
    "firm": "Firm",
    "tight": "Tight",
    "aggressive": "Aggressive",
}

# Dark-to-light colour ramp: nocost (darkest) → aggressive (lightest)
LEVEL_COLORS = {
    "nocost":     "#1B2838",
    "gentle":     "#2A5F8F",
    "moderate":   "#3D8EBA",
    "firm":       "#6DB8D4",
    "tight":      "#A0D4E4",
    "aggressive": "#D0ECF4",
}

CATEGORY_ORDER = ["bio_stable", "bio_unstable", "nonbio_stable", "nonbio_unstable"]
CATEGORY_LABELS = {
    "bio_stable": "Biodeg., Stable",
    "bio_unstable": "Biodeg., Unstable",
    "nonbio_stable": "Non-biodeg., Stable",
    "nonbio_unstable": "Non-biodeg., Unstable",
}


# ------------------------------------------------------------------
# Pareto front computation
# ------------------------------------------------------------------
def pareto_front_max(x, y):
    """Pareto front for maximising both x and y. Returns index array."""
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dom = (x[i] >= x) & (y[i] >= y) & ((x[i] > x) | (y[i] > y))
        dom[i] = False
        is_pareto[dom] = False
    return np.where(is_pareto)[0]


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_fp_fom1_by_level(summary_df, baseline_df, out_dir):
    """Single panel: Pareto fronts coloured by cost level + baseline."""

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 8,
    })

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Pool all seeds per level, deduplicate globally per level
    for level in LEVEL_ORDER:
        ldf = summary_df[summary_df["cost_level"] == level].copy()
        if ldf.empty:
            continue

        # Deduplicate across seeds
        ldf = ldf.sort_values("fom1_40C", ascending=False)
        ldf = ldf.drop_duplicates(subset=["SMILES"], keep="first")

        fp_c = ldf["fp"].values - 273.15  # K → °C
        fom1 = ldf["fom1_40C"].values

        # Light scatter
        ax.scatter(fp_c, fom1, c=LEVEL_COLORS[level], s=4, alpha=0.08,
                   edgecolors="none", zorder=1)

        # Pareto front
        pidx = pareto_front_max(fp_c, fom1)
        if len(pidx) > 0:
            pfp, pfom = fp_c[pidx], fom1[pidx]
            order = np.argsort(pfp)
            ax.plot(pfp[order], pfom[order], "o-",
                    color=LEVEL_COLORS[level], markersize=3.5, linewidth=1.5,
                    label=LEVEL_LABELS[level], zorder=3 + LEVEL_ORDER.index(level))

    # Baseline survivors
    if baseline_df is not None and len(baseline_df) > 0:
        bfp = baseline_df["fp"].values  # already in K
        bfp_c = bfp - 273.15
        bfom = baseline_df["FOM1_exp_40"].values

        ax.scatter(bfp_c, bfom, c="#E63946", s=25, alpha=0.8,
                   edgecolors="black", linewidths=0.4, marker="D",
                   zorder=10, label=f"Baseline ({len(baseline_df)})")

        # Baseline Pareto front
        pidx = pareto_front_max(bfp_c, bfom)
        if len(pidx) > 0:
            pfp, pfom = bfp_c[pidx], bfom[pidx]
            order = np.argsort(pfp)
            ax.plot(pfp[order], pfom[order], "D--", color="#E63946",
                    markersize=4, linewidth=1.2, zorder=11)

    ax.set_xlabel("Flash Point / \u00b0C")
    ax.set_ylabel("FOM1 / W m\u207b\u00b9 K\u207b\u00b9")
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"molprice_fp_posthoc.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(out_dir, 'molprice_fp_posthoc.{{png,pdf}}')}")


def plot_fp_fom1_2x2(summary_df, baseline_df, out_dir):
    """2x2 grid: one panel per category, fronts coloured by cost level."""

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 7,
    })

    panel_map = {
        "nonbio_unstable": (0, 0),
        "bio_unstable": (0, 1),
        "nonbio_stable": (1, 0),
        "bio_stable": (1, 1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)

    for category, (row, col) in panel_map.items():
        ax = axes[row, col]
        cdf = summary_df[summary_df["category"] == category]

        for level in LEVEL_ORDER:
            ldf = cdf[cdf["cost_level"] == level].copy()
            if ldf.empty:
                continue

            ldf = ldf.sort_values("fom1_40C", ascending=False)
            ldf = ldf.drop_duplicates(subset=["SMILES"], keep="first")

            fp_c = ldf["fp"].values - 273.15
            fom1 = ldf["fom1_40C"].values

            ax.scatter(fp_c, fom1, c=LEVEL_COLORS[level], s=4, alpha=0.08,
                       edgecolors="none", zorder=1)

            pidx = pareto_front_max(fp_c, fom1)
            if len(pidx) > 0:
                pfp, pfom = fp_c[pidx], fom1[pidx]
                order = np.argsort(pfp)
                ax.plot(pfp[order], pfom[order], "o-",
                        color=LEVEL_COLORS[level], markersize=3, linewidth=1.3,
                        label=LEVEL_LABELS[level],
                        zorder=3 + LEVEL_ORDER.index(level))

        # Baseline (filtered for this category)
        if baseline_df is not None:
            bdf = _filter_baseline_for_category(baseline_df, category)
            if len(bdf) > 0 and "fp" in bdf.columns and "FOM1_exp_40" in bdf.columns:
                bfp_c = bdf["fp"].values - 273.15
                bfom = bdf["FOM1_exp_40"].values

                ax.scatter(bfp_c, bfom, c="#E63946", s=20, alpha=0.8,
                           edgecolors="black", linewidths=0.3, marker="D",
                           zorder=10, label=f"Baseline ({len(bdf)})")

        ax.legend(frameon=False, loc="upper left")
        ax.set_title(CATEGORY_LABELS[category], fontsize=10)

    # Shared labels
    for ax in axes[1, :]:
        ax.set_xlabel("Flash Point / \u00b0C")
    for ax in axes[:, 0]:
        ax.set_ylabel("FOM1 / W m\u207b\u00b9 K\u207b\u00b9")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"molprice_fp_posthoc_2x2.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(out_dir, 'molprice_fp_posthoc_2x2.{{png,pdf}}')}")


def _filter_baseline_for_category(baseline_df, category):
    """Filter baseline to molecules satisfying a category's constraints."""
    mask = pd.Series(True, index=baseline_df.index)
    if category.startswith("bio_"):
        if "is_biodegradable" in baseline_df.columns:
            mask &= baseline_df["is_biodegradable"]
    if category.endswith("_stable"):
        if "is_stable" in baseline_df.columns:
            mask &= baseline_df["is_stable"]
    return baseline_df[mask].copy()


# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------
def print_summary(summary_df):
    """Per-level summary statistics."""
    print(f"\n{'='*80}")
    print(f"  SUMMARY STATISTICS BY COST LEVEL")
    print(f"{'='*80}\n")

    header = (f"{'Level':<12} {'Unique':>8} {'FP>=125':>8} {'FP>=150':>8} "
              f"{'FP>=175':>8} {'Max FOM1':>10} {'Mean MP (Pareto)':>16}")
    print(header)
    print("-" * len(header))

    for level in LEVEL_ORDER:
        ldf = summary_df[summary_df["cost_level"] == level].copy()
        if ldf.empty:
            continue

        # Deduplicate across seeds
        ldf = ldf.sort_values("fom1_40C", ascending=False)
        ldf = ldf.drop_duplicates(subset=["SMILES"], keep="first")

        fp_c = ldf["fp"].values - 273.15
        fom1 = ldf["fom1_40C"].values

        n_unique = len(ldf)
        n_fp125 = int(np.sum(fp_c >= 125))
        n_fp150 = int(np.sum(fp_c >= 150))
        n_fp175 = int(np.sum(fp_c >= 175))
        max_fom1 = fom1.max()

        # Pareto front mean MolPrice
        pidx = pareto_front_max(fp_c, fom1)
        if len(pidx) > 0 and "MolPrice" in ldf.columns:
            mean_mp = ldf.iloc[pidx]["MolPrice"].mean()
            mp_str = f"{mean_mp:.2f}"
        else:
            mp_str = "n/a"

        print(f"{level:<12} {n_unique:>8} {n_fp125:>8} {n_fp150:>8} "
              f"{n_fp175:>8} {max_fom1:>10.2f} {mp_str:>16}")

    # Per-category breakdown
    print(f"\n{'='*80}")
    print(f"  PER CATEGORY x LEVEL: Max FOM1")
    print(f"{'='*80}\n")

    header2 = f"{'Category':<22} " + " ".join(f"{l:>12}" for l in LEVEL_ORDER)
    print(header2)
    print("-" * len(header2))

    for cat in CATEGORY_ORDER:
        vals = []
        for level in LEVEL_ORDER:
            sub = summary_df[(summary_df["cost_level"] == level) &
                             (summary_df["category"] == cat)]
            if sub.empty:
                vals.append(f"{'---':>12}")
            else:
                vals.append(f"{sub['fom1_40C'].max():>12.2f}")
        print(f"{cat:<22} " + " ".join(vals))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc FP vs FOM1 analysis of MolPrice sweep")
    parser.add_argument(
        "--summary", type=str,
        default="../outputs/molprice_sweep_nist8100/valid_molecules_summary.csv",
        help="Path to valid_molecules_summary.csv from extract script",
    )
    parser.add_argument(
        "--baseline", type=str,
        default="../BaselineFOM1Eval/output/fom1_dataset_categories.csv",
        help="Baseline FOM1 dataset CSV",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default="../outputs/molprice_sweep_nist8100/analysis",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    # Load summary
    print(f"Loading summary: {args.summary}")
    summary_df = pd.read_csv(args.summary)
    print(f"  {len(summary_df)} rows, "
          f"{summary_df['cost_level'].nunique()} levels, "
          f"{summary_df['category'].nunique()} categories")

    # Load baseline
    baseline_df = None
    if os.path.exists(args.baseline):
        print(f"Loading baseline: {args.baseline}")
        baseline_df = pd.read_csv(args.baseline)
        if "passes_base_constraints" in baseline_df.columns:
            baseline_df = baseline_df[
                baseline_df["passes_base_constraints"] == True].copy()
        print(f"  {len(baseline_df)} baseline survivors")
    else:
        print(f"  Baseline not found: {args.baseline}")

    # Output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Summary stats
    print_summary(summary_df)

    # Figures
    print(f"\nGenerating figures...")
    plot_fp_fom1_by_level(summary_df, baseline_df, args.out_dir)
    plot_fp_fom1_2x2(summary_df, baseline_df, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
