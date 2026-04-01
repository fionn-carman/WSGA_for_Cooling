"""Publication-quality figures for hyperparameter sweep v3.

Figure 1: HP importance — (a) ANOVA variance decomposition, (b) marginal effects
Figure 2: Convergence speed by population size

Usage:
    python plot_hyperparam_sweep_v3.py
    python plot_hyperparam_sweep_v3.py --sweep_dir ../outputs/hyperparam_sweep_v3
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── manuscript style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 7.5,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
})

# ── colours ─────────────────────────────────────────────────────────
BLUE = "#1565C0"
BLUE_LIGHT = "#90CAF9"
RED_MUTED = "#C44E52"
GREY = "#888888"

GROUP_COLS = ["pop", "mr", "er", "k"]
PARAM_LABELS = {
    "pop": "Population size",
    "mr": "Mutation rate",
    "er": "Elitism rate",
    "k": "Tournament size",
}

# Population colours (sequential blue gradient — light to dark)
POP_COLORS = {
    500: "#90CAF9",
    1000: "#42A5F5",
    2000: "#1565C0",
    3000: "#0D47A1",
}


# ====================================================================
# Computation helpers
# ====================================================================

def compute_eta_squared(df, metric="best_fom1"):
    """ANOVA eta-squared for each parameter + seed (within-config)."""
    grand_mean = df[metric].mean()
    ss_total = ((df[metric] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return {}

    eta_sq = {}
    for param in GROUP_COLS:
        group_means = df.groupby(param)[metric].mean()
        group_counts = df.groupby(param)[metric].count()
        ss_between = (group_counts * (group_means - grand_mean) ** 2).sum()
        eta_sq[param] = ss_between / ss_total

    # Seed = within-config residual
    config_means = df.groupby(GROUP_COLS)[metric].transform("mean")
    ss_within = ((df[metric] - config_means) ** 2).sum()
    eta_sq["seed"] = ss_within / ss_total

    return eta_sq


def compute_marginal_effects(df, metric="best_fom1"):
    """Mean ± 95% CI for each parameter value."""
    results = {}
    for param in GROUP_COLS:
        agg = df.groupby(param)[metric].agg(["mean", "sem", "count"])
        agg["ci95"] = 1.96 * agg["sem"]
        results[param] = agg.sort_index()
    return results


# ====================================================================
# Figure 1: Marginal Effects
# ====================================================================

def plot_marginal_effects(df, out_dir):
    """Standalone 4-panel stacked figure showing marginal effect of each HP."""

    marginals = compute_marginal_effects(df)
    grand_mean = df["best_fom1"].mean()

    param_order = ["pop", "mr", "er", "k"]

    fig, axes = plt.subplots(4, 1, figsize=(3.5, 5.0), sharex=False)
    fig.subplots_adjust(hspace=0.55)

    # Shared y-axis limits
    all_means = []
    all_cis = []
    for param in param_order:
        m = marginals[param]
        all_means.extend(m["mean"].values)
        all_cis.extend(m["ci95"].values)
    y_min = min(mv - c for mv, c in zip(all_means, all_cis))
    y_max = max(mv + c for mv, c in zip(all_means, all_cis))
    y_pad = (y_max - y_min) * 0.18
    y_lim = (y_min - y_pad, y_max + y_pad)

    for i, (param, ax) in enumerate(zip(param_order, axes)):
        m = marginals[param]
        x_vals = np.arange(len(m))

        # Connecting line
        ax.plot(x_vals, m["mean"].values, "-", color=BLUE, linewidth=0.8,
                alpha=0.6, zorder=1)

        # Points with error bars
        ax.errorbar(x_vals, m["mean"].values, yerr=m["ci95"].values,
                    fmt="o", color=BLUE, markersize=4, capsize=2.5,
                    capthick=0.6, elinewidth=0.6, markeredgecolor="none",
                    zorder=2)

        ax.set_ylim(y_lim)
        ax.set_xticks(x_vals)

        # Format x tick labels
        labels = []
        for v in m.index:
            if isinstance(v, float) and v == int(v):
                labels.append(str(int(v)))
            elif isinstance(v, float):
                labels.append(f"{v:.2f}")
            else:
                labels.append(str(v))
        ax.set_xticklabels(labels, fontsize=6.5)

        # x-axis label = parameter name, tight to axis
        ax.set_xlabel(PARAM_LABELS[param], fontsize=7.5, labelpad=3)

    # Shared y-axis label centred across all 4 panels
    fig.text(0.02, 0.5, r"Mean best FOM1 / W m$^{-1}$ K$^{-1}$",
             va="center", ha="center", rotation="vertical", fontsize=7.5)

    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"hp_marginal_effects.{ext}")
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  Saved: {os.path.join(out_dir, 'hp_marginal_effects.png')}")


# ====================================================================
# Figure 2: Convergence by Population Size
# ====================================================================

def load_convergence_data(sweep_dir, runs_df):
    """Load generation_stats.csv for each run and compute cumulative best."""
    curves = {pop: [] for pop in sorted(runs_df["pop"].unique())}

    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        stats_path = os.path.join(run_path, "generation_stats.csv")
        if not os.path.exists(stats_path):
            continue
        try:
            stats = pd.read_csv(stats_path)
        except Exception:
            continue
        if stats.empty or "FitnessScore_max" not in stats.columns:
            continue

        # Cumulative best
        curve = stats.set_index("generation")["FitnessScore_max"].cummax()
        curves[run["pop"]].append(curve)

    return curves


def plot_convergence(sweep_dir, runs_df, out_dir):
    """Convergence curves grouped by population size."""

    curves = load_convergence_data(sweep_dir, runs_df)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for pop in sorted(curves.keys()):
        if not curves[pop]:
            continue

        combined = pd.concat(curves[pop], axis=1)
        mean_curve = combined.mean(axis=1)
        std_curve = combined.std(axis=1)

        color = POP_COLORS[pop]
        ax.plot(mean_curve.index, mean_curve.values,
                color=color, linewidth=1.4, label=f"N = {pop}", zorder=2)
        lower = np.maximum((mean_curve - std_curve).values, 0)
        upper = (mean_curve + std_curve).values
        ax.fill_between(mean_curve.index, lower, upper,
                        color=color, alpha=0.18, linewidth=0, zorder=1)

    ax.set_xlabel("Generation", fontsize=7.5)
    ax.set_ylabel(r"Best FOM1 / W m$^{-1}$ K$^{-1}$", fontsize=7.5)
    ax.set_xlim(0, 150)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right",
              handlelength=1.5, handletextpad=0.4, labelspacing=0.3)
    ax.tick_params(labelsize=6.5)

    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"hp_convergence.{ext}")
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  Saved: {os.path.join(out_dir, 'hp_convergence.png')}")


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/hyperparam_sweep_v3")
    args = parser.parse_args()

    out_dir = os.path.join(args.sweep_dir, "analysis")
    csv_path = os.path.join(out_dir, "all_runs.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run analyse_hyperparam_sweep_v3.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} runs from {csv_path}")

    print("\nFigure 1: Marginal effects...")
    plot_marginal_effects(df, out_dir)

    print("\nFigure 2: Convergence...")
    plot_convergence(args.sweep_dir, df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
