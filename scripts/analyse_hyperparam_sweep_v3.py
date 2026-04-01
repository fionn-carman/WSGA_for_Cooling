"""
Analyse hyperparameter sweep v3 results.

Directory naming format:
  pop{}_mr{}_er{}_k{}_seed{}

Parameters swept:
  - Population size:   [500, 1000, 2000, 3000]
  - Mutation rate:     [0.3, 0.5, 0.7, 0.8, 0.9]
  - Elitism rate:      [0.1, 0.25, 0.5, 0.75, 0.9]
  - Tournament k:      [2, 3, 5]
  - Seeds:             [42, 123, 7, 256, 999]

Fixed: Tau=0.15, BER=0.5, Generations=150, Init=GRU, Target=FOM1

Ranking approach:
  - For each run, read top_25_molecules_FOM1.csv and take the #1 molecule
    (by FitnessScore) and the mean FOM1 of the top 10.
  - Group seeds per hyperparameter config, compute mean/std.
  - Rank by mean #1 FitnessScore across seeds.
  - Use generation_stats.csv for convergence curves (FitnessScore_max).

Figures generated:
  1. Pairwise heatmaps of mean best FOM1 for each parameter pair
  2. Marginal violin plots for each parameter
  3. Convergence curves for top configs (from generation_stats.csv)
  4. Parallel coordinates plot
  5. Structural similarity heatmap of top molecules
  6. Murcko scaffold distribution
  7. ANOVA variance decomposition
  8. Seed vs config variance comparison

Usage:
    python analyse_hyperparam_sweep_v3.py --sweep_dir ../outputs/hyperparam_sweep_v3
    python analyse_hyperparam_sweep_v3.py --sweep_dir ../outputs/hyperparam_sweep_v3 --top_configs 20
"""

import os
import re
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from itertools import combinations
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# Configuration
# ======================================================================

GROUP_COLS = ["pop", "mr", "er", "k"]

PARAM_LABELS = {
    "pop": "Population Size",
    "mr": "Mutation Rate",
    "er": "Elitism Rate",
    "k": "Tournament Size (k)",
}

PARAM_LABELS_SHORT = {"pop": "pop", "mr": "mr", "er": "er", "k": "k"}

PARAM_HEADERS = {
    "pop": ("Pop", 6),
    "mr": ("MR", 5),
    "er": ("ER", 5),
    "k": ("k", 4),
}

# Figure style (from CLAUDE.md)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": False,
})


# ======================================================================
# Data loading
# ======================================================================

def parse_run_dir(dirname):
    """Extract hyperparameters from directory name.

    Expected format: pop{}_mr{}_er{}_k{}_seed{}
    """
    m = re.match(r"pop(\d+)_mr([\d.]+)_er([\d.]+)_k(\d+)_seed(\d+)", dirname)
    if m:
        return {
            "pop": int(m.group(1)),
            "mr": float(m.group(2)),
            "er": float(m.group(3)),
            "k": int(m.group(4)),
            "seed": int(m.group(5)),
        }
    return None


def safe_read_csv(path):
    """Read a CSV, returning None if it's empty or corrupt."""
    try:
        if os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path, low_memory=False)
        if df.empty or len(df.columns) < 2:
            return None
        return df
    except Exception:
        return None


def load_top_molecules(run_path):
    """Load top_25_molecules_FOM1.csv from a run directory."""
    path = os.path.join(run_path, "top_25_molecules_FOM1.csv")
    if not os.path.exists(path):
        return None
    return safe_read_csv(path)


def load_generation_stats(run_path):
    """Load generation_stats.csv from a run directory."""
    path = os.path.join(run_path, "generation_stats.csv")
    if not os.path.exists(path):
        return None
    return safe_read_csv(path)


# ======================================================================
# Figures
# ======================================================================

def plot_pairwise_heatmaps(runs_df, out_dir):
    """Heatmap of mean best FOM1 for each pair of parameters."""
    pairs = list(combinations(GROUP_COLS, 2))
    n = len(pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for idx, (p1, p2) in enumerate(pairs):
        ax = axes[idx]
        pivot = runs_df.groupby([p1, p2])["best_fom1"].mean().reset_index()
        pivot = pivot.pivot(index=p2, columns=p1, values="best_fom1")
        pivot = pivot.sort_index(ascending=False)

        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" if isinstance(v, float) else str(v)
                            for v in pivot.index], fontsize=9)
        ax.set_xlabel(PARAM_LABELS.get(p1, p1), fontsize=11)
        ax.set_ylabel(PARAM_LABELS.get(p2, p2), fontsize=11)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    mean_val = pivot.values[~np.isnan(pivot.values)].mean()
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if val < mean_val else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "heatmaps_pairwise.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_marginal_violins(runs_df, out_dir):
    """Violin plot of best FOM1 for each parameter value."""
    fig, axes = plt.subplots(1, len(GROUP_COLS), figsize=(4.5 * len(GROUP_COLS), 5))
    if len(GROUP_COLS) == 1:
        axes = [axes]

    for ax, param in zip(axes, GROUP_COLS):
        values = sorted(runs_df[param].unique())
        data = [runs_df.loc[runs_df[param] == v, "best_fom1"].dropna().values
                for v in values]

        parts = ax.violinplot(data, positions=range(len(values)), showmeans=True,
                              showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("red")
        parts["cmedians"].set_color("black")

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], fontsize=10)
        ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=12)
        ax.set_ylabel("Best FOM1 / W m$^{-1}$ K$^{-1}$" if param == GROUP_COLS[0] else "",
                       fontsize=12)
        ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "violins_marginal.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_convergence_curves(runs_df, sweep_dir, out_dir, top_n=5):
    """Convergence curves for top N configs from generation_stats.csv."""
    config_means = runs_df.groupby(GROUP_COLS)["best_fom1"].mean()
    top_configs = config_means.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    for rank, (config, _) in enumerate(top_configs.items()):
        if not isinstance(config, tuple):
            config = (config,)

        mask = pd.Series(True, index=runs_df.index)
        for col, val in zip(GROUP_COLS, config):
            mask &= (runs_df[col] == val)
        seed_dirs = runs_df.loc[mask, "dir"].tolist()

        all_curves = []
        for d in seed_dirs:
            gen_stats = load_generation_stats(os.path.join(sweep_dir, d))
            if gen_stats is not None and "FitnessScore_max" in gen_stats.columns:
                curve = gen_stats.set_index("generation")["FitnessScore_max"].cummax()
                all_curves.append(curve)

        if not all_curves:
            continue

        combined = pd.concat(all_curves, axis=1)
        mean_curve = combined.mean(axis=1)
        std_curve = combined.std(axis=1)

        label = ", ".join(f"{PARAM_LABELS_SHORT.get(c, c)}={v}"
                          for c, v in zip(GROUP_COLS, config))
        color = cmap(rank)
        ax.plot(mean_curve.index, mean_curve.values, label=label,
                color=color, linewidth=2)
        if len(all_curves) > 1:
            ax.fill_between(mean_curve.index,
                            (mean_curve - std_curve).values,
                            (mean_curve + std_curve).values,
                            alpha=0.15, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best FOM1 / W m$^{-1}$ K$^{-1}$", fontsize=12)
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_parallel_coordinates(configs_df, out_dir):
    """Parallel coordinates plot coloured by mean best FOM1."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_col = "best_fom1_mean"
    if metric_col not in configs_df.columns:
        plt.close(fig)
        return

    plot_df = configs_df.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        plt.close(fig)
        return

    norm = Normalize(vmin=plot_df[metric_col].min(), vmax=plot_df[metric_col].max())
    cmap = plt.cm.viridis

    normed = {}
    for col in GROUP_COLS:
        vals = plot_df[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            normed[col] = (vals - vmin) / (vmax - vmin)
        else:
            normed[col] = np.zeros_like(vals)

    x_positions = np.arange(len(GROUP_COLS))

    order = plot_df[metric_col].argsort().values
    for idx in order:
        y = [normed[col][idx] for col in GROUP_COLS]
        color = cmap(norm(plot_df[metric_col].iloc[idx]))
        ax.plot(x_positions, y, color=color, alpha=0.5, linewidth=1.2)

    for i, col in enumerate(GROUP_COLS):
        unique_vals = sorted(plot_df[col].unique())
        vals_float = np.array(unique_vals, dtype=float)
        vmin, vmax = vals_float.min(), vals_float.max()
        if vmax > vmin:
            tick_positions = (vals_float - vmin) / (vmax - vmin)
        else:
            tick_positions = np.zeros_like(vals_float)

        for tp, tv in zip(tick_positions, unique_vals):
            ax.plot([i - 0.02, i + 0.02], [tp, tp], color="black", linewidth=0.8)
            label_str = f"{tv:.2f}" if isinstance(tv, float) and tv != int(tv) else str(int(tv))
            ax.text(i - 0.06, tp, label_str, ha="right", va="center", fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([PARAM_LABELS.get(c, c) for c in GROUP_COLS], fontsize=11)
    ax.set_yticks([])
    ax.set_xlim(-0.3, len(GROUP_COLS) - 0.7)
    ax.set_ylim(-0.05, 1.05)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Mean Best FOM1 / W m$^{-1}$ K$^{-1}$", fontsize=11)

    fig.tight_layout()
    path = os.path.join(out_dir, "parallel_coordinates.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_structural_similarity(top_unique, smiles_col, out_dir, n_mols=50):
    """Pairwise Tanimoto similarity heatmap of top molecules."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        print("  Skipping structural similarity (RDKit not available)")
        return

    smiles_list = top_unique.head(n_mols)[smiles_col].tolist()
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            valid_smiles.append(smi)

    if len(fps) < 3:
        print("  Skipping structural similarity (too few valid molecules)")
        return

    n = len(fps)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])

    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    print(f"\n  Structural Diversity (top {n} molecules):")
    print(f"    Mean pairwise Tanimoto: {upper_tri.mean():.3f}")
    print(f"    Median:                 {np.median(upper_tri):.3f}")
    print(f"    Min:                    {upper_tri.min():.3f}")
    print(f"    Max:                    {upper_tri.max():.3f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Molecule Rank", fontsize=12)
    ax.set_ylabel("Molecule Rank", fontsize=12)

    tick_step = max(1, n // 10)
    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels(range(1, n + 1, tick_step), fontsize=9)
    ax.set_yticks(range(0, n, tick_step))
    ax.set_yticklabels(range(1, n + 1, tick_step), fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tanimoto Similarity", fontsize=11)

    fig.tight_layout()
    path = os.path.join(out_dir, "structural_similarity_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_scaffold_distribution(top_unique, smiles_col, out_dir, n_mols=50,
                                top_scaffolds=10):
    """Murcko scaffold decomposition and frequency bar chart."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        print("  Skipping scaffold analysis (RDKit not available)")
        return

    smiles_list = top_unique.head(n_mols)[smiles_col].tolist()
    scaffolds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                core = MurckoScaffold.GetScaffoldForMol(mol)
                generic = MurckoScaffold.MakeScaffoldGeneric(core)
                scaffolds.append(Chem.MolToSmiles(generic))
            except Exception:
                scaffolds.append("(failed)")

    if not scaffolds:
        print("  Skipping scaffold analysis (no valid molecules)")
        return

    scaffold_counts = Counter(scaffolds)
    n_unique_scaffolds = len(scaffold_counts)
    top_sc = scaffold_counts.most_common(top_scaffolds)

    print(f"\n  Scaffold Analysis (top {len(smiles_list)} molecules):")
    print(f"    Unique Murcko scaffolds: {n_unique_scaffolds}")
    print(f"    Most common scaffolds:")
    for sc, count in top_sc:
        pct = 100 * count / len(scaffolds)
        print(f"      {sc:<50s}  {count:>3d} ({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [sc for sc, _ in top_sc]
    counts = [c for _, c in top_sc]
    display_labels = [s if len(s) <= 30 else s[:27] + "..." for s in labels]

    bars = ax.barh(range(len(counts)), counts, color="#4C72B0", alpha=0.8)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(display_labels, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Count", fontsize=12)
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "scaffold_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_variance_decomposition(runs_df, out_dir):
    """ANOVA-style variance decomposition (eta-squared per parameter)."""
    metric = "best_fom1"
    grand_mean = runs_df[metric].mean()
    ss_total = ((runs_df[metric] - grand_mean) ** 2).sum()

    if ss_total == 0:
        print("  Skipping variance decomposition (no variance in metric)")
        return

    eta_sq = {}
    for param in GROUP_COLS:
        group_means = runs_df.groupby(param)[metric].mean()
        group_counts = runs_df.groupby(param)[metric].count()
        ss_between = (group_counts * (group_means - grand_mean) ** 2).sum()
        eta_sq[param] = ss_between / ss_total

    # Seed variance
    config_means = runs_df.groupby(GROUP_COLS)[metric].transform("mean")
    ss_within = ((runs_df[metric] - config_means) ** 2).sum()
    eta_sq["seed"] = ss_within / ss_total

    fig, ax = plt.subplots(figsize=(8, 5))
    params = list(eta_sq.keys())
    values = [eta_sq[p] for p in params]
    labels = [PARAM_LABELS.get(p, p) for p in params]

    colors = ["#4C72B0"] * len(GROUP_COLS) + ["#C44E52"]
    bars = ax.bar(range(len(params)), values, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel(r"$\eta^2$ (fraction of total variance)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.25)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(out_dir, "variance_decomposition.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")

    print(f"\n  Variance Decomposition (eta-squared):")
    for p, v in sorted(eta_sq.items(), key=lambda x: x[1], reverse=True):
        label = PARAM_LABELS.get(p, p)
        print(f"    {label:<25s}  {v:.4f}  ({v*100:.1f}%)")


def plot_seed_vs_config_variance(runs_df, out_dir):
    """Compare variance due to seeds vs hyperparameter config."""
    metric = "best_fom1"

    config_group = runs_df.groupby(GROUP_COLS)[metric]
    config_means = config_group.mean()
    config_stds = config_group.std().dropna()

    between_config_std = config_means.std()
    within_config_std = config_stds.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(config_stds.values, bins=30, color="#4C72B0", alpha=0.8,
            edgecolor="black", linewidth=0.5)
    ax.axvline(within_config_std, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {within_config_std:.3f}")
    ax.set_xlabel("Std of Best FOM1 across Seeds", fontsize=12)
    ax.set_ylabel("Number of Configs", fontsize=12)
    ax.legend(fontsize=10, frameon=False)

    ax = axes[1]
    bars = ax.bar(["Between\nConfigs", "Within Config\n(Seed Variance)"],
                  [between_config_std, within_config_std],
                  color=["#4C72B0", "#C44E52"], alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, [between_config_std, within_config_std]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Std of Best FOM1 / W m$^{-1}$ K$^{-1}$", fontsize=12)

    fig.tight_layout()
    path = os.path.join(out_dir, "seed_vs_config_variance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")

    print(f"\n  Variance Comparison:")
    print(f"    Between-config std:  {between_config_std:.4f}")
    print(f"    Within-config std:   {within_config_std:.4f} (mean across configs)")
    ratio = between_config_std / within_config_std if within_config_std > 0 else float("inf")
    print(f"    Ratio:               {ratio:.2f}x")
    if ratio < 1.5:
        print(f"    NOTE: Seed variance is comparable to config variance — "
              f"GA is relatively robust to hyperparameters.")


def plot_mr_er_interaction(runs_df, out_dir):
    """Dedicated heatmap of mutation rate x elitism rate interaction."""
    metric = "best_fom1"
    pivot = runs_df.groupby(["mr", "er"])[metric].mean().reset_index()
    pivot = pivot.pivot(index="mr", columns="er", values=metric)
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index], fontsize=11)
    ax.set_xlabel("Elitism Rate", fontsize=13)
    ax.set_ylabel("Mutation Rate", fontsize=13)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                mean_val = pivot.values[~np.isnan(pivot.values)].mean()
                text_color = "white" if val < mean_val else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Best FOM1 / W m$^{-1}$ K$^{-1}$", fontsize=11)

    fig.tight_layout()
    path = os.path.join(out_dir, "heatmap_mr_er.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


# ======================================================================
# Main analysis
# ======================================================================

def analyse_sweep(sweep_dir, top_n_configs=10, top_n_molecules=50):

    # ------------------------------------------------------------------
    # 1. Scan all run directories
    # ------------------------------------------------------------------
    runs = []
    skipped = 0
    for dirname in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue
        params = parse_run_dir(dirname)
        if params is None:
            continue

        top_mols = load_top_molecules(run_path)
        if top_mols is None:
            skipped += 1
            continue

        gen_stats = load_generation_stats(run_path)

        # Best molecule by FitnessScore
        sort_col = "FitnessScore" if "FitnessScore" in top_mols.columns else "fom1_40C"
        top_mols = top_mols.sort_values(sort_col, ascending=False)
        best = top_mols.iloc[0]
        top10 = top_mols.head(10)

        run_info = dict(params)
        run_info.update({
            "dir": dirname,
            "best_fom1": best.get("FitnessScore", best.get("fom1_40C", np.nan)),
            "best_fom1_40": best.get("FOM1_40", best.get("fom1_40C", np.nan)),
            "best_fitness": best.get("FitnessScore", np.nan),
            "best_smiles": best.get("CanonicalSMILES", best.get("SMILES", "")),
            "best_mp": best.get("mp", np.nan),
            "best_fp": best.get("fp", np.nan),
            "best_sc": best.get("SCScore", np.nan),
            "top10_mean_fom1": top10["FitnessScore"].mean() if "FitnessScore" in top10.columns else np.nan,
            "n_top_mols": len(top_mols),
        })

        if gen_stats is not None and not gen_stats.empty:
            run_info["final_gen"] = int(gen_stats["generation"].max())
            if "FitnessScore_max" in gen_stats.columns:
                run_info["final_best_fitness"] = gen_stats["FitnessScore_max"].iloc[-1]
        else:
            run_info["final_gen"] = np.nan
            run_info["final_best_fitness"] = np.nan

        runs.append(run_info)

    if not runs:
        print("No completed runs found.")
        return

    runs_df = pd.DataFrame(runs)
    n_configs = runs_df.groupby(GROUP_COLS).ngroups
    n_seeds = len(runs_df["seed"].unique())
    print(f"Found {len(runs_df)} completed runs across {n_configs} configs "
          f"({n_seeds} seeds, skipped {skipped} incomplete)")

    seeds_per_config = runs_df.groupby(GROUP_COLS)["seed"].count()
    incomplete = seeds_per_config[seeds_per_config < n_seeds]
    if len(incomplete) > 0:
        print(f"  {len(incomplete)} configs have fewer than {n_seeds} seeds")

    print()

    # ------------------------------------------------------------------
    # 2. Aggregate across seeds per config
    # ------------------------------------------------------------------
    metric_cols = ["best_fom1", "best_fom1_40", "top10_mean_fom1", "best_fitness"]
    agg_dict = {col: ["mean", "std", "count"] for col in metric_cols}
    configs = runs_df.groupby(GROUP_COLS).agg(agg_dict).reset_index()
    configs.columns = [f"{a}_{b}" if b else a for a, b in configs.columns]

    # ------------------------------------------------------------------
    # 3. Report best configs
    # ------------------------------------------------------------------
    configs = configs.sort_values("best_fom1_mean", ascending=False)
    show = configs.head(top_n_configs)

    print(f"{'='*100}")
    print(f"  TOP {top_n_configs} HYPERPARAMETER CONFIGS (ranked by mean best FOM1 across seeds)")
    print(f"{'='*100}\n")

    header = f"{'Rank':<5} "
    header += " ".join(f"{PARAM_HEADERS[c][0]:<{PARAM_HEADERS[c][1]}}" for c in GROUP_COLS)
    header += f" {'Best FOM1':>16}   {'Top10 FOM1':>16}   {'Seeds':>5}"
    print(header)
    print("-" * 100)

    for i, (_, row) in enumerate(show.iterrows(), 1):
        std_1 = row["best_fom1_std"]
        std_10 = row["top10_mean_fom1_std"]
        s1 = f"{std_1:.3f}" if not np.isnan(std_1) else "  n/a"
        s10 = f"{std_10:.3f}" if not np.isnan(std_10) else "  n/a"

        line = f"{i:<5} "
        for c in GROUP_COLS:
            w = PARAM_HEADERS[c][1]
            val = row[c]
            if isinstance(val, float) and val == int(val):
                line += f"{int(val):<{w}} "
            elif isinstance(val, float):
                line += f"{val:<{w}.2f} "
            else:
                line += f"{val:<{w}} "
        line += (f"{row['best_fom1_mean']:>8.3f} +/- {s1}"
                 f"   {row['top10_mean_fom1_mean']:>8.3f} +/- {s10}"
                 f"   {int(row['best_fom1_count']):>5}")
        print(line)

    # ------------------------------------------------------------------
    # 4. Per-parameter marginal analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  MARGINAL EFFECT OF EACH PARAMETER (mean best FOM1)")
    print(f"{'='*100}\n")

    for param in GROUP_COLS:
        marginal = runs_df.groupby(param)["best_fom1"].agg(["mean", "std", "count"])
        marginal = marginal.sort_values("mean", ascending=False)
        print(f"  {PARAM_LABELS[param]}:")
        for val, row in marginal.iterrows():
            print(f"    {val:<8}  mean={row['mean']:.3f}  std={row['std']:.3f}  n={int(row['count'])}")
        print()

    # ------------------------------------------------------------------
    # 5. Collect best unique molecules across all runs
    # ------------------------------------------------------------------
    all_top_mols = []
    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        top_df = load_top_molecules(run_path)
        if top_df is None:
            continue
        top_df = top_df.copy()
        config_parts = [f"{c}{run[c]}" for c in GROUP_COLS]
        top_df["source_config"] = "_".join(config_parts)
        top_df["source_seed"] = run["seed"]
        all_top_mols.append(top_df)

    top_unique = None
    smiles_col = "CanonicalSMILES"
    if all_top_mols:
        combined = pd.concat(all_top_mols, ignore_index=True)
        if smiles_col not in combined.columns:
            smiles_col = "SMILES"
        sort_col = "FitnessScore" if "FitnessScore" in combined.columns else "fom1_40C"

        combined = combined.sort_values(sort_col, ascending=False)
        unique = combined.drop_duplicates(subset=[smiles_col], keep="first")
        top_unique = unique.head(top_n_molecules)

        print(f"{'='*130}")
        print(f"  TOP {top_n_molecules} UNIQUE MOLECULES ACROSS ALL RUNS ({len(unique)} unique total)")
        print(f"{'='*130}\n")

        print(f"{'Rank':<5} {'SMILES':<50} {'FOM1':>8} {'MP':>7} {'FP':>7} {'SC':>5}  {'Config'}")
        print("-" * 130)

        for i, (_, mol) in enumerate(top_unique.iterrows(), 1):
            smi = str(mol.get(smiles_col, ""))
            if len(smi) > 47:
                smi = smi[:44] + "..."
            print(
                f"{i:<5} {smi:<50} "
                f"{mol.get('FitnessScore', mol.get('fom1_40C', np.nan)):>8.3f} "
                f"{mol.get('mp', np.nan):>7.1f} "
                f"{mol.get('fp', np.nan):>7.1f} "
                f"{mol.get('SCScore', np.nan):>5.2f}  "
                f"{mol.get('source_config', '')}"
            )

    # ------------------------------------------------------------------
    # 6. Save CSVs
    # ------------------------------------------------------------------
    out_dir = os.path.join(sweep_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    runs_df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
    configs.to_csv(os.path.join(out_dir, "configs_aggregated.csv"), index=False)

    if top_unique is not None:
        top_unique.to_csv(os.path.join(out_dir, "top_molecules.csv"), index=False)

    print(f"\nCSVs saved to: {out_dir}/")
    print(f"  all_runs.csv            - per-run metrics ({len(runs_df)} runs)")
    print(f"  configs_aggregated.csv  - configs averaged over seeds ({len(configs)} configs)")
    if top_unique is not None:
        print(f"  top_molecules.csv       - best {top_n_molecules} unique molecules")

    # ------------------------------------------------------------------
    # 7. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")

    plot_pairwise_heatmaps(runs_df, out_dir)
    plot_marginal_violins(runs_df, out_dir)
    plot_convergence_curves(runs_df, sweep_dir, out_dir, top_n=5)
    plot_parallel_coordinates(configs, out_dir)
    plot_variance_decomposition(runs_df, out_dir)
    plot_seed_vs_config_variance(runs_df, out_dir)
    plot_mr_er_interaction(runs_df, out_dir)

    if top_unique is not None:
        plot_structural_similarity(top_unique, smiles_col, out_dir, n_mols=top_n_molecules)
        plot_scaffold_distribution(top_unique, smiles_col, out_dir, n_mols=top_n_molecules)

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse WSGA v3 hyperparameter sweep results"
    )
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/hyperparam_sweep_v3")
    parser.add_argument("--top_configs", type=int, default=10)
    parser.add_argument("--top_molecules", type=int, default=50)
    args = parser.parse_args()

    analyse_sweep(args.sweep_dir, args.top_configs, args.top_molecules)
