"""
Analyse hyperparameter sweep results.

Parses output directories from hyperparam_sweep_stage2.sh, extracts the top
molecules and generation stats from each run, and identifies the best
hyperparameter configurations.

Directory naming format:
  pop{}_er{}_ber{}_k{}_seed{}

Parameters swept:
  - Population size:   [500, 1000, 2000, 3000]
  - Elitism rate:      [0.1, 0.25, 0.5, 0.75]
  - Best elite ratio:  [0.0, 0.25, 0.5, 0.75, 1.0]
  - Tournament k:      [2, 3, 5, 7]
  - Seeds:             [42, 123, 7]

Ranking approach:
  - For each run, read the final top-N CSV and take the #1 molecule
    (by FitnessScore) and the mean FOM1_avg of the top 10.
  - Group the repeat seeds per hyperparameter config and compute
    the mean and std of both metrics across repeats.
  - Rank configurations by mean #1 FOM1_avg across repeats.
  - Also report FOM1_40 and FOM1_100 breakdowns for the top configs.
  - Collect the best unique molecules across all runs.

Figures generated:
  1. Pairwise heatmaps of mean #1 FOM1_avg for each parameter pair
  2. Marginal box plots showing isolated effect of each parameter
  3. Convergence curves (best fitness vs generation) for top configs
  4. Parallel coordinates plot coloured by FOM1_avg
  5. Structural similarity heatmap of top molecules (Tanimoto)
  6. Murcko scaffold distribution bar chart
  7. ANOVA-style variance decomposition (parameter importance)
  8. Seed variance vs config variance comparison

Usage:
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep_stage2
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep_stage2 --top_configs 20
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
import matplotlib.ticker as ticker
from itertools import combinations
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# Configuration
# ======================================================================

GROUP_COLS = ["pop", "er", "ber", "k"]

PARAM_LABELS = {
    "pop": "Population Size",
    "er": "Elitism Rate",
    "ber": "Best Elite Ratio",
    "k": "Tournament Size (k)",
}

PARAM_LABELS_SHORT = {"pop": "pop", "er": "er", "ber": "ber", "k": "k"}

PARAM_HEADERS = {
    "pop": ("Pop", 6),
    "er": ("ER", 5),
    "ber": ("BER", 5),
    "k": ("k", 4),
}


# ======================================================================
# Data loading
# ======================================================================

def parse_run_dir(dirname):
    """Extract hyperparameters from directory name.

    Expected format: pop{}_er{}_ber{}_k{}_seed{}
    """
    m = re.match(r"pop(\d+)_er([\d.]+)_ber([\d.]+)_k(\d+)_seed(\d+)", dirname)
    if m:
        return {
            "pop": int(m.group(1)),
            "er": float(m.group(2)),
            "ber": float(m.group(3)),
            "k": int(m.group(4)),
            "seed": int(m.group(5)),
        }
    return None


def safe_read_csv(path):
    """Read a CSV, returning None if it's empty or corrupt."""
    try:
        if os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        if df.empty or len(df.columns) < 2:
            return None
        return df
    except Exception:
        return None


def load_top_molecules(run_path):
    """Load the final top-N CSV from a run directory."""
    candidates = [f for f in os.listdir(run_path)
                  if f.startswith("top_") and f.endswith(".csv")]
    if not candidates:
        return None
    return safe_read_csv(os.path.join(run_path, candidates[0]))


def load_generation_stats(run_path):
    """Load generation_stats.csv from a run directory."""
    path = os.path.join(run_path, "generation_stats.csv")
    if not os.path.exists(path):
        return None
    return safe_read_csv(path)


def load_best_fom1_per_generation(run_path, mp_hard=-30):
    """Load all_evaluated_molecules.csv and compute best FOM1_avg per generation.

    Filters to molecules that pass all hard constraints:
      - is_valid == 1
      - FitnessScore > 0
      - MP-Measured < mp_hard (soft threshold, default -30°C)

    Returns a Series indexed by generation with the best (max) FOM1_avg,
    or None if the file doesn't exist.
    """
    path = os.path.join(run_path, "all_evaluated_molecules.csv")
    if not os.path.exists(path):
        return None
    df = safe_read_csv(path)
    if df is None or df.empty:
        return None

    # Filter to valid molecules passing hard MP threshold
    mask = pd.Series(True, index=df.index)
    if "is_valid" in df.columns:
        mask &= (df["is_valid"] == 1)
    if "FitnessScore" in df.columns:
        mask &= (df["FitnessScore"] > 0)
    if "MP-Measured" in df.columns:
        mask &= (df["MP-Measured"] < mp_hard)
    df = df[mask]

    if df.empty or "FOM1_avg" not in df.columns or "generation" not in df.columns:
        return None

    # Compute cumulative best FOM1_avg up to each generation
    best_per_gen = df.groupby("generation")["FOM1_avg"].max()
    best_per_gen = best_per_gen.sort_index().cummax()
    return best_per_gen


# ======================================================================
# Figures
# ======================================================================

def plot_pairwise_heatmaps(runs_df, out_dir):
    """Heatmap of mean #1 FOM1_avg for each pair of parameters."""
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
        pivot = runs_df.groupby([p1, p2])["best_FOM1_avg"].mean().reset_index()
        pivot = pivot.pivot(index=p2, columns=p1, values="best_FOM1_avg")
        pivot = pivot.sort_index(ascending=False)

        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" if isinstance(v, float) else str(v)
                            for v in pivot.index], fontsize=9)
        ax.set_xlabel(PARAM_LABELS.get(p1, p1), fontsize=11)
        ax.set_ylabel(PARAM_LABELS.get(p2, p2), fontsize=11)

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if val < pivot.values[~np.isnan(pivot.values)].mean() else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Pairwise Hyperparameter Interaction — Mean Best FOM1", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "heatmaps_pairwise.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_marginal_boxplots(runs_df, out_dir):
    """Box plot of #1 FOM1_avg for each parameter value."""
    fig, axes = plt.subplots(1, len(GROUP_COLS), figsize=(4 * len(GROUP_COLS), 5))
    if len(GROUP_COLS) == 1:
        axes = [axes]

    for ax, param in zip(axes, GROUP_COLS):
        values = sorted(runs_df[param].unique())
        data = [runs_df.loc[runs_df[param] == v, "best_FOM1_avg"].dropna().values
                for v in values]
        bp = ax.boxplot(data, labels=[str(v) for v in values], patch_artist=True,
                        widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.7)
        ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=12)
        ax.set_ylabel("Best FOM1 (avg)" if param == GROUP_COLS[0] else "", fontsize=12)
        ax.tick_params(labelsize=10)

    fig.suptitle("Marginal Effect of Each Hyperparameter on Best FOM1", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "boxplots_marginal.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_marginal_violins(runs_df, out_dir):
    """Violin plot of #1 FOM1_avg for each parameter value.

    Shows full distribution shape rather than just quartiles.
    """
    fig, axes = plt.subplots(1, len(GROUP_COLS), figsize=(4.5 * len(GROUP_COLS), 5))
    if len(GROUP_COLS) == 1:
        axes = [axes]

    for ax, param in zip(axes, GROUP_COLS):
        values = sorted(runs_df[param].unique())
        data = [runs_df.loc[runs_df[param] == v, "best_FOM1_avg"].dropna().values
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
        ax.set_ylabel("Best FOM1 (avg)" if param == GROUP_COLS[0] else "", fontsize=12)
        ax.tick_params(labelsize=10)

    fig.suptitle("Marginal Distribution of Best FOM1 per Hyperparameter", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "violins_marginal.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_convergence_curves(runs_df, sweep_dir, out_dir, top_n=5):
    """Convergence curves for the top N configs (averaged over seeds).

    Computes the cumulative best FOM1_avg per generation from the
    all_evaluated_molecules.csv files, filtering to molecules that pass
    all hard constraints (is_valid, FitnessScore > 0, MP < -30°C).
    """
    config_means = runs_df.groupby(GROUP_COLS)["best_FOM1_avg"].mean()
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

        all_gen_curves = []
        for d in seed_dirs:
            d_path = os.path.join(sweep_dir, d)
            if not os.path.isdir(d_path):
                continue
            curve = load_best_fom1_per_generation(d_path)
            if curve is not None and not curve.empty:
                all_gen_curves.append(curve)

        if not all_gen_curves:
            continue

        combined = pd.concat(all_gen_curves, axis=1)
        mean_curve = combined.mean(axis=1)
        std_curve = combined.std(axis=1)

        label = ", ".join(f"{PARAM_LABELS_SHORT.get(c, c)}={v}"
                          for c, v in zip(GROUP_COLS, config))
        color = cmap(rank)
        ax.plot(mean_curve.index, mean_curve.values, label=label, color=color, linewidth=2)
        if len(all_gen_curves) > 1:
            ax.fill_between(mean_curve.index,
                            (mean_curve - std_curve).values,
                            (mean_curve + std_curve).values,
                            alpha=0.15, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best FOM1 (avg)", fontsize=12)
    ax.set_title(f"Convergence Curves — Top {top_n} Configurations", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_parallel_coordinates(configs_df, out_dir):
    """Parallel coordinates plot coloured by mean #1 FOM1_avg."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_col = "best_FOM1_avg_mean"
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
    ax.set_title("Parallel Coordinates — Hyperparameter Configurations", fontsize=14)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Mean Best FOM1 (avg)", fontsize=11)

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

    # Report summary stats
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    print(f"\n  Structural Diversity (top {n} molecules):")
    print(f"    Mean pairwise Tanimoto: {upper_tri.mean():.3f}")
    print(f"    Median:                 {np.median(upper_tri):.3f}")
    print(f"    Min:                    {upper_tri.min():.3f}")
    print(f"    Max:                    {upper_tri.max():.3f}")
    if upper_tri.mean() > 0.6:
        print(f"    WARNING: High mean similarity suggests convergence to a local optimum")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Molecule Rank", fontsize=12)
    ax.set_ylabel("Molecule Rank", fontsize=12)
    ax.set_title(f"Pairwise Tanimoto Similarity — Top {n} Molecules", fontsize=14)

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

    return sim_matrix, valid_smiles


def plot_scaffold_distribution(top_unique, smiles_col, out_dir, n_mols=50, top_scaffolds=10):
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

    if n_unique_scaffolds <= 3:
        print(f"    WARNING: Very few unique scaffolds — strong evidence of local optimum")

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [sc for sc, _ in top_sc]
    counts = [c for _, c in top_sc]
    display_labels = [s if len(s) <= 30 else s[:27] + "..." for s in labels]

    bars = ax.barh(range(len(counts)), counts, color="#4C72B0", alpha=0.8)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(display_labels, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title(f"Top {top_scaffolds} Murcko Scaffolds (from top {len(smiles_list)} molecules)", fontsize=14)
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
    """ANOVA-style variance decomposition showing how much each
    parameter contributes to variance in best FOM1_avg.

    Computes the between-group sum of squares for each parameter as a
    fraction of the total sum of squares. This is equivalent to eta-squared.
    """
    metric = "best_FOM1_avg"
    grand_mean = runs_df[metric].mean()
    ss_total = ((runs_df[metric] - grand_mean) ** 2).sum()

    if ss_total == 0:
        print("  Skipping variance decomposition (no variance in metric)")
        return

    # Between-group SS for each parameter
    eta_sq = {}
    for param in GROUP_COLS:
        group_means = runs_df.groupby(param)[metric].mean()
        group_counts = runs_df.groupby(param)[metric].count()
        ss_between = (group_counts * (group_means - grand_mean) ** 2).sum()
        eta_sq[param] = ss_between / ss_total

    # Seed variance (within-config variance)
    config_means = runs_df.groupby(GROUP_COLS)[metric].transform("mean")
    ss_within = ((runs_df[metric] - config_means) ** 2).sum()
    eta_sq["seed"] = ss_within / ss_total

    fig, ax = plt.subplots(figsize=(8, 5))
    params = list(eta_sq.keys())
    values = [eta_sq[p] for p in params]
    labels = [PARAM_LABELS.get(p, p) for p in params]

    colors = ["#4C72B0"] * len(GROUP_COLS) + ["#C44E52"]
    bars = ax.bar(range(len(params)), values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Fraction of Total Variance (eta-squared)", fontsize=12)
    ax.set_title("Variance Decomposition — Parameter Importance", fontsize=14)
    ax.set_ylim(0, max(values) * 1.25)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(out_dir, "variance_decomposition.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")

    # Print text summary
    print(f"\n  Variance Decomposition (eta-squared):")
    for p, v in sorted(eta_sq.items(), key=lambda x: x[1], reverse=True):
        label = PARAM_LABELS.get(p, p)
        print(f"    {label:<25s}  {v:.4f}  ({v*100:.1f}%)")


def plot_seed_vs_config_variance(runs_df, out_dir):
    """Compare variance due to seeds vs hyperparameter config.

    For each config (unique combo of group_cols), compute std across seeds.
    Then compare: are configs more different from each other, or are seeds
    within a config more different?
    """
    metric = "best_FOM1_avg"

    config_group = runs_df.groupby(GROUP_COLS)[metric]
    config_means = config_group.mean()
    config_stds = config_group.std().dropna()

    # Between-config std: std of the config means
    between_config_std = config_means.std()
    # Within-config std: mean of the per-config stds (seed variance)
    within_config_std = config_stds.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of per-config seed stds
    ax = axes[0]
    ax.hist(config_stds.values, bins=30, color="#4C72B0", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axvline(within_config_std, color="red", linestyle="--", linewidth=2,
               label=f"Mean within-config std = {within_config_std:.4f}")
    ax.set_xlabel("Std of Best FOM1 across Seeds", fontsize=12)
    ax.set_ylabel("Number of Configs", fontsize=12)
    ax.set_title("Seed Variance per Configuration", fontsize=13)
    ax.legend(fontsize=10)

    # Right: comparison bar chart
    ax = axes[1]
    bars = ax.bar(["Between\nConfigs", "Within Config\n(Seed Variance)"],
                  [between_config_std, within_config_std],
                  color=["#4C72B0", "#C44E52"], alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, [between_config_std, within_config_std]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Std of Best FOM1 (avg)", fontsize=12)
    ax.set_title("Config Variance vs Seed Variance", fontsize=13)

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


def plot_ber_er_interaction(runs_df, out_dir):
    """Dedicated heatmap of the BER x ER interaction.

    This is the most theoretically interesting interaction:
    elitism rate controls how many elites are kept, and best_elite_ratio
    controls how those elites are selected (exploitation vs diversity).
    """
    metric = "best_FOM1_avg"
    pivot = runs_df.groupby(["ber", "er"])[metric].mean().reset_index()
    pivot = pivot.pivot(index="ber", columns="er", values=metric)
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=11)
    ax.set_xlabel("Elitism Rate", fontsize=13)
    ax.set_ylabel("Best Elite Ratio", fontsize=13)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < pivot.values[~np.isnan(pivot.values)].mean() else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Best FOM1 (avg)", fontsize=11)

    ax.set_title("Elitism Rate vs Best Elite Ratio — Mean Best FOM1", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "heatmap_ber_er.png")
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

        top_df = load_top_molecules(run_path)
        if top_df is None:
            skipped += 1
            continue

        gen_stats = load_generation_stats(run_path)

        # Filter to valid molecules that pass the hard MP threshold
        if "is_valid" in top_df.columns:
            valid_df = top_df[top_df["is_valid"] == 1]
        else:
            valid_df = top_df

        if "FitnessScore" in valid_df.columns:
            valid_df = valid_df[valid_df["FitnessScore"] > 0]

        mp_col = "MP-Measured"
        if mp_col in valid_df.columns:
            valid_df = valid_df[valid_df[mp_col] < -30]

        if valid_df.empty:
            skipped += 1
            continue

        best = valid_df.iloc[0]
        top10 = valid_df.head(10)

        run_info = dict(params)
        run_info.update({
            "dir": dirname,
            "best_FOM1_avg": best.get("FOM1_avg", np.nan),
            "best_FOM1_40": best.get("FOM1_40", np.nan),
            "best_FOM1_100": best.get("FOM1_100", np.nan),
            "best_fitness": best.get("FitnessScore", np.nan),
            "best_smiles": best.get("SMILES", best.get("CanonicalSMILES", "")),
            "top10_mean_FOM1_avg": top10["FOM1_avg"].mean() if "FOM1_avg" in top10.columns else np.nan,
            "top10_mean_FOM1_40": top10["FOM1_40"].mean() if "FOM1_40" in top10.columns else np.nan,
            "top10_mean_FOM1_100": top10["FOM1_100"].mean() if "FOM1_100" in top10.columns else np.nan,
            "n_valid_top25": int(top_df["is_valid"].sum()) if "is_valid" in top_df.columns else len(top_df),
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

    # Check for incomplete configs
    seeds_per_config = runs_df.groupby(GROUP_COLS)["seed"].count()
    incomplete = seeds_per_config[seeds_per_config < n_seeds]
    if len(incomplete) > 0:
        print(f"  {len(incomplete)} configs have fewer than {n_seeds} seeds")

    print()

    # ------------------------------------------------------------------
    # 2. Aggregate across seeds per config
    # ------------------------------------------------------------------
    metric_cols = [
        "best_FOM1_avg", "best_FOM1_40", "best_FOM1_100",
        "top10_mean_FOM1_avg", "top10_mean_FOM1_40", "top10_mean_FOM1_100",
        "best_fitness",
    ]

    agg_dict = {col: ["mean", "std", "count"] for col in metric_cols}
    configs = runs_df.groupby(GROUP_COLS).agg(agg_dict).reset_index()

    configs.columns = [
        f"{a}_{b}" if b else a
        for a, b in configs.columns
    ]

    # ------------------------------------------------------------------
    # 3. Report best configs
    # ------------------------------------------------------------------
    configs = configs.sort_values("best_FOM1_avg_mean", ascending=False)
    show = configs.head(top_n_configs)

    print(f"{'='*100}")
    print(f"  TOP {top_n_configs} HYPERPARAMETER CONFIGS (ranked by mean #1 FOM1_avg across seeds)")
    print(f"{'='*100}\n")

    header = f"{'Rank':<5} "
    header += " ".join(f"{PARAM_HEADERS[c][0]:<{PARAM_HEADERS[c][1]}}" for c in GROUP_COLS)
    header += f" {'#1 FOM1_avg':>16}   {'Top10 FOM1_avg':>18}   {'#1 FOM1_40':>10} {'#1 FOM1_100':>11}"
    print(header)
    print("-" * 100)

    for i, (_, row) in enumerate(show.iterrows(), 1):
        std_1 = row["best_FOM1_avg_std"]
        std_10 = row["top10_mean_FOM1_avg_std"]
        s1 = f"{std_1:.4f}" if not np.isnan(std_1) else "  n/a"
        s10 = f"{std_10:.4f}" if not np.isnan(std_10) else "  n/a"

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
        line += (f"{row['best_FOM1_avg_mean']:>10.4f} +/- {s1}"
                 f"   {row['top10_mean_FOM1_avg_mean']:>10.4f} +/- {s10}"
                 f"   {row['best_FOM1_40_mean']:>10.4f}"
                 f"   {row['best_FOM1_100_mean']:>10.4f}")
        print(line)

    # ------------------------------------------------------------------
    # 4. Per-parameter marginal analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  MARGINAL EFFECT OF EACH PARAMETER (mean #1 FOM1_avg)")
    print(f"{'='*100}\n")

    for param in GROUP_COLS:
        marginal = runs_df.groupby(param)["best_FOM1_avg"].agg(["mean", "std", "count"])
        marginal = marginal.sort_values("mean", ascending=False)
        print(f"  {PARAM_LABELS[param]}:")
        for val, row in marginal.iterrows():
            print(f"    {val:<8}  mean={row['mean']:.4f}  std={row['std']:.4f}  n={int(row['count'])}")
        print()

    # ------------------------------------------------------------------
    # 5. Collect best unique molecules across all runs
    # ------------------------------------------------------------------
    all_top_mols = []
    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        if not os.path.isdir(run_path):
            continue
        top_df = load_top_molecules(run_path)
        if top_df is None:
            continue
        top_df = top_df.copy()
        config_parts = [f"{c}{run[c]}" for c in GROUP_COLS]
        top_df["source_config"] = "_".join(config_parts)
        top_df["source_seed"] = run["seed"]
        all_top_mols.append(top_df)

    top_unique = None
    if all_top_mols:
        combined = pd.concat(all_top_mols, ignore_index=True)
        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in combined.columns else "SMILES"
        sort_col = "FOM1_avg" if "FOM1_avg" in combined.columns else "FitnessScore"

        # Filter to valid molecules
        if "is_valid" in combined.columns:
            combined = combined[combined["is_valid"] == 1]
        if "FitnessScore" in combined.columns:
            combined = combined[combined["FitnessScore"] > 0]
        if "MP-Measured" in combined.columns:
            combined = combined[combined["MP-Measured"] < -30]

        combined = combined.sort_values(sort_col, ascending=False)
        unique = combined.drop_duplicates(subset=[smiles_col], keep="first")
        top_unique = unique.head(top_n_molecules)

        print(f"{'='*130}")
        print(f"  TOP {top_n_molecules} UNIQUE MOLECULES ACROSS ALL RUNS ({len(unique)} unique total)")
        print(f"{'='*130}\n")

        print(f"{'Rank':<5} {'SMILES':<45} {'FOM1_avg':>10} {'FOM1_40':>10} "
              f"{'FOM1_100':>10} {'MP':>7} {'FP':>7} {'SC':>5}  {'Config'}")
        print("-" * 130)

        for i, (_, mol) in enumerate(top_unique.iterrows(), 1):
            smi = str(mol.get(smiles_col, ""))
            if len(smi) > 42:
                smi = smi[:39] + "..."
            print(
                f"{i:<5} {smi:<45} "
                f"{mol.get('FOM1_avg', np.nan):>10.4f} "
                f"{mol.get('FOM1_40', np.nan):>10.4f} "
                f"{mol.get('FOM1_100', np.nan):>10.4f} "
                f"{mol.get('MP-Measured', np.nan):>7.1f} "
                f"{mol.get('flashpoint', np.nan):>7.1f} "
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
        top_unique.to_csv(
            os.path.join(out_dir, "top_molecules.csv"), index=False
        )

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
    plot_marginal_boxplots(runs_df, out_dir)
    plot_marginal_violins(runs_df, out_dir)
    plot_convergence_curves(runs_df, sweep_dir, out_dir, top_n=5)
    plot_parallel_coordinates(configs, out_dir)
    plot_variance_decomposition(runs_df, out_dir)
    plot_seed_vs_config_variance(runs_df, out_dir)
    plot_ber_er_interaction(runs_df, out_dir)

    # Structural diversity analysis
    if top_unique is not None:
        plot_structural_similarity(top_unique, smiles_col, out_dir, n_mols=top_n_molecules)
        plot_scaffold_distribution(top_unique, smiles_col, out_dir, n_mols=top_n_molecules)

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse WSGA Stage 2 hyperparameter sweep results"
    )
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/hyperparam_sweep_stage2")
    parser.add_argument("--top_configs", type=int, default=10)
    parser.add_argument("--top_molecules", type=int, default=50)
    args = parser.parse_args()

    analyse_sweep(args.sweep_dir, args.top_configs, args.top_molecules)
