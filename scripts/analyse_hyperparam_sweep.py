"""
Analyse hyperparameter sweep results.

Parses output directories from hyperparam_sweep.sh, extracts the top
molecules and generation stats from each run, and identifies the best
hyperparameter configurations.

Supports three directory naming formats:
  Stage 1: pop{}_mr{}_er{}_k{}_seed{}    (GA mechanics sweep)
  Stage 2: pop{}_er{}_ber{}_k{}_seed{}    (full sweep with best elite ratio)
  Legacy:  pop{}_mr{}_er{}_tau{}_seed{}   (original sweep)

Ranking approach:
  - For each run, read the final top-25 CSV and take the #1 molecule
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

Usage:
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep --top_configs 20
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep --format stage1
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep_stage2 --format stage2
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
# Data loading
# ======================================================================

def parse_run_dir(dirname):
    """Extract hyperparameters from directory name.

    Supports Stage 1 (with k), Stage 2 (with ber), and legacy (with tau).
    """
    # Stage 1 format: pop{}_mr{}_er{}_k{}_seed{}
    m = re.match(r"pop(\d+)_mr([\d.]+)_er([\d.]+)_k(\d+)_seed(\d+)", dirname)
    if m:
        return {
            "pop": int(m.group(1)),
            "mr": float(m.group(2)),
            "er": float(m.group(3)),
            "k": int(m.group(4)),
            "seed": int(m.group(5)),
            "format": "stage1",
        }

    # Stage 2 format: pop{}_er{}_ber{}_k{}_seed{}
    m = re.match(r"pop(\d+)_er([\d.]+)_ber([\d.]+)_k(\d+)_seed(\d+)", dirname)
    if m:
        return {
            "pop": int(m.group(1)),
            "er": float(m.group(2)),
            "ber": float(m.group(3)),
            "k": int(m.group(4)),
            "seed": int(m.group(5)),
            "format": "stage2",
        }

    # Legacy format: pop{}_mr{}_er{}_tau{}_seed{}
    m = re.match(r"pop(\d+)_mr([\d.]+)_er([\d.]+)_tau([\d.]+)_seed(\d+)", dirname)
    if m:
        return {
            "pop": int(m.group(1)),
            "mr": float(m.group(2)),
            "er": float(m.group(3)),
            "tau": float(m.group(4)),
            "seed": int(m.group(5)),
            "format": "legacy",
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
    """Load the final top-25 CSV from a run directory."""
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


# ======================================================================
# Figures
# ======================================================================

def plot_pairwise_heatmaps(runs_df, group_cols, out_dir):
    """Heatmap of mean #1 FOM1_avg for each pair of parameters."""
    pairs = list(combinations(group_cols, 2))
    n = len(pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    param_labels = {"pop": "Population Size", "mr": "Mutation Rate",
                    "er": "Elitism Rate", "ber": "Best Elite Ratio",
                    "k": "Tournament Size (k)", "tau": "Niching Threshold (τ)"}

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
        ax.set_xlabel(param_labels.get(p1, p1), fontsize=11)
        ax.set_ylabel(param_labels.get(p2, p2), fontsize=11)

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if val < pivot.values[~np.isnan(pivot.values)].mean() else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Pairwise Hyperparameter Interaction — Mean Best FOM1", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "heatmaps_pairwise.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_marginal_boxplots(runs_df, group_cols, out_dir):
    """Box plot of #1 FOM1_avg for each parameter value."""
    param_labels = {"pop": "Population Size", "mr": "Mutation Rate",
                    "er": "Elitism Rate", "ber": "Best Elite Ratio",
                    "k": "Tournament Size (k)", "tau": "Niching Threshold (τ)"}

    fig, axes = plt.subplots(1, len(group_cols), figsize=(4 * len(group_cols), 5))
    if len(group_cols) == 1:
        axes = [axes]

    for ax, param in zip(axes, group_cols):
        values = sorted(runs_df[param].unique())
        data = [runs_df.loc[runs_df[param] == v, "best_FOM1_avg"].dropna().values
                for v in values]
        bp = ax.boxplot(data, labels=[str(v) for v in values], patch_artist=True,
                        widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.7)
        ax.set_xlabel(param_labels.get(param, param), fontsize=12)
        ax.set_ylabel("Best FOM1 (avg)" if param == group_cols[0] else "", fontsize=12)
        ax.tick_params(labelsize=10)

    fig.suptitle("Marginal Effect of Each Hyperparameter on Best FOM1", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "boxplots_marginal.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_convergence_curves(runs_df, group_cols, sweep_dir, out_dir, top_n=5):
    """Convergence curves for the top N configs (averaged over seeds)."""
    config_means = runs_df.groupby(group_cols)["best_FOM1_avg"].mean()
    top_configs = config_means.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    param_labels_short = {"pop": "pop", "mr": "mr", "er": "er",
                          "ber": "ber", "k": "k", "tau": "τ"}

    for rank, (config, _) in enumerate(top_configs.items()):
        if not isinstance(config, tuple):
            config = (config,)

        mask = pd.Series(True, index=runs_df.index)
        for col, val in zip(group_cols, config):
            mask &= (runs_df[col] == val)
        seed_dirs = runs_df.loc[mask, "dir"].tolist()

        all_gen_fitness = []
        for d in seed_dirs:
            d_path = os.path.join(sweep_dir, d)
            if not os.path.isdir(d_path):
                continue
            gs = load_generation_stats(d_path)
            if gs is None:
                continue
            if "FitnessScore_max" in gs.columns:
                all_gen_fitness.append(gs.set_index("generation")["FitnessScore_max"])

        if not all_gen_fitness:
            continue

        combined = pd.concat(all_gen_fitness, axis=1)
        mean_curve = combined.mean(axis=1)
        std_curve = combined.std(axis=1)

        label = ", ".join(f"{param_labels_short.get(c, c)}={v}"
                          for c, v in zip(group_cols, config))
        color = cmap(rank)
        ax.plot(mean_curve.index, mean_curve.values, label=label, color=color, linewidth=2)
        if len(all_gen_fitness) > 1:
            ax.fill_between(mean_curve.index,
                            (mean_curve - std_curve).values,
                            (mean_curve + std_curve).values,
                            alpha=0.15, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness Score", fontsize=12)
    ax.set_title(f"Convergence Curves — Top {top_n} Configurations", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_parallel_coordinates(configs_df, group_cols, out_dir):
    """Parallel coordinates plot coloured by mean #1 FOM1_avg."""
    param_labels = {"pop": "Population\nSize", "mr": "Mutation\nRate",
                    "er": "Elitism\nRate", "ber": "Best Elite\nRatio",
                    "k": "Tournament\nSize (k)", "tau": "Niching\nThreshold (τ)"}

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
    for col in group_cols:
        vals = plot_df[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            normed[col] = (vals - vmin) / (vmax - vmin)
        else:
            normed[col] = np.zeros_like(vals)

    x_positions = np.arange(len(group_cols))

    order = plot_df[metric_col].argsort().values
    for idx in order:
        y = [normed[col][idx] for col in group_cols]
        color = cmap(norm(plot_df[metric_col].iloc[idx]))
        ax.plot(x_positions, y, color=color, alpha=0.5, linewidth=1.2)

    for i, col in enumerate(group_cols):
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
    ax.set_xticklabels([param_labels.get(c, c) for c in group_cols], fontsize=11)
    ax.set_yticks([])
    ax.set_xlim(-0.3, len(group_cols) - 0.7)
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

    # Tick every 5 molecules
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

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [sc for sc, _ in top_sc]
    counts = [c for _, c in top_sc]

    # Truncate long SMILES for display
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


# ======================================================================
# Main analysis
# ======================================================================

def analyse_sweep(sweep_dir, top_n_configs=10, top_n_molecules=50,
                   format_filter=None):

    # ------------------------------------------------------------------
    # 1. Scan all run directories
    # ------------------------------------------------------------------
    runs = []
    skipped = 0
    detected_format = None
    format_counts = {"stage1": 0, "stage2": 0, "legacy": 0}
    for dirname in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue
        params = parse_run_dir(dirname)
        if params is None:
            continue

        format_counts[params["format"]] += 1

        # Skip directories that don't match the requested format
        if format_filter and params["format"] != format_filter:
            continue

        if detected_format is None:
            detected_format = params["format"]
        elif params["format"] != detected_format:
            # Mixed formats without explicit filter — auto-select the
            # dominant format and restart the scan
            if format_filter is None:
                dominant = max(format_counts, key=format_counts.get)
                counts_str = ", ".join(f"{k}={v}" for k, v in format_counts.items() if v > 0)
                print(f"WARNING: Mixed directory formats detected "
                      f"({counts_str}). "
                      f"Auto-selecting '{dominant}' format.\n"
                      f"  Use --format stage1|stage2|legacy to override.\n")
                return analyse_sweep(sweep_dir, top_n_configs,
                                     top_n_molecules,
                                     format_filter=dominant)
            continue

        top_df = load_top_molecules(run_path)
        if top_df is None:
            skipped += 1
            continue

        gen_stats = load_generation_stats(run_path)

        # Filter to molecules with MP < -30 before selecting best
        mp_col = "MP-Measured"
        if mp_col in top_df.columns:
            mp_valid = top_df[top_df[mp_col] < -30]
        else:
            mp_valid = top_df

        if mp_valid.empty:
            skipped += 1
            continue

        best = mp_valid.iloc[0]
        top10 = mp_valid.head(10)

        run_info = {
            k: v for k, v in params.items() if k != "format"
        }
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
    print(f"Found {len(runs_df)} completed runs (skipped {skipped} incomplete)")
    print(f"Directory format: {detected_format}\n")

    # ------------------------------------------------------------------
    # 2. Determine group columns based on detected format
    # ------------------------------------------------------------------
    if detected_format == "stage1":
        group_cols = ["pop", "mr", "er", "k"]
    elif detected_format == "stage2":
        group_cols = ["pop", "er", "ber", "k"]
    else:
        group_cols = ["pop", "mr", "er", "tau"]

    # ------------------------------------------------------------------
    # 3. Aggregate across seeds per config
    # ------------------------------------------------------------------
    metric_cols = [
        "best_FOM1_avg", "best_FOM1_40", "best_FOM1_100",
        "top10_mean_FOM1_avg", "top10_mean_FOM1_40", "top10_mean_FOM1_100",
        "best_fitness",
    ]

    agg_dict = {col: ["mean", "std", "count"] for col in metric_cols}
    configs = runs_df.groupby(group_cols).agg(agg_dict).reset_index()

    configs.columns = [
        f"{a}_{b}" if b else a
        for a, b in configs.columns
    ]

    # ------------------------------------------------------------------
    # 4. Report best configs
    # ------------------------------------------------------------------
    configs = configs.sort_values("best_FOM1_avg_mean", ascending=False)
    show = configs.head(top_n_configs)

    n_seeds = int(show["best_FOM1_avg_count"].iloc[0])

    # Build header dynamically based on group cols
    param_headers = {"pop": ("Pop", 6), "mr": ("MR", 5), "er": ("ER", 5),
                     "ber": ("BER", 5), "k": ("k", 4), "tau": ("Tau", 6)}

    print(f"{'='*90}")
    print(f"  TOP {top_n_configs} HYPERPARAMETER CONFIGS (ranked by mean #1 FOM1_avg across {n_seeds} seeds)")
    print(f"{'='*90}\n")

    header = f"{'Rank':<5} "
    header += " ".join(f"{param_headers[c][0]:<{param_headers[c][1]}}" for c in group_cols)
    header += f" {'#1 FOM1_avg':>16}   {'Top10 FOM1_avg':>18}   {'#1 FOM1_40':>10} {'#1 FOM1_100':>11}"
    print(header)
    print("-" * 105)

    for i, (_, row) in enumerate(show.iterrows(), 1):
        std_1 = row["best_FOM1_avg_std"]
        std_10 = row["top10_mean_FOM1_avg_std"]
        s1 = f"{std_1:.4f}" if not np.isnan(std_1) else "  n/a"
        s10 = f"{std_10:.4f}" if not np.isnan(std_10) else "  n/a"

        line = f"{i:<5} "
        for c in group_cols:
            w = param_headers[c][1]
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
    # 5. Per-parameter marginal analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  MARGINAL EFFECT OF EACH PARAMETER (mean #1 FOM1_avg)")
    print(f"{'='*90}\n")

    for param in group_cols:
        marginal = runs_df.groupby(param)["best_FOM1_avg"].agg(["mean", "std", "count"])
        marginal = marginal.sort_values("mean", ascending=False)
        print(f"  {param}:")
        for val, row in marginal.iterrows():
            print(f"    {val:<8}  mean={row['mean']:.4f}  std={row['std']:.4f}  n={int(row['count'])}")
        print()

    # ------------------------------------------------------------------
    # 6. Collect best unique molecules across all runs
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
        config_parts = [f"{c}{run[c]}" for c in group_cols]
        top_df["source_config"] = "_".join(config_parts)
        top_df["source_seed"] = run["seed"]
        all_top_mols.append(top_df)

    top_unique = None
    if all_top_mols:
        combined = pd.concat(all_top_mols, ignore_index=True)
        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in combined.columns else "SMILES"
        sort_col = "FOM1_avg" if "FOM1_avg" in combined.columns else "FitnessScore"
        # Filter to molecules with MP < -30
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
    # 7. Save CSVs
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
    # 8. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")

    plot_pairwise_heatmaps(runs_df, group_cols, out_dir)
    plot_marginal_boxplots(runs_df, group_cols, out_dir)
    plot_convergence_curves(runs_df, group_cols, sweep_dir, out_dir, top_n=5)
    plot_parallel_coordinates(configs, group_cols, out_dir)

    # Structural diversity analysis
    if top_unique is not None:
        plot_structural_similarity(top_unique, smiles_col, out_dir, n_mols=top_n_molecules)
        plot_scaffold_distribution(top_unique, smiles_col, out_dir, n_mols=top_n_molecules)

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse WSGA hyperparameter sweep results"
    )
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/hyperparam_sweep")
    parser.add_argument("--top_configs", type=int, default=10)
    parser.add_argument("--top_molecules", type=int, default=50)
    parser.add_argument("--format", type=str, choices=["stage1", "stage2", "legacy"],
                        default=None,
                        help="Only analyse dirs matching this format "
                             "(auto-detects if omitted)")
    args = parser.parse_args()

    analyse_sweep(args.sweep_dir, args.top_configs, args.top_molecules,
                  format_filter=args.format)
