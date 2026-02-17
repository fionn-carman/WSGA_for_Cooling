"""
Analyse hyperparameter sweep results.

Parses output directories from hyperparam_sweep.sh, extracts the top
molecules and generation stats from each run, and identifies the best
hyperparameter configurations.

Ranking approach:
  - For each run, read the final top-25 CSV and take the #1 molecule
    (by FitnessScore) and the mean FOM1_avg of the top 10.
  - Group the 3 repeat seeds per hyperparameter config and compute
    the mean and std of both metrics across repeats.
  - Rank configurations by mean #1 FOM1_avg across repeats (most
    robust indicator -- a config that flukes one good molecule but
    has poor top-10 average is less useful than one that reliably
    produces a strong population).
  - Also report FOM1_40 and FOM1_100 breakdowns for the top configs.
  - Collect the best unique molecules across all runs.

Figures generated:
  1. Pairwise heatmaps of mean #1 FOM1_avg for each parameter pair
  2. Marginal box plots showing isolated effect of each parameter
  3. Convergence curves (best fitness vs generation) for top configs
  4. Parallel coordinates plot coloured by FOM1_avg

Usage:
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep
    python analyse_hyperparam_sweep.py --sweep_dir ../outputs/hyperparam_sweep --top_configs 20
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

warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# Data loading
# ======================================================================

def parse_run_dir(dirname):
    """Extract hyperparameters from directory name."""
    pattern = r"pop(\d+)_mr([\d.]+)_er([\d.]+)_tau([\d.]+)_seed(\d+)"
    m = re.match(pattern, dirname)
    if not m:
        return None
    return {
        "pop": int(m.group(1)),
        "mr": float(m.group(2)),
        "er": float(m.group(3)),
        "tau": float(m.group(4)),
        "seed": int(m.group(5)),
    }


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
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    param_labels = {"pop": "Population Size", "mr": "Mutation Rate",
                    "er": "Elitism Rate", "tau": "Niching Threshold (τ)"}

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

    # Hide unused subplot if odd number of pairs
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Pairwise Hyperparameter Interaction — Mean Best FOM1", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmaps_pairwise.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "heatmaps_pairwise.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: heatmaps_pairwise.png/pdf")


def plot_marginal_boxplots(runs_df, group_cols, out_dir):
    """Box plot of #1 FOM1_avg for each parameter value."""
    param_labels = {"pop": "Population Size", "mr": "Mutation Rate",
                    "er": "Elitism Rate", "tau": "Niching Threshold (τ)"}

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
    fig.savefig(os.path.join(out_dir, "boxplots_marginal.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "boxplots_marginal.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: boxplots_marginal.png/pdf")


def plot_convergence_curves(runs_df, sweep_dir, out_dir, top_n=5):
    """Convergence curves for the top N configs (averaged over seeds)."""
    group_cols = ["pop", "mr", "er", "tau"]
    config_means = runs_df.groupby(group_cols)["best_FOM1_avg"].mean()
    top_configs = config_means.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    for rank, (config, _) in enumerate(top_configs.items()):
        pop, mr, er, tau = config
        mask = ((runs_df["pop"] == pop) & (runs_df["mr"] == mr) &
                (runs_df["er"] == er) & (runs_df["tau"] == tau))
        seed_dirs = runs_df.loc[mask, "dir"].tolist()

        all_gen_fitness = []
        for d in seed_dirs:
            gs = load_generation_stats(os.path.join(sweep_dir, d))
            if gs is None:
                continue
            if "FitnessScore_max" in gs.columns:
                all_gen_fitness.append(gs.set_index("generation")["FitnessScore_max"])

        if not all_gen_fitness:
            continue

        combined = pd.concat(all_gen_fitness, axis=1)
        mean_curve = combined.mean(axis=1)
        std_curve = combined.std(axis=1)

        label = f"pop={pop}, mr={mr}, er={er}, τ={tau}"
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
    fig.savefig(os.path.join(out_dir, "convergence_curves.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "convergence_curves.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: convergence_curves.png/pdf")


def plot_parallel_coordinates(configs_df, group_cols, out_dir):
    """Parallel coordinates plot coloured by mean #1 FOM1_avg."""
    param_labels = {"pop": "Population\nSize", "mr": "Mutation\nRate",
                    "er": "Elitism\nRate", "tau": "Niching\nThreshold (τ)"}

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

    # Normalise each parameter to [0, 1] for plotting
    normed = {}
    for col in group_cols:
        vals = plot_df[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            normed[col] = (vals - vmin) / (vmax - vmin)
        else:
            normed[col] = np.zeros_like(vals)

    x_positions = np.arange(len(group_cols))

    # Sort by metric so best lines are drawn on top
    order = plot_df[metric_col].argsort().values
    for idx in order:
        y = [normed[col][idx] for col in group_cols]
        color = cmap(norm(plot_df[metric_col].iloc[idx]))
        ax.plot(x_positions, y, color=color, alpha=0.5, linewidth=1.2)

    # Axis ticks: show actual parameter values
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
    fig.savefig(os.path.join(out_dir, "parallel_coordinates.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "parallel_coordinates.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: parallel_coordinates.png/pdf")


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

        best = top_df.iloc[0]
        top10 = top_df.head(10)

        run_info = {
            **params,
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
        }

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
    print(f"Found {len(runs_df)} completed runs (skipped {skipped} incomplete)\n")

    # ------------------------------------------------------------------
    # 2. Aggregate across seeds per config
    # ------------------------------------------------------------------
    group_cols = ["pop", "mr", "er", "tau"]
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
    # 3. Report best configs
    # ------------------------------------------------------------------
    configs = configs.sort_values("best_FOM1_avg_mean", ascending=False)
    show = configs.head(top_n_configs)

    n_seeds = int(show["best_FOM1_avg_count"].iloc[0])

    print(f"{'='*90}")
    print(f"  TOP {top_n_configs} HYPERPARAMETER CONFIGS (ranked by mean #1 FOM1_avg across {n_seeds} seeds)")
    print(f"{'='*90}\n")

    print(f"{'Rank':<5} {'Pop':<6} {'MR':<5} {'ER':<5} {'Tau':<6} "
          f"{'#1 FOM1_avg':>16}   {'Top10 FOM1_avg':>18}   "
          f"{'#1 FOM1_40':>10} {'#1 FOM1_100':>11}")
    print("-" * 105)

    for i, (_, row) in enumerate(show.iterrows(), 1):
        std_1 = row["best_FOM1_avg_std"]
        std_10 = row["top10_mean_FOM1_avg_std"]
        s1 = f"{std_1:.4f}" if not np.isnan(std_1) else "  n/a"
        s10 = f"{std_10:.4f}" if not np.isnan(std_10) else "  n/a"
        print(
            f"{i:<5} {row['pop']:<6.0f} {row['mr']:<5.1f} {row['er']:<5.1f} {row['tau']:<6.2f} "
            f"{row['best_FOM1_avg_mean']:>10.4f} +/- {s1}"
            f"   {row['top10_mean_FOM1_avg_mean']:>10.4f} +/- {s10}"
            f"   {row['best_FOM1_40_mean']:>10.4f}"
            f"   {row['best_FOM1_100_mean']:>10.4f}"
        )

    # ------------------------------------------------------------------
    # 4. Per-parameter marginal analysis
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
    # 5. Collect best unique molecules across all runs
    # ------------------------------------------------------------------
    all_top_mols = []
    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        top_df = load_top_molecules(run_path)
        if top_df is None:
            continue
        top_df = top_df.copy()
        top_df["source_config"] = (
            f"pop{run['pop']}_mr{run['mr']}_er{run['er']}_tau{run['tau']}"
        )
        top_df["source_seed"] = run["seed"]
        all_top_mols.append(top_df)

    if all_top_mols:
        combined = pd.concat(all_top_mols, ignore_index=True)
        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in combined.columns else "SMILES"
        sort_col = "FOM1_avg" if "FOM1_avg" in combined.columns else "FitnessScore"
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

    if all_top_mols:
        top_unique.to_csv(
            os.path.join(out_dir, "top_molecules.csv"), index=False
        )

    print(f"\nCSVs saved to: {out_dir}/")
    print(f"  all_runs.csv            - per-run metrics ({len(runs_df)} runs)")
    print(f"  configs_aggregated.csv  - configs averaged over seeds ({len(configs)} configs)")
    print(f"  top_molecules.csv       - best {top_n_molecules} unique molecules")

    # ------------------------------------------------------------------
    # 7. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")

    plot_pairwise_heatmaps(runs_df, group_cols, out_dir)
    plot_marginal_boxplots(runs_df, group_cols, out_dir)
    plot_convergence_curves(runs_df, sweep_dir, out_dir, top_n=5)
    plot_parallel_coordinates(configs, group_cols, out_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse WSGA hyperparameter sweep results"
    )
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/hyperparam_sweep")
    parser.add_argument("--top_configs", type=int, default=10)
    parser.add_argument("--top_molecules", type=int, default=50)
    args = parser.parse_args()

    analyse_sweep(args.sweep_dir, args.top_configs, args.top_molecules)
