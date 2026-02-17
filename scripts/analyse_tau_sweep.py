"""
Analyse tau (niching) sweep results — Stage 2.

Parses output directories from tau_sweep.sh, which uses the best GA
parameters from Stage 1 and sweeps tau across a broad range.

Directory format: tau{}_seed{}

Figures generated:
  1. FOM1 vs tau (mean +/- std across seeds)
  2. Convergence curves per tau value
  3. Structural similarity heatmap of top molecules per tau level
  4. Scaffold diversity vs tau

Usage:
    python analyse_tau_sweep.py --sweep_dir ../outputs/tau_sweep
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
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# Data loading
# ======================================================================

def parse_tau_dir(dirname):
    """Extract tau and seed from directory name."""
    m = re.match(r"tau([\d.]+)_seed(\d+)", dirname)
    if not m:
        return None
    return {
        "tau": float(m.group(1)),
        "seed": int(m.group(2)),
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

def plot_fom1_vs_tau(runs_df, out_dir):
    """FOM1 vs tau with mean +/- std error bars."""
    agg = runs_df.groupby("tau")["best_FOM1_avg"].agg(["mean", "std", "count"])
    agg = agg.sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                fmt="o-", capsize=5, linewidth=2, markersize=8,
                color="#4C72B0", ecolor="#999999")

    ax.set_xlabel("Tau (Niching Threshold)", fontsize=12)
    ax.set_ylabel("Best FOM1 (avg)", fontsize=12)
    ax.set_title("Effect of Niching Threshold on Best FOM1", fontsize=14)
    ax.tick_params(labelsize=10)

    # Annotate with n per point
    for tau_val, row in agg.iterrows():
        ax.annotate(f"n={int(row['count'])}", (tau_val, row["mean"]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fom1_vs_tau.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fom1_vs_tau.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fom1_vs_tau.png/pdf")


def plot_top10_vs_tau(runs_df, out_dir):
    """Top-10 mean FOM1 vs tau (population quality, not just best individual)."""
    agg = runs_df.groupby("tau")["top10_mean_FOM1_avg"].agg(["mean", "std"])
    agg = agg.sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                fmt="s-", capsize=5, linewidth=2, markersize=8,
                color="#DD8452", ecolor="#999999")

    ax.set_xlabel("Tau (Niching Threshold)", fontsize=12)
    ax.set_ylabel("Top-10 Mean FOM1 (avg)", fontsize=12)
    ax.set_title("Effect of Niching on Population Quality (Top-10 Mean)", fontsize=14)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "top10_vs_tau.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "top10_vs_tau.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: top10_vs_tau.png/pdf")


def plot_convergence_by_tau(runs_df, sweep_dir, out_dir):
    """Convergence curves grouped by tau, averaged over seeds."""
    tau_values = sorted(runs_df["tau"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(tau_values), vmax=max(tau_values))

    for tau_val in tau_values:
        mask = runs_df["tau"] == tau_val
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

        color = cmap(norm(tau_val))
        ax.plot(mean_curve.index, mean_curve.values,
                label=f"τ={tau_val}", color=color, linewidth=2)
        if len(all_gen_fitness) > 1:
            ax.fill_between(mean_curve.index,
                            (mean_curve - std_curve).values,
                            (mean_curve + std_curve).values,
                            alpha=0.12, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness Score", fontsize=12)
    ax.set_title("Convergence Curves by Niching Threshold (τ)", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "convergence_by_tau.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "convergence_by_tau.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: convergence_by_tau.png/pdf")


def plot_diversity_vs_tau(runs_df, sweep_dir, out_dir, n_mols=25):
    """Structural diversity (mean pairwise Tanimoto) of top molecules per tau level."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        print("  Skipping diversity vs tau (RDKit not available)")
        return

    tau_values = sorted(runs_df["tau"].unique())
    diversity_data = []

    for tau_val in tau_values:
        mask = runs_df["tau"] == tau_val
        tau_runs = runs_df.loc[mask]

        for _, run in tau_runs.iterrows():
            run_path = os.path.join(sweep_dir, run["dir"])
            top_df = load_top_molecules(run_path)
            if top_df is None:
                continue

            smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in top_df.columns else "SMILES"
            smiles_list = top_df.head(n_mols)[smiles_col].tolist()

            fps = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

            if len(fps) < 3:
                continue

            # Compute mean pairwise Tanimoto
            n = len(fps)
            sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

            diversity_data.append({
                "tau": tau_val,
                "seed": run["seed"],
                "mean_tanimoto": np.mean(sims),
                "n_molecules": n,
            })

    if not diversity_data:
        print("  Skipping diversity vs tau (no data)")
        return

    div_df = pd.DataFrame(diversity_data)
    agg = div_df.groupby("tau")["mean_tanimoto"].agg(["mean", "std"])
    agg = agg.sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                fmt="D-", capsize=5, linewidth=2, markersize=8,
                color="#55A868", ecolor="#999999")

    ax.set_xlabel("Tau (Niching Threshold)", fontsize=12)
    ax.set_ylabel("Mean Pairwise Tanimoto Similarity", fontsize=12)
    ax.set_title("Population Diversity vs Niching Threshold", fontsize=14)
    ax.tick_params(labelsize=10)
    ax.set_ylim(0, 1)

    # Add interpretation guide
    ax.axhline(y=0.6, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(agg.index.max(), 0.61, "Low diversity threshold",
            ha="right", fontsize=9, color="red", alpha=0.7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diversity_vs_tau.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "diversity_vs_tau.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: diversity_vs_tau.png/pdf")

    return div_df


def plot_scaffold_count_vs_tau(runs_df, sweep_dir, out_dir, n_mols=25):
    """Number of unique Murcko scaffolds in top molecules per tau level."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        print("  Skipping scaffold count vs tau (RDKit not available)")
        return

    tau_values = sorted(runs_df["tau"].unique())
    scaffold_data = []

    for tau_val in tau_values:
        mask = runs_df["tau"] == tau_val
        tau_runs = runs_df.loc[mask]

        for _, run in tau_runs.iterrows():
            run_path = os.path.join(sweep_dir, run["dir"])
            top_df = load_top_molecules(run_path)
            if top_df is None:
                continue

            smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in top_df.columns else "SMILES"
            smiles_list = top_df.head(n_mols)[smiles_col].tolist()

            scaffolds = set()
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    try:
                        core = MurckoScaffold.GetScaffoldForMol(mol)
                        generic = MurckoScaffold.MakeScaffoldGeneric(core)
                        scaffolds.add(Chem.MolToSmiles(generic))
                    except Exception:
                        pass

            scaffold_data.append({
                "tau": tau_val,
                "seed": run["seed"],
                "n_unique_scaffolds": len(scaffolds),
                "n_molecules": len(smiles_list),
            })

    if not scaffold_data:
        print("  Skipping scaffold count vs tau (no data)")
        return

    sc_df = pd.DataFrame(scaffold_data)
    agg = sc_df.groupby("tau")["n_unique_scaffolds"].agg(["mean", "std"])
    agg = agg.sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                fmt="^-", capsize=5, linewidth=2, markersize=8,
                color="#C44E52", ecolor="#999999")

    ax.set_xlabel("Tau (Niching Threshold)", fontsize=12)
    ax.set_ylabel("Unique Murcko Scaffolds (top 25 molecules)", fontsize=12)
    ax.set_title("Scaffold Diversity vs Niching Threshold", fontsize=14)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scaffold_count_vs_tau.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "scaffold_count_vs_tau.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: scaffold_count_vs_tau.png/pdf")


# ======================================================================
# Main analysis
# ======================================================================

def analyse_tau_sweep(sweep_dir, top_n_molecules=50):

    # ------------------------------------------------------------------
    # 1. Scan all run directories
    # ------------------------------------------------------------------
    runs = []
    skipped = 0
    for dirname in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue
        params = parse_tau_dir(dirname)
        if params is None:
            continue

        top_df = load_top_molecules(run_path)
        if top_df is None:
            skipped += 1
            continue

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
        }

        runs.append(run_info)

    if not runs:
        print("No completed runs found.")
        return

    runs_df = pd.DataFrame(runs)
    print(f"Found {len(runs_df)} completed runs (skipped {skipped} incomplete)\n")

    # ------------------------------------------------------------------
    # 2. Summary table
    # ------------------------------------------------------------------
    agg = runs_df.groupby("tau").agg(
        FOM1_mean=("best_FOM1_avg", "mean"),
        FOM1_std=("best_FOM1_avg", "std"),
        top10_mean=("top10_mean_FOM1_avg", "mean"),
        top10_std=("top10_mean_FOM1_avg", "std"),
        n_seeds=("best_FOM1_avg", "count"),
    ).sort_index()

    print(f"{'='*80}")
    print(f"  TAU SWEEP RESULTS")
    print(f"{'='*80}\n")

    print(f"{'Tau':<8} {'#1 FOM1_avg':>18}   {'Top10 FOM1_avg':>18}   {'Seeds':>5}")
    print("-" * 60)

    for tau_val, row in agg.iterrows():
        s1 = f"{row['FOM1_std']:.4f}" if not np.isnan(row['FOM1_std']) else "  n/a"
        s10 = f"{row['top10_std']:.4f}" if not np.isnan(row['top10_std']) else "  n/a"
        print(f"{tau_val:<8.2f} {row['FOM1_mean']:>10.4f} +/- {s1}"
              f"   {row['top10_mean']:>10.4f} +/- {s10}"
              f"   {int(row['n_seeds']):>5}")

    # ------------------------------------------------------------------
    # 3. Best unique molecules
    # ------------------------------------------------------------------
    all_top_mols = []
    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        top_df = load_top_molecules(run_path)
        if top_df is None:
            continue
        top_df = top_df.copy()
        top_df["source_tau"] = run["tau"]
        top_df["source_seed"] = run["seed"]
        all_top_mols.append(top_df)

    if all_top_mols:
        combined = pd.concat(all_top_mols, ignore_index=True)
        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in combined.columns else "SMILES"
        sort_col = "FOM1_avg" if "FOM1_avg" in combined.columns else "FitnessScore"
        combined = combined.sort_values(sort_col, ascending=False)
        unique = combined.drop_duplicates(subset=[smiles_col], keep="first")
        top_unique = unique.head(top_n_molecules)

        print(f"\n{'='*100}")
        print(f"  TOP {top_n_molecules} UNIQUE MOLECULES ({len(unique)} unique total)")
        print(f"{'='*100}\n")

        print(f"{'Rank':<5} {'SMILES':<45} {'FOM1_avg':>10} {'FOM1_40':>10} "
              f"{'FOM1_100':>10} {'Tau':>6}")
        print("-" * 95)

        for i, (_, mol) in enumerate(top_unique.iterrows(), 1):
            smi = str(mol.get(smiles_col, ""))
            if len(smi) > 42:
                smi = smi[:39] + "..."
            print(
                f"{i:<5} {smi:<45} "
                f"{mol.get('FOM1_avg', np.nan):>10.4f} "
                f"{mol.get('FOM1_40', np.nan):>10.4f} "
                f"{mol.get('FOM1_100', np.nan):>10.4f} "
                f"{mol.get('source_tau', np.nan):>6.2f}"
            )

    # ------------------------------------------------------------------
    # 4. Save CSVs
    # ------------------------------------------------------------------
    out_dir = os.path.join(sweep_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    runs_df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
    agg.to_csv(os.path.join(out_dir, "tau_summary.csv"))

    if all_top_mols:
        top_unique.to_csv(os.path.join(out_dir, "top_molecules.csv"), index=False)

    print(f"\nCSVs saved to: {out_dir}/")

    # ------------------------------------------------------------------
    # 5. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")

    plot_fom1_vs_tau(runs_df, out_dir)
    plot_top10_vs_tau(runs_df, out_dir)
    plot_convergence_by_tau(runs_df, sweep_dir, out_dir)
    plot_diversity_vs_tau(runs_df, sweep_dir, out_dir)
    plot_scaffold_count_vs_tau(runs_df, sweep_dir, out_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse WSGA tau/niching sweep results (Stage 2)"
    )
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/tau_sweep")
    parser.add_argument("--top_molecules", type=int, default=50)
    args = parser.parse_args()

    analyse_tau_sweep(args.sweep_dir, args.top_molecules)
