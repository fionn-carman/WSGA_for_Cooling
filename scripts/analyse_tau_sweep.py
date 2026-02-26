"""
Analyse tau (niching) sweep results — Stage 2.

Parses output directories from tau_sweep.sh, which uses the best GA
parameters from Stage 1 and sweeps tau across a broad range.

Directory format: tau{}_seed{}

All analysis uses `all_evaluated_molecules.csv` as the single source of
truth.  Molecules are filtered to those that:
  - pass all hard constraints  (is_valid == 1)
  - have positive fitness      (FitnessScore > 0)
  - pass the MP threshold      (MP-Measured < -30 °C)

This guarantees that the convergence curves, FOM1-vs-tau plot, top-10
plot, and top-molecules table are all derived from the same data and
the same filtering logic.

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


def load_all_evaluated(run_path, mp_hard=-30):
    """Load all_evaluated_molecules.csv and filter to valid, MP-passing molecules.

    Filters to molecules that:
      - pass all hard constraints  (is_valid == 1)
      - have positive fitness      (FitnessScore > 0)
      - pass the MP threshold      (MP-Measured < mp_hard, default −30 °C)

    Returns the filtered DataFrame (sorted by FOM1_avg descending) or None.
    """
    path = os.path.join(run_path, "all_evaluated_molecules.csv")
    if not os.path.exists(path):
        return None
    df = safe_read_csv(path)
    if df is None:
        return None

    mask = pd.Series(True, index=df.index)
    if "is_valid" in df.columns:
        mask &= (df["is_valid"] == 1)
    if "FitnessScore" in df.columns:
        mask &= (df["FitnessScore"] > 0)
    if "MP-Measured" in df.columns:
        mask &= (df["MP-Measured"] < mp_hard)

    df = df.loc[mask].copy()
    if df.empty:
        return None

    df = df.sort_values("FOM1_avg", ascending=False)
    return df


def get_convergence_curve(valid_df):
    """Compute per-generation best valid FOM1 (cumulative max).

    Parameters
    ----------
    valid_df : pd.DataFrame
        Already filtered to valid, MP-passing molecules.  Must contain
        'generation' and 'FOM1_avg' columns.

    Returns
    -------
    pd.Series  (index = generation, values = running-best FOM1_avg)
    """
    best_per_gen = valid_df.groupby("generation")["FOM1_avg"].max()
    best_per_gen = best_per_gen.sort_index()
    running_best = best_per_gen.cummax()
    return running_best


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
    path = os.path.join(out_dir, "fom1_vs_tau.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


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
    path = os.path.join(out_dir, "top10_vs_tau.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_convergence_by_tau(runs_df, sweep_dir, out_dir):
    """Convergence curves grouped by tau, averaged over seeds.

    Uses all_evaluated_molecules.csv filtered to valid, MP-passing
    molecules.  Per-generation running-best FOM1_avg is computed for
    each seed, then averaged across seeds per tau.
    """
    tau_values = sorted(runs_df["tau"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(tau_values), vmax=max(tau_values))

    for tau_val in tau_values:
        mask = runs_df["tau"] == tau_val
        seed_dirs = runs_df.loc[mask, "dir"].tolist()

        all_curves = []
        for d in seed_dirs:
            valid_df = load_all_evaluated(os.path.join(sweep_dir, d))
            if valid_df is None:
                continue
            curve = get_convergence_curve(valid_df)
            all_curves.append(curve)

        if not all_curves:
            continue

        combined = pd.concat(all_curves, axis=1)
        # Forward-fill so seeds that converge early carry their best value
        combined = combined.ffill()
        mean_curve = combined.mean(axis=1)
        std_curve = combined.std(axis=1)

        color = cmap(norm(tau_val))
        ax.plot(mean_curve.index, mean_curve.values,
                label=f"τ={tau_val}", color=color, linewidth=2)
        if len(all_curves) > 1:
            ax.fill_between(mean_curve.index,
                            (mean_curve - std_curve).values,
                            (mean_curve + std_curve).values,
                            alpha=0.12, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Valid FOM1 (avg)", fontsize=12)
    ax.set_title("Convergence Curves by Niching Threshold (τ)\n"
                 "(valid molecules only, MP ≤ −30 °C)", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_by_tau.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_diversity_vs_tau(runs_df, sweep_dir, out_dir, n_mols=25):
    """Structural diversity (mean pairwise Tanimoto) of top valid molecules per tau level."""
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
            valid_df = load_all_evaluated(run_path)
            if valid_df is None:
                continue

            smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in valid_df.columns else "SMILES"
            # Deduplicate and take top n_mols by FOM1_avg (already sorted)
            top_df = valid_df.drop_duplicates(subset=[smiles_col]).head(n_mols)
            smiles_list = top_df[smiles_col].tolist()

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
    path = os.path.join(out_dir, "diversity_vs_tau.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")

    return div_df


def plot_scaffold_count_vs_tau(runs_df, sweep_dir, out_dir, n_mols=25):
    """Number of unique Murcko scaffolds in top valid molecules per tau level."""
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
            valid_df = load_all_evaluated(run_path)
            if valid_df is None:
                continue

            smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in valid_df.columns else "SMILES"
            # Deduplicate and take top n_mols by FOM1_avg (already sorted)
            top_df = valid_df.drop_duplicates(subset=[smiles_col]).head(n_mols)
            smiles_list = top_df[smiles_col].tolist()

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
    path = os.path.join(out_dir, "scaffold_count_vs_tau.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


# ======================================================================
# Main analysis
# ======================================================================

def analyse_tau_sweep(sweep_dir, top_n_molecules=50):

    # ------------------------------------------------------------------
    # 1. Scan all run directories — use all_evaluated_molecules.csv
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

        valid_df = load_all_evaluated(run_path)
        if valid_df is None:
            skipped += 1
            continue

        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in valid_df.columns else "SMILES"

        # Deduplicate by SMILES within this run (keep best FOM1_avg)
        unique_df = valid_df.drop_duplicates(subset=[smiles_col], keep="first")

        best = unique_df.iloc[0]
        top10 = unique_df.head(10)

        run_info = {
            **params,
            "dir": dirname,
            "best_FOM1_avg": best["FOM1_avg"],
            "best_FOM1_40": best.get("FOM1_40", np.nan),
            "best_FOM1_100": best.get("FOM1_100", np.nan),
            "best_fitness": best.get("FitnessScore", np.nan),
            "best_smiles": best.get(smiles_col, ""),
            "top10_mean_FOM1_avg": top10["FOM1_avg"].mean(),
            "n_valid_molecules": len(unique_df),
        }

        runs.append(run_info)

    if not runs:
        print("No completed runs found.")
        return

    runs_df = pd.DataFrame(runs)
    print(f"Found {len(runs_df)} completed runs (skipped {skipped} incomplete)")
    print(f"  [Using all_evaluated_molecules.csv filtered to is_valid=1, FitnessScore>0, MP<-30°C]\n")

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
    print(f"  TAU SWEEP RESULTS  (valid molecules only, MP ≤ −30 °C)")
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
    # 3. Best unique molecules across all runs
    # ------------------------------------------------------------------
    all_valid_mols = []
    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        valid_df = load_all_evaluated(run_path)
        if valid_df is None:
            continue
        valid_df = valid_df.copy()
        valid_df["source_tau"] = run["tau"]
        valid_df["source_seed"] = run["seed"]
        all_valid_mols.append(valid_df)

    if all_valid_mols:
        combined = pd.concat(all_valid_mols, ignore_index=True)
        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in combined.columns else "SMILES"
        combined = combined.sort_values("FOM1_avg", ascending=False)
        unique = combined.drop_duplicates(subset=[smiles_col], keep="first")
        top_unique = unique.head(top_n_molecules)

        print(f"\n{'='*100}")
        print(f"  TOP {top_n_molecules} UNIQUE VALID MOLECULES ({len(unique)} unique total)")
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

    if all_valid_mols:
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
