"""
Analyse REINVENT scaffold diversity sweep results.

Parses output directories from reinvent_scaffold_sweep.sh, which sweeps
max_per_scaffold across [5, 10, 25, 50, 100, disabled] x 5 seeds.

Directory format: mps{value}_seed{seed}  (or mps_disabled_seed{seed})

Uses top_n_tracking.csv for convergence and per-step metrics, and
all_evaluated_molecules.csv for total unique molecule counts.

Figures generated:
  1. Best FOM1 vs max_per_scaffold  (mean +/- std across seeds)
  2. Top-10 mean FOM1 vs max_per_scaffold
  3. Convergence curves per scaffold setting
  4. Unique Murcko scaffolds discovered vs setting
  5. Total unique valid molecules vs setting

Usage:
    python analyse_reinvent_scaffold_sweep.py --sweep_dir ../outputs/reinvent_scaffold_sweep
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

warnings.filterwarnings("ignore", category=FutureWarning)

# Label for x-axis ticks (order matters for plotting)
MPS_ORDER = [5, 10, 25, 50, 100, "disabled"]
MPS_LABELS = ["5", "10", "25", "50", "100", "Off"]


# ======================================================================
# Data loading
# ======================================================================

def parse_scaffold_dir(dirname):
    """Extract max_per_scaffold and seed from directory name."""
    m = re.match(r"mps(\d+|disabled)_seed(\d+)", dirname)
    if not m:
        return None
    mps_raw = m.group(1)
    mps = "disabled" if mps_raw == "disabled" else int(mps_raw)
    return {"max_per_scaffold": mps, "seed": int(m.group(2))}


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


def load_valid_from_tracking(run_path, fp_min=373, mp_hard=-30):
    """Load top_n_tracking.csv and filter to valid, constraint-passing molecules.

    Returns the filtered DataFrame (sorted by fom1_40C descending) or None.
    """
    path = os.path.join(run_path, "top_n_tracking.csv")
    if not os.path.exists(path):
        return None
    df = safe_read_csv(path)
    if df is None:
        return None

    req = {"fp", "fom1_40C", "is_valid", "FitnessScore", "mp"}
    if not req.issubset(df.columns):
        return None

    mask = (
        (df["is_valid"] == 1)
        & (df["FitnessScore"] > 0)
        & (df["mp"] < mp_hard)
        & (df["fp"] >= fp_min)
    )
    df = df.loc[mask].copy()
    if df.empty:
        return None

    df = df.sort_values("fom1_40C", ascending=False)
    return df


def get_convergence_curve(valid_df):
    """Compute per-step running-best valid FOM1 (cumulative max).

    Returns pd.Series (index=generation/step, values=running-best fom1_40C).
    """
    best_per_step = valid_df.groupby("generation")["fom1_40C"].max()
    best_per_step = best_per_step.sort_index()
    return best_per_step.cummax()


def mps_sort_key(val):
    """Sort key: numeric values in order, 'disabled' last."""
    if val == "disabled":
        return 999
    return val


# ======================================================================
# Figures
# ======================================================================

def plot_fom1_vs_scaffold(runs_df, out_dir):
    """Best FOM1 vs max_per_scaffold with mean +/- std error bars."""
    agg = runs_df.groupby("max_per_scaffold")["best_fom1"].agg(["mean", "std", "count"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(MPS_ORDER))
    means = [agg.loc[k, "mean"] if k in agg.index else np.nan for k in MPS_ORDER]
    stds = [agg.loc[k, "std"] if k in agg.index else 0 for k in MPS_ORDER]

    ax.errorbar(x_pos, means, yerr=stds,
                fmt="o-", capsize=5, linewidth=2, markersize=8,
                color="#4C72B0", ecolor="#999999")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(MPS_LABELS)
    ax.set_xlabel("max_per_scaffold", fontsize=12)
    ax.set_ylabel("Best FOM1", fontsize=12)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "fom1_vs_max_per_scaffold.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_top10_vs_scaffold(runs_df, out_dir):
    """Top-10 mean FOM1 vs max_per_scaffold."""
    agg = runs_df.groupby("max_per_scaffold")["top10_mean_fom1"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(MPS_ORDER))
    means = [agg.loc[k, "mean"] if k in agg.index else np.nan for k in MPS_ORDER]
    stds = [agg.loc[k, "std"] if k in agg.index else 0 for k in MPS_ORDER]

    ax.errorbar(x_pos, means, yerr=stds,
                fmt="s-", capsize=5, linewidth=2, markersize=8,
                color="#DD8452", ecolor="#999999")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(MPS_LABELS)
    ax.set_xlabel("max_per_scaffold", fontsize=12)
    ax.set_ylabel("Top-10 Mean FOM1", fontsize=12)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "top10_vs_max_per_scaffold.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_convergence_by_scaffold(runs_df, sweep_dir, out_dir):
    """Convergence curves grouped by max_per_scaffold, averaged over seeds."""
    mps_values = sorted(runs_df["max_per_scaffold"].unique(), key=mps_sort_key)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mps_values)))

    for mps_val, color in zip(mps_values, colors):
        mask = runs_df["max_per_scaffold"] == mps_val
        seed_dirs = runs_df.loc[mask, "dir"].tolist()

        all_curves = []
        for d in seed_dirs:
            valid_df = load_valid_from_tracking(os.path.join(sweep_dir, d))
            if valid_df is None:
                continue
            curve = get_convergence_curve(valid_df)
            all_curves.append(curve)

        if not all_curves:
            continue

        combined = pd.concat(all_curves, axis=1)
        combined = combined.ffill()
        mean_curve = combined.mean(axis=1)
        std_curve = combined.std(axis=1)

        label = f"mps={mps_val}" if mps_val != "disabled" else "Off"
        ax.plot(mean_curve.index, mean_curve.values,
                label=label, color=color, linewidth=2)
        if len(all_curves) > 1:
            ax.fill_between(mean_curve.index,
                            (mean_curve - std_curve).values,
                            (mean_curve + std_curve).values,
                            alpha=0.12, color=color)

    ax.set_xlabel("RL Step", fontsize=12)
    ax.set_ylabel("Best Valid FOM1", fontsize=12)
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_by_scaffold.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_scaffold_count_vs_setting(runs_df, sweep_dir, out_dir, n_mols=50):
    """Unique Murcko scaffolds in top valid molecules per setting."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        print("  Skipping scaffold count plot (RDKit not available)")
        return

    scaffold_data = []
    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        valid_df = load_valid_from_tracking(run_path)
        if valid_df is None:
            continue

        top_df = valid_df.drop_duplicates(subset=["SMILES"]).head(n_mols)
        smiles_list = top_df["SMILES"].tolist()

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
            "max_per_scaffold": run["max_per_scaffold"],
            "seed": run["seed"],
            "n_unique_scaffolds": len(scaffolds),
        })

    if not scaffold_data:
        print("  Skipping scaffold count plot (no data)")
        return

    sc_df = pd.DataFrame(scaffold_data)
    agg = sc_df.groupby("max_per_scaffold")["n_unique_scaffolds"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(MPS_ORDER))
    means = [agg.loc[k, "mean"] if k in agg.index else np.nan for k in MPS_ORDER]
    stds = [agg.loc[k, "std"] if k in agg.index else 0 for k in MPS_ORDER]

    ax.errorbar(x_pos, means, yerr=stds,
                fmt="^-", capsize=5, linewidth=2, markersize=8,
                color="#C44E52", ecolor="#999999")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(MPS_LABELS)
    ax.set_xlabel("max_per_scaffold", fontsize=12)
    ax.set_ylabel(f"Unique Murcko Scaffolds (top {n_mols} molecules)", fontsize=12)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "scaffold_count_vs_setting.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_n_unique_vs_setting(runs_df, out_dir):
    """Total unique valid molecules vs max_per_scaffold."""
    agg = runs_df.groupby("max_per_scaffold")["n_valid"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(MPS_ORDER))
    means = [agg.loc[k, "mean"] if k in agg.index else np.nan for k in MPS_ORDER]
    stds = [agg.loc[k, "std"] if k in agg.index else 0 for k in MPS_ORDER]

    ax.errorbar(x_pos, means, yerr=stds,
                fmt="D-", capsize=5, linewidth=2, markersize=8,
                color="#55A868", ecolor="#999999")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(MPS_LABELS)
    ax.set_xlabel("max_per_scaffold", fontsize=12)
    ax.set_ylabel("Unique Valid Molecules", fontsize=12)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "n_unique_vs_setting.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


# ======================================================================
# Main analysis
# ======================================================================

def analyse_scaffold_sweep(sweep_dir, top_n_molecules=50):

    # ------------------------------------------------------------------
    # 1. Scan all run directories
    # ------------------------------------------------------------------
    runs = []
    skipped = 0
    for dirname in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path) or dirname == "analysis":
            continue
        params = parse_scaffold_dir(dirname)
        if params is None:
            continue

        valid_df = load_valid_from_tracking(run_path)
        if valid_df is None:
            skipped += 1
            continue

        unique_df = valid_df.drop_duplicates(subset=["SMILES"], keep="first")
        best = unique_df.iloc[0]
        top10 = unique_df.head(10)

        runs.append({
            **params,
            "dir": dirname,
            "best_fom1": best["fom1_40C"],
            "best_smiles": best["SMILES"],
            "best_fp": best["fp"],
            "top10_mean_fom1": top10["fom1_40C"].mean(),
            "n_valid": len(unique_df),
        })

    if not runs:
        print("No completed runs found.")
        return

    runs_df = pd.DataFrame(runs)
    print(f"Found {len(runs_df)} completed runs (skipped {skipped} incomplete)")

    # ------------------------------------------------------------------
    # 2. Summary table
    # ------------------------------------------------------------------
    agg = runs_df.groupby("max_per_scaffold").agg(
        FOM1_mean=("best_fom1", "mean"),
        FOM1_std=("best_fom1", "std"),
        top10_mean=("top10_mean_fom1", "mean"),
        top10_std=("top10_mean_fom1", "std"),
        n_valid_mean=("n_valid", "mean"),
        n_seeds=("best_fom1", "count"),
    )
    # Sort with disabled last
    agg = agg.loc[sorted(agg.index, key=mps_sort_key)]

    print(f"\n{'='*80}")
    print(f"  REINVENT SCAFFOLD DIVERSITY SWEEP RESULTS")
    print(f"{'='*80}\n")

    print(f"{'MPS':<12} {'Best FOM1':>18}   {'Top10 FOM1':>18}   {'Valid':>8}  {'Seeds':>5}")
    print("-" * 75)

    for mps_val, row in agg.iterrows():
        label = str(mps_val) if mps_val != "disabled" else "Off"
        s1 = f"{row['FOM1_std']:.4f}" if not np.isnan(row['FOM1_std']) else "  n/a"
        s10 = f"{row['top10_std']:.4f}" if not np.isnan(row['top10_std']) else "  n/a"
        print(f"{label:<12} {row['FOM1_mean']:>10.4f} +/- {s1}"
              f"   {row['top10_mean']:>10.4f} +/- {s10}"
              f"   {row['n_valid_mean']:>8.0f}"
              f"  {int(row['n_seeds']):>5}")

    # ------------------------------------------------------------------
    # 3. Best unique molecules across all runs
    # ------------------------------------------------------------------
    all_valid_chunks = []
    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        valid_df = load_valid_from_tracking(run_path)
        if valid_df is None:
            continue
        valid_df = valid_df.copy()
        valid_df["source_mps"] = run["max_per_scaffold"]
        valid_df["source_seed"] = run["seed"]
        all_valid_chunks.append(valid_df)

    if all_valid_chunks:
        combined = pd.concat(all_valid_chunks, ignore_index=True)
        combined = combined.sort_values("fom1_40C", ascending=False)
        unique = combined.drop_duplicates(subset=["SMILES"], keep="first")
        top_unique = unique.head(top_n_molecules)

        print(f"\n{'='*100}")
        print(f"  TOP {top_n_molecules} UNIQUE VALID MOLECULES ({len(unique)} unique total)")
        print(f"{'='*100}\n")

        print(f"{'Rank':<5} {'SMILES':<45} {'FOM1':>10} {'FP/K':>8} {'MPS':>8}")
        print("-" * 85)

        for i, (_, mol) in enumerate(top_unique.iterrows(), 1):
            smi = str(mol.get("SMILES", ""))
            if len(smi) > 42:
                smi = smi[:39] + "..."
            mps_label = str(mol["source_mps"]) if mol["source_mps"] != "disabled" else "Off"
            print(f"{i:<5} {smi:<45} "
                  f"{mol['fom1_40C']:>10.4f} "
                  f"{mol['fp']:>8.1f} "
                  f"{mps_label:>8}")

    # ------------------------------------------------------------------
    # 4. Save CSVs
    # ------------------------------------------------------------------
    out_dir = os.path.join(sweep_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    runs_df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
    agg.to_csv(os.path.join(out_dir, "scaffold_summary.csv"))

    if all_valid_chunks:
        top_unique.to_csv(os.path.join(out_dir, "top_molecules.csv"), index=False)

    print(f"\nCSVs saved to: {out_dir}/")

    # ------------------------------------------------------------------
    # 5. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")

    plot_fom1_vs_scaffold(runs_df, out_dir)
    plot_top10_vs_scaffold(runs_df, out_dir)
    plot_convergence_by_scaffold(runs_df, sweep_dir, out_dir)
    plot_scaffold_count_vs_setting(runs_df, sweep_dir, out_dir)
    plot_n_unique_vs_setting(runs_df, out_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse REINVENT scaffold diversity sweep results"
    )
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/reinvent_scaffold_sweep")
    parser.add_argument("--top_molecules", type=int, default=50)
    args = parser.parse_args()

    analyse_scaffold_sweep(args.sweep_dir, args.top_molecules)
