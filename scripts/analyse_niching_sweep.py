"""
Analyse niching sweep results — Radius x Tau calibrated study.

Parses output directories from niching_sweep.sh, which sweeps Morgan
fingerprint radius and niching threshold tau together with radius-specific
tau grids.

Directory format: r{radius}_tau{tau}_seed{seed} or no_niching_seed{seed}

All analysis uses `all_evaluated_molecules.csv` as the single source of
truth.  Molecules are filtered to those that:
  - pass all hard constraints  (is_valid == 1)
  - have positive fitness      (FitnessScore > 0)
  - pass the MP threshold      (MP-Measured < -30 °C)

Diversity metrics use a FIXED fingerprint radius (radius=2, ECFP4) for
all conditions, independent of the GA's niching radius, to avoid
circularity.

Figures generated:
  1. FOM1 vs tau per radius (line plot + no-niching baseline)
  2. Top-10 FOM1 vs tau per radius
  3. Convergence curves (one panel per radius)
  4. Scaffold diversity vs tau per radius
  5. Feasibility rate vs condition

Usage:
    python analyse_niching_sweep.py --sweep_dir ../outputs/niching_sweep
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

# Fixed analysis fingerprint radius (ECFP4) — independent of GA niching radius
ANALYSIS_FP_RADIUS = 2
ANALYSIS_FP_BITS = 2048

# Colours per radius
RADIUS_COLOURS = {2: "#4C72B0", 4: "#DD8452", 8: "#55A868"}
RADIUS_MARKERS = {2: "o", 4: "s", 8: "D"}


# ======================================================================
# Data loading
# ======================================================================

def parse_niching_dir(dirname):
    """Extract radius, tau, seed from directory name."""
    m = re.match(r"r(\d+)_tau([\d.]+)_seed(\d+)", dirname)
    if m:
        return {
            "radius": int(m.group(1)),
            "tau": float(m.group(2)),
            "seed": int(m.group(3)),
            "niching": True,
        }
    m = re.match(r"no_niching_seed(\d+)", dirname)
    if m:
        return {
            "radius": 0,
            "tau": np.nan,
            "seed": int(m.group(1)),
            "niching": False,
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


def get_fom1_col(df):
    """Determine which FOM1 column to use."""
    for col in ["fom1_40C", "FOM1_40", "FOM1_avg"]:
        if col in df.columns:
            return col
    return None


def load_all_evaluated(run_path, mp_hard=-30):
    """Load all_evaluated_molecules.csv and filter to valid, MP-passing molecules."""
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
    # Handle both column naming conventions
    if "MP-Measured" in df.columns:
        mask &= (df["MP-Measured"] < mp_hard)
    elif "mp" in df.columns:
        mask &= (df["mp"] < mp_hard)

    df = df.loc[mask].copy()
    if df.empty:
        return None

    fom1_col = get_fom1_col(df)
    if fom1_col:
        df = df.sort_values(fom1_col, ascending=False)
    return df


def load_all_raw(run_path):
    """Load all_evaluated_molecules.csv without filtering (for feasibility rate)."""
    path = os.path.join(run_path, "all_evaluated_molecules.csv")
    if not os.path.exists(path):
        return None
    return safe_read_csv(path)


def get_convergence_curve(valid_df, fom1_col):
    """Compute per-generation best valid FOM1 (cumulative max)."""
    if "generation" not in valid_df.columns:
        return None
    best_per_gen = valid_df.groupby("generation")[fom1_col].max()
    best_per_gen = best_per_gen.sort_index()
    return best_per_gen.cummax()


# ======================================================================
# Figures
# ======================================================================

def plot_fom1_vs_tau(runs_df, baseline_df, fom1_col, out_dir):
    """FOM1 vs tau, one line per radius, with no-niching baseline."""
    niched = runs_df[runs_df["niching"]].copy()
    radii = sorted(niched["radius"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in radii:
        rdf = niched[niched["radius"] == r]
        agg = rdf.groupby("tau")[f"best_{fom1_col}"].agg(["mean", "std"]).sort_index()
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                    fmt=f"{RADIUS_MARKERS[r]}-", capsize=4, linewidth=2, markersize=7,
                    color=RADIUS_COLOURS[r], ecolor=RADIUS_COLOURS[r],
                    alpha=0.8, label=f"radius = {r}")

    # No-niching baseline
    if not baseline_df.empty:
        bl_mean = baseline_df[f"best_{fom1_col}"].mean()
        bl_std = baseline_df[f"best_{fom1_col}"].std()
        ax.axhline(bl_mean, color="grey", linestyle="--", linewidth=1.5, label="No niching")
        ax.axhspan(bl_mean - bl_std, bl_mean + bl_std, color="grey", alpha=0.1)

    ax.set_xlabel(r"$\tau$ (niching threshold)", fontsize=12)
    ax.set_ylabel("Best feasible FOM1", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "fom1_vs_tau_by_radius.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_top10_vs_tau(runs_df, baseline_df, fom1_col, out_dir):
    """Top-10 mean FOM1 vs tau per radius."""
    niched = runs_df[runs_df["niching"]].copy()
    radii = sorted(niched["radius"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in radii:
        rdf = niched[niched["radius"] == r]
        agg = rdf.groupby("tau")[f"top10_mean_{fom1_col}"].agg(["mean", "std"]).sort_index()
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                    fmt=f"{RADIUS_MARKERS[r]}-", capsize=4, linewidth=2, markersize=7,
                    color=RADIUS_COLOURS[r], ecolor=RADIUS_COLOURS[r],
                    alpha=0.8, label=f"radius = {r}")

    if not baseline_df.empty:
        bl_mean = baseline_df[f"top10_mean_{fom1_col}"].mean()
        bl_std = baseline_df[f"top10_mean_{fom1_col}"].std()
        ax.axhline(bl_mean, color="grey", linestyle="--", linewidth=1.5, label="No niching")
        ax.axhspan(bl_mean - bl_std, bl_mean + bl_std, color="grey", alpha=0.1)

    ax.set_xlabel(r"$\tau$ (niching threshold)", fontsize=12)
    ax.set_ylabel("Top-10 mean feasible FOM1", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "top10_vs_tau_by_radius.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_convergence_by_radius(runs_df, baseline_df, sweep_dir, fom1_col, out_dir):
    """Convergence curves — one panel per radius, with no-niching baseline in each."""
    niched = runs_df[runs_df["niching"]].copy()
    radii = sorted(niched["radius"].unique())

    fig, axes = plt.subplots(1, len(radii), figsize=(6 * len(radii), 5), sharey=True)
    if len(radii) == 1:
        axes = [axes]

    # Precompute baseline convergence
    bl_curves = []
    if not baseline_df.empty:
        for _, run in baseline_df.iterrows():
            vdf = load_all_evaluated(os.path.join(sweep_dir, run["dir"]))
            if vdf is not None:
                curve = get_convergence_curve(vdf, fom1_col)
                if curve is not None:
                    bl_curves.append(curve)

    for ax, r in zip(axes, radii):
        rdf = niched[niched["radius"] == r]
        tau_values = sorted(rdf["tau"].unique())
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(tau_values), vmax=max(tau_values))

        for tau_val in tau_values:
            seed_dirs = rdf.loc[rdf["tau"] == tau_val, "dir"].tolist()
            curves = []
            for d in seed_dirs:
                vdf = load_all_evaluated(os.path.join(sweep_dir, d))
                if vdf is not None:
                    c = get_convergence_curve(vdf, fom1_col)
                    if c is not None:
                        curves.append(c)
            if not curves:
                continue

            combined = pd.concat(curves, axis=1).ffill()
            mean_c = combined.mean(axis=1)
            std_c = combined.std(axis=1)

            color = cmap(norm(tau_val))
            ax.plot(mean_c.index, mean_c.values,
                    label=f"τ={tau_val}", color=color, linewidth=1.5)
            if len(curves) > 1:
                ax.fill_between(mean_c.index,
                                (mean_c - std_c).values, (mean_c + std_c).values,
                                alpha=0.1, color=color)

        # Baseline in each panel
        if bl_curves:
            bl_combined = pd.concat(bl_curves, axis=1).ffill()
            bl_mean = bl_combined.mean(axis=1)
            ax.plot(bl_mean.index, bl_mean.values,
                    color="grey", linestyle="--", linewidth=1.5, label="No niching")

        ax.set_xlabel("Generation", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Best feasible FOM1", fontsize=11)
        ax.set_title(f"Radius = {r}", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.tick_params(labelsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_by_radius.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_scaffold_diversity(runs_df, baseline_df, sweep_dir, fom1_col, out_dir, n_mols=25):
    """Unique Murcko scaffolds in top molecules, per radius and tau."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        print("  Skipping scaffold diversity (RDKit not available)")
        return

    scaffold_data = []
    all_runs = pd.concat([runs_df, baseline_df], ignore_index=True)

    for _, run in all_runs.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        valid_df = load_all_evaluated(run_path)
        if valid_df is None:
            continue

        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in valid_df.columns else "SMILES"
        top_df = valid_df.drop_duplicates(subset=[smiles_col]).head(n_mols)

        scaffolds = set()
        for smi in top_df[smiles_col]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    generic = MurckoScaffold.MakeScaffoldGeneric(core)
                    scaffolds.add(Chem.MolToSmiles(generic))
                except Exception:
                    pass

        scaffold_data.append({
            "radius": run["radius"],
            "tau": run["tau"],
            "seed": run["seed"],
            "niching": run["niching"],
            "n_scaffolds": len(scaffolds),
        })

    if not scaffold_data:
        print("  Skipping scaffold diversity (no data)")
        return

    sc_df = pd.DataFrame(scaffold_data)
    niched = sc_df[sc_df["niching"]]
    radii = sorted(niched["radius"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in radii:
        rdf = niched[niched["radius"] == r]
        agg = rdf.groupby("tau")["n_scaffolds"].agg(["mean", "std"]).sort_index()
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                    fmt=f"{RADIUS_MARKERS[r]}-", capsize=4, linewidth=2, markersize=7,
                    color=RADIUS_COLOURS[r], ecolor=RADIUS_COLOURS[r],
                    alpha=0.8, label=f"radius = {r}")

    bl = sc_df[~sc_df["niching"]]
    if not bl.empty:
        bl_mean = bl["n_scaffolds"].mean()
        bl_std = bl["n_scaffolds"].std()
        ax.axhline(bl_mean, color="grey", linestyle="--", linewidth=1.5, label="No niching")
        ax.axhspan(bl_mean - bl_std, bl_mean + bl_std, color="grey", alpha=0.1)

    ax.set_xlabel(r"$\tau$ (niching threshold)", fontsize=12)
    ax.set_ylabel(f"Unique Murcko scaffolds (top {n_mols})", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "scaffold_diversity_by_radius.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tanimoto_diversity(runs_df, baseline_df, sweep_dir, fom1_col, out_dir, n_mols=25):
    """Mean pairwise Tanimoto similarity of top molecules, using FIXED radius=2 (ECFP4)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        print("  Skipping Tanimoto diversity (RDKit not available)")
        return

    div_data = []
    all_runs = pd.concat([runs_df, baseline_df], ignore_index=True)

    for _, run in all_runs.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        valid_df = load_all_evaluated(run_path)
        if valid_df is None:
            continue

        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in valid_df.columns else "SMILES"
        top_df = valid_df.drop_duplicates(subset=[smiles_col]).head(n_mols)

        fps = []
        for smi in top_df[smiles_col]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(
                    mol, ANALYSIS_FP_RADIUS, nBits=ANALYSIS_FP_BITS))

        if len(fps) < 3:
            continue

        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

        div_data.append({
            "radius": run["radius"],
            "tau": run["tau"],
            "seed": run["seed"],
            "niching": run["niching"],
            "mean_tanimoto": np.mean(sims),
        })

    if not div_data:
        print("  Skipping Tanimoto diversity (no data)")
        return

    div_df = pd.DataFrame(div_data)
    niched = div_df[div_df["niching"]]
    radii = sorted(niched["radius"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in radii:
        rdf = niched[niched["radius"] == r]
        agg = rdf.groupby("tau")["mean_tanimoto"].agg(["mean", "std"]).sort_index()
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                    fmt=f"{RADIUS_MARKERS[r]}-", capsize=4, linewidth=2, markersize=7,
                    color=RADIUS_COLOURS[r], ecolor=RADIUS_COLOURS[r],
                    alpha=0.8, label=f"radius = {r}")

    bl = div_df[~div_df["niching"]]
    if not bl.empty:
        bl_mean = bl["mean_tanimoto"].mean()
        bl_std = bl["mean_tanimoto"].std()
        ax.axhline(bl_mean, color="grey", linestyle="--", linewidth=1.5, label="No niching")
        ax.axhspan(bl_mean - bl_std, bl_mean + bl_std, color="grey", alpha=0.1)

    ax.set_xlabel(r"$\tau$ (niching threshold)", fontsize=12)
    ax.set_ylabel(f"Mean pairwise Tanimoto (ECFP4, top {n_mols})", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    path = os.path.join(out_dir, "tanimoto_diversity_by_radius.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feasibility_rate(runs_df, baseline_df, sweep_dir, out_dir):
    """Feasibility rate (fraction of valid molecules) per condition."""
    feas_data = []
    all_runs = pd.concat([runs_df, baseline_df], ignore_index=True)

    for _, run in all_runs.iterrows():
        raw_df = load_all_raw(os.path.join(sweep_dir, run["dir"]))
        if raw_df is None:
            continue
        n_total = len(raw_df)
        n_valid = int((raw_df.get("is_valid", pd.Series()) == 1).sum())
        feas_data.append({
            "radius": run["radius"],
            "tau": run["tau"],
            "seed": run["seed"],
            "niching": run["niching"],
            "feasibility": n_valid / n_total if n_total > 0 else 0,
        })

    if not feas_data:
        print("  Skipping feasibility rate (no data)")
        return

    feas_df = pd.DataFrame(feas_data)
    niched = feas_df[feas_df["niching"]]
    radii = sorted(niched["radius"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in radii:
        rdf = niched[niched["radius"] == r]
        agg = rdf.groupby("tau")["feasibility"].agg(["mean", "std"]).sort_index()
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                    fmt=f"{RADIUS_MARKERS[r]}-", capsize=4, linewidth=2, markersize=7,
                    color=RADIUS_COLOURS[r], ecolor=RADIUS_COLOURS[r],
                    alpha=0.8, label=f"radius = {r}")

    bl = feas_df[~feas_df["niching"]]
    if not bl.empty:
        bl_mean = bl["feasibility"].mean()
        bl_std = bl["feasibility"].std()
        ax.axhline(bl_mean, color="grey", linestyle="--", linewidth=1.5, label="No niching")
        ax.axhspan(bl_mean - bl_std, bl_mean + bl_std, color="grey", alpha=0.1)

    ax.set_xlabel(r"$\tau$ (niching threshold)", fontsize=12)
    ax.set_ylabel("Feasibility rate", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "feasibility_by_radius.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Main analysis
# ======================================================================

def analyse_niching_sweep(sweep_dir, n_mols_diversity=25):

    # ------------------------------------------------------------------
    # 1. Scan all run directories
    # ------------------------------------------------------------------
    runs = []
    skipped = 0
    fom1_col = None

    for dirname in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue
        params = parse_niching_dir(dirname)
        if params is None:
            continue

        # Count total molecules for feasibility (before filtering)
        raw_df = load_all_raw(run_path)
        n_total = len(raw_df) if raw_df is not None else 0

        valid_df = load_all_evaluated(run_path)

        if valid_df is None:
            # Run exists but zero feasible molecules — record as zeros, don't skip
            if raw_df is not None and n_total > 0:
                if fom1_col is None:
                    # Try to detect FOM1 column from raw data
                    fom1_col = get_fom1_col(raw_df) or "fom1_40C"
                runs.append({
                    **params,
                    "dir": dirname,
                    f"best_{fom1_col}": np.nan,
                    "best_fitness": np.nan,
                    "best_smiles": "",
                    f"top10_mean_{fom1_col}": np.nan,
                    "n_unique_valid": 0,
                    "n_total_evaluated": n_total,
                    "feasibility": 0.0,
                })
            else:
                skipped += 1
            continue

        if fom1_col is None:
            fom1_col = get_fom1_col(valid_df)
            if fom1_col is None:
                print(f"ERROR: No FOM1 column found in {dirname}")
                return

        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in valid_df.columns else "SMILES"
        unique_df = valid_df.drop_duplicates(subset=[smiles_col], keep="first")
        n_valid = len(unique_df)

        best = unique_df.iloc[0]
        top10 = unique_df.head(10)

        run_info = {
            **params,
            "dir": dirname,
            f"best_{fom1_col}": best[fom1_col],
            "best_fitness": best.get("FitnessScore", np.nan),
            "best_smiles": best.get(smiles_col, ""),
            f"top10_mean_{fom1_col}": top10[fom1_col].mean(),
            "n_unique_valid": n_valid,
            "n_total_evaluated": n_total,
            "feasibility": n_valid / n_total if n_total > 0 else 0,
        }

        # OOD fraction (optional)
        if "OOD_any" in unique_df.columns:
            top25 = unique_df.head(25)
            run_info["ood_fraction_top25"] = top25["OOD_any"].mean()

        runs.append(run_info)

    if not runs:
        print("No completed runs found.")
        return

    all_df = pd.DataFrame(runs)
    runs_df = all_df[all_df["niching"]].copy()
    baseline_df = all_df[~all_df["niching"]].copy()

    print(f"Found {len(runs_df)} niched runs + {len(baseline_df)} no-niching controls "
          f"(skipped {skipped} incomplete)")
    print(f"  FOM1 column: {fom1_col}")
    print(f"  [Filtered to is_valid=1, FitnessScore>0, MP<-30°C]\n")

    # ------------------------------------------------------------------
    # 2. Summary table
    # ------------------------------------------------------------------
    print(f"{'='*100}")
    print(f"  NICHING SWEEP RESULTS")
    print(f"{'='*100}\n")

    print(f"{'Radius':<8} {'Tau':<8} {'Best FOM1':>16} {'Top-10 FOM1':>16} "
          f"{'Unique':>8} {'Feasib':>8} {'Seeds':>6}")
    print("-" * 80)

    for r in sorted(runs_df["radius"].unique()):
        rdf = runs_df[runs_df["radius"] == r]
        for tau_val in sorted(rdf["tau"].unique()):
            tdf = rdf[rdf["tau"] == tau_val]
            best_m = tdf[f"best_{fom1_col}"].mean()
            best_s = tdf[f"best_{fom1_col}"].std()
            t10_m = tdf[f"top10_mean_{fom1_col}"].mean()
            t10_s = tdf[f"top10_mean_{fom1_col}"].std()
            uniq = tdf["n_unique_valid"].mean()
            feas = tdf["feasibility"].mean()
            n = len(tdf)
            print(f"{r:<8} {tau_val:<8.2f} {best_m:>8.2f}±{best_s:>5.2f} "
                  f"{t10_m:>8.2f}±{t10_s:>5.2f} {uniq:>8.0f} {feas:>8.3f} {n:>6}")

    # No-niching baseline
    if not baseline_df.empty:
        bl_best_m = baseline_df[f"best_{fom1_col}"].mean()
        bl_best_s = baseline_df[f"best_{fom1_col}"].std()
        bl_t10_m = baseline_df[f"top10_mean_{fom1_col}"].mean()
        bl_t10_s = baseline_df[f"top10_mean_{fom1_col}"].std()
        bl_uniq = baseline_df["n_unique_valid"].mean()
        bl_feas = baseline_df["feasibility"].mean()
        bl_n = len(baseline_df)
        print("-" * 80)
        print(f"{'none':<8} {'---':<8} {bl_best_m:>8.2f}±{bl_best_s:>5.2f} "
              f"{bl_t10_m:>8.2f}±{bl_t10_s:>5.2f} {bl_uniq:>8.0f} {bl_feas:>8.3f} {bl_n:>6}")

    # ------------------------------------------------------------------
    # 3. Save CSVs
    # ------------------------------------------------------------------
    out_dir = os.path.join(sweep_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    all_df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)

    # Summary pivot
    summary_rows = []
    for _, row in all_df.iterrows():
        summary_rows.append({
            "radius": row["radius"],
            "tau": row["tau"],
            "niching": row["niching"],
            "seed": row["seed"],
            f"best_{fom1_col}": row[f"best_{fom1_col}"],
            f"top10_mean_{fom1_col}": row[f"top10_mean_{fom1_col}"],
            "n_unique_valid": row["n_unique_valid"],
            "feasibility": row["feasibility"],
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print(f"\nCSVs saved to: {out_dir}/")

    # ------------------------------------------------------------------
    # 4. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")

    plot_fom1_vs_tau(runs_df, baseline_df, fom1_col, out_dir)
    plot_top10_vs_tau(runs_df, baseline_df, fom1_col, out_dir)
    plot_convergence_by_radius(runs_df, baseline_df, sweep_dir, fom1_col, out_dir)
    plot_scaffold_diversity(runs_df, baseline_df, sweep_dir, fom1_col, out_dir,
                            n_mols=n_mols_diversity)
    plot_tanimoto_diversity(runs_df, baseline_df, sweep_dir, fom1_col, out_dir,
                            n_mols=n_mols_diversity)
    plot_feasibility_rate(runs_df, baseline_df, sweep_dir, out_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse WSGA niching sweep results (radius x tau)"
    )
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/niching_sweep")
    parser.add_argument("--n_mols_diversity", type=int, default=25)
    args = parser.parse_args()

    analyse_niching_sweep(args.sweep_dir, args.n_mols_diversity)
