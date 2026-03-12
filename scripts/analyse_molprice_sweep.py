"""
Analyse MolPrice threshold sweep results.

Parses output directories from molprice_sweep.sh, which sweeps MolPrice
cost-penalty thresholds across 6 levels × 2 biodeg settings × 5 seeds.

Directory format: {level}_{biodeg}_seed{SEED}

All analysis uses `all_evaluated_molecules.csv` as the single source of
truth.  Molecules are filtered to those that:
  - pass all hard constraints  (is_valid == 1)
  - have positive fitness      (FitnessScore > 0)
  - pass the MP threshold      (MP-Measured < -30 °C)

Figures generated:
  1. FOM1 vs MolPrice threshold level (mean ± std, bio/nonbio lines)
  2. Pareto front: FOM1 vs MolPrice (scatter, coloured by level)
  3. MolPrice distribution by threshold (box plot)
  4. Convergence curves by threshold level (separate panels for bio/nonbio)
  5. Cost vs FOM1 Pareto fronts (1st/2nd/3rd) + baseline FOM1 dataset

Usage:
    python analyse_molprice_sweep.py --sweep_dir ../outputs/molprice_sweep
"""

import os
import sys
import re
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


# Ordered threshold levels for consistent plotting
LEVEL_ORDER = ["nocost", "gentle", "moderate", "firm", "tight", "aggressive"]
LEVEL_DESCRIPTIONS = {
    "nocost": "No cost",
    "gentle": "Gentle (3.5/6.0)",
    "moderate": "Moderate (3.0/5.0)",
    "firm": "Firm (2.5/4.5)",
    "tight": "Tight (2.5/4.0)",
    "aggressive": "Aggressive (2.0/3.5)",
}


# ======================================================================
# Data loading
# ======================================================================

def parse_sweep_dir(dirname):
    """Extract level, biodeg, seed from directory name."""
    m = re.match(r"([a-z]+)_(bio|nonbio)_seed(\d+)", dirname)
    if not m:
        return None
    level = m.group(1)
    if level not in LEVEL_ORDER:
        return None
    return {
        "level": level,
        "biodeg": m.group(2),
        "seed": int(m.group(3)),
    }


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
    """Compute per-generation best valid FOM1 (cumulative max)."""
    best_per_gen = valid_df.groupby("generation")["FOM1_avg"].max()
    best_per_gen = best_per_gen.sort_index()
    running_best = best_per_gen.cummax()
    return running_best


def _pareto_fronts(x, y, n_fronts=3):
    """Compute successive Pareto fronts for maximising both x and y.

    Returns a list of index arrays (into the original x, y), one per front.
    """
    remaining = np.ones(len(x), dtype=bool)
    fronts = []
    for _ in range(n_fronts):
        idxs = np.where(remaining)[0]
        if len(idxs) == 0:
            break

        xi, yi = x[idxs], y[idxs]
        is_pareto = np.ones(len(idxs), dtype=bool)
        for i in range(len(idxs)):
            if not is_pareto[i]:
                continue
            dom = (xi >= xi[i]) & (yi >= yi[i]) & ((xi > xi[i]) | (yi > yi[i]))
            dom[i] = False
            if dom.any():
                is_pareto[i] = False

        front = idxs[is_pareto]
        fronts.append(front)
        remaining[front] = False

    return fronts


# ======================================================================
# Figures
# ======================================================================

def plot_fom1_vs_threshold(runs_df, out_dir):
    """FOM1 vs threshold level with mean ± std, separate lines for bio/nonbio."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"bio": "#4C72B0", "nonbio": "#DD8452"}
    markers = {"bio": "o", "nonbio": "s"}

    for biodeg in ["bio", "nonbio"]:
        sub = runs_df[runs_df["biodeg"] == biodeg]
        if sub.empty:
            continue

        agg = sub.groupby("level")["best_FOM1_40"].agg(["mean", "std", "count"])
        # Reindex to level order
        agg = agg.reindex([l for l in LEVEL_ORDER if l in agg.index])

        x = np.arange(len(agg))
        label = "Biodegradable" if biodeg == "bio" else "Non-biodegradable"
        offset = -0.1 if biodeg == "bio" else 0.1

        ax.errorbar(x + offset, agg["mean"], yerr=agg["std"],
                    fmt=f"{markers[biodeg]}-", capsize=5, linewidth=2,
                    markersize=8, color=colors[biodeg], ecolor="#999999",
                    label=label)

        for i, (level, row) in enumerate(agg.iterrows()):
            ax.annotate(f"n={int(row['count'])}", (x[i] + offset, row["mean"]),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=7, color="gray")

    ax.set_xticks(np.arange(len([l for l in LEVEL_ORDER
                                 if l in runs_df["level"].unique()])))
    present_levels = [l for l in LEVEL_ORDER if l in runs_df["level"].unique()]
    ax.set_xticklabels([LEVEL_DESCRIPTIONS.get(l, l) for l in present_levels],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Best FOM1_40", fontsize=12)
    ax.set_title("Best FOM1 vs MolPrice Threshold Level", fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "fom1_vs_threshold.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_pareto_front(runs_df, sweep_dir, out_dir):
    """Pareto front: FOM1 vs MolPrice scatter of top-10 molecules per run."""
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap = plt.cm.viridis
    present_levels = [l for l in LEVEL_ORDER if l in runs_df["level"].unique()]
    level_to_idx = {l: i for i, l in enumerate(present_levels)}
    norm = plt.Normalize(vmin=0, vmax=max(len(present_levels) - 1, 1))

    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        valid_df = load_all_evaluated(run_path)
        if valid_df is None:
            continue

        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in valid_df.columns else "SMILES"
        top10 = valid_df.drop_duplicates(subset=[smiles_col]).head(10)

        if "MolPrice" not in top10.columns:
            continue

        color = cmap(norm(level_to_idx[run["level"]]))
        ax.scatter(top10["MolPrice"], top10["FOM1_avg"],
                   c=[color], alpha=0.5, s=30, edgecolors="none")

    # Add legend via dummy scatter
    for level in present_levels:
        color = cmap(norm(level_to_idx[level]))
        ax.scatter([], [], c=[color], s=60,
                   label=LEVEL_DESCRIPTIONS.get(level, level))

    ax.set_xlabel("Predicted MolPrice (log $/mol)", fontsize=12)
    ax.set_ylabel("FOM1_avg", fontsize=12)
    ax.set_title("FOM1 vs MolPrice Trade-off (Top-10 per Run)", fontsize=14)
    ax.legend(fontsize=9, loc="best", title="Threshold Level")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "pareto_fom1_vs_molprice.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_molprice_distribution(runs_df, sweep_dir, out_dir):
    """Box plot of MolPrice distribution of top-10 molecules by threshold level."""
    data_for_plot = []

    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        valid_df = load_all_evaluated(run_path)
        if valid_df is None:
            continue

        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in valid_df.columns else "SMILES"
        top10 = valid_df.drop_duplicates(subset=[smiles_col]).head(10)

        if "MolPrice" not in top10.columns:
            continue

        for _, mol in top10.iterrows():
            data_for_plot.append({
                "level": run["level"],
                "biodeg": run["biodeg"],
                "MolPrice": mol["MolPrice"],
            })

    if not data_for_plot:
        print("  Skipping MolPrice distribution (no MolPrice data)")
        return

    plot_df = pd.DataFrame(data_for_plot)
    present_levels = [l for l in LEVEL_ORDER if l in plot_df["level"].unique()]

    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [plot_df[plot_df["level"] == l]["MolPrice"].dropna().values
                for l in present_levels]
    bp = ax.boxplot(box_data, labels=[LEVEL_DESCRIPTIONS.get(l, l)
                                       for l in present_levels],
                    patch_artist=True, showfliers=True)

    cmap = plt.cm.viridis
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / max(len(present_levels) - 1, 1)))
        patch.set_alpha(0.7)

    ax.set_xticklabels([LEVEL_DESCRIPTIONS.get(l, l) for l in present_levels],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Predicted MolPrice (log $/mol)", fontsize=12)
    ax.set_title("MolPrice Distribution of Top-10 Molecules by Threshold", fontsize=14)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "molprice_distribution_by_threshold.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_convergence_by_threshold(runs_df, sweep_dir, out_dir):
    """Convergence curves by threshold level, separate panels for bio/nonbio."""
    biodeg_values = sorted(runs_df["biodeg"].unique())
    present_levels = [l for l in LEVEL_ORDER if l in runs_df["level"].unique()]

    fig, axes = plt.subplots(1, len(biodeg_values),
                             figsize=(8 * len(biodeg_values), 6),
                             sharey=True, squeeze=False)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max(len(present_levels) - 1, 1))

    for col, biodeg in enumerate(biodeg_values):
        ax = axes[0, col]
        sub = runs_df[runs_df["biodeg"] == biodeg]

        for level in present_levels:
            level_runs = sub[sub["level"] == level]
            if level_runs.empty:
                continue

            all_curves = []
            for _, run in level_runs.iterrows():
                valid_df = load_all_evaluated(os.path.join(sweep_dir, run["dir"]))
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

            color = cmap(norm(present_levels.index(level)))
            ax.plot(mean_curve.index, mean_curve.values,
                    label=LEVEL_DESCRIPTIONS.get(level, level),
                    color=color, linewidth=2)
            if len(all_curves) > 1:
                ax.fill_between(mean_curve.index,
                                (mean_curve - std_curve).values,
                                (mean_curve + std_curve).values,
                                alpha=0.12, color=color)

        title = "Biodegradable" if biodeg == "bio" else "Non-biodegradable"
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Generation", fontsize=12)
        if col == 0:
            ax.set_ylabel("Best Valid FOM1 (avg)", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        ax.tick_params(labelsize=10)

    fig.suptitle("Convergence by MolPrice Threshold\n"
                 "(valid molecules only, MP < −30 °C)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_by_threshold.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_pareto_cost_vs_fom1(runs_df, sweep_dir, out_dir,
                              baseline_csv=None, molprice_model_path=None):
    """Cost vs FOM1 Pareto fronts: 1st/2nd/3rd WSGA + baseline comparison.

    X-axis: −MolPrice (higher = cheaper).
    Y-axis: FOM1_avg.
    All molecules pass hard constraints + MP < −30 °C.
    """
    # ------------------------------------------------------------------
    # 1. Collect all valid WSGA molecules (deduplicated)
    # ------------------------------------------------------------------
    all_mols = []
    for _, run in runs_df.iterrows():
        run_path = os.path.join(sweep_dir, run["dir"])
        valid_df = load_all_evaluated(run_path)
        if valid_df is None or "MolPrice" not in valid_df.columns:
            continue
        all_mols.append(valid_df)

    if not all_mols:
        print("  Skipping Pareto cost/FOM1 plot (no MolPrice data)")
        return

    combined = pd.concat(all_mols, ignore_index=True)
    smiles_col = ("CanonicalSMILES" if "CanonicalSMILES" in combined.columns
                  else "SMILES")
    combined = combined.sort_values("FOM1_avg", ascending=False)
    wsga = combined.drop_duplicates(subset=[smiles_col], keep="first").copy()
    wsga = wsga.dropna(subset=["MolPrice", "FOM1_avg"])

    afford_wsga = -wsga["MolPrice"].values
    fom1_wsga = wsga["FOM1_avg"].values
    wsga_fronts = _pareto_fronts(afford_wsga, fom1_wsga, n_fronts=3)

    # ------------------------------------------------------------------
    # 2. Baseline FOM1 dataset (optional)
    # ------------------------------------------------------------------
    baseline_ok = False
    if baseline_csv and os.path.exists(baseline_csv):
        base_df = pd.read_csv(baseline_csv)

        # Apply same hard constraints as WSGA (where columns exist)
        mask = pd.Series(True, index=base_df.index)
        if "MP-Measured" in base_df.columns:
            mask &= base_df["MP-Measured"] < -30
        if "BP-Measured" in base_df.columns:
            mask &= base_df["BP-Measured"] >= 100
        if "DC_exp" in base_df.columns:
            mask &= base_df["DC_exp"] <= 7
        if "flashpoint" in base_df.columns:
            mask &= base_df["flashpoint"] >= 423        # K  (150 °C)
        if "Tox21_Score" in base_df.columns:
            mask &= base_df["Tox21_Score"] <= 3

        base_df = base_df[mask].copy()

        if len(base_df) > 0 and molprice_model_path:
            try:
                _src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "src")
                if _src not in sys.path:
                    sys.path.insert(0, _src)
                from molprice import MolPriceModel

                mp_model = MolPriceModel(molprice_model_path)
                base_df["MolPrice"] = mp_model.predict_batch(
                    base_df["SMILES"].tolist()
                )
                base_df = base_df.dropna(subset=["FOM1_exp_avg", "MolPrice"])

                if len(base_df) > 0:
                    afford_base = -base_df["MolPrice"].values
                    fom1_base = base_df["FOM1_exp_avg"].values
                    base_front = _pareto_fronts(afford_base, fom1_base,
                                                n_fronts=1)
                    baseline_ok = True
                    print(f"  Baseline: {len(base_df)} molecules pass all "
                          f"constraints")
            except Exception as e:
                print(f"  WARNING: could not load MolPrice model for "
                      f"baseline: {e}")

    # ------------------------------------------------------------------
    # 3. Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    # Background: all WSGA points
    ax.scatter(afford_wsga, fom1_wsga,
               c="#CCCCCC", s=12, alpha=0.35, edgecolors="none",
               zorder=1, label="WSGA molecules")

    # WSGA Pareto fronts (1st, 2nd, 3rd)
    front_colors = ["#E63946", "#457B9D", "#2A9D8F"]
    front_labels = ["1st Pareto front (WSGA)",
                    "2nd Pareto front (WSGA)",
                    "3rd Pareto front (WSGA)"]
    for i, front_idx in enumerate(wsga_fronts):
        if len(front_idx) == 0:
            continue
        fx = afford_wsga[front_idx]
        fy = fom1_wsga[front_idx]
        order = np.argsort(fx)
        ax.plot(fx[order], fy[order], "o-",
                color=front_colors[i], markersize=5, linewidth=1.5,
                label=front_labels[i], zorder=5 - i)

    # Baseline overlay
    if baseline_ok:
        ax.scatter(afford_base, fom1_base,
                   c="#FFD166", s=20, alpha=0.5, edgecolors="none",
                   marker="D", zorder=2, label="FOM1 dataset")

        if base_front and len(base_front[0]) > 0:
            bx = afford_base[base_front[0]]
            by = fom1_base[base_front[0]]
            order = np.argsort(bx)
            ax.plot(bx[order], by[order], "D--",
                    color="#F77F00", markersize=6, linewidth=1.5,
                    label="Pareto front (FOM1 dataset)", zorder=6)

    ax.set_xlabel("−MolPrice (−log $/mmol)  →  cheaper", fontsize=12)
    ax.set_ylabel("FOM1 (avg)", fontsize=12)
    ax.set_title("Cost vs FOM1 Pareto Fronts\n"
                 "(all constraints, MP < −30 °C)", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "pareto_cost_vs_fom1.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


# ======================================================================
# Main analysis
# ======================================================================

def analyse_molprice_sweep(sweep_dir, top_n_molecules=50,
                           baseline_csv=None, molprice_model=None):

    # ------------------------------------------------------------------
    # 1. Scan all run directories
    # ------------------------------------------------------------------
    runs = []
    skipped = 0
    for dirname in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, dirname)
        if not os.path.isdir(run_path):
            continue
        params = parse_sweep_dir(dirname)
        if params is None:
            continue

        valid_df = load_all_evaluated(run_path)
        if valid_df is None:
            skipped += 1
            continue

        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in valid_df.columns else "SMILES"
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
            "mean_molprice_top10": top10["MolPrice"].mean() if "MolPrice" in top10.columns else np.nan,
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
    print(f"{'='*90}")
    print(f"  MOLPRICE THRESHOLD SWEEP RESULTS  (valid molecules only, MP < −30 °C)")
    print(f"{'='*90}\n")

    print(f"{'Level':<12} {'Biodeg':<8} {'Best FOM1_40':>18}   "
          f"{'Mean MolPrice':>16}   {'Seeds':>5}")
    print("-" * 75)

    for biodeg in ["bio", "nonbio"]:
        for level in LEVEL_ORDER:
            sub = runs_df[(runs_df["level"] == level) & (runs_df["biodeg"] == biodeg)]
            if sub.empty:
                continue
            fom1_mean = sub["best_FOM1_40"].mean()
            fom1_std = sub["best_FOM1_40"].std()
            mp_mean = sub["mean_molprice_top10"].mean()
            s1 = f"{fom1_std:.4f}" if not np.isnan(fom1_std) else "  n/a"
            mp_str = f"{mp_mean:.2f}" if not np.isnan(mp_mean) else "n/a"
            print(f"{level:<12} {biodeg:<8} {fom1_mean:>10.4f} +/- {s1}"
                  f"   {mp_str:>16}   {len(sub):>5}")
        print()

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
        valid_df["source_level"] = run["level"]
        valid_df["source_biodeg"] = run["biodeg"]
        valid_df["source_seed"] = run["seed"]
        all_valid_mols.append(valid_df)

    if all_valid_mols:
        combined = pd.concat(all_valid_mols, ignore_index=True)
        smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in combined.columns else "SMILES"
        combined = combined.sort_values("FOM1_avg", ascending=False)
        unique = combined.drop_duplicates(subset=[smiles_col], keep="first")
        top_unique = unique.head(top_n_molecules)

        print(f"{'='*110}")
        print(f"  TOP {top_n_molecules} UNIQUE VALID MOLECULES ({len(unique)} unique total)")
        print(f"{'='*110}\n")

        mp_col = "MolPrice" if "MolPrice" in top_unique.columns else None
        header = (f"{'Rank':<5} {'SMILES':<42} {'FOM1_avg':>10} {'FOM1_40':>10} "
                  f"{'MolPrice':>10} {'Level':>12} {'Biodeg':>8}")
        print(header)
        print("-" * len(header))

        for i, (_, mol) in enumerate(top_unique.iterrows(), 1):
            smi = str(mol.get(smiles_col, ""))
            if len(smi) > 39:
                smi = smi[:36] + "..."
            mp_val = f"{mol['MolPrice']:.2f}" if mp_col and not np.isnan(mol.get("MolPrice", np.nan)) else "n/a"
            print(
                f"{i:<5} {smi:<42} "
                f"{mol.get('FOM1_avg', np.nan):>10.4f} "
                f"{mol.get('FOM1_40', np.nan):>10.4f} "
                f"{mp_val:>10} "
                f"{mol.get('source_level', ''):>12} "
                f"{mol.get('source_biodeg', ''):>8}"
            )

    # ------------------------------------------------------------------
    # 4. Save CSVs
    # ------------------------------------------------------------------
    out_dir = os.path.join(sweep_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    runs_df.to_csv(os.path.join(out_dir, "summary_by_level.csv"), index=False)

    if all_valid_mols:
        top_unique.to_csv(os.path.join(out_dir, "top_molecules.csv"), index=False)

    print(f"\nCSVs saved to: {out_dir}/")

    # ------------------------------------------------------------------
    # 5. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")

    plot_fom1_vs_threshold(runs_df, out_dir)
    plot_pareto_front(runs_df, sweep_dir, out_dir)
    plot_molprice_distribution(runs_df, sweep_dir, out_dir)
    plot_convergence_by_threshold(runs_df, sweep_dir, out_dir)
    plot_pareto_cost_vs_fom1(runs_df, sweep_dir, out_dir,
                              baseline_csv, molprice_model)

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse WSGA MolPrice threshold sweep results"
    )
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/molprice_sweep")
    parser.add_argument("--top_molecules", type=int, default=50)
    parser.add_argument("--baseline_csv", type=str,
                        default="../BaselineFOM1Eval/output/baseline_fom1_results.csv",
                        help="Baseline FOM1 dataset CSV (experimental)")
    parser.add_argument("--molprice_model", type=str,
                        default="../models/MolPrice/MP_Morgan_hybrid.pkl",
                        help="MolPrice model weights (.pkl)")
    args = parser.parse_args()

    analyse_molprice_sweep(args.sweep_dir, args.top_molecules,
                           args.baseline_csv, args.molprice_model)
