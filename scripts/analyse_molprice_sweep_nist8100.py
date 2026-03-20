"""
Analyse NIST 8100 MolPrice sweep results (4 categories).

Parses output directories from molprice_sweep_nist8100.sh, which sweeps
MolPrice cost-penalty thresholds across 6 levels × 4 categories × 5 seeds.
Uses fom1_40C (direct prediction) as the FOM1 column.

Directory format: {level}_{category}_seed{SEED}

Categories: bio_stable, bio_unstable, nonbio_stable, nonbio_unstable

Usage:
    python analyse_molprice_sweep_nist8100.py --sweep_dir ../outputs/molprice_sweep_nist8100
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

# Add src to path for MolPrice model
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# Ordered threshold levels for consistent plotting
LEVEL_ORDER = ["nocost", "gentle", "moderate", "firm", "tight", "aggressive"]
LEVEL_DESCRIPTIONS = {
    "nocost": "No cost",
    "gentle": "Gentle (4.0/6.0)",
    "moderate": "Moderate (3.5/5.5)",
    "firm": "Firm (3.0/5.0)",
    "tight": "Tight (2.5/4.5)",
    "aggressive": "Aggressive (2.0/4.0)",
}

CATEGORY_ORDER = ["bio_stable", "bio_unstable", "nonbio_stable", "nonbio_unstable"]
CATEGORY_LABELS = {
    "bio_stable": "Biodegradable = True, Stable = True",
    "bio_unstable": "Biodegradable = True, Stable = False",
    "nonbio_stable": "Biodegradable = False, Stable = True",
    "nonbio_unstable": "Biodegradable = False, Stable = False",
}
CATEGORY_COLORS = {
    "bio_stable": "#2A9D8F",
    "bio_unstable": "#E76F51",
    "nonbio_stable": "#264653",
    "nonbio_unstable": "#E9C46A",
}
CATEGORY_MARKERS = {
    "bio_stable": "o",
    "bio_unstable": "s",
    "nonbio_stable": "D",
    "nonbio_unstable": "^",
}


# ======================================================================
# Data loading
# ======================================================================

def parse_sweep_dir(dirname):
    """Extract level, category, seed from directory name."""
    m = re.match(r"([a-z]+)_(bio_stable|bio_unstable|nonbio_stable|nonbio_unstable)_seed(\d+)",
                 dirname)
    if not m:
        return None
    level = m.group(1)
    if level not in LEVEL_ORDER:
        return None
    return {
        "level": level,
        "category": m.group(2),
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


_LOADED_CACHE = {}
_MOLPRICE_MODEL = None


def _get_molprice_model():
    """Lazy-load the MolPrice model (singleton)."""
    global _MOLPRICE_MODEL
    if _MOLPRICE_MODEL is None:
        from molprice import MolPriceModel
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "models", "MolPrice", "MP_Morgan_hybrid.pkl")
        if os.path.exists(model_path):
            _MOLPRICE_MODEL = MolPriceModel(model_path)
            print(f"  Loaded MolPrice model for backfilling")
        else:
            print(f"  WARNING: MolPrice model not found at {model_path}")
    return _MOLPRICE_MODEL


def _backfill_molprice(df):
    """Predict MolPrice for rows where it is missing."""
    smiles_col = ("CanonicalSMILES" if "CanonicalSMILES" in df.columns
                  else "SMILES")
    if "MolPrice" not in df.columns:
        df["MolPrice"] = np.nan

    missing = df["MolPrice"].isna() | (df["MolPrice"].astype(str) == "")
    if not missing.any():
        return df

    mp_model = _get_molprice_model()
    if mp_model is None:
        return df

    smiles_to_predict = df.loc[missing, smiles_col].tolist()
    predictions = mp_model.predict_batch(smiles_to_predict)
    df.loc[missing, "MolPrice"] = predictions
    return df


def load_all_evaluated(run_path, mp_hard=-30):
    """Load all_evaluated_molecules.csv (or top_n_tracking.csv fallback).

    Falls back to top_n_tracking.csv when all_evaluated is unavailable.
    Backfills missing MolPrice values using the MolPrice model.
    """
    cache_key = (run_path, mp_hard)
    if cache_key in _LOADED_CACHE:
        return _LOADED_CACHE[cache_key]

    path = os.path.join(run_path, "all_evaluated_molecules.csv")
    if not os.path.exists(path):
        path = os.path.join(run_path, "top_n_tracking.csv")
    if not os.path.exists(path):
        _LOADED_CACHE[cache_key] = None
        return None
    df = safe_read_csv(path)
    if df is None:
        _LOADED_CACHE[cache_key] = None
        return None

    mask = pd.Series(True, index=df.index)
    if "is_valid" in df.columns:
        mask &= (df["is_valid"] == 1)
    if "FitnessScore" in df.columns:
        mask &= (df["FitnessScore"] > 0)
    if "mp" in df.columns:
        mask &= (df["mp"] < mp_hard)

    df = df.loc[mask].copy()
    if df.empty:
        _LOADED_CACHE[cache_key] = None
        return None

    # Backfill missing MolPrice (e.g. nocost runs)
    df = _backfill_molprice(df)

    df = df.sort_values("fom1_40C", ascending=False)
    _LOADED_CACHE[cache_key] = df
    return df


def get_convergence_curve(valid_df):
    """Compute per-generation best valid FOM1 (cumulative max)."""
    best_per_gen = valid_df.groupby("generation")["fom1_40C"].max()
    best_per_gen = best_per_gen.sort_index()
    running_best = best_per_gen.cummax()
    return running_best


def _pareto_fronts(x, y, n_fronts=1):
    """Compute successive Pareto fronts for maximising both x and y."""
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
            dom = ((xi[i] >= xi) & (yi[i] >= yi)
                   & ((xi[i] > xi) | (yi[i] > yi)))
            dom[i] = False
            is_pareto[dom] = False
        front = idxs[is_pareto]
        fronts.append(front)
        remaining[front] = False
    return fronts


# ======================================================================
# Figures
# ======================================================================

def _filter_baseline_for_category(baseline_df, category):
    """Filter baseline molecules to those satisfying a category's constraints.

    Each category implies a set of filters:
      - bio_*:    biodegradable molecules only
      - nonbio_*: no biodeg filter (all molecules)
      - *_stable: stable molecules only
      - *_unstable: no stability filter (all molecules)

    So nonbio_unstable gets ALL baseline molecules, bio_stable gets only
    those that are both biodegradable AND stable, etc.
    """
    mask = pd.Series(True, index=baseline_df.index)
    if category.startswith("bio_"):
        if "is_biodegradable" in baseline_df.columns:
            mask &= baseline_df["is_biodegradable"]
    if category.endswith("_stable"):
        if "is_stable" in baseline_df.columns:
            mask &= baseline_df["is_stable"]
    return baseline_df[mask].copy()


def plot_pareto_2x2(runs_df, sweep_dir, out_dir, baseline_df=None):
    """2x2 grid of Pareto front panels — one per category.

    Each panel shows:
      - Gray scatter: all valid WSGA molecules
      - Red line: WSGA 1st Pareto front
      - Blue diamonds: baseline experimental FOM1
      - Orange squares: baseline predicted FOM1
      - Dashed lines: baseline Pareto fronts

    Baseline molecules are filtered to those satisfying the category's
    constraints (not the exclusive category assignment), so e.g. the
    "no filters" panel shows all 15 baseline molecules.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)

    # Layout: columns = biodegradable (left=False, right=True)
    #         rows    = stable (top=False, bottom=True)
    panel_map = {
        "nonbio_unstable": (0, 0),
        "bio_unstable": (0, 1),
        "nonbio_stable": (1, 0),
        "bio_stable": (1, 1),
    }

    for category, (row, col) in panel_map.items():
        ax = axes[row, col]
        # --- Collect WSGA molecules for this category ---
        cat_runs = runs_df[runs_df["category"] == category]
        all_mols = []
        for _, run in cat_runs.iterrows():
            run_path = os.path.join(sweep_dir, run["dir"])
            valid_df = load_all_evaluated(run_path)
            if valid_df is None or "MolPrice" not in valid_df.columns:
                continue
            all_mols.append(valid_df)

        if not all_mols:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray")
            continue

        wsga = pd.concat(all_mols, ignore_index=True)
        smiles_col = ("CanonicalSMILES" if "CanonicalSMILES" in wsga.columns
                      else "SMILES")
        wsga = wsga.sort_values("fom1_40C", ascending=False)
        wsga = wsga.drop_duplicates(subset=[smiles_col], keep="first")
        wsga = wsga.dropna(subset=["MolPrice", "fom1_40C"])

        if wsga.empty:
            continue

        afford = -wsga["MolPrice"].values
        fom1 = wsga["fom1_40C"].values

        # All WSGA molecules (gray)
        ax.scatter(afford, fom1, c="#CCCCCC", s=10, alpha=0.3,
                   edgecolors="none", zorder=1, label="WSGA molecules")

        # WSGA Pareto front (red)
        fronts = _pareto_fronts(afford, fom1, n_fronts=1)
        if fronts and len(fronts[0]) > 0:
            fidx = fronts[0]
            fx, fy = afford[fidx], fom1[fidx]
            order = np.argsort(fx)
            ax.plot(fx[order], fy[order], "o-", color="#E63946",
                    markersize=5, linewidth=1.5, label="Pareto (WSGA)",
                    zorder=5)

        # --- Baseline for this category ---
        # Filter to molecules satisfying this category's constraints
        # (inclusive, not exclusive category assignment)
        if baseline_df is not None:
            bdf = _filter_baseline_for_category(baseline_df, category)

            if len(bdf) > 0 and "MolPrice" in bdf.columns:
                bdf = bdf.dropna(subset=["MolPrice"])
                ab = -bdf["MolPrice"].values

                # Experimental FOM1
                if "FOM1_exp_avg" in bdf.columns:
                    fb_exp = bdf["FOM1_exp_avg"].values
                    ax.scatter(ab, fb_exp, c="#1F77B4", s=30, alpha=0.7,
                               edgecolors="black", linewidths=0.4,
                               marker="D", zorder=2,
                               label=f"Baseline exp. ({len(bdf)})")
                    bfronts = _pareto_fronts(ab, fb_exp, n_fronts=1)
                    for bidx in bfronts:
                        if len(bidx) == 0:
                            continue
                        bx, by = ab[bidx], fb_exp[bidx]
                        order = np.argsort(bx)
                        ax.plot(bx[order], by[order], "D--",
                                color="#0B5394", markersize=5,
                                linewidth=1.5,
                                label="Pareto (baseline exp.)",
                                zorder=6)

                # Predicted FOM1
                if "FOM1_pred_avg" in bdf.columns:
                    fb_pred = bdf["FOM1_pred_avg"].values
                    ax.scatter(ab, fb_pred, c="#FF7F0E", s=30, alpha=0.7,
                               edgecolors="black", linewidths=0.4,
                               marker="s", zorder=3,
                               label="Baseline pred.")
                    bfronts = _pareto_fronts(ab, fb_pred, n_fronts=1)
                    for bidx in bfronts:
                        if len(bidx) == 0:
                            continue
                        bx, by = ab[bidx], fb_pred[bidx]
                        order = np.argsort(bx)
                        ax.plot(bx[order], by[order], "s--",
                                color="#CC5500", markersize=5,
                                linewidth=1.5,
                                label="Pareto (baseline pred.)",
                                zorder=7)

        ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax.tick_params(labelsize=10)

    # Axis limits
    for row_axes in axes:
        for ax in row_axes:
            ax.set_xlim(-6, -2)
            ax.set_ylim(55, None)

    # Column / row labels
    axes[0, 0].annotate("Biodegradable = False", xy=(0.5, 1.15),
                         xycoords="axes fraction", ha="center", fontsize=13,
                         fontweight="bold")
    axes[0, 1].annotate("Biodegradable = True", xy=(0.5, 1.15),
                         xycoords="axes fraction", ha="center", fontsize=13,
                         fontweight="bold")
    axes[0, 0].annotate("Stable = False", xy=(-0.22, 0.5),
                         xycoords="axes fraction", ha="center", va="center",
                         fontsize=13, fontweight="bold", rotation=90)
    axes[1, 0].annotate("Stable = True", xy=(-0.22, 0.5),
                         xycoords="axes fraction", ha="center", va="center",
                         fontsize=13, fontweight="bold", rotation=90)

    # Shared axis labels
    for ax in axes[1, :]:
        ax.set_xlabel("\u2212MolPrice / \u2212log($ mmol\u207b\u00b9)  \u2192  cheaper",
                      fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("FOM1 / W m\u207b\u00b9 K\u207b\u00b9", fontsize=11)

    fig.tight_layout(rect=[0.04, 0, 1, 0.96])
    path = os.path.join(out_dir, "pareto_2x2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_fom1_vs_threshold(runs_df, out_dir):
    """FOM1 vs threshold level with 4 category lines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for category in CATEGORY_ORDER:
        sub = runs_df[runs_df["category"] == category]
        if sub.empty:
            continue

        agg = sub.groupby("level")["best_fom1_40C"].agg(["mean", "std", "count"])
        agg = agg.reindex([l for l in LEVEL_ORDER if l in agg.index])

        x = np.arange(len(agg))
        offsets = {"bio_stable": -0.15, "bio_unstable": -0.05,
                   "nonbio_stable": 0.05, "nonbio_unstable": 0.15}
        offset = offsets.get(category, 0)

        ax.errorbar(x + offset, agg["mean"], yerr=agg["std"],
                    fmt=f"{CATEGORY_MARKERS[category]}-", capsize=4,
                    linewidth=2, markersize=7,
                    color=CATEGORY_COLORS[category],
                    ecolor="#999999",
                    label=CATEGORY_LABELS[category])

    present_levels = [l for l in LEVEL_ORDER if l in runs_df["level"].unique()]
    ax.set_xticks(np.arange(len(present_levels)))
    ax.set_xticklabels([LEVEL_DESCRIPTIONS.get(l, l) for l in present_levels],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Best FOM1 (direct)", fontsize=12)
    ax.set_title("Best FOM1 vs MolPrice Threshold Level", fontsize=14)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "fom1_vs_threshold.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_molprice_distribution(runs_df, sweep_dir, out_dir):
    """MolPrice distribution of top-10 molecules, 4 panels by category."""
    present_levels = [l for l in LEVEL_ORDER if l in runs_df["level"].unique()]
    present_cats = [c for c in CATEGORY_ORDER if c in runs_df["category"].unique()]
    ncols = len(present_cats)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6),
                             sharey=True, squeeze=False)

    for col_idx, category in enumerate(present_cats):
        ax = axes[0, col_idx]
        data_for_plot = []

        cat_runs = runs_df[runs_df["category"] == category]
        for _, run in cat_runs.iterrows():
            valid_df = load_all_evaluated(os.path.join(sweep_dir, run["dir"]))
            if valid_df is None or "MolPrice" not in valid_df.columns:
                continue
            smiles_col = ("CanonicalSMILES" if "CanonicalSMILES"
                          in valid_df.columns else "SMILES")
            top10 = valid_df.drop_duplicates(subset=[smiles_col]).head(10)
            for _, mol in top10.iterrows():
                data_for_plot.append({
                    "level": run["level"],
                    "MolPrice": mol["MolPrice"],
                })

        if not data_for_plot:
            ax.set_title(CATEGORY_LABELS[category], fontsize=11)
            continue

        plot_df = pd.DataFrame(data_for_plot)
        box_data = [plot_df[plot_df["level"] == l]["MolPrice"].dropna().values
                    for l in present_levels]
        bp = ax.boxplot(box_data, tick_labels=[l[:6] for l in present_levels],
                        patch_artist=True, showfliers=True)

        cmap = plt.cm.viridis
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(cmap(i / max(len(present_levels) - 1, 1)))
            patch.set_alpha(0.7)

        ax.set_xticklabels([l[:6] for l in present_levels],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(CATEGORY_LABELS[category], fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("MolPrice (log $/mol)", fontsize=11)

    fig.suptitle("MolPrice Distribution of Top-10 Molecules", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "molprice_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


def plot_convergence_2x2(runs_df, sweep_dir, out_dir):
    """Convergence curves in 2x2 grid (one per category)."""
    present_levels = [l for l in LEVEL_ORDER if l in runs_df["level"].unique()]
    # Same layout as Pareto: cols=biodeg, rows=stable
    panel_map = {
        "nonbio_unstable": (0, 0),
        "bio_unstable": (0, 1),
        "nonbio_stable": (1, 0),
        "bio_stable": (1, 1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max(len(present_levels) - 1, 1))

    for category, (row, col) in panel_map.items():
        ax = axes[row, col]
        cat_runs = runs_df[runs_df["category"] == category]

        for level in present_levels:
            level_runs = cat_runs[cat_runs["level"] == level]
            if level_runs.empty:
                continue

            all_curves = []
            for _, run in level_runs.iterrows():
                valid_df = load_all_evaluated(
                    os.path.join(sweep_dir, run["dir"]))
                if valid_df is None:
                    continue
                curve = get_convergence_curve(valid_df)
                all_curves.append(curve)

            if not all_curves:
                continue

            combined = pd.concat(all_curves, axis=1)
            combined = combined.ffill().bfill()
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

        ax.set_title(CATEGORY_LABELS[category], fontsize=12)
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(labelsize=9)

    for ax in axes[1, :]:
        ax.set_xlabel("Generation", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Best Valid FOM1 (direct)", fontsize=11)

    fig.suptitle("Convergence by MolPrice Threshold\n"
                 "(valid molecules only, MP < \u221230 \u00b0C)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_2x2.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(path)}")


# ======================================================================
# Main analysis
# ======================================================================

def analyse_stability_sweep(sweep_dir, top_n_molecules=50,
                             baseline_csv=None):
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

        smiles_col = ("CanonicalSMILES" if "CanonicalSMILES"
                      in valid_df.columns else "SMILES")
        unique_df = valid_df.drop_duplicates(subset=[smiles_col], keep="first")
        best = unique_df.iloc[0]
        top10 = unique_df.head(10)

        run_info = {
            **params,
            "dir": dirname,
            "best_fom1_40C": best["fom1_40C"],
            "best_FOM1_40": best.get("FOM1_40", np.nan),
            "best_fitness": best.get("FitnessScore", np.nan),
            "best_smiles": best.get(smiles_col, ""),
            "mean_molprice_top10": (top10["MolPrice"].mean()
                                    if "MolPrice" in top10.columns
                                    else np.nan),
            "top10_mean_fom1_40C": top10["fom1_40C"].mean(),
            "n_valid_molecules": len(unique_df),
        }
        runs.append(run_info)

    if not runs:
        print("No completed runs found.")
        return

    runs_df = pd.DataFrame(runs)
    print(f"Found {len(runs_df)} completed runs (skipped {skipped} incomplete)")

    # ------------------------------------------------------------------
    # 2. Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  STABILITY x MOLPRICE SWEEP RESULTS  (valid molecules, MP < -30 C)")
    print(f"{'='*100}\n")

    print(f"{'Level':<12} {'Category':<20} {'Best fom1_40C':>18}   "
          f"{'Mean MolPrice':>16}   {'Seeds':>5}")
    print("-" * 85)

    for category in CATEGORY_ORDER:
        for level in LEVEL_ORDER:
            sub = runs_df[(runs_df["level"] == level) &
                          (runs_df["category"] == category)]
            if sub.empty:
                continue
            fom1_mean = sub["best_fom1_40C"].mean()
            fom1_std = sub["best_fom1_40C"].std()
            mp_mean = sub["mean_molprice_top10"].mean()
            s1 = f"{fom1_std:.4f}" if not np.isnan(fom1_std) else "  n/a"
            mp_str = (f"{mp_mean:.2f}" if not np.isnan(mp_mean) else "n/a")
            print(f"{level:<12} {category:<20} {fom1_mean:>10.4f} +/- {s1}"
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
        valid_df["source_category"] = run["category"]
        valid_df["source_seed"] = run["seed"]
        all_valid_mols.append(valid_df)

    if all_valid_mols:
        combined = pd.concat(all_valid_mols, ignore_index=True)
        smiles_col = ("CanonicalSMILES" if "CanonicalSMILES"
                      in combined.columns else "SMILES")
        combined = combined.sort_values("fom1_40C", ascending=False)
        unique = combined.drop_duplicates(subset=[smiles_col], keep="first")
        top_unique = unique.head(top_n_molecules)

        print(f"\n{'='*120}")
        print(f"  TOP {top_n_molecules} UNIQUE VALID MOLECULES "
              f"({len(unique)} unique total)")
        print(f"{'='*120}\n")

        header = (f"{'Rank':<5} {'SMILES':<42} {'fom1_40C':>10} "
                  f"{'MolPrice':>10} {'Category':>20} {'Level':>12}")
        print(header)
        print("-" * len(header))

        for i, (_, mol) in enumerate(top_unique.iterrows(), 1):
            smi = str(mol.get(smiles_col, ""))
            if len(smi) > 39:
                smi = smi[:36] + "..."
            mp_val = (f"{mol['MolPrice']:.2f}"
                      if "MolPrice" in mol and not np.isnan(
                          mol.get("MolPrice", np.nan))
                      else "n/a")
            print(
                f"{i:<5} {smi:<42} "
                f"{mol.get('fom1_40C', np.nan):>10.4f} "
                f"{mp_val:>10} "
                f"{mol.get('source_category', ''):>20} "
                f"{mol.get('source_level', ''):>12}"
            )

    # ------------------------------------------------------------------
    # 4. Save CSVs
    # ------------------------------------------------------------------
    out_dir = os.path.join(sweep_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    runs_df.to_csv(os.path.join(out_dir, "summary_by_level.csv"), index=False)
    if all_valid_mols:
        top_unique.to_csv(os.path.join(out_dir, "top_molecules.csv"),
                          index=False)

    print(f"\nCSVs saved to: {out_dir}/")

    # ------------------------------------------------------------------
    # 5. Load baseline (if available)
    # ------------------------------------------------------------------
    baseline_df = None
    if baseline_csv and os.path.exists(baseline_csv):
        print(f"\nLoading baseline: {baseline_csv}")
        baseline_df = pd.read_csv(baseline_csv)
        # Only keep molecules that pass base constraints
        if "passes_base_constraints" in baseline_df.columns:
            baseline_df = baseline_df[
                baseline_df["passes_base_constraints"] == True].copy()
        print(f"  Baseline: {len(baseline_df)} molecules passing constraints")
    elif baseline_csv:
        print(f"  WARNING: baseline CSV not found: {baseline_csv}")

    # ------------------------------------------------------------------
    # 6. Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")

    plot_pareto_2x2(runs_df, sweep_dir, out_dir, baseline_df)
    plot_fom1_vs_threshold(runs_df, out_dir)
    plot_molprice_distribution(runs_df, sweep_dir, out_dir)
    plot_convergence_2x2(runs_df, sweep_dir, out_dir)

    # ------------------------------------------------------------------
    # 7. Per-category Pareto CSVs
    # ------------------------------------------------------------------
    if all_valid_mols:
        for category in CATEGORY_ORDER:
            cat_mols = combined[combined["source_category"] == category]
            if cat_mols.empty or "MolPrice" not in cat_mols.columns:
                continue
            cat_unique = cat_mols.drop_duplicates(
                subset=[smiles_col], keep="first")
            cat_unique = cat_unique.dropna(subset=["MolPrice", "fom1_40C"])
            if cat_unique.empty:
                continue

            afford = -cat_unique["MolPrice"].values
            fom1 = cat_unique["fom1_40C"].values
            fronts = _pareto_fronts(afford, fom1, n_fronts=1)

            if fronts and len(fronts[0]) > 0:
                pareto_df = cat_unique.iloc[fronts[0]].sort_values(
                    "fom1_40C", ascending=False)
                csv_path = os.path.join(out_dir,
                                        f"pareto_front_{category}.csv")
                save_cols = [c for c in [smiles_col, "fom1_40C", "FOM1_40",
                                         "MolPrice",
                                         "source_level", "source_seed"]
                             if c in pareto_df.columns]
                pareto_df[save_cols].to_csv(csv_path, index=False)
                print(f"  Saved: {os.path.abspath(csv_path)}")

    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse WSGA stability x MolPrice sweep results")
    parser.add_argument("--sweep_dir", type=str,
                        default="../outputs/molprice_sweep_nist8100")
    parser.add_argument("--top_molecules", type=int, default=50)
    parser.add_argument("--baseline_csv", type=str,
                        default="../BaselineFOM1Eval/output/fom1_dataset_categories.csv",
                        help="Pre-categorised FOM1 dataset CSV")
    args = parser.parse_args()

    analyse_stability_sweep(args.sweep_dir, args.top_molecules,
                             args.baseline_csv)
