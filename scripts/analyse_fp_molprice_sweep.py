#!/usr/bin/env python
"""
Analyse FP x MolPrice production sweep results (NIST 8100 models).

Reads the compact valid_molecules_summary.csv and generation_stats_combined.csv
produced by extract_fp_molprice_sweep_valid.py. Generates summary tables,
hypervolume analysis, convergence curves, OOD analysis, and candidate ranking.

Usage:
    python scripts/analyse_fp_molprice_sweep.py \
        --sweep_dir outputs/fp_molprice_sweep_nist8100
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# ======================================================================
# Constants
# ======================================================================

FP_LABELS = ["fp100", "fp125", "fp150", "fp175"]
FP_VALUES_C = {"fp100": 100, "fp125": 125, "fp150": 150, "fp175": 175}

LEVEL_ORDER = ["nocost", "gentle", "aggressive"]
LEVEL_LABELS = {
    "nocost": "No cost",
    "gentle": "Gentle (4/6)",
    "aggressive": "Aggressive (2/4)",
}
LEVEL_COLORS = {
    "nocost": "#1B2838",
    "gentle": "#2A5F8F",
    "aggressive": "#6DB8D4",
}
LEVEL_MARKERS = {
    "nocost": "o",
    "gentle": "s",
    "aggressive": "D",
}

SEEDS = [42, 123, 7, 256, 999]

BASELINE_BEST_FOM1 = 87.6

CRITICAL_OOD_COLS = ["OOD_fom1_40C", "OOD_fp", "OOD_dc", "OOD_mp", "OOD_bp"]
ALL_OOD_COLS = CRITICAL_OOD_COLS + [
    "OOD_density_40C", "OOD_viscosity_40C", "OOD_tc_40C",
    "OOD_cpsat_40C", "OOD_beta_40C",
]

OOD_SHORT_NAMES = {
    "OOD_fom1_40C": "FOM1",
    "OOD_fp": "FP",
    "OOD_dc": "DC",
    "OOD_mp": "MP",
    "OOD_bp": "BP",
    "OOD_density_40C": "Density",
    "OOD_viscosity_40C": "Viscosity",
    "OOD_tc_40C": "TC",
    "OOD_cpsat_40C": "Cp,sat",
    "OOD_beta_40C": "Beta",
}

HV_REF_POINT = (30.0, -8.0)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 9,
})


# ======================================================================
# Data loading
# ======================================================================

def load_molecules(sweep_dir):
    path = os.path.join(sweep_dir, "valid_molecules_summary.csv")
    print(f"Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df)} rows, {df['fp_label'].nunique()} FP levels, "
          f"{df['cost_level'].nunique()} cost levels, "
          f"{df['seed'].nunique()} seeds")
    return df


def load_gen_stats(sweep_dir):
    path = os.path.join(sweep_dir, "generation_stats_combined.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping convergence analysis")
        return None
    df = pd.read_csv(path)
    print(f"  Generation stats: {len(df)} rows")
    return df


def load_baseline():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "BaselineFOM1Eval", "output", "fom1_dataset_categories.csv")
    if not os.path.exists(path):
        print(f"  WARNING: baseline not found at {path}")
        return None
    df = pd.read_csv(path)
    if "passes_base_constraints" in df.columns:
        df = df[df["passes_base_constraints"] == True].copy()
    print(f"  Baseline: {len(df)} molecules passing constraints")
    return df


def backfill_molprice(df):
    """Predict MolPrice for nocost rows where it's missing."""
    if "MolPrice" not in df.columns:
        df["MolPrice"] = np.nan
    missing = df["MolPrice"].isna()
    if not missing.any():
        return df
    try:
        from molprice import MolPriceModel
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "models", "MolPrice", "MP_Morgan_hybrid.pkl")
        if not os.path.exists(model_path):
            print("  WARNING: MolPrice model not found, cannot backfill")
            return df
        mp_model = MolPriceModel(model_path)
        smiles_list = df.loc[missing, "SMILES"].tolist()
        predictions = mp_model.predict_batch(smiles_list)
        df.loc[missing, "MolPrice"] = predictions
        print(f"  Backfilled MolPrice for {missing.sum()} molecules")
    except ImportError:
        print("  WARNING: molprice not importable, skipping backfill")
    return df


# ======================================================================
# Pareto front & hypervolume (from plot_hv_fp_prototype.py)
# ======================================================================

def pareto_front_max(x, y):
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dom = (x[i] >= x) & (y[i] >= y) & ((x[i] > x) | (y[i] > y))
        dom[i] = False
        is_pareto[dom] = False
    return np.where(is_pareto)[0]


def compute_hv_2d(fom1, neg_mp, ref_point):
    pidx = pareto_front_max(fom1, neg_mp)
    if len(pidx) == 0:
        return 0.0
    pf = np.column_stack([fom1[pidx], neg_mp[pidx]])
    pf = pf[(pf[:, 0] > ref_point[0]) & (pf[:, 1] > ref_point[1])]
    if len(pf) == 0:
        return 0.0
    pf = pf[pf[:, 0].argsort()[::-1]]
    hv = 0.0
    for i in range(len(pf)):
        f_next = ref_point[0] if i == len(pf) - 1 else pf[i + 1, 0]
        hv += (pf[i, 0] - f_next) * (pf[i, 1] - ref_point[1])
    return hv


# ======================================================================
# Section A: Summary table
# ======================================================================

def compute_summary(df):
    """Compute per-config summary statistics, aggregated across seeds."""
    rows = []
    for fp in FP_LABELS:
        for lv in LEVEL_ORDER:
            sub = df[(df["fp_label"] == fp) & (df["cost_level"] == lv)]
            if sub.empty:
                continue

            per_seed = []
            for seed in SEEDS:
                ssub = sub[sub["seed"] == seed]
                if ssub.empty:
                    continue
                top10 = ssub.nlargest(10, "fom1_40C")
                per_seed.append({
                    "seed": seed,
                    "best_fom1": ssub["fom1_40C"].max(),
                    "mean_top10_fom1": top10["fom1_40C"].mean(),
                    "n_valid": len(ssub),
                    "mean_molprice_top10": top10["MolPrice"].mean()
                        if "MolPrice" in top10.columns else np.nan,
                    "total_evaluated": ssub["total_evaluated"].iloc[0]
                        if "total_evaluated" in ssub.columns else np.nan,
                })

            if not per_seed:
                continue

            ps = pd.DataFrame(per_seed)
            rows.append({
                "fp_label": fp,
                "fp_C": FP_VALUES_C[fp],
                "cost_level": lv,
                "n_seeds": len(ps),
                "best_fom1_mean": ps["best_fom1"].mean(),
                "best_fom1_std": ps["best_fom1"].std(),
                "mean_top10_fom1_mean": ps["mean_top10_fom1"].mean(),
                "mean_top10_fom1_std": ps["mean_top10_fom1"].std(),
                "n_valid_mean": ps["n_valid"].mean(),
                "n_valid_std": ps["n_valid"].std(),
                "molprice_top10_mean": ps["mean_molprice_top10"].mean(),
                "total_evaluated_mean": ps["total_evaluated"].mean(),
            })

    summary = pd.DataFrame(rows)
    return summary


# ======================================================================
# Section B: Hypervolume analysis
# ======================================================================

def compute_hv_grid(df):
    """Compute HV per (fp_label, cost_level, seed) and aggregate."""
    hv_rows = []
    for fp in FP_LABELS:
        for lv in LEVEL_ORDER:
            seed_hvs = []
            for seed in SEEDS:
                sub = df[(df["fp_label"] == fp) & (df["cost_level"] == lv)
                         & (df["seed"] == seed)]
                if sub.empty or sub["MolPrice"].isna().all():
                    continue
                sub_clean = sub.dropna(subset=["fom1_40C", "MolPrice"])
                if sub_clean.empty:
                    continue
                fom1 = sub_clean["fom1_40C"].values
                neg_mp = -sub_clean["MolPrice"].values
                hv = compute_hv_2d(fom1, neg_mp, HV_REF_POINT)
                seed_hvs.append(hv)

            if seed_hvs:
                hv_rows.append({
                    "fp_label": fp,
                    "fp_C": FP_VALUES_C[fp],
                    "cost_level": lv,
                    "hv_mean": np.mean(seed_hvs),
                    "hv_std": np.std(seed_hvs),
                    "n_seeds": len(seed_hvs),
                })

    return pd.DataFrame(hv_rows)


def plot_hv_heatmap(hv_df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    grid = np.full((len(FP_LABELS), len(LEVEL_ORDER)), np.nan)
    annot = np.empty((len(FP_LABELS), len(LEVEL_ORDER)), dtype=object)

    for _, row in hv_df.iterrows():
        i = FP_LABELS.index(row["fp_label"])
        j = LEVEL_ORDER.index(row["cost_level"])
        grid[i, j] = row["hv_mean"]
        annot[i, j] = f"{row['hv_mean']:.1f}\n\u00b1{row['hv_std']:.1f}"

    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto", origin="upper")
    for i in range(len(FP_LABELS)):
        for j in range(len(LEVEL_ORDER)):
            if annot[i, j] is not None:
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        fontsize=9, color="black" if grid[i, j] < np.nanmax(grid) * 0.7 else "white")

    ax.set_xticks(range(len(LEVEL_ORDER)))
    ax.set_xticklabels([LEVEL_LABELS[l] for l in LEVEL_ORDER])
    ax.set_yticks(range(len(FP_LABELS)))
    ax.set_yticklabels([f"FP \u2265 {FP_VALUES_C[f]} \u00b0C" for f in FP_LABELS])
    ax.set_xlabel("MolPrice regime")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Hypervolume (FOM1, \u2212MolPrice)")

    fig.tight_layout()
    path = os.path.join(out_dir, "hv_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_hv_vs_fp(hv_df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for lv in LEVEL_ORDER:
        sub = hv_df[hv_df["cost_level"] == lv].sort_values("fp_C")
        if sub.empty:
            continue
        ax.errorbar(sub["fp_C"], sub["hv_mean"], yerr=sub["hv_std"],
                    fmt=f"{LEVEL_MARKERS[lv]}-", capsize=4, linewidth=2,
                    markersize=6, color=LEVEL_COLORS[lv],
                    label=LEVEL_LABELS[lv])

    ax.set_xlabel("Min flash point / \u00b0C")
    ax.set_ylabel("Hypervolume (FOM1, \u2212MolPrice)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(out_dir, "hv_vs_fp.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Section C: FOM1 vs FP trade-off
# ======================================================================

def plot_fom1_vs_fp(summary_df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for lv in LEVEL_ORDER:
        sub = summary_df[summary_df["cost_level"] == lv].sort_values("fp_C")
        if sub.empty:
            continue
        ax.errorbar(sub["fp_C"], sub["best_fom1_mean"],
                    yerr=sub["best_fom1_std"],
                    fmt=f"{LEVEL_MARKERS[lv]}-", capsize=4, linewidth=2,
                    markersize=6, color=LEVEL_COLORS[lv],
                    label=LEVEL_LABELS[lv])

    ax.axhline(BASELINE_BEST_FOM1, color="#E63946", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Baseline best ({BASELINE_BEST_FOM1})")
    ax.set_xlabel("Min flash point / \u00b0C")
    ax.set_ylabel("Best FOM1")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(out_dir, "fom1_vs_fp.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Section D: Pareto front grid
# ======================================================================

def plot_pareto_grid(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    fp_colors = {
        "fp100": "#264653",
        "fp125": "#2A9D8F",
        "fp150": "#E9C46A",
        "fp175": "#E76F51",
    }

    for j, lv in enumerate(LEVEL_ORDER):
        ax = axes[j]
        lv_data = df[df["cost_level"] == lv].dropna(subset=["fom1_40C", "MolPrice"])
        if lv_data.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        # Gray scatter of all molecules
        ax.scatter(-lv_data["MolPrice"].values, lv_data["fom1_40C"].values,
                   c="#CCCCCC", s=6, alpha=0.2, edgecolors="none", zorder=1)

        for fp in FP_LABELS:
            fp_data = lv_data[lv_data["fp_label"] == fp]
            fp_data = fp_data.sort_values("fom1_40C", ascending=False)
            fp_data = fp_data.drop_duplicates(subset=["SMILES"], keep="first")
            if fp_data.empty:
                continue

            fom1 = fp_data["fom1_40C"].values
            neg_mp = -fp_data["MolPrice"].values
            pidx = pareto_front_max(fom1, neg_mp)

            if len(pidx) > 0:
                fx, fy = neg_mp[pidx], fom1[pidx]
                order = np.argsort(fx)
                ax.plot(fx[order], fy[order], "o-",
                        color=fp_colors[fp], markersize=4, linewidth=1.5,
                        label=f"FP \u2265 {FP_VALUES_C[fp]} \u00b0C", zorder=5)

        ax.axhline(BASELINE_BEST_FOM1, color="#E63946", linestyle="--",
                   linewidth=1, alpha=0.5)
        ax.set_xlabel("\u2212MolPrice / \u2212log($ mmol\u207b\u00b9)")
        if j == 0:
            ax.set_ylabel("FOM1")
        ax.annotate(LEVEL_LABELS[lv], xy=(0.5, 1.03), xycoords="axes fraction",
                    ha="center", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", frameon=False)

    fig.tight_layout()
    path = os.path.join(out_dir, "pareto_fp_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Section E: Convergence analysis
# ======================================================================

def plot_convergence_grid(gen_stats, out_dir):
    if gen_stats is None:
        print("  Skipping convergence (no gen stats)")
        return None

    gen_col = None
    for candidate in ["generation", "Generation", "gen"]:
        if candidate in gen_stats.columns:
            gen_col = candidate
            break
    if gen_col is None:
        print("  WARNING: no generation column in gen stats")
        return None

    fom1_col = None
    for candidate in ["FitnessScore_max", "fom1_40C_max", "best_fom1"]:
        if candidate in gen_stats.columns:
            fom1_col = candidate
            break
    if fom1_col is None:
        print(f"  WARNING: no FOM1 max column in gen stats. Columns: {list(gen_stats.columns)}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    convergence_info = []

    for idx, fp in enumerate(FP_LABELS):
        ax = axes_flat[idx]
        for lv in LEVEL_ORDER:
            sub = gen_stats[(gen_stats["fp_label"] == fp) &
                            (gen_stats["cost_level"] == lv)]
            if sub.empty:
                continue

            curves = []
            best_gens = []
            for seed in SEEDS:
                ssub = sub[sub["seed"] == seed].sort_values(gen_col)
                if ssub.empty:
                    continue
                running_best = ssub[fom1_col].cummax()
                curves.append(running_best.values)

                best_gen = ssub.loc[running_best.idxmax(), gen_col]
                best_gens.append(best_gen)

            if not curves:
                continue

            max_len = max(len(c) for c in curves)
            padded = np.full((len(curves), max_len), np.nan)
            for i, c in enumerate(curves):
                padded[i, :len(c)] = c

            mean_curve = np.nanmean(padded, axis=0)
            std_curve = np.nanstd(padded, axis=0)
            gens = np.arange(max_len)

            ax.plot(gens, mean_curve, color=LEVEL_COLORS[lv],
                    linewidth=1.5, label=LEVEL_LABELS[lv])
            ax.fill_between(gens, mean_curve - std_curve,
                            mean_curve + std_curve,
                            color=LEVEL_COLORS[lv], alpha=0.15)

            convergence_info.append({
                "fp_label": fp,
                "cost_level": lv,
                "best_gen_mean": np.mean(best_gens),
                "best_gen_std": np.std(best_gens),
                "mols_to_best_mean": np.mean(best_gens) * 3000,
            })

        ax.axhline(BASELINE_BEST_FOM1, color="#E63946", linestyle="--",
                   linewidth=1, alpha=0.5)
        ax.annotate(f"FP \u2265 {FP_VALUES_C[fp]} \u00b0C",
                    xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", va="top")
        if idx >= 2:
            ax.set_xlabel("Generation")
        if idx % 2 == 0:
            ax.set_ylabel("Best FOM1")
        if idx == 0:
            ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    if convergence_info:
        conv_df = pd.DataFrame(convergence_info)
        return conv_df
    return None


def plot_convergence_heatmap(conv_df, out_dir):
    if conv_df is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    grid = np.full((len(FP_LABELS), len(LEVEL_ORDER)), np.nan)
    annot = np.empty((len(FP_LABELS), len(LEVEL_ORDER)), dtype=object)

    for _, row in conv_df.iterrows():
        i = FP_LABELS.index(row["fp_label"])
        j = LEVEL_ORDER.index(row["cost_level"])
        grid[i, j] = row["best_gen_mean"]
        annot[i, j] = f"Gen {row['best_gen_mean']:.0f}\n\u00b1{row['best_gen_std']:.0f}"

    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="upper")
    for i in range(len(FP_LABELS)):
        for j in range(len(LEVEL_ORDER)):
            if annot[i, j] is not None:
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        fontsize=9)

    ax.set_xticks(range(len(LEVEL_ORDER)))
    ax.set_xticklabels([LEVEL_LABELS[l] for l in LEVEL_ORDER])
    ax.set_yticks(range(len(FP_LABELS)))
    ax.set_yticklabels([f"FP \u2265 {FP_VALUES_C[f]} \u00b0C" for f in FP_LABELS])
    ax.set_xlabel("MolPrice regime")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Generation of best molecule")

    fig.tight_layout()
    path = os.path.join(out_dir, "convergence_gen_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Section F: Top molecules
# ======================================================================

def extract_top_molecules(df, n=50):
    deduped = df.sort_values("fom1_40C", ascending=False)
    deduped = deduped.drop_duplicates(subset=["SMILES"], keep="first")
    return deduped.head(n).copy()


# ======================================================================
# OOD analysis
# ======================================================================

def compute_ood_rates(df, label="all valid"):
    """Compute OOD rate per model."""
    available = [c for c in ALL_OOD_COLS if c in df.columns and not df[c].isna().all()]
    if not available:
        print(f"  No OOD columns found for {label}")
        return None

    rates = {}
    for col in available:
        rates[col] = df[col].mean()
    return rates


def plot_ood_rates(ood_all, ood_top50, out_dir):
    if ood_all is None or ood_top50 is None:
        print("  Skipping OOD rate plot (no data)")
        return

    cols = sorted(set(ood_all.keys()) & set(ood_top50.keys()),
                  key=lambda c: ALL_OOD_COLS.index(c) if c in ALL_OOD_COLS else 99)

    labels = [OOD_SHORT_NAMES.get(c, c.replace("OOD_", "")) for c in cols]
    vals_all = [ood_all[c] * 100 for c in cols]
    vals_top50 = [ood_top50[c] * 100 for c in cols]

    x = np.arange(len(cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, vals_all, width, color="#2A5F8F", alpha=0.8,
           label="All valid molecules")
    ax.bar(x + width / 2, vals_top50, width, color="#E76F51", alpha=0.8,
           label="Top-50 candidates")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("OOD rate / %")
    ax.legend(frameon=False)

    for i, (v1, v2) in enumerate(zip(vals_all, vals_top50)):
        ax.text(i - width / 2, v1 + 1, f"{v1:.0f}", ha="center", fontsize=7)
        ax.text(i + width / 2, v2 + 1, f"{v2:.0f}", ha="center", fontsize=7)

    fig.tight_layout()
    path = os.path.join(out_dir, "ood_rates_top50.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_ood_by_config(df):
    """OOD rate per (fp_label, cost_level) for critical models."""
    rows = []
    for fp in FP_LABELS:
        for lv in LEVEL_ORDER:
            sub = df[(df["fp_label"] == fp) & (df["cost_level"] == lv)]
            if sub.empty:
                continue
            row = {"fp_label": fp, "cost_level": lv, "n": len(sub)}
            for col in CRITICAL_OOD_COLS:
                if col in sub.columns:
                    row[col] = sub[col].mean()
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


# ======================================================================
# Candidate selection
# ======================================================================

def rank_candidates(df, out_dir):
    """Select and rank synthesis candidates."""
    deduped = df.sort_values("fom1_40C", ascending=False)
    deduped = deduped.drop_duplicates(subset=["SMILES"], keep="first")

    n_above = (deduped["fom1_40C"] > BASELINE_BEST_FOM1).sum()
    print(f"\n  Molecules above baseline ({BASELINE_BEST_FOM1}): {n_above}")

    above_baseline = deduped.head(max(50, n_above)).copy()
    print(f"  Ranking top {len(above_baseline)} candidates")

    critical_ood = [c for c in CRITICAL_OOD_COLS if c in above_baseline.columns]
    noncritical_ood = [c for c in ALL_OOD_COLS
                       if c in above_baseline.columns and c not in critical_ood]

    above_baseline["n_critical_ood"] = above_baseline[critical_ood].sum(axis=1) \
        if critical_ood else 0
    above_baseline["n_total_ood"] = above_baseline[
        [c for c in ALL_OOD_COLS if c in above_baseline.columns]].sum(axis=1) \
        if any(c in above_baseline.columns for c in ALL_OOD_COLS) else 0

    def assign_tier(row):
        fom1_ood = row.get("OOD_fom1_40C", 0)
        n_crit = row.get("n_critical_ood", 0)
        sc = row.get("SCScore", 99)
        mp = row.get("MolPrice", 99)
        if fom1_ood == 0 and n_crit == 0 and sc < 3 and mp < 5:
            return 1
        if fom1_ood == 0 and n_crit <= 1:
            return 2
        return 3

    above_baseline["tier"] = above_baseline.apply(assign_tier, axis=1)

    above_baseline = above_baseline.sort_values(
        ["tier", "fom1_40C"], ascending=[True, False])

    output_cols = [
        "SMILES", "fom1_40C", "fp", "bp", "mp", "dc",
        "MolPrice", "SCScore", "Tox21_Score", "Biodegradable",
        "tier", "n_critical_ood", "n_total_ood",
    ] + [c for c in ALL_OOD_COLS if c in above_baseline.columns] + [
        "fp_label", "cost_level",
    ]
    available_cols = [c for c in output_cols if c in above_baseline.columns]

    cand_path = os.path.join(out_dir, "candidates_ranked.csv")
    above_baseline[available_cols].to_csv(cand_path, index=False)
    print(f"  Saved: {cand_path}")

    for tier in [1, 2, 3]:
        n = (above_baseline["tier"] == tier).sum()
        print(f"    Tier {tier}: {n} candidates")

    return above_baseline


def plot_candidate_grid(candidates, out_dir, n=8):
    """Molecule grid of top Tier 1/2 candidates."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        print("  Skipping candidate grid (RDKit not available)")
        return

    top = candidates[candidates["tier"] <= 2].head(n)
    if top.empty:
        top = candidates.head(n)

    mols = []
    legends = []
    for _, row in top.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        mols.append(mol)
        props = (f"FOM1={row['fom1_40C']:.1f}"
                 f"  FP={row.get('fp', 0):.0f}K"
                 f"  MP={row.get('MolPrice', 0):.1f}"
                 f"  SC={row.get('SCScore', 0):.1f}"
                 f"  T{int(row.get('tier', 0))}")
        legends.append(props)

    if not mols:
        return

    n_cols = min(4, len(mols))
    n_rows = (len(mols) + n_cols - 1) // n_cols
    img = Draw.MolsToGridImage(
        mols, molsPerRow=n_cols, subImgSize=(400, 300), legends=legends)

    path = os.path.join(out_dir, "candidates_grid.png")
    img.save(path)
    print(f"  Saved: {path}")


# ======================================================================
# MolPrice distribution
# ======================================================================

def plot_molprice_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = []
    data = []
    labels = []
    colors_list = []
    pos = 0

    for fp in FP_LABELS:
        for lv in LEVEL_ORDER:
            sub = df[(df["fp_label"] == fp) & (df["cost_level"] == lv)]
            if sub.empty:
                continue
            top10 = sub.nlargest(10, "fom1_40C")
            if top10["MolPrice"].isna().all():
                continue
            data.append(top10["MolPrice"].dropna().values)
            positions.append(pos)
            labels.append(f"{FP_VALUES_C[fp]}\u00b0C\n{lv}")
            colors_list.append(LEVEL_COLORS[lv])
            pos += 1

    if not data:
        plt.close(fig)
        return

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("MolPrice / log($ mmol\u207b\u00b9)")
    fig.tight_layout()
    path = os.path.join(out_dir, "molprice_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Training set cross-reference
# ======================================================================

def check_training_overlap(top_mols, baseline_df):
    """Check if any top molecules appear in the training set."""
    if baseline_df is None:
        return

    smiles_col = "SMILES" if "SMILES" in baseline_df.columns else "smiles"
    if smiles_col not in baseline_df.columns:
        return

    training_smiles = set(baseline_df[smiles_col].dropna().str.strip())
    top_smiles = set(top_mols["SMILES"].dropna().str.strip())
    overlap = training_smiles & top_smiles

    if overlap:
        print(f"\n  WARNING: {len(overlap)} top molecules found in training set!")
        for smi in list(overlap)[:5]:
            print(f"    {smi}")
    else:
        print(f"\n  No overlap between top molecules and training set")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyse FP x MolPrice sweep")
    parser.add_argument("--sweep_dir", type=str,
                        default="outputs/fp_molprice_sweep_nist8100")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    out_dir = os.path.join(sweep_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("FP x MolPrice Sweep Analysis")
    print("=" * 60)

    # Load data
    df = load_molecules(sweep_dir)
    gen_stats = load_gen_stats(sweep_dir)
    baseline_df = load_baseline()

    # Backfill MolPrice for nocost runs
    print("\nBackfilling MolPrice for nocost runs...")
    df = backfill_molprice(df)

    # -- Section A: Summary table --
    print("\n--- Section A: Summary Table ---")
    summary = compute_summary(df)
    summary_path = os.path.join(out_dir, "summary_by_config.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")
    print(summary.to_string(index=False))

    total_unique = df.drop_duplicates(subset=["SMILES"]).shape[0]
    if "total_evaluated" in df.columns:
        per_run = df.groupby(["fp_label", "cost_level", "seed"])["total_evaluated"].first()
        total_evaluated = per_run.sum()
    else:
        total_evaluated = "N/A"
    print(f"\n  Total unique valid molecules across all runs: {total_unique:,}")
    print(f"  Total molecules evaluated (incl. invalid): {total_evaluated:,.0f}")

    # -- Section B: Hypervolume --
    print("\n--- Section B: Hypervolume Analysis ---")
    hv_df = compute_hv_grid(df)
    if not hv_df.empty:
        hv_path = os.path.join(out_dir, "hv_by_config.csv")
        hv_df.to_csv(hv_path, index=False)
        print(f"  Saved: {hv_path}")
        print(hv_df.to_string(index=False))
        plot_hv_heatmap(hv_df, out_dir)
        plot_hv_vs_fp(hv_df, out_dir)

    # -- Section C: FOM1 vs FP --
    print("\n--- Section C: FOM1 vs FP Trade-off ---")
    plot_fom1_vs_fp(summary, out_dir)

    # -- Section D: Pareto front grid --
    print("\n--- Section D: Pareto Front Grid ---")
    plot_pareto_grid(df, out_dir)

    # -- Section E: Convergence --
    print("\n--- Section E: Convergence Analysis ---")
    conv_df = plot_convergence_grid(gen_stats, out_dir)
    plot_convergence_heatmap(conv_df, out_dir)
    if conv_df is not None:
        conv_path = os.path.join(out_dir, "convergence_by_config.csv")
        conv_df.to_csv(conv_path, index=False)
        print(f"  Saved: {conv_path}")
        print(conv_df.to_string(index=False))

    # -- Section F: Top molecules --
    print("\n--- Section F: Top Molecules ---")
    top50 = extract_top_molecules(df, n=50)
    top_path = os.path.join(out_dir, "top_molecules.csv")
    top50.to_csv(top_path, index=False)
    print(f"  Saved: {top_path}")
    print(f"  Top 5:")
    for _, row in top50.head(5).iterrows():
        print(f"    FOM1={row['fom1_40C']:.1f}  FP={row.get('fp', 0):.0f}K"
              f"  MP={row.get('MolPrice', 'N/A')}  {row['SMILES'][:60]}")

    check_training_overlap(top50, baseline_df)

    # -- OOD Analysis --
    print("\n--- OOD Analysis ---")
    ood_all = compute_ood_rates(df, "all valid")
    ood_top50 = compute_ood_rates(top50, "top 50")
    if ood_all:
        print("  OOD rates (all valid molecules):")
        for col, rate in sorted(ood_all.items()):
            print(f"    {OOD_SHORT_NAMES.get(col, col):<12} {rate*100:.1f}%")
    if ood_top50:
        print("  OOD rates (top 50):")
        for col, rate in sorted(ood_top50.items()):
            print(f"    {OOD_SHORT_NAMES.get(col, col):<12} {rate*100:.1f}%")
    plot_ood_rates(ood_all, ood_top50, out_dir)

    ood_config = compute_ood_by_config(df)
    if ood_config is not None:
        ood_config_path = os.path.join(out_dir, "ood_rates_by_config.csv")
        ood_config.to_csv(ood_config_path, index=False)
        print(f"  Saved: {ood_config_path}")

    # -- Candidate Selection --
    print("\n--- Candidate Selection ---")
    candidates = rank_candidates(df, out_dir)
    plot_candidate_grid(candidates, out_dir)

    # -- MolPrice Distribution --
    print("\n--- MolPrice Distribution ---")
    plot_molprice_distribution(df, out_dir)

    print(f"\n{'='*60}")
    print(f"Analysis complete. Outputs: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
