"""
Baseline FOM1 Evaluation — NIST 8,100-molecule dataset

Evaluates the new NIST dataset to establish baseline FOM1 performance:

1. Load pre-computed FOM1 from training/data/nist_8100/fom1_cho_cleaned.csv
   (7,380 molecules with quadratic beta, Sat Cp, computed FOM1)
2. Load multi-temperature experimental properties (density, viscosity, TC, Cp)
3. Predict safety constraint properties (MP, FP, BP, DC, Tox21, Biodeg)
   using XGBoost models where available; use NIST experimental values where not
4. FOM1 distribution plots for bio and no-bio subsets
5. FP/MP constraint heatmaps (2x2 grids): number of valid molecules
   and best FOM1 for different flash point and melting point thresholds
6. Top molecules table for each constraint combination

Usage:
    cd BaselineFOM1Eval/
    python baseline_fom1_eval.py
    python baseline_fom1_eval.py --out_dir ./custom_output
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", message=".*PerformanceWarning.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Compatibility fix: models saved with newer numpy/sklearn may fail to
# deserialize numpy RandomState objects on older versions.
import numpy.random as _nr

def _randomstate_ctor(bit_generator_name="MT19937"):
    return np.random.RandomState()

_nr.__dict__["__RandomState_ctor"] = _randomstate_ctor
np.random.__dict__["__RandomState_ctor"] = _randomstate_ctor

# Compatibility fix: older RDKit versions use np.float / np.int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# Add src/ to path for model loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsga_helper import (
    load_regression_models_with_aux,
    load_tox21_predictor,
    load_biodeg_model,
    evaluate_molecules,
)
from SCScorer import SCScorer


# ======================================================================
# Configuration
# ======================================================================

# All models (Mordred+Optuna pipeline) — need thermo models for derived properties
ALL_TARGETS = [
    "bp", "mp", "dc", "fp",
    "density_40C", "viscosity_40C", "tc_40C", "cpsat_40C", "beta_40C",
    "fom1_40C",
]

# FP/MP sweep grids (flash point in degC to match old convention)
FP_VALUES_C = list(range(50, 180, 10))   # [50, 60, ..., 170]
MP_VALUES_C = list(range(-50, 25, 5))    # [-50, -45, ..., 20]

# Convert FP to Kelvin for internal use (model predicts in K)
FP_VALUES_K = [t + 273.15 for t in FP_VALUES_C]

# Additional hard constraints
DC_THRESHOLD = 7.0
BP_THRESHOLD = 100.0  # degC
TOX_THRESHOLD = 3.0


# ======================================================================
# Data loading
# ======================================================================

def load_nist_dataset(nist_dir):
    """Load and merge the NIST 8,100-molecule multi-temp datasets.

    Returns a DataFrame with FOM1, thermophysical properties at 40C and 100C,
    and molecular weight.
    """
    # Primary: FOM1 dataset (7,380 molecules with pre-computed FOM1)
    fom1_path = os.path.join(nist_dir, "fom1_cho_cleaned.csv")
    df = pd.read_csv(fom1_path)
    print(f"  FOM1 dataset: {len(df)} molecules")

    # Rename FOM1 column for consistency
    df = df.rename(columns={"fom1_40C": "FOM1_exp_40"})

    # Keep only SMILES, name, FOM1, and the component properties
    keep_cols = ["SMILES", "name", "FOM1_exp_40",
                 "density_40C_g_cm3", "kv_40C_cSt", "tc_40C_W_mK",
                 "cpsat_40C_J_K_mol", "beta_40C", "MolWt"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Load density for 100C (for FOM1_100C computation)
    density_path = os.path.join(nist_dir, "density_cho_cleaned.csv")
    if os.path.exists(density_path):
        dens = pd.read_csv(density_path)[["SMILES", "density_100C_g_cm3"]]
        df = df.merge(dens, on="SMILES", how="left")

    # Load viscosity for 100C
    visc_path = os.path.join(nist_dir, "viscosity_cho_cleaned.csv")
    if os.path.exists(visc_path):
        visc = pd.read_csv(visc_path)[["SMILES", "kv_100C_cSt"]]
        df = df.merge(visc, on="SMILES", how="left")

    # Load TC for 100C
    tc_path = os.path.join(nist_dir, "tc_cho_cleaned.csv")
    if os.path.exists(tc_path):
        tc = pd.read_csv(tc_path)[["SMILES", "tc_100C_W_mK"]]
        df = df.merge(tc, on="SMILES", how="left")

    # Load Cp for 100C
    cp_path = os.path.join(nist_dir, "cpsat_cho_cleaned.csv")
    if os.path.exists(cp_path):
        cp = pd.read_csv(cp_path)[["SMILES", "cpsat_100C_J_K_mol"]]
        df = df.merge(cp, on="SMILES", how="left")

    # Load beta (for 100C FOM1)
    beta_path = os.path.join(nist_dir, "beta_cho_cleaned.csv")
    if os.path.exists(beta_path):
        beta_df = pd.read_csv(beta_path)
        # Merge beta_40C if not already in df (from FOM1 file)
        if "beta_40C" not in df.columns and "beta_40C" in beta_df.columns:
            df = df.merge(beta_df[["SMILES", "beta_40C"]], on="SMILES", how="left")
        # Also merge beta_100C for FOM1_100C computation
        if "beta_100C" in beta_df.columns:
            df = df.merge(beta_df[["SMILES", "beta_100C"]], on="SMILES", how="left")

    return df


def compute_fom1_100c(df):
    """Compute FOM1 at 100C from component properties."""
    # Use beta_100C if available, fall back to beta_40C
    beta_col = "beta_100C" if "beta_100C" in df.columns else "beta_40C"
    required = ["tc_100C_W_mK", "density_100C_g_cm3", "cpsat_100C_J_K_mol",
                 "kv_100C_cSt", beta_col, "MolWt"]
    mask = df[required].notna().all(axis=1)

    k = df["tc_100C_W_mK"]
    Cp = df["cpsat_100C_J_K_mol"] / df["MolWt"] * 1000  # J/(kg K)
    nu = df["kv_100C_cSt"] * 1e-6  # m^2/s
    rho = df["density_100C_g_cm3"]
    beta = df[beta_col]
    rho_kg = rho * 1000

    fom1_100 = k * ((beta * Cp * rho_kg) / (nu * k)) ** 0.2813
    df["FOM1_exp_100"] = np.where(mask, fom1_100, np.nan)

    # Average
    df["FOM1_exp_avg"] = np.where(
        df["FOM1_exp_40"].notna() & df["FOM1_exp_100"].notna(),
        (df["FOM1_exp_40"] + df["FOM1_exp_100"]) / 2,
        df["FOM1_exp_40"]  # fallback to 40C only
    )
    return df


# ======================================================================
# Constraint filtering
# ======================================================================

def apply_fp_mp_filter(df, fp_k, mp_c, use_biodeg):
    """Filter by flash point (K) and melting point (degC) + biodeg."""
    mask = (df["fp"] >= fp_k) & (df["mp"] <= mp_c)
    if use_biodeg:
        mask = mask & (df["Biodegradable"] == True)
    return df[mask]


def apply_all_constraints(df, fp_k, mp_c, use_biodeg):
    """Filter by all constraints: FP, MP, BP, DC, Tox + biodeg."""
    mask = (
        (df["fp"] >= fp_k) &
        (df["mp"] <= mp_c) &
        (df["bp"] >= BP_THRESHOLD) &
        (df["dc"] <= DC_THRESHOLD) &
        (df["Tox21_Score"] <= TOX_THRESHOLD)
    )
    if use_biodeg:
        mask = mask & (df["Biodegradable"] == True)
    return df[mask]


# ======================================================================
# Plotting
# ======================================================================

def plot_fom1_distributions(df, out_dir):
    """Plot FOM1 distributions for bio and no-bio subsets."""

    bio_mask = df["Biodegradable"] == True
    bio_fom = df.loc[bio_mask, "FOM1_exp_avg"].dropna()
    all_fom = df["FOM1_exp_avg"].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: all molecules
    ax = axes[0]
    ax.hist(all_fom, bins=50, color="#4C72B0", alpha=0.8,
            edgecolor="black", linewidth=0.5)
    ax.axvline(all_fom.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {all_fom.mean():.2f}")
    ax.axvline(all_fom.median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {all_fom.median():.2f}")
    ax.set_xlabel("FOM1 / W m$^{-1}$ K$^{-1}$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(frameon=False, fontsize=9)

    # Panel 2: biodegradable only
    ax = axes[1]
    if len(bio_fom) > 0:
        ax.hist(bio_fom, bins=40, color="#55A868", alpha=0.8,
                edgecolor="black", linewidth=0.5)
        ax.axvline(bio_fom.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean = {bio_fom.mean():.2f}")
        ax.axvline(bio_fom.median(), color="orange", linestyle="--", linewidth=1.5,
                   label=f"Median = {bio_fom.median():.2f}")
    ax.set_xlabel("FOM1 / W m$^{-1}$ K$^{-1}$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(frameon=False, fontsize=9)

    # Panel 3: overlay
    ax = axes[2]
    ax.hist(all_fom, bins=50, alpha=0.5, color="#4C72B0",
            edgecolor="black", linewidth=0.3, label=f"All (n={len(all_fom)})")
    if len(bio_fom) > 0:
        ax.hist(bio_fom, bins=40, alpha=0.5, color="#55A868",
                edgecolor="black", linewidth=0.3, label=f"Biodeg (n={len(bio_fom)})")
    ax.set_xlabel("FOM1 / W m$^{-1}$ K$^{-1}$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "fom1_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _build_heatmap_grids(df, filter_func):
    """Build n_valid and best_fom1 grids for both bio and no-bio."""
    results = {}
    for use_biodeg in [True, False]:
        tag = "bio" if use_biodeg else "nobio"
        n_valid = np.full((len(MP_VALUES_C), len(FP_VALUES_K)), np.nan)
        best_fom1 = np.full((len(MP_VALUES_C), len(FP_VALUES_K)), np.nan)

        for i, mp in enumerate(MP_VALUES_C):
            for j, fp in enumerate(FP_VALUES_K):
                valid = filter_func(df, fp, mp, use_biodeg)
                n_valid[i, j] = len(valid)
                if len(valid) > 0 and "FOM1_exp_avg" in valid.columns:
                    fom_vals = valid["FOM1_exp_avg"].dropna()
                    if len(fom_vals) > 0:
                        best_fom1[i, j] = fom_vals.max()

        results[tag] = {"n_valid": n_valid, "best_fom1": best_fom1}
    return results


def plot_2x2_heatmaps(results, out_dir, filename):
    """Plot a 2x2 seaborn heatmap grid.

    Layout:
        (a) Bio Best FOM1       (b) Bio Molecule Count
        (c) NoBio Best FOM1     (d) NoBio Molecule Count
    """
    bio_fom1 = results["bio"]["best_fom1"]
    bio_count = results["bio"]["n_valid"]
    nobio_fom1 = results["nobio"]["best_fom1"]
    nobio_count = results["nobio"]["n_valid"]

    mp_labels = [f"{mp:.0f}" for mp in MP_VALUES_C]
    fp_labels = [f"{fp:.0f}" for fp in FP_VALUES_C]

    df_fom1_bio = pd.DataFrame(bio_fom1, index=mp_labels, columns=fp_labels)
    df_count_bio = pd.DataFrame(bio_count, index=mp_labels, columns=fp_labels)
    df_fom1_nobio = pd.DataFrame(nobio_fom1, index=mp_labels, columns=fp_labels)
    df_count_nobio = pd.DataFrame(nobio_count, index=mp_labels, columns=fp_labels)

    for d in [df_fom1_bio, df_count_bio, df_fom1_nobio, df_count_nobio]:
        d.sort_index(ascending=False, inplace=True,
                     key=lambda x: x.astype(int))

    fom1_vmin = min(np.nanmin(bio_fom1), np.nanmin(nobio_fom1))
    fom1_vmax = max(np.nanmax(bio_fom1), np.nanmax(nobio_fom1))
    count_vmax = max(np.nanmax(bio_count), np.nanmax(nobio_count))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.heatmap(df_fom1_bio, annot=True, fmt=".0f", cmap="RdYlGn",
                ax=axes[0, 0], vmin=fom1_vmin, vmax=fom1_vmax,
                cbar_kws={"label": "Best FOM1 / W m$^{-1}$ K$^{-1}$"})
    axes[0, 0].set_xlabel("Min Flash Point / \u00b0C")
    axes[0, 0].set_ylabel("Max Melting Point / \u00b0C")
    axes[0, 0].text(-0.15, 1.02, "(a) Biodegradable", transform=axes[0, 0].transAxes,
                    fontsize=12, va="bottom")

    sns.heatmap(df_count_bio, annot=True, fmt=".0f", cmap="Blues",
                ax=axes[0, 1], vmin=0, vmax=count_vmax,
                cbar_kws={"label": "Number of Molecules"})
    axes[0, 1].set_xlabel("Min Flash Point / \u00b0C")
    axes[0, 1].set_ylabel("Max Melting Point / \u00b0C")
    axes[0, 1].text(-0.15, 1.02, "(b) Biodegradable", transform=axes[0, 1].transAxes,
                    fontsize=12, va="bottom")

    sns.heatmap(df_fom1_nobio, annot=True, fmt=".0f", cmap="RdYlGn",
                ax=axes[1, 0], vmin=fom1_vmin, vmax=fom1_vmax,
                cbar_kws={"label": "Best FOM1 / W m$^{-1}$ K$^{-1}$"})
    axes[1, 0].set_xlabel("Min Flash Point / \u00b0C")
    axes[1, 0].set_ylabel("Max Melting Point / \u00b0C")
    axes[1, 0].text(-0.15, 1.02, "(c) No Biodeg Filter", transform=axes[1, 0].transAxes,
                    fontsize=12, va="bottom")

    sns.heatmap(df_count_nobio, annot=True, fmt=".0f", cmap="Blues",
                ax=axes[1, 1], vmin=0, vmax=count_vmax,
                cbar_kws={"label": "Number of Molecules"})
    axes[1, 1].set_xlabel("Min Flash Point / \u00b0C")
    axes[1, 1].set_ylabel("Max Melting Point / \u00b0C")
    axes[1, 1].text(-0.15, 1.02, "(d) No Biodeg Filter", transform=axes[1, 1].transAxes,
                    fontsize=12, va="bottom")

    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_top_molecules(df, filter_func, out_dir, use_biodeg, fp_k, mp_c, n_top=10):
    """Print and save the top molecules for given constraints."""
    tag = "bio" if use_biodeg else "nobio"
    label = "Biodegradable" if use_biodeg else "No Biodeg Filter"
    fp_c = fp_k - 273.15

    valid = filter_func(df, fp_k, mp_c, use_biodeg)
    valid = valid.sort_values("FOM1_exp_avg", ascending=False)
    valid = valid.drop_duplicates(subset=["SMILES"], keep="first")

    print(f"\n  Top {n_top} molecules ({label}, FP>={fp_c:.0f}\u00b0C, MP<={mp_c}\u00b0C):")
    print(f"  {'Rank':<5} {'SMILES':<45} {'FOM1_avg':>10} {'FOM1_40':>10} "
          f"{'MP':>7} {'FP':>8} {'Bio':>4}")
    print(f"  {'-'*95}")

    for i, (_, mol) in enumerate(valid.head(n_top).iterrows(), 1):
        smi = str(mol.get("SMILES", ""))
        if len(smi) > 42:
            smi = smi[:39] + "..."
        bio = "Y" if mol.get("Biodegradable", False) else "N"
        fp_display = mol.get("fp", np.nan) - 273.15  # K to C
        print(
            f"  {i:<5} {smi:<45} "
            f"{mol.get('FOM1_exp_avg', np.nan):>10.2f} "
            f"{mol.get('FOM1_exp_40', np.nan):>10.2f} "
            f"{mol.get('mp', np.nan):>7.1f} "
            f"{fp_display:>8.1f} "
            f"{bio:>4}"
        )

    # Save CSV
    save_cols = ["SMILES", "name", "FOM1_exp_avg", "FOM1_exp_40", "FOM1_exp_100",
                 "mp", "fp", "bp", "dc",
                 "Tox21_Score", "Biodegradable"]
    save_cols = [c for c in save_cols if c in valid.columns]
    out_path = os.path.join(out_dir, f"top_molecules_{tag}.csv")
    valid.head(50)[save_cols].to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# ======================================================================
# Main
# ======================================================================

def main(out_dir, nist_dir, model_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load NIST dataset
    # ------------------------------------------------------------------
    print("Loading NIST 8,100-molecule dataset...")
    df = load_nist_dataset(nist_dir)
    print(f"  {len(df)} molecules with FOM1")

    # Deduplicate by SMILES
    n_before = len(df)
    df = df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    if n_before != len(df):
        print(f"  Deduplicated: {n_before} -> {len(df)}")

    # ------------------------------------------------------------------
    # 2. Compute FOM1 at 100C and average
    # ------------------------------------------------------------------
    print("\nComputing FOM1 at 100C...")
    df = compute_fom1_100c(df)
    n_100c = df["FOM1_exp_100"].notna().sum()
    print(f"  FOM1_100C computed for {n_100c} / {len(df)} molecules")

    fom1_valid = df["FOM1_exp_avg"].dropna()
    print(f"  FOM1_avg: Mean={fom1_valid.mean():.2f}, "
          f"Median={fom1_valid.median():.2f}, Max={fom1_valid.max():.2f}")

    # ------------------------------------------------------------------
    # 3. Predict safety constraint properties
    # ------------------------------------------------------------------
    print("\nLoading safety/constraint models...")

    # Check which models are available
    available_targets = []
    for target in ALL_TARGETS:
        model_path = os.path.join(model_dir, target, "model", "xgb_model.joblib")
        if os.path.exists(model_path):
            available_targets.append(target)
        else:
            print(f"  WARNING: Model not found for {target}")

    if available_targets:
        thermo_models = load_regression_models_with_aux(available_targets, model_dir)
    else:
        thermo_models = {}

    sc_model = SCScorer()
    sc_ckpt = os.path.join(model_dir, "SCScorer/scscore/models/"
                           "full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz")
    if os.path.exists(sc_ckpt):
        sc_model.restore(sc_ckpt)
    else:
        print("  WARNING: SCScorer checkpoint not found")
        sc_model = None

    tox21_dir = os.path.join(model_dir, "tox21_gt4sd")
    tox21_models = None
    if os.path.isdir(tox21_dir):
        try:
            tox21_models = load_tox21_predictor(tox21_dir)
        except Exception as e:
            print(f"  WARNING: Could not load Tox21 models: {e}")

    biodeg_model = None
    biodeg_dir = os.path.join(model_dir, "biodeg")
    if os.path.isdir(biodeg_dir):
        try:
            biodeg_model = load_biodeg_model(biodeg_dir)
        except Exception as e:
            print(f"  WARNING: Could not load biodeg model: {e}")

    print("\nPredicting constraint properties...")
    pred_df = evaluate_molecules(
        pd.DataFrame({"SMILES": df["SMILES"]}),
        thermo_models=thermo_models,
        sc_model=sc_model,
        tox21_models=tox21_models,
        biodeg_model=biodeg_model,
        drop_descriptors=True,
    )
    print(f"  {len(pred_df)} molecules evaluated")

    # Merge predicted constraint columns
    constraint_cols = ["SMILES", "mp", "fp", "bp",
                       "dc", "Tox21_Score", "Biodegradable", "SCScore"]
    constraint_cols = [c for c in constraint_cols if c in pred_df.columns]
    df = df.merge(pred_df[constraint_cols], on="SMILES", how="left")

    # Deduplicate after merge
    df = df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)

    n_bio = (df.get("Biodegradable", pd.Series(dtype=bool)) == True).sum()
    print(f"  Biodegradable: {n_bio} / {len(df)}")
    if "fp" in df.columns:
        fp = df["fp"].dropna()
        print(f"  FP range: {fp.min():.1f} \u2013 {fp.max():.1f} K "
              f"({fp.min()-273.15:.0f} \u2013 {fp.max()-273.15:.0f} \u00b0C)")
    if "mp" in df.columns:
        mp = df["mp"].dropna()
        print(f"  MP range: {mp.min():.1f} \u2013 {mp.max():.1f} \u00b0C")

    # ------------------------------------------------------------------
    # 4. FOM1 distribution plots
    # ------------------------------------------------------------------
    print(f"\nGenerating plots...")
    plot_fom1_distributions(df, out_dir)

    # ------------------------------------------------------------------
    # 5. FP/MP constraint heatmaps (FP + MP only)
    # ------------------------------------------------------------------
    if "fp" in df.columns and "mp" in df.columns:
        print(f"\nFP/MP heatmaps (FP + MP constraints only)...")
        results_simple = _build_heatmap_grids(df, apply_fp_mp_filter)
        plot_2x2_heatmaps(results_simple, out_dir, "heatmap_fpmp.png")

        # ------------------------------------------------------------------
        # 6. FP/MP constraint heatmaps (all constraints)
        # ------------------------------------------------------------------
        if all(c in df.columns for c in ["bp", "dc", "Tox21_Score"]):
            print(f"\nFP/MP heatmaps (all constraints: FP, MP, BP>={BP_THRESHOLD}, "
                  f"DC<={DC_THRESHOLD}, Tox<={TOX_THRESHOLD})...")
            results_all = _build_heatmap_grids(df, apply_all_constraints)
            plot_2x2_heatmaps(results_all, out_dir, "heatmap_all_constraints.png")
        else:
            print("\n  Skipping all-constraints heatmap (missing model predictions)")

        # ------------------------------------------------------------------
        # 7. Top molecules tables
        # ------------------------------------------------------------------
        default_fp = 373.15
        default_mp = -10

        print(f"\n--- Top molecules (FP + MP only) ---")
        for use_biodeg in [True, False]:
            print_top_molecules(df, apply_fp_mp_filter, out_dir, use_biodeg,
                                default_fp, default_mp)

        if all(c in df.columns for c in ["bp", "dc", "Tox21_Score"]):
            print(f"\n--- Top molecules (all constraints) ---")
            for use_biodeg in [True, False]:
                print_top_molecules(df, apply_all_constraints, out_dir, use_biodeg,
                                    default_fp, default_mp)
    else:
        print("\n  Skipping constraint analysis (safety models not available)")

    # ------------------------------------------------------------------
    # 8. Save full results
    # ------------------------------------------------------------------
    save_cols = ["SMILES", "name", "FOM1_exp_40", "FOM1_exp_100", "FOM1_exp_avg",
                 "density_40C_g_cm3", "kv_40C_cSt", "tc_40C_W_mK",
                 "cpsat_40C_J_K_mol", "beta_40C", "MolWt"]
    for c in constraint_cols:
        if c != "SMILES" and c not in save_cols:
            save_cols.append(c)
    save_cols = [c for c in save_cols if c in df.columns]

    out_csv = os.path.join(out_dir, "baseline_fom1_results.csv")
    df[save_cols].to_csv(out_csv, index=False)
    print(f"\nFull results saved to: {out_csv}")
    print(f"  {len(df)} molecules, {len(save_cols)} columns")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline FOM1 Evaluation (NIST 8,100)")
    parser.add_argument("--out_dir", type=str, default="./output",
                        help="Output directory for plots and CSVs")
    parser.add_argument("--nist_dir", type=str,
                        default="../training/data/nist_8100",
                        help="NIST 8,100-molecule dataset directory")
    parser.add_argument("--model_dir", type=str, default="../models",
                        help="Pre-trained model directory")
    args = parser.parse_args()

    main(args.out_dir, args.nist_dir, args.model_dir)
