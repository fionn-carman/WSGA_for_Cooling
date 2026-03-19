#!/usr/bin/env python3
"""
Train single-temp XGBoost models at 40°C and compare direct vs pipeline FOM1 prediction.

- Step 3: Train 5-fold models for density, viscosity, tc, cpsat, FOM1_improved (all 40°C)
- Step 4: Train temp-feature density model → predicted beta from OOF density at 11 temps
- Step 5: Compare direct FOM1 vs pipeline FOM1 (assembled from predicted components)

Requires: data/FOM1_improved_40C.csv (from compute_improved_fom1.py)

Usage:
    python train_pipeline_comparison.py
"""

import json
import logging
import re
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUT_DIR = Path("pipeline_comparison")
DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]
TEMPS_C = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

FIXED_PARAMS = {
    "n_estimators": 600, "max_depth": 4, "learning_rate": 0.1,
    "subsample": 1.0, "colsample_bytree": 0.7,
    "min_child_weight": 10, "reg_alpha": 0, "reg_lambda": 0.1,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def is_glyme(smi):
    """Detect glymes: ≥3 ether linkages (OD2 bonded to 2 carbons), no carbonyl."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    carbonyl = Chem.MolFromSmarts("[#6]=[#8]")
    if mol.HasSubstructMatch(carbonyl):
        return False
    ether = Chem.MolFromSmarts("[OD2]([#6])[#6]")
    matches = mol.GetSubstructMatches(ether)
    return len(matches) >= 3


def get_finite_feature_mask(X):
    """Return boolean mask of columns that are finite everywhere."""
    return np.all(np.isfinite(X), axis=0)


def compute_fom1_from_components(rho_gcm3, kv_cSt, k_WmK, cp_molar_JKmol, beta_K, mw):
    """Compute FOM1 = k × ((β × Cp_specific × ρ_SI) / (ν_SI × k))^0.2813."""
    cp_specific = cp_molar_JKmol / mw * 1000  # J/(kg·K)
    rho_SI = rho_gcm3 * 1000                  # kg/m³
    nu_SI = kv_cSt * 1e-6                     # m²/s
    inner = (beta_K * cp_specific * rho_SI) / (nu_SI * k_WmK)
    return k_WmK * np.power(inner, 0.2813)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_intersection_data():
    """Load all component CSVs + FOM1_improved, return intersection dataset."""
    logger.info("Loading datasets...")

    density_df = pd.read_csv(DATA_DIR / "density_cho_cleaned.csv")
    viscosity_df = pd.read_csv(DATA_DIR / "viscosity_cho_cleaned.csv")
    tc_df = pd.read_csv(DATA_DIR / "tc_cho_cleaned.csv")
    cpsat_df = pd.read_csv(DATA_DIR / "cpsat_cho_cleaned.csv")
    fom1_df = pd.read_csv(DATA_DIR / "FOM1_improved_40C.csv")

    logger.info("Raw sizes: density=%d, viscosity=%d, tc=%d, cpsat=%d, FOM1_improved=%d",
                len(density_df), len(viscosity_df), len(tc_df), len(cpsat_df), len(fom1_df))

    # Intersection SMILES: must be in all 5 datasets
    smiles_sets = [
        set(density_df["SMILES"]),
        set(viscosity_df["SMILES"]),
        set(tc_df["SMILES"]),
        set(cpsat_df["SMILES"]),
        set(fom1_df["SMILES"]),
    ]
    intersection = sorted(set.intersection(*smiles_sets))
    logger.info("Intersection SMILES: %d", len(intersection))

    return {
        "density": density_df[density_df["SMILES"].isin(intersection)].copy(),
        "viscosity": viscosity_df[viscosity_df["SMILES"].isin(intersection)].copy(),
        "tc": tc_df[tc_df["SMILES"].isin(intersection)].copy(),
        "cpsat": cpsat_df[cpsat_df["SMILES"].isin(intersection)].copy(),
        "fom1": fom1_df[fom1_df["SMILES"].isin(intersection)].copy(),
        "intersection_smiles": intersection,
    }


def assign_folds(smiles_list, n_splits=5):
    """Deterministic fold assignment using GroupKFold on sorted SMILES."""
    sorted_smiles = sorted(smiles_list)
    dummy_X = np.arange(len(sorted_smiles)).reshape(-1, 1)
    dummy_y = np.zeros(len(sorted_smiles))

    gkf = GroupKFold(n_splits=n_splits)
    fold_map = {}
    for fold_i, (_, test_idx) in enumerate(gkf.split(dummy_X, dummy_y, sorted_smiles)):
        for idx in test_idx:
            fold_map[sorted_smiles[idx]] = fold_i

    return fold_map


# ---------------------------------------------------------------------------
# Single-temp model training
# ---------------------------------------------------------------------------

def train_single_temp(df, target_col, fold_map, log_transform, prop_name):
    """Train 5-fold single-temp XGBoost model. Returns OOF predictions aligned to df."""
    logger.info("Training single-temp model: %s (target=%s, log=%s)",
                prop_name, target_col, log_transform)

    # Drop NaN target rows
    mask = df[target_col].notna()
    df_valid = df[mask].copy().reset_index(drop=True)

    # Features: RDKit descriptors (no T_K for single-temp)
    desc_cols = [c for c in DESCRIPTOR_NAMES if c in df_valid.columns]
    X = df_valid[desc_cols].values.astype(float)
    y_raw = df_valid[target_col].values.astype(float)

    # Remove non-finite feature columns
    finite_mask = get_finite_feature_mask(X)
    if not np.all(finite_mask):
        logger.info("  Removing %d non-finite features", (~finite_mask).sum())
        desc_cols = [f for f, v in zip(desc_cols, finite_mask) if v]
        X = X[:, finite_mask]

    if log_transform:
        y = np.log1p(y_raw)
    else:
        y = y_raw.copy()

    smiles = df_valid["SMILES"].values
    folds = np.array([fold_map[s] for s in smiles])

    oof_preds = np.full(len(y), np.nan)
    models = []

    for fold_i in range(5):
        train_mask = folds != fold_i
        test_mask = folds == fold_i

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1, random_state=fold_i, verbosity=0,
            **FIXED_PARAMS,
        )
        model.fit(X_train_s, y_train)
        oof_preds[test_mask] = model.predict(X_test_s)
        models.append({"model": model, "scaler": scaler, "feature_cols": desc_cols})

    # Metrics
    if log_transform:
        y_real = np.expm1(y)
        pred_real = np.expm1(np.clip(oof_preds, 0, np.log1p(y_raw.max())))
    else:
        y_real = y
        pred_real = oof_preds

    r2 = r2_score(y_real, pred_real)
    rmse = np.sqrt(mean_squared_error(y_real, pred_real))
    mae = mean_absolute_error(y_real, pred_real)
    logger.info("  R²=%.4f, RMSE=%.4f, MAE=%.4f (n=%d)", r2, rmse, mae, len(y))

    return {
        "smiles": smiles,
        "y_true": y_real,
        "y_pred": pred_real,
        "r2": r2, "rmse": rmse, "mae": mae,
        "models": models,
        "n": len(y),
    }


# ---------------------------------------------------------------------------
# Temp-feature density model (for beta prediction)
# ---------------------------------------------------------------------------

def train_density_temp_feature(density_df, fold_map):
    """Train temp-feature density model. Return OOF predictions at ALL 11 temps."""
    logger.info("Training temp-feature density model for beta prediction...")

    density_cols = {f"density_{t}C_g_cm3": t + 273.15 for t in TEMPS_C}
    desc_cols = [c for c in DESCRIPTOR_NAMES if c in density_df.columns]

    # Melt to long format: (SMILES, T_K, density, descriptors)
    rows = []
    for _, row in density_df.iterrows():
        smi = row["SMILES"]
        descs = {d: row[d] for d in desc_cols}
        for col_name, t_k in density_cols.items():
            val = row[col_name]
            if pd.notna(val):
                r = {"SMILES": smi, "T_K": t_k, "density": val}
                r.update(descs)
                rows.append(r)
    long_df = pd.DataFrame(rows)
    logger.info("  Long format: %d rows from %d molecules",
                len(long_df), long_df["SMILES"].nunique())

    feature_cols = ["T_K"] + desc_cols
    X_all = long_df[feature_cols].values.astype(float)
    y_all = long_df["density"].values.astype(float)
    smiles_long = long_df["SMILES"].values

    # Remove non-finite features
    finite_mask = get_finite_feature_mask(X_all)
    if not np.all(finite_mask):
        logger.info("  Removing %d non-finite features", (~finite_mask).sum())
        feature_cols = [f for f, v in zip(feature_cols, finite_mask) if v]
        X_all = X_all[:, finite_mask]

    folds_long = np.array([fold_map[s] for s in smiles_long])

    # For OOF: predict at ALL 11 temps for each test molecule
    # Build prediction grid: for each unique SMILES, create 11 rows (one per temp)
    unique_smiles = sorted(density_df["SMILES"].unique())
    pred_grid_rows = []
    for smi in unique_smiles:
        row = density_df[density_df["SMILES"] == smi].iloc[0]
        descs = {d: row[d] for d in desc_cols}
        for t_k in sorted(density_cols.values()):
            r = {"SMILES": smi, "T_K": t_k}
            r.update(descs)
            pred_grid_rows.append(r)
    pred_grid = pd.DataFrame(pred_grid_rows)
    X_pred_grid = pred_grid[feature_cols].values.astype(float)
    if not np.all(finite_mask):
        X_pred_grid = X_pred_grid[:, finite_mask]
    pred_grid_smiles = pred_grid["SMILES"].values
    pred_grid_tk = pred_grid["T_K"].values
    pred_grid_folds = np.array([fold_map[s] for s in pred_grid_smiles])

    # OOF prediction at all temps
    oof_density_pred = np.full(len(pred_grid), np.nan)

    for fold_i in range(5):
        # Train on long-format data from other folds
        train_mask = folds_long != fold_i
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1, random_state=fold_i, verbosity=0,
            **FIXED_PARAMS,
        )
        model.fit(X_train_s, y_train)

        # Predict on test fold molecules at all 11 temps
        test_mask = pred_grid_folds == fold_i
        X_test = X_pred_grid[test_mask]
        X_test_s = scaler.transform(X_test)
        oof_density_pred[test_mask] = model.predict(X_test_s)

    # Evaluate on actual data points
    actual_mask = np.zeros(len(pred_grid), dtype=bool)
    actual_density = np.full(len(pred_grid), np.nan)
    for i, (smi, t_k) in enumerate(zip(pred_grid_smiles, pred_grid_tk)):
        t_c = int(round(t_k - 273.15))
        col = f"density_{t_c}C_g_cm3"
        row = density_df[density_df["SMILES"] == smi]
        if len(row) > 0:
            val = row.iloc[0][col]
            if pd.notna(val):
                actual_mask[i] = True
                actual_density[i] = val

    r2 = r2_score(actual_density[actual_mask], oof_density_pred[actual_mask])
    rmse = np.sqrt(mean_squared_error(actual_density[actual_mask], oof_density_pred[actual_mask]))
    logger.info("  Temp-feature density: R²=%.4f, RMSE=%.6f g/cm³", r2, rmse)

    # Reshape: per molecule, densities at 11 temps
    n_mols = len(unique_smiles)
    density_pred_matrix = oof_density_pred.reshape(n_mols, len(TEMPS_C))

    return {
        "smiles": np.array(unique_smiles),
        "temps_c": np.array(TEMPS_C, dtype=float),
        "density_pred": density_pred_matrix,  # (n_mols, 11)
        "r2": r2,
    }


def compute_predicted_beta(density_result):
    """Fit quadratic to predicted density at 11 temps → β(40°C) for each molecule."""
    logger.info("Computing predicted beta from OOF density predictions...")

    smiles = density_result["smiles"]
    temps = density_result["temps_c"]
    density_pred = density_result["density_pred"]

    betas = np.full(len(smiles), np.nan)

    for i in range(len(smiles)):
        dens = density_pred[i]
        # All 11 temps should have predictions (model can predict at any T)
        valid = np.isfinite(dens)
        if valid.sum() < 3:
            continue

        t_valid = temps[valid]
        d_valid = dens[valid]

        coeffs = np.polyfit(t_valid, d_valid, 2)
        c_coeff, b_coeff, a_coeff = coeffs

        rho_40 = a_coeff + b_coeff * 40 + c_coeff * 40**2
        drho_dT_40 = b_coeff + 2 * c_coeff * 40
        beta_40 = -(1.0 / rho_40) * drho_dT_40
        betas[i] = beta_40

    valid_count = np.isfinite(betas).sum()
    logger.info("  Predicted beta computed for %d/%d molecules", valid_count, len(smiles))
    logger.info("  Beta range: %.6f – %.6f",
                np.nanmin(betas), np.nanmax(betas))

    return dict(zip(smiles, betas))


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_parity_side_by_side(true_fom1, direct_pred, pipeline_pred, glyme_mask, smiles,
                              direct_metrics, pipeline_metrics):
    """Side-by-side parity plots: direct vs pipeline FOM1 predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    for ax, pred, label, metrics in [
        (axes[0], direct_pred, "Direct FOM1 Model", direct_metrics),
        (axes[1], pipeline_pred, "Pipeline FOM1", pipeline_metrics),
    ]:
        ax.scatter(true_fom1[~glyme_mask], pred[~glyme_mask],
                   alpha=0.3, s=10, color="steelblue",
                   label=f"Other ({(~glyme_mask).sum()})")
        if glyme_mask.any():
            ax.scatter(true_fom1[glyme_mask], pred[glyme_mask],
                       alpha=0.8, s=40, color="crimson", marker="D",
                       label=f"Glymes ({glyme_mask.sum()})")

        lims = [min(true_fom1.min(), pred.min()), max(true_fom1.max(), pred.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True FOM1 (40°C)")
        ax.set_ylabel(f"Predicted FOM1 (40°C) — {label}")
        ax.legend(frameon=False, loc="upper left")
        ax.set_aspect("equal")

        ax.text(0.95, 0.05,
                f"R² = {metrics['r2_all']:.4f}\n"
                f"MAE = {metrics['mae_all']:.1f}\n"
                f"R²(glyme) = {metrics['r2_glyme']:.4f}\n"
                f"MAE(glyme) = {metrics['mae_glyme']:.1f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "direct_vs_pipeline_parity.png")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "direct_vs_pipeline_parity.png")


def plot_residuals(true_fom1, direct_pred, pipeline_pred, glyme_mask, smiles):
    """Residual comparison: direct vs pipeline, with glyme highlights."""
    direct_resid = direct_pred - true_fom1
    pipeline_resid = pipeline_pred - true_fom1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    for ax, resid, label in [
        (axes[0], direct_resid, "Direct Model Residuals"),
        (axes[1], pipeline_resid, "Pipeline Residuals"),
    ]:
        ax.scatter(true_fom1[~glyme_mask], resid[~glyme_mask],
                   alpha=0.3, s=10, color="steelblue", label="Other")
        if glyme_mask.any():
            ax.scatter(true_fom1[glyme_mask], resid[glyme_mask],
                       alpha=0.8, s=40, color="crimson", marker="D", label="Glymes")
        ax.axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("True FOM1 (40°C)")
        ax.set_ylabel(f"Residual — {label}")
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "residual_comparison.png")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "residual_comparison.png")


def plot_glyme_residuals(true_fom1, direct_pred, pipeline_pred, glyme_mask, smiles):
    """Bar chart of glyme residuals: direct vs pipeline side by side."""
    if not glyme_mask.any():
        logger.info("No glymes found — skipping glyme residual bar chart.")
        return

    glyme_smi = smiles[glyme_mask]
    glyme_true = true_fom1[glyme_mask]
    glyme_direct = direct_pred[glyme_mask] - glyme_true
    glyme_pipeline = pipeline_pred[glyme_mask] - glyme_true

    # Sort by true FOM1
    order = np.argsort(glyme_true)
    glyme_smi = glyme_smi[order]
    glyme_true = glyme_true[order]
    glyme_direct = glyme_direct[order]
    glyme_pipeline = glyme_pipeline[order]

    # Truncate long SMILES for labels
    labels = [s if len(s) <= 30 else s[:27] + "..." for s in glyme_smi]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6), dpi=150)
    ax.bar(x - width/2, glyme_direct, width, label="Direct", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, glyme_pipeline, width, label="Pipeline", color="darkorange", alpha=0.8)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Residual (predicted − true)")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "glyme_residuals.png")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "glyme_residuals.png")


def plot_component_errors_glymes(component_results, glyme_mask, smiles):
    """For glymes: which component property drives the FOM1 error?"""
    if not glyme_mask.any():
        return

    props = ["density", "viscosity", "tc", "cpsat"]
    prop_labels = ["ρ / g cm⁻³", "ν / cSt", "k / W m⁻¹ K⁻¹", "Cp / J K⁻¹ mol⁻¹"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=150)

    for ax, prop, plabel in zip(axes, props, prop_labels):
        res = component_results[prop]
        # Match SMILES ordering
        smi_to_idx = {s: i for i, s in enumerate(res["smiles"])}
        glyme_smi = smiles[glyme_mask]

        glyme_true = []
        glyme_pred = []
        for s in glyme_smi:
            if s in smi_to_idx:
                idx = smi_to_idx[s]
                glyme_true.append(res["y_true"][idx])
                glyme_pred.append(res["y_pred"][idx])

        glyme_true = np.array(glyme_true)
        glyme_pred = np.array(glyme_pred)

        if len(glyme_true) == 0:
            continue

        pct_error = 100 * (glyme_pred - glyme_true) / glyme_true

        ax.bar(range(len(pct_error)), pct_error, color="crimson", alpha=0.8)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_ylabel(f"% error — {plabel}")
        ax.set_xticks(range(len(glyme_smi)))
        ax.set_xticklabels(
            [s if len(s) <= 20 else s[:17] + "..." for s in glyme_smi],
            rotation=45, ha="right", fontsize=6,
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "glyme_component_errors.png")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "glyme_component_errors.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # ------------------------------------------------------------------
    # Load data and assign folds
    # ------------------------------------------------------------------
    data = load_intersection_data()
    fold_map = assign_folds(data["intersection_smiles"])

    logger.info("Fold distribution: %s",
                {f: sum(1 for v in fold_map.values() if v == f) for f in range(5)})

    # ------------------------------------------------------------------
    # Step 3: Train single-temp models at 40°C
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Training single-temp models at 40°C")
    logger.info("=" * 60)

    component_results = {}

    component_results["density"] = train_single_temp(
        data["density"], "density_40C_g_cm3", fold_map,
        log_transform=False, prop_name="density_40C",
    )
    component_results["viscosity"] = train_single_temp(
        data["viscosity"], "kv_40C_cSt", fold_map,
        log_transform=True, prop_name="viscosity_40C",
    )
    component_results["tc"] = train_single_temp(
        data["tc"], "tc_40C_W_mK", fold_map,
        log_transform=False, prop_name="tc_40C",
    )
    component_results["cpsat"] = train_single_temp(
        data["cpsat"], "cpsat_40C_J_K_mol", fold_map,
        log_transform=False, prop_name="cpsat_40C",
    )

    # Direct FOM1 model (trained on improved FOM1 target)
    direct_result = train_single_temp(
        data["fom1"], "FOM1_improved_40C", fold_map,
        log_transform=True, prop_name="FOM1_direct_40C",
    )

    # ------------------------------------------------------------------
    # Step 4: Temp-feature density → predicted beta
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Temp-feature density model for beta prediction")
    logger.info("=" * 60)

    density_tf_result = train_density_temp_feature(data["density"], fold_map)
    beta_pred_map = compute_predicted_beta(density_tf_result)

    # ------------------------------------------------------------------
    # Assemble pipeline FOM1 from OOF component predictions
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Assembling pipeline FOM1 and comparing")
    logger.info("=" * 60)

    # Build aligned arrays: same molecule order across all models
    # Use the FOM1 improved dataset as the reference (it defines the intersection)
    fom1_df = data["fom1"]
    ref_smiles = direct_result["smiles"]

    # Map from SMILES to OOF predictions for each component
    density_map = dict(zip(component_results["density"]["smiles"],
                           component_results["density"]["y_pred"]))
    visc_map = dict(zip(component_results["viscosity"]["smiles"],
                        component_results["viscosity"]["y_pred"]))
    tc_map = dict(zip(component_results["tc"]["smiles"],
                      component_results["tc"]["y_pred"]))
    cp_map = dict(zip(component_results["cpsat"]["smiles"],
                      component_results["cpsat"]["y_pred"]))

    # Get MW for each molecule (from FOM1_improved CSV)
    mw_map = dict(zip(fom1_df["SMILES"], fom1_df["MolWt"]))

    # Build aligned arrays
    valid_smiles = []
    rho_pred_arr, visc_pred_arr, tc_pred_arr, cp_pred_arr = [], [], [], []
    beta_pred_arr, mw_arr, fom1_true_arr, fom1_direct_arr = [], [], [], []

    direct_map = dict(zip(direct_result["smiles"], direct_result["y_pred"]))
    true_map = dict(zip(direct_result["smiles"], direct_result["y_true"]))

    for smi in ref_smiles:
        if all(smi in m for m in [density_map, visc_map, tc_map, cp_map, beta_pred_map, mw_map]):
            beta_val = beta_pred_map[smi]
            if not np.isfinite(beta_val) or beta_val <= 0:
                continue
            valid_smiles.append(smi)
            rho_pred_arr.append(density_map[smi])
            visc_pred_arr.append(visc_map[smi])
            tc_pred_arr.append(tc_map[smi])
            cp_pred_arr.append(cp_map[smi])
            beta_pred_arr.append(beta_val)
            mw_arr.append(mw_map[smi])
            fom1_true_arr.append(true_map[smi])
            fom1_direct_arr.append(direct_map[smi])

    valid_smiles = np.array(valid_smiles)
    rho_pred = np.array(rho_pred_arr)
    visc_pred = np.array(visc_pred_arr)
    tc_pred = np.array(tc_pred_arr)
    cp_pred = np.array(cp_pred_arr)
    beta_pred = np.array(beta_pred_arr)
    mw = np.array(mw_arr)
    fom1_true = np.array(fom1_true_arr)
    fom1_direct = np.array(fom1_direct_arr)

    logger.info("Molecules with all pipeline components: %d", len(valid_smiles))

    # Compute pipeline FOM1
    fom1_pipeline = compute_fom1_from_components(
        rho_pred, visc_pred, tc_pred, cp_pred, beta_pred, mw
    )

    # ------------------------------------------------------------------
    # Verification: pipeline from NIST values should match FOM1_improved
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Verification: pipeline from NIST components vs FOM1_improved")

    # Get actual NIST component values for these molecules
    nist_rho = np.array([fom1_df[fom1_df["SMILES"] == s]["density_40C_g_cm3"].values[0]
                         for s in valid_smiles])
    nist_visc = np.array([fom1_df[fom1_df["SMILES"] == s]["kv_40C_cSt"].values[0]
                          for s in valid_smiles])
    nist_tc = np.array([fom1_df[fom1_df["SMILES"] == s]["tc_40C_W_mK"].values[0]
                        for s in valid_smiles])
    nist_cp = np.array([fom1_df[fom1_df["SMILES"] == s]["cpsat_40C_J_K_mol"].values[0]
                        for s in valid_smiles])
    nist_beta = np.array([fom1_df[fom1_df["SMILES"] == s]["beta_40C"].values[0]
                          for s in valid_smiles])

    fom1_from_nist = compute_fom1_from_components(
        nist_rho, nist_visc, nist_tc, nist_cp, nist_beta, mw
    )
    verif_r2 = r2_score(fom1_true, fom1_from_nist)
    verif_mae = mean_absolute_error(fom1_true, fom1_from_nist)
    logger.info("  NIST-recomputed vs FOM1_improved: R²=%.6f, MAE=%.4f", verif_r2, verif_mae)

    # ------------------------------------------------------------------
    # Metrics: direct vs pipeline
    # ------------------------------------------------------------------
    glyme_mask = np.array([is_glyme(s) for s in valid_smiles])
    n_glymes = glyme_mask.sum()
    logger.info("Glymes in comparison set: %d", n_glymes)

    def compute_metrics(y_true, y_pred, mask=None):
        if mask is None:
            mask = np.ones(len(y_true), dtype=bool)
        if mask.sum() < 2:
            return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "n": mask.sum()}
        return {
            "r2": r2_score(y_true[mask], y_pred[mask]),
            "rmse": np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
            "mae": mean_absolute_error(y_true[mask], y_pred[mask]),
            "n": int(mask.sum()),
        }

    direct_metrics = {
        "all": compute_metrics(fom1_true, fom1_direct),
        "glyme": compute_metrics(fom1_true, fom1_direct, glyme_mask),
        "non_glyme": compute_metrics(fom1_true, fom1_direct, ~glyme_mask),
    }
    pipeline_metrics = {
        "all": compute_metrics(fom1_true, fom1_pipeline),
        "glyme": compute_metrics(fom1_true, fom1_pipeline, glyme_mask),
        "non_glyme": compute_metrics(fom1_true, fom1_pipeline, ~glyme_mask),
    }

    logger.info("")
    logger.info("%-20s  %-10s  %-10s  %-10s  %s", "", "R²", "RMSE", "MAE", "N")
    logger.info("-" * 65)
    for label, d_m, p_m in [("All", direct_metrics["all"], pipeline_metrics["all"]),
                             ("Glymes", direct_metrics["glyme"], pipeline_metrics["glyme"]),
                             ("Non-glymes", direct_metrics["non_glyme"], pipeline_metrics["non_glyme"])]:
        logger.info("Direct %-13s  %-10.4f  %-10.2f  %-10.2f  %d",
                     label, d_m["r2"], d_m["rmse"], d_m["mae"], d_m["n"])
        logger.info("Pipeline %-11s  %-10.4f  %-10.2f  %-10.2f  %d",
                     label, p_m["r2"], p_m["rmse"], p_m["mae"], p_m["n"])
        logger.info("")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_metrics_direct = {
        "r2_all": direct_metrics["all"]["r2"],
        "mae_all": direct_metrics["all"]["mae"],
        "r2_glyme": direct_metrics["glyme"]["r2"],
        "mae_glyme": direct_metrics["glyme"]["mae"],
    }
    plot_metrics_pipeline = {
        "r2_all": pipeline_metrics["all"]["r2"],
        "mae_all": pipeline_metrics["all"]["mae"],
        "r2_glyme": pipeline_metrics["glyme"]["r2"],
        "mae_glyme": pipeline_metrics["glyme"]["mae"],
    }

    plot_parity_side_by_side(fom1_true, fom1_direct, fom1_pipeline, glyme_mask,
                              valid_smiles, plot_metrics_direct, plot_metrics_pipeline)
    plot_residuals(fom1_true, fom1_direct, fom1_pipeline, glyme_mask, valid_smiles)
    plot_glyme_residuals(fom1_true, fom1_direct, fom1_pipeline, glyme_mask, valid_smiles)
    plot_component_errors_glymes(component_results, glyme_mask, valid_smiles)

    # ------------------------------------------------------------------
    # Save predictions and results
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "SMILES": valid_smiles,
        "FOM1_true": fom1_true,
        "FOM1_direct_pred": fom1_direct,
        "FOM1_pipeline_pred": fom1_pipeline,
        "density_pred": rho_pred,
        "viscosity_pred": visc_pred,
        "tc_pred": tc_pred,
        "cpsat_pred": cp_pred,
        "beta_pred": beta_pred,
        "is_glyme": glyme_mask,
    })
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

    results = {
        "n_molecules": len(valid_smiles),
        "n_glymes": int(n_glymes),
        "verification_r2": round(verif_r2, 6),
        "direct_metrics": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                for kk, vv in v.items()}
                           for k, v in direct_metrics.items()},
        "pipeline_metrics": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                  for kk, vv in v.items()}
                             for k, v in pipeline_metrics.items()},
        "component_r2": {
            prop: round(res["r2"], 4) for prop, res in component_results.items()
        },
        "density_temp_feature_r2": round(density_tf_result["r2"], 4),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved predictions and results to %s/", OUT_DIR)
    logger.info("Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
