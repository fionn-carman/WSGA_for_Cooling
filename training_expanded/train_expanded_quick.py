#!/usr/bin/env python3
"""
Quick training on NIST WTT dataset (WTT-only, no merge with old data).

Trains 10 XGBoost models with default hyperparameters (no RFE, no tuning):
  - Density 40C/100C
  - Kinematic Viscosity 40C/100C
  - Thermal Conductivity 40C/100C
  - Cpsat 40C/100C
  - FOM1_sat 40C/100C

Usage:
    cd training && python train_expanded_quick.py
"""

import json
import logging
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

WTT_CSV = os.path.join(os.path.dirname(__file__),
                        "..", "NIST_WTT_Collection", "data", "nist_wtt_cleaned.csv")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BACKUP_DIR = os.path.join(DATA_DIR, "pre_expanded_backup")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_expanded_quick")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Property definitions: (wtt_column, target_name, csv_filename, log_transform)
PROPERTIES = [
    ("density_40C_g_cm3",   "Density_40C_g_cm^3",   "Density_40C_g_cm^3_cleaned.csv",   False),
    ("density_100C_g_cm3",  "Density_100C_g_cm^3",  "Density_100C_g_cm^3_cleaned.csv",  False),
    ("kv_40C_cSt",          "Kinematic_Viscosity_40C",  "Kinematic_Viscosity_40C_cleaned.csv",  True),
    ("kv_100C_cSt",         "Kinematic_Viscosity_100C", "Kinematic_Viscosity_100C_cleaned.csv", True),
    ("tc_40C_W_mK",         "Thermal_Conductivity_40C",  "Thermal_Conductivity_40C_cleaned.csv",  False),
    ("tc_100C_W_mK",        "Thermal_Conductivity_100C", "Thermal_Conductivity_100C_cleaned.csv", False),
    ("cpsat_40C_J_K_mol",   "Cpsat_40C_J_K_Mol",   "Cpsat_40C_J_K_Mol_cleaned.csv",   True),
    ("cpsat_100C_J_K_mol",  "Cpsat_100C_J_K_Mol",  "Cpsat_100C_J_K_Mol_cleaned.csv",  True),
    ("FOM1_sat_40",         "FOM1_sat_40",          "FOM1_sat_40_cleaned.csv",          False),
    ("FOM1_sat_100",        "FOM1_sat_100",         "FOM1_sat_100_cleaned.csv",         False),
]

# RDKit descriptor list
ALL_DESC_NAMES = [desc[0] for desc in Descriptors._descList]
ALL_DESC_FUNCS = [desc[1] for desc in Descriptors._descList]

VISCOSITY_THRESHOLD = 40.0  # cSt — remove molecules above this


# ── Helpers ──────────────────────────────────────────────────────────────

def canonicalise(smiles):
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compute_all_descriptors(smiles):
    """Compute all RDKit descriptors. Returns dict or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    result = {}
    for name, func in zip(ALL_DESC_NAMES, ALL_DESC_FUNCS):
        try:
            result[name] = func(mol)
        except Exception:
            result[name] = np.nan
    return result


# ── Phase 1: Create Training CSVs ────────────────────────────────────────

def phase1_create_csvs():
    """Extract per-property CSVs from WTT data with descriptors."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Create training CSVs from WTT data")
    logger.info("=" * 60)

    wtt = pd.read_csv(WTT_CSV)
    logger.info("Loaded WTT data: %d rows", len(wtt))

    # Canonicalise SMILES
    wtt["canonical_SMILES"] = wtt["SMILES"].apply(canonicalise)
    invalid = wtt["canonical_SMILES"].isna()
    if invalid.any():
        logger.warning("%d invalid SMILES removed", invalid.sum())
    wtt = wtt[~invalid].copy()

    # Compute descriptors for all unique molecules
    logger.info("Computing RDKit descriptors for %d unique molecules...",
                wtt["canonical_SMILES"].nunique())
    desc_cache = {}
    for smi in wtt["canonical_SMILES"].unique():
        desc_cache[smi] = compute_all_descriptors(smi)
    failed = sum(1 for v in desc_cache.values() if v is None)
    if failed:
        logger.warning("%d molecules failed descriptor computation", failed)

    # Identify high-viscosity molecules to remove from KV datasets
    kv40_data = wtt[wtt["kv_40C_cSt"].notna()].copy()
    high_visc_smiles = set(
        kv40_data[kv40_data["kv_40C_cSt"] > VISCOSITY_THRESHOLD]["canonical_SMILES"]
    )
    logger.info("High-viscosity molecules (kv_40C > %.0f cSt): %d",
                VISCOSITY_THRESHOLD, len(high_visc_smiles))

    # Back up existing files that will be overwritten
    os.makedirs(BACKUP_DIR, exist_ok=True)
    files_to_backup = [
        "Density_40C_g_cm^3_cleaned.csv",
        "Density_100C_g_cm^3_cleaned.csv",
        "Kinematic_Viscosity_40C_cleaned.csv",
        "Kinematic_Viscosity_100C_cleaned.csv",
        "Thermal_Conductivity_40C_cleaned.csv",
        "Thermal_Conductivity_100C_cleaned.csv",
    ]
    for fname in files_to_backup:
        src = os.path.join(DATA_DIR, fname)
        dst = os.path.join(BACKUP_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info("Backed up %s", fname)

    # Create each CSV
    csv_paths = {}
    for wtt_col, target_name, csv_filename, log_transform in PROPERTIES:
        logger.info("-" * 40)
        logger.info("Processing: %s -> %s", wtt_col, target_name)

        # Filter to rows with data
        subset = wtt[wtt[wtt_col].notna()].copy()
        if len(subset) == 0:
            logger.warning("No data for %s, skipping", wtt_col)
            continue

        # Apply viscosity outlier filter
        if wtt_col in ("kv_40C_cSt", "kv_100C_cSt"):
            before = len(subset)
            subset = subset[~subset["canonical_SMILES"].isin(high_visc_smiles)]
            logger.info("Removed %d high-viscosity molecules", before - len(subset))

        # Deduplicate by canonical SMILES (take median)
        subset = subset.groupby("canonical_SMILES").agg({wtt_col: "median"}).reset_index()
        subset = subset.rename(columns={"canonical_SMILES": "SMILES", wtt_col: target_name})

        # Compute descriptors
        rows = []
        for _, row in subset.iterrows():
            smi = row["SMILES"]
            desc = desc_cache.get(smi)
            if desc is None:
                continue
            row_data = {"SMILES": smi, target_name: row[target_name]}
            row_data.update(desc)
            rows.append(row_data)

        df = pd.DataFrame(rows)
        # Ensure column order: SMILES, target, descriptors
        cols = ["SMILES", target_name] + ALL_DESC_NAMES
        df = df[[c for c in cols if c in df.columns]]

        # Clean NaN/inf in descriptors
        desc_cols = [c for c in df.columns if c not in ("SMILES", target_name)]
        df[desc_cols] = df[desc_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        csv_path = os.path.join(DATA_DIR, csv_filename)
        df.to_csv(csv_path, index=False)
        csv_paths[target_name] = csv_path
        logger.info("Saved %s: %d molecules", csv_filename, len(df))

    return csv_paths


# ── Phase 2: Train Models ────────────────────────────────────────────────

def phase2_train_models(csv_paths):
    """Train default XGBoost for each property (no RFE, no hyperparam tuning)."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Train XGBoost models (default hyperparams)")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for wtt_col, target_name, csv_filename, log_transform in PROPERTIES:
        csv_path = csv_paths.get(target_name)
        if csv_path is None or not os.path.exists(csv_path):
            logger.warning("Skipping %s: no CSV", target_name)
            continue

        logger.info("-" * 40)
        logger.info("Training: %s (log=%s)", target_name, log_transform)
        t0 = time.time()

        df = pd.read_csv(csv_path)
        descriptor_cols = [c for c in df.columns if c not in ("SMILES", target_name)]
        X = df[descriptor_cols].values.astype(np.float64)
        y = df[target_name].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if log_transform:
            y = np.log1p(y)

        # 90/10 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        # Train with defaults
        model = XGBRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        if log_transform:
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred

        r2 = r2_score(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        elapsed = time.time() - t0

        logger.info("=> R2=%.4f  RMSE=%.4f  MAE=%.4f  (%.0fs)", r2, rmse, mae, elapsed)

        # Save model
        out_dir = os.path.join(OUTPUT_DIR, target_name, "model")
        os.makedirs(out_dir, exist_ok=True)

        model_dict = {
            "model": model,
            "features": descriptor_cols,
            "log_target": log_transform,
            "test_r2": float(r2),
            "test_rmse": float(rmse),
            "test_mae": float(mae),
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "training_time_s": float(elapsed),
        }
        joblib.dump(model_dict, os.path.join(out_dir, "xgb_model.joblib"))

        metrics = {
            "target": target_name,
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_features": len(descriptor_cols),
            "n_samples": len(df),
            "log_transform": log_transform,
            "training_time_s": float(elapsed),
        }
        with open(os.path.join(OUTPUT_DIR, target_name, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        results[target_name] = metrics

    return results


# ── Phase 3: Comparison Table ────────────────────────────────────────────

def phase3_comparison(results):
    """Compare new WTT models against old baseline models."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Comparison with old models")
    logger.info("=" * 60)

    import warnings
    warnings.filterwarnings("ignore")

    OLD_MODELS = {
        "Density_40C_g_cm^3":        ("Density_40C_g_cm^3",        "single", "Density 40C"),
        "Density_100C_g_cm^3":       ("Density_100C_g_cm^3",       "single", "Density 100C"),
        "Kinematic_Viscosity_40C":   ("Kinematic_Viscosity_40C",   "single", "KV 40C"),
        "Kinematic_Viscosity_100C":  ("Kinematic_Viscosity_100C",  "single", "KV 100C"),
        "Thermal_Conductivity_40C":  ("Thermal_Conductivity_40C",  "single", "TC 40C"),
        "Thermal_Conductivity_100C": ("Thermal_Conductivity_100C", "single", "TC 100C"),
        "Cpsat_40C_J_K_Mol":        ("Heat_Capacity_Constant_Pressure_40C_J_K_Mol", "single", "Cpsat 40C (vs IG Cp)"),
        "Cpsat_100C_J_K_Mol":       ("Heat_Capacity_Constant_Pressure_100C_J_K_Mol", "single", "Cpsat 100C (vs IG Cp)"),
        "FOM1_sat_40":              ("FOM1_direct_5fold_40C",     "5fold",  "FOM1_sat 40 (vs IG FOM1)"),
        "FOM1_sat_100":             ("FOM1_direct_5fold_100C",    "5fold",  "FOM1_sat 100 (vs IG FOM1)"),
    }

    rows = []
    for target_name, (old_dir, model_type, desc) in OLD_MODELS.items():
        new = results.get(target_name, {})
        new_r2 = new.get("r2", None)
        new_rmse = new.get("rmse", None)
        new_mae = new.get("mae", None)
        new_n = new.get("n_samples", None)

        old_r2 = old_rmse = old_mae = old_n = None

        if model_type == "single":
            old_path = os.path.join(MODELS_DIR, old_dir, "model", "xgb_model.joblib")
            if os.path.exists(old_path):
                old_d = joblib.load(old_path)
                old_r2 = old_d.get("test_r2")
                old_rmse = old_d.get("test_rmse")
                old_mae = old_d.get("test_mae")
                old_csv_map = {
                    "Density_40C_g_cm^3": "Density_40C_g_cm^3_cleaned.csv",
                    "Density_100C_g_cm^3": "Density_100C_g_cm^3_cleaned.csv",
                    "Kinematic_Viscosity_40C": "Kinematic_Viscosity_40C_cleaned.csv",
                    "Kinematic_Viscosity_100C": "Kinematic_Viscosity_100C_cleaned.csv",
                    "Thermal_Conductivity_40C": "Thermal_Conductivity_40C_cleaned.csv",
                    "Thermal_Conductivity_100C": "Thermal_Conductivity_100C_cleaned.csv",
                    "Heat_Capacity_Constant_Pressure_40C_J_K_Mol": "Heat_Capacity_Constant_Pressure_40C_J_K_Mol_cleaned.csv",
                    "Heat_Capacity_Constant_Pressure_100C_J_K_Mol": "Heat_Capacity_Constant_Pressure_100C_J_K_Mol_cleaned.csv",
                }
                for backup in [BACKUP_DIR, os.path.join(DATA_DIR, "pre_wtt_backup")]:
                    csv_p = os.path.join(backup, old_csv_map.get(old_dir, ""))
                    if os.path.exists(csv_p):
                        old_n = len(pd.read_csv(csv_p))
                        break

        elif model_type == "5fold":
            fold_r2s = []
            for fold in range(5):
                mp = os.path.join(MODELS_DIR, old_dir,
                                  f"xgboost_descriptors_fold{fold}_metrics.json")
                if os.path.exists(mp):
                    with open(mp) as f:
                        m = json.load(f)
                    fold_r2s.append(m["r2"])
            if fold_r2s:
                old_r2 = np.mean(fold_r2s)

        delta_r2 = None
        if new_r2 is not None and old_r2 is not None:
            delta_r2 = new_r2 - old_r2

        rows.append({
            "Property": desc,
            "Target": target_name,
            "Old_N": old_n,
            "New_N": new_n,
            "Old_R2": old_r2,
            "New_R2": new_r2,
            "Delta_R2": delta_r2,
            "New_RMSE": new_rmse,
            "New_MAE": new_mae,
        })

    comparison_df = pd.DataFrame(rows)

    # Print table
    print("\n" + "=" * 90)
    print("COMPARISON: Old Models vs WTT-Trained Models (default XGBoost)")
    print("=" * 90)
    print(f"{'Property':<25} {'Old N':>6} {'New N':>6} {'Old R²':>8} {'New R²':>8} {'ΔR²':>8} {'RMSE':>10}")
    print("-" * 90)
    for _, row in comparison_df.iterrows():
        old_n_str = str(int(row["Old_N"])) if pd.notna(row["Old_N"]) else "—"
        new_n_str = str(int(row["New_N"])) if pd.notna(row["New_N"]) else "—"
        old_r2_str = f"{row['Old_R2']:.4f}" if pd.notna(row["Old_R2"]) else "—"
        new_r2_str = f"{row['New_R2']:.4f}" if pd.notna(row["New_R2"]) else "—"
        delta_str = f"{row['Delta_R2']:+.4f}" if pd.notna(row["Delta_R2"]) else "—"
        rmse_str = f"{row['New_RMSE']:.4f}" if pd.notna(row["New_RMSE"]) else "—"
        print(f"{row['Property']:<25} {old_n_str:>6} {new_n_str:>6} {old_r2_str:>8} {new_r2_str:>8} {delta_str:>8} {rmse_str:>10}")
    print("=" * 90)

    comparison_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Saved comparison to %s", comparison_path)

    return comparison_df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    logger.info("Quick Training on WTT Dataset (default XGBoost, no RFE/tuning)")
    logger.info("WTT source: %s", os.path.abspath(WTT_CSV))

    if not os.path.exists(WTT_CSV):
        logger.error("WTT CSV not found: %s", WTT_CSV)
        return

    csv_paths = phase1_create_csvs()
    results = phase2_train_models(csv_paths)
    phase3_comparison(results)

    logger.info("Done!")


if __name__ == "__main__":
    main()
