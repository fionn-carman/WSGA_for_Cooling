#!/usr/bin/env python3
"""
Train non-temperature-dependent XGBoost models at 40C and 100C on clipped2 data.
For comparison with temp-as-feature models.

Usage:
    python train_single_temp.py
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]

FIXED_PARAMS = {
    "n_estimators": 600, "max_depth": 4, "learning_rate": 0.1,
    "subsample": 1.0, "colsample_bytree": 0.7,
    "min_child_weight": 10, "reg_alpha": 0, "reg_lambda": 0.1,
}

MODELS = [
    ("density",   "../../data/nist_8100/density_cho_cleaned.csv",   "density_40C", "density_100C", False),
    ("viscosity", "../../data/nist_8100/viscosity_cho_cleaned.csv", "kv_40C", "kv_100C", True),
    ("tc",        "../../data/nist_8100/tc_cho_cleaned.csv",        "tc_40C", "tc_100C", False),
    ("cpsat",     "../../data/nist_8100/cpsat_cho_cleaned.csv",     "cpsat_40C", "cpsat_100C", False),
    ("beta",      "../../data/nist_8100/beta_cho_cleaned.csv",      "beta_40C", "beta_100C", False),
    ("fom1",      "../../data/nist_8100/fom1_cho_cleaned.csv",      "FOM1_40C", "FOM1_100C", True),
]


def train_single(prop_name, csv_path, col_name, log_transform):
    """Train 5-fold XGBoost for a single property at a single temperature."""
    df = pd.read_csv(csv_path)

    # Find the target column
    target_col = None
    for c in df.columns:
        if c == col_name or c.replace("C", "C") == col_name:
            target_col = c
            break
    if target_col is None:
        # Try partial match
        for c in df.columns:
            if col_name.lower() in c.lower():
                target_col = c
                break
    if target_col is None:
        logger.error("Column %s not found in %s. Columns: %s", col_name, csv_path,
                      [c for c in df.columns if not c.startswith("fr_") and c not in DESCRIPTOR_NAMES])
        return None

    # Drop rows with NaN target
    mask = df[target_col].notna()
    df = df[mask].copy()

    desc_cols = [c for c in DESCRIPTOR_NAMES if c in df.columns]
    X = df[desc_cols].values
    y_raw = df[target_col].values

    if log_transform:
        y = np.log1p(y_raw)
    else:
        y = y_raw.copy()

    groups = df["SMILES"].values

    # Remove non-finite features
    valid_mask = np.all(np.isfinite(X), axis=0)
    if not np.all(valid_mask):
        desc_cols = [f for f, v in zip(desc_cols, valid_mask) if v]
        X = X[:, valid_mask]

    gkf = GroupKFold(n_splits=5)
    all_preds = np.full(len(y), np.nan)

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=fold_i,
            verbosity=0,
            **FIXED_PARAMS,
        )
        model.fit(X_train_s, y_train)
        all_preds[test_idx] = model.predict(X_test_s)

    valid = ~np.isnan(all_preds)
    if log_transform:
        preds_clipped = np.clip(all_preds[valid], 0, np.log1p(y_raw.max()))
        r2_real = r2_score(np.expm1(y[valid]), np.expm1(preds_clipped))
        rmse_real = np.sqrt(mean_squared_error(np.expm1(y[valid]), np.expm1(preds_clipped)))
    else:
        r2_real = r2_score(y[valid], all_preds[valid])
        rmse_real = np.sqrt(mean_squared_error(y[valid], all_preds[valid]))

    return {
        "r2_real": round(r2_real, 4),
        "rmse_real": round(rmse_real, 4),
        "n_molecules": int(valid.sum()),
    }


def main():
    results = {}

    for prop_name, csv_path, col_40, col_100, log_transform in MODELS:
        for temp, col_name in [("40C", col_40), ("100C", col_100)]:
            key = f"{prop_name}_{temp}"
            logger.info("Training: %s (log=%s)", key, log_transform)
            t0 = time.time()
            result = train_single(prop_name, csv_path, col_name, log_transform)
            elapsed = time.time() - t0
            if result:
                result["time_s"] = round(elapsed, 1)
                results[key] = result
                logger.info("  => R²(real)=%.4f  RMSE=%.4f  N=%d  (%.1fs)",
                            result["r2_real"], result["rmse_real"],
                            result["n_molecules"], elapsed)

    # Print comparison table
    print("\n" + "=" * 70)
    print("SINGLE-TEMP MODELS (5-fold CV, same HPs, clipped2 data)")
    print("=" * 70)
    print(f"{'Model':<20} {'R²(real)':>10} {'RMSE':>10} {'N':>8}")
    print("-" * 70)
    for key, r in results.items():
        print(f"{key:<20} {r['r2_real']:>10.4f} {r['rmse_real']:>10.4f} {r['n_molecules']:>8}")
    print("=" * 70)

    with open("single_temp_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
