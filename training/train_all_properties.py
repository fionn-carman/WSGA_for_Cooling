#!/usr/bin/env python3
"""
Train temperature-as-feature XGBoost for all 5 properties using pre-computed descriptors.

Usage:
    python train_all_properties.py
    python train_all_properties.py --property density
"""

import argparse
import json
import logging
import re
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

# Temperature pattern: column names containing e.g. _0C_, _10C_, _100C
TEMP_PATTERN = re.compile(r'_(\d+)C')

PROPERTIES = {
    "viscosity": {
        "file": "data/viscosity_cho_cleaned.csv",
        "prefix": "kv_",
        "unit": "cSt",
        "log_transform": True,
        "fixed_params": {
            "n_estimators": 600, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 1.0, "colsample_bytree": 0.7,
            "min_child_weight": 10, "reg_alpha": 0, "reg_lambda": 0.1,
        },
    },
    "density": {
        "file": "data/density_cho_cleaned.csv",
        "prefix": "density_",
        "unit": "g/cm3",
        "log_transform": False,
        "fixed_params": {
            "n_estimators": 600, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 1.0, "colsample_bytree": 0.7,
            "min_child_weight": 10, "reg_alpha": 0, "reg_lambda": 0.1,
        },
    },
    "tc": {
        "file": "data/tc_cho_cleaned.csv",
        "prefix": "tc_",
        "unit": "W/mK",
        "log_transform": False,
        "fixed_params": {
            "n_estimators": 600, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 1.0, "colsample_bytree": 0.7,
            "min_child_weight": 10, "reg_alpha": 0, "reg_lambda": 0.1,
        },
    },
    "cpsat": {
        "file": "data/cpsat_cho_cleaned.csv",
        "prefix": "cpsat_",
        "unit": "J/K/mol",
        "log_transform": False,
        "fixed_params": {
            "n_estimators": 600, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 1.0, "colsample_bytree": 0.7,
            "min_child_weight": 10, "reg_alpha": 0, "reg_lambda": 0.1,
        },
    },
    "FOM1": {
        "file": "data/FOM1_cho_cleaned.csv",
        "prefix": "FOM1_",
        "unit": "",
        "log_transform": True,
        "fixed_params": {
            "n_estimators": 600, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 1.0, "colsample_bytree": 0.7,
            "min_child_weight": 10, "reg_alpha": 0, "reg_lambda": 0.1,
        },
    },
}


def detect_property_cols(df, prefix):
    """Find property columns and extract temperature in K."""
    prop_cols = {}
    for col in df.columns:
        if col.startswith(prefix):
            m = TEMP_PATTERN.search(col)
            if m:
                temp_c = int(m.group(1))
                prop_cols[col] = temp_c + 273.15
    return dict(sorted(prop_cols.items(), key=lambda x: x[1]))


def melt_with_descriptors(df, prop_cols, prop_name):
    """Melt wide-format to long, keeping pre-computed descriptors."""
    desc_cols = [c for c in DESCRIPTOR_NAMES if c in df.columns]
    rows = []
    for idx, row in df.iterrows():
        smi = row["SMILES"]
        descs = {d: row[d] for d in desc_cols}
        for col_name, t_k in prop_cols.items():
            val = row[col_name]
            if pd.notna(val):
                r = {"SMILES": smi, "T_K": t_k, "value": val}
                r.update(descs)
                rows.append(r)
    return pd.DataFrame(rows)


def train_property(prop_name, config):
    """Train 5-fold XGBoost for a single property."""
    logger.info("=" * 60)
    logger.info("TRAINING: %s", prop_name.upper())
    logger.info("=" * 60)

    df = pd.read_csv(config["file"])
    logger.info("Loaded %s: %d molecules, %d columns", config["file"], len(df), len(df.columns))

    prop_cols = detect_property_cols(df, config["prefix"])
    logger.info("Property columns: %d temperatures (%.0f–%.0f K)",
                len(prop_cols), min(prop_cols.values()), max(prop_cols.values()))

    # Melt to long format
    df_long = melt_with_descriptors(df, prop_cols, prop_name)
    df_long = df_long.dropna()
    logger.info("Long format: %d rows", len(df_long))

    # Features
    desc_cols = [c for c in DESCRIPTOR_NAMES if c in df_long.columns]
    feature_cols = ["T_K"] + desc_cols
    X = df_long[feature_cols].values
    y_raw = df_long["value"].values

    if config["log_transform"]:
        y = np.log1p(y_raw)
        logger.info("Target: log1p(%s)", prop_name)
    else:
        y = y_raw.copy()
        logger.info("Target: %s (no transform)", prop_name)

    groups = df_long["SMILES"].values

    # Remove non-finite features
    valid_mask = np.all(np.isfinite(X), axis=0)
    if not np.all(valid_mask):
        n_removed = (~valid_mask).sum()
        logger.info("Removing %d non-finite features", n_removed)
        feature_cols = [f for f, v in zip(feature_cols, valid_mask) if v]
        X = X[:, valid_mask]

    logger.info("Features: %d (T_K + %d descriptors)", len(feature_cols), len(feature_cols) - 1)

    output_dir = Path(f"{prop_name}_temp_feature")
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_params = config["fixed_params"]
    logger.info("Fixed HPs: %s", fixed_params)

    gkf = GroupKFold(n_splits=5)
    fold_results = []
    all_preds = np.full(len(y), np.nan)
    all_temps = df_long["T_K"].values

    t0 = time.time()
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info("--- Fold %d/5 ---", fold_i + 1)
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
            **fixed_params,
        )
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        all_preds[test_idx] = y_pred

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Real-space metrics
        if config["log_transform"]:
            y_test_real = np.expm1(y_test)
            y_pred_real = np.expm1(np.clip(y_pred, 0, np.log1p(y_raw.max())))
            r2_real = r2_score(y_test_real, y_pred_real)
            rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        else:
            r2_real = r2
            rmse_real = rmse

        logger.info("  R²: %.4f, RMSE: %.4f | R²(real): %.4f, RMSE(real): %.4f %s",
                     r2, rmse, r2_real, rmse_real, config["unit"])

        fold_results.append({
            "fold": fold_i,
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_real": round(r2_real, 4),
            "rmse_real": round(rmse_real, 4),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

        joblib.dump({
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "log_transform": config["log_transform"],
        }, output_dir / f"fold{fold_i}_model.joblib")

    elapsed = time.time() - t0

    # Overall metrics
    valid = ~np.isnan(all_preds)
    overall_r2 = r2_score(y[valid], all_preds[valid])
    overall_rmse = np.sqrt(mean_squared_error(y[valid], all_preds[valid]))

    if config["log_transform"]:
        preds_clipped = np.clip(all_preds[valid], 0, np.log1p(y_raw.max()))
        overall_r2_real = r2_score(np.expm1(y[valid]), np.expm1(preds_clipped))
        overall_rmse_real = np.sqrt(mean_squared_error(np.expm1(y[valid]), np.expm1(preds_clipped)))
    else:
        overall_r2_real = overall_r2
        overall_rmse_real = overall_rmse

    # Per-temperature
    per_temp = {}
    for col_name, t_k in sorted(prop_cols.items(), key=lambda x: x[1]):
        mask = (all_temps == t_k) & valid
        if mask.sum() > 10:
            r2_t = r2_score(y[mask], all_preds[mask])
            if config["log_transform"]:
                r2_t_real = r2_score(np.expm1(y[mask]),
                                     np.expm1(np.clip(all_preds[mask], 0, np.log1p(y_raw.max()))))
            else:
                r2_t_real = r2_t
            temp_c = int(round(t_k - 273.15))
            per_temp[f"{temp_c}C"] = {
                "r2": round(r2_t, 4),
                "r2_real": round(r2_t_real, 4),
                "n_samples": int(mask.sum()),
            }

    summary = {
        "property": prop_name,
        "overall_r2": round(overall_r2, 4),
        "overall_rmse": round(overall_rmse, 4),
        "overall_r2_real": round(overall_r2_real, 4),
        "overall_rmse_real": round(overall_rmse_real, 4),
        "unit": config["unit"],
        "log_transform": config["log_transform"],
        "per_temperature": per_temp,
        "fold_results": fold_results,
        "n_molecules": int(df_long["SMILES"].nunique()),
        "n_rows": len(df_long),
        "n_features": len(feature_cols),
        "training_time_seconds": round(elapsed, 1),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save predictions
    pred_df = df_long[["SMILES", "T_K", "value"]].copy()
    pred_df["pred"] = all_preds
    if config["log_transform"]:
        pred_df["value_real"] = np.expm1(y)
        pred_df["pred_real"] = np.expm1(np.clip(all_preds, 0, np.log1p(y_raw.max())))
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    # Print summary
    logger.info("")
    logger.info("%-6s  %-8s  %-8s  %s", "Temp", "R²", "R²(real)", "N")
    for temp_label, vals in per_temp.items():
        logger.info("%-6s  %-8.4f  %-8.4f  %d",
                     temp_label, vals["r2"], vals["r2_real"], vals["n_samples"])
    logger.info("")
    logger.info("Overall R²: %.4f | R²(real): %.4f | RMSE(real): %.4f %s | %.1fs",
                overall_r2, overall_r2_real, overall_rmse_real, config["unit"], elapsed)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", type=str, default=None,
                        help="Single property to train (default: all)")
    args = parser.parse_args()

    if args.property:
        props = {args.property: PROPERTIES[args.property]}
    else:
        props = PROPERTIES

    results = {}
    for name, config in props.items():
        results[name] = train_property(name, config)

    # Final comparison table
    if len(results) > 1:
        logger.info("")
        logger.info("=" * 60)
        logger.info("ALL PROPERTIES SUMMARY")
        logger.info("=" * 60)
        logger.info("%-12s  %-8s  %-8s  %-10s  %-6s  %s", "Property", "R²", "R²(real)", "RMSE(real)", "N_mol", "Time")
        for name, s in results.items():
            logger.info("%-12s  %-8.4f  %-8.4f  %-10.4f  %-6d  %.1fs",
                         name, s["overall_r2"], s["overall_r2_real"],
                         s["overall_rmse_real"], s["n_molecules"], s["training_time_seconds"])


if __name__ == "__main__":
    main()
