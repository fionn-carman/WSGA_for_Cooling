#!/usr/bin/env python3
"""
Train a single temperature-as-feature XGBoost model for kinematic viscosity.

Workflow:
  1. Read wide-format viscosity CSV (one row per molecule, 11 KV columns)
  2. Melt to long format (one row per molecule × temperature)
  3. Compute RDKit descriptors (cached to kv_long.csv)
  4. Train 5-fold GroupKFold XGBoost (molecule-level splits)
  5. RandomizedSearchCV with 50 iterations per fold
  6. Evaluate overall + per-temperature R²
  7. Save fold models + summary JSON

Usage:
    python train_quick.py
    python train_quick.py --input ../NISTDataFull/viscosity_cho.csv --skip-prep  # if kv_long.csv exists
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
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

# RDKit descriptor definitions
DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]
DESCRIPTOR_FUNCS = [desc[1] for desc in Descriptors._descList]

# Temperature columns in the wide CSV
TEMP_COLS = {
    "kv_0C_cSt": 273.15,
    "kv_10C_cSt": 283.15,
    "kv_20C_cSt": 293.15,
    "kv_30C_cSt": 303.15,
    "kv_40C_cSt": 313.15,
    "kv_50C_cSt": 323.15,
    "kv_60C_cSt": 333.15,
    "kv_70C_cSt": 343.15,
    "kv_80C_cSt": 353.15,
    "kv_90C_cSt": 363.15,
    "kv_100C_cSt": 373.15,
}


def calc_descriptors(mol):
    """Compute all RDKit descriptors for a single Mol object."""
    if mol is None:
        return [np.nan] * len(DESCRIPTOR_FUNCS)
    vals = []
    for func in DESCRIPTOR_FUNCS:
        try:
            v = func(mol)
            vals.append(v)
        except Exception:
            vals.append(np.nan)
    return vals


def prepare_long_format(input_csv: Path, output_csv: Path):
    """Convert wide CSV to long format with RDKit descriptors."""
    logger.info("Reading wide-format CSV: %s", input_csv)
    df_wide = pd.read_csv(input_csv)
    logger.info("  %d molecules, %d columns", len(df_wide), len(df_wide.columns))

    # Canonicalize SMILES
    logger.info("Canonicalizing SMILES...")
    canon = []
    for smi in df_wide["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        canon.append(Chem.MolToSmiles(mol) if mol else None)
    df_wide["SMILES"] = canon
    n_before = len(df_wide)
    df_wide = df_wide.dropna(subset=["SMILES"])
    logger.info("  %d/%d valid SMILES", len(df_wide), n_before)

    # Compute descriptors (one per unique molecule)
    logger.info("Computing RDKit descriptors for %d molecules...", len(df_wide))
    mols = [Chem.MolFromSmiles(smi) for smi in df_wide["SMILES"]]
    desc_data = [calc_descriptors(mol) for mol in mols]
    desc_df = pd.DataFrame(desc_data, columns=DESCRIPTOR_NAMES, index=df_wide.index)

    # Melt wide → long
    logger.info("Melting to long format...")
    rows = []
    for idx, row in df_wide.iterrows():
        smi = row["SMILES"]
        descs = desc_df.loc[idx]
        for col_name, t_k in TEMP_COLS.items():
            kv = row[col_name]
            if pd.notna(kv) and kv > 0:
                r = {"SMILES": smi, "T_K": t_k, "kv_cSt": kv}
                for d_name in DESCRIPTOR_NAMES:
                    r[d_name] = descs[d_name]
                rows.append(r)

    df_long = pd.DataFrame(rows)
    logger.info("  Long format: %d rows (%d molecules × up to %d temps)",
                len(df_long), df_wide["SMILES"].nunique(), len(TEMP_COLS))

    # Drop rows with NaN descriptors
    n_before = len(df_long)
    df_long = df_long.dropna()
    if len(df_long) < n_before:
        logger.info("  Dropped %d rows with NaN descriptors", n_before - len(df_long))

    df_long.to_csv(output_csv, index=False)
    logger.info("  Saved to %s", output_csv)
    return df_long


def train_model(df_long: pd.DataFrame, output_dir: Path):
    """Train 5-fold GroupKFold XGBoost with RandomizedSearchCV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Features: T_K + all RDKit descriptors
    feature_cols = ["T_K"] + DESCRIPTOR_NAMES
    # Filter to columns that actually exist in df_long
    feature_cols = [c for c in feature_cols if c in df_long.columns]
    logger.info("Features: %d (T_K + %d descriptors)", len(feature_cols), len(feature_cols) - 1)

    X = df_long[feature_cols].values
    y_raw = df_long["kv_cSt"].values
    y = np.log1p(y_raw)  # log-transform target
    groups = df_long["SMILES"].values  # molecule-level grouping

    # Remove constant or all-NaN features
    valid_mask = np.all(np.isfinite(X), axis=0)
    if not np.all(valid_mask):
        n_removed = (~valid_mask).sum()
        logger.info("  Removing %d non-finite features", n_removed)
        feature_cols = [f for f, v in zip(feature_cols, valid_mask) if v]
        X = X[:, valid_mask]

    # Fixed HPs from previous run (fold 0 best)
    fixed_params = {
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "reg_alpha": 0,
        "reg_lambda": 0.1,
    }
    logger.info("Using fixed HPs: %s", fixed_params)

    gkf = GroupKFold(n_splits=5)
    fold_results = []
    all_preds = np.full(len(y), np.nan)
    all_temps = df_long["T_K"].values

    t0 = time.time()
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info("=== Fold %d/5 ===", fold_i + 1)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train with fixed HPs
        best_model = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=fold_i,
            verbosity=0,
            **fixed_params,
        )
        best_model.fit(X_train_s, y_train)
        logger.info("  Trained with fixed params")

        # Evaluate on held-out fold
        y_pred = best_model.predict(X_test_s)
        all_preds[test_idx] = y_pred

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Back-transform for real-unit metrics (clip predictions to data range)
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(np.clip(y_pred, 0, np.log1p(y_raw.max())))
        r2_real = r2_score(y_test_real, y_pred_real)
        rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

        logger.info("  Test R² (log): %.4f, RMSE (log): %.4f", r2, rmse)
        logger.info("  Test R² (real): %.4f, RMSE (real): %.4f cSt", r2_real, rmse_real)

        # Per-temperature R² for this fold
        per_temp_r2 = {}
        for t_k in sorted(TEMP_COLS.values()):
            mask = all_temps[test_idx] == t_k
            if mask.sum() > 10:
                r2_t = r2_score(y_test[mask], y_pred[mask])
                per_temp_r2[f"{t_k:.2f}K"] = round(r2_t, 4)

        fold_result = {
            "fold": fold_i,
            "r2_log": round(r2, 4),
            "rmse_log": round(rmse, 4),
            "mae_log": round(mae, 4),
            "r2_real": round(r2_real, 4),
            "rmse_real_cSt": round(rmse_real, 4),
            "best_params": fixed_params,
            "per_temp_r2_log": per_temp_r2,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        fold_results.append(fold_result)

        # Save fold model
        model_path = output_dir / f"fold{fold_i}_model.joblib"
        joblib.dump({
            "model": best_model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "log_transform": True,
        }, model_path)
        logger.info("  Saved %s", model_path)

    elapsed = time.time() - t0
    logger.info("Training complete in %.1f seconds", elapsed)

    # Overall metrics (across all folds)
    valid = ~np.isnan(all_preds)
    overall_r2_log = r2_score(y[valid], all_preds[valid])
    overall_rmse_log = np.sqrt(mean_squared_error(y[valid], all_preds[valid]))
    preds_clipped = np.clip(all_preds[valid], 0, np.log1p(y_raw.max()))
    overall_r2_real = r2_score(np.expm1(y[valid]), np.expm1(preds_clipped))
    overall_rmse_real = np.sqrt(mean_squared_error(np.expm1(y[valid]), np.expm1(preds_clipped)))

    # Per-temperature overall R²
    per_temp_overall = {}
    for col_name, t_k in sorted(TEMP_COLS.items(), key=lambda x: x[1]):
        mask = (all_temps == t_k) & valid
        if mask.sum() > 10:
            r2_t = r2_score(y[mask], all_preds[mask])
            r2_t_real = r2_score(np.expm1(y[mask]), np.expm1(np.clip(all_preds[mask], 0, np.log1p(y_raw.max()))))
            temp_c = int(round(t_k - 273.15))
            per_temp_overall[f"{temp_c}C"] = {
                "r2_log": round(r2_t, 4),
                "r2_real": round(r2_t_real, 4),
                "n_samples": int(mask.sum()),
            }

    summary = {
        "overall_r2_log": round(overall_r2_log, 4),
        "overall_rmse_log": round(overall_rmse_log, 4),
        "overall_r2_real": round(overall_r2_real, 4),
        "overall_rmse_real_cSt": round(overall_rmse_real, 4),
        "per_temperature": per_temp_overall,
        "fold_results": fold_results,
        "n_molecules": int(df_long["SMILES"].nunique()),
        "n_rows": len(df_long),
        "n_features": len(feature_cols),
        "training_time_seconds": round(elapsed, 1),
        "baseline_kv40_r2": 0.688,
        "baseline_kv100_r2": 0.653,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Print results table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Overall R² (log):  %.4f", overall_r2_log)
    logger.info("Overall R² (real): %.4f", overall_r2_real)
    logger.info("Overall RMSE (real): %.4f cSt", overall_rmse_real)
    logger.info("")
    logger.info("Per-temperature R² (log-space, all folds):")
    logger.info("  %-6s  %-8s  %-8s  %s", "Temp", "R²(log)", "R²(real)", "N")
    for temp_label, vals in per_temp_overall.items():
        logger.info("  %-6s  %-8.4f  %-8.4f  %d",
                     temp_label, vals["r2_log"], vals["r2_real"], vals["n_samples"])
    logger.info("")
    logger.info("Baseline comparison:")
    logger.info("  KV 40°C:  old R²=0.688, new R²=%.4f", per_temp_overall.get("40C", {}).get("r2_log", float("nan")))
    logger.info("  KV 100°C: old R²=0.653, new R²=%.4f", per_temp_overall.get("100C", {}).get("r2_log", float("nan")))

    # Save predictions for analysis
    pred_df = df_long[["SMILES", "T_K", "kv_cSt"]].copy()
    pred_df["log_kv_true"] = y
    pred_df["log_kv_pred"] = all_preds
    pred_df["kv_pred_cSt"] = np.expm1(np.clip(all_preds, 0, np.log1p(y_raw.max())))
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    logger.info("Predictions saved to %s", output_dir / "predictions.csv")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train temp-as-feature XGBoost for KV")
    parser.add_argument("--input", type=str,
                        default="../NISTDataFull/viscosity_cho.csv",
                        help="Wide-format viscosity CSV")
    parser.add_argument("--output-dir", type=str,
                        default="kv_temp_feature",
                        help="Output directory for models")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Skip data prep, load existing kv_long.csv")
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    long_csv = Path("kv_long.csv")

    if args.skip_prep and long_csv.exists():
        logger.info("Loading cached long-format data: %s", long_csv)
        df_long = pd.read_csv(long_csv)
        logger.info("  %d rows, %d columns", len(df_long), len(df_long.columns))
    else:
        df_long = prepare_long_format(input_csv, long_csv)

    summary = train_model(df_long, output_dir)
    return summary


if __name__ == "__main__":
    main()
