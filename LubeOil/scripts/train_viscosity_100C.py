#!/usr/bin/env python3
"""Train viscosity_100C XGBoost model, reusing features+hyperparams from the
40C model. Saves a drop-in replacement for models/viscosity_40C that wsga_helper
can load the same way.

Run from repo root:
    python LubeOil/scripts/train_viscosity_100C.py
"""

import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "training" / "XGBoost" / "nist8100"))
from fold_utils import assign_folds, get_fold_indices  # noqa: E402

DATA_CSV = REPO / "training" / "data" / "nist_8100" / "viscosity_cho_cleaned.csv"
DESC_CSV = REPO / "training" / "data" / "nist_8100" / "descriptors_rdkit_mordred.csv"
REF_MODEL = REPO / "models" / "viscosity_40C" / "model" / "xgb_model.joblib"
OUT_DIR = REPO / "models" / "viscosity_100C" / "model"

TARGET_COL = "kv_100C_cSt"
N_FOLDS = 5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = joblib.load(REF_MODEL)
    selected = ref["selected_features"]
    best_params = ref["best_params"]
    log_transform = ref["log_transform"]
    print(f"Reusing {len(selected)} features and params from viscosity_40C:")
    print(f"  {best_params}")

    print(f"Loading {DATA_CSV.name}...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    df = df[df[TARGET_COL].notna()].copy()

    print(f"Loading descriptors {DESC_CSV.name}...")
    desc = pd.read_csv(DESC_CSV, low_memory=False)

    missing = [f for f in selected if f not in desc.columns]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} descriptors in CSV, first 5: {missing[:5]}")

    merged = df[["SMILES", TARGET_COL]].merge(desc[["SMILES", *selected]], on="SMILES", how="inner")
    print(f"{len(df)} SMILES in CSV, {len(merged)} after descriptor join")

    smiles = merged["SMILES"].values
    y_raw = merged[TARGET_COL].values.astype(np.float64)
    X = np.nan_to_num(merged[selected].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.log1p(y_raw) if log_transform else y_raw.copy()

    folds = assign_folds(smiles, n_folds=N_FOLDS)

    fold_metrics = []
    all_preds = np.full(len(y), np.nan)

    for fold_i in range(N_FOLDS):
        t0 = time.time()
        train_idx, test_idx = get_fold_indices(folds, fold_i)
        model = XGBRegressor(
            objective="reg:squarederror",
            **best_params,
            random_state=fold_i,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        all_preds[test_idx] = preds

        if log_transform:
            y_real = np.expm1(y[test_idx])
            preds_real = np.expm1(preds)
        else:
            y_real, preds_real = y[test_idx], preds

        r2 = r2_score(y_real, preds_real)
        rmse = math.sqrt(mean_squared_error(y_real, preds_real))
        mae = mean_absolute_error(y_real, preds_real)
        fold_metrics.append({"fold": fold_i, "r2": r2, "rmse": rmse, "mae": mae,
                              "n_test": int(len(test_idx))})
        print(f"  Fold {fold_i}: R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f} ({time.time()-t0:.1f}s)")

        joblib.dump({
            "model": model,
            "selected_features": selected,
            "best_params": best_params,
            "log_transform": log_transform,
            "property": "viscosity",
        }, OUT_DIR / f"fold{fold_i}_model.joblib")

    valid = ~np.isnan(all_preds)
    y_real = np.expm1(y[valid]) if log_transform else y[valid]
    pred_real = np.expm1(all_preds[valid]) if log_transform else all_preds[valid]
    oof_r2 = r2_score(y_real, pred_real)
    oof_rmse = math.sqrt(mean_squared_error(y_real, pred_real))
    oof_mae = mean_absolute_error(y_real, pred_real)
    print(f"OOF 5-fold: R2={oof_r2:.4f} RMSE={oof_rmse:.4f} MAE={oof_mae:.4f}  N={valid.sum()}")

    pd.DataFrame({
        "SMILES": smiles, "fold": folds,
        "y_true": np.expm1(y) if log_transform else y,
        "y_pred": np.expm1(all_preds) if log_transform else all_preds,
    }).to_csv(OUT_DIR / "predictions.csv", index=False)

    print("Training full-data model...")
    full_model = XGBRegressor(
        objective="reg:squarederror",
        **best_params, random_state=42, n_jobs=-1, verbosity=0,
    )
    full_model.fit(X, y)
    joblib.dump({
        "model": full_model,
        "selected_features": selected,
        "best_params": best_params,
        "log_transform": log_transform,
        "property": "viscosity",
    }, OUT_DIR / "xgb_model.joblib")

    summary = {
        "property": "viscosity",
        "temperature": "100C",
        "r2": round(oof_r2, 4),
        "rmse": round(oof_rmse, 4),
        "mae": round(oof_mae, 4),
        "n_molecules": int(valid.sum()),
        "n_features": len(selected),
        "log_transform": log_transform,
        "best_params": best_params,
        "feature_source": "reused from viscosity_40C",
        "fold_metrics": fold_metrics,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {OUT_DIR}")
    print(f"  OOF R2={oof_r2:.4f} RMSE={oof_rmse:.4f} on {valid.sum()} molecules")


if __name__ == "__main__":
    main()
