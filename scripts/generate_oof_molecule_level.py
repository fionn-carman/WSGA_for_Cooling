#!/usr/bin/env python3
"""
Generate molecule_level.csv (per-molecule OOF predictions + Mahalanobis distance)
for constraint properties (DC, FP, MP) using deployed model configs.

Reuses the deployed model's selected features and best_params for speed —
no feature selection or hyperparameter tuning, just 5-fold CV with OOF predictions.

Usage:
    python generate_oof_molecule_level.py          # all 3 properties
    python generate_oof_molecule_level.py --prop dc # single property
"""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

REPO = Path(__file__).resolve().parent.parent
OOD_DIR = REPO / "OOD_testing" / "r2_vs_ood_percentile_hpc"

PROPERTIES = {
    "dc": {
        "data": REPO / "training" / "data" / "constraints" / "DC_exp_cleaned.csv",
        "target_col": "DC_exp",
        "model": REPO / "models" / "DC_exp" / "model" / "xgb_model.joblib",
        "out_dir": OOD_DIR / "dc",
    },
    "fp": {
        "data": REPO / "training" / "data" / "constraints" / "flashpoint_cleaned.csv",
        "target_col": "flashpoint",
        "model": REPO / "models" / "flashpoint" / "model" / "xgb_model.joblib",
        "out_dir": OOD_DIR / "fp",
    },
    "mp": {
        "data": REPO / "training" / "data" / "constraints" / "MP-Measured_cleaned.csv",
        "target_col": "MP-Measured",
        "model": REPO / "models" / "MP-Measured" / "model" / "xgb_model.joblib",
        "out_dir": OOD_DIR / "mp",
    },
}

SEED = 42


def compute_mahalanobis(X: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from covariance centre (LedoitWolf shrinkage)."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    # Decorrelate: drop features with |r| > 0.95
    corr = np.nan_to_num(np.corrcoef(Xs, rowvar=False), nan=0.0)
    keep = list(range(Xs.shape[1]))
    drop = set()
    for i in range(len(keep)):
        if i in drop:
            continue
        for j in range(i + 1, len(keep)):
            if j in drop:
                continue
            if abs(corr[i, j]) > 0.95:
                drop.add(j)
    keep = [i for i in keep if i not in drop]
    Xs = Xs[:, keep]
    lw = LedoitWolf().fit(Xs)
    diff = Xs - lw.location_
    return np.sqrt(np.maximum(np.sum(diff @ lw.precision_ * diff, axis=1), 0))


def generate_molecule_level(prop_key: str) -> None:
    cfg = PROPERTIES[prop_key]
    print(f"\n{'='*60}")
    print(f"Generating molecule_level.csv for {prop_key}")
    print(f"{'='*60}")

    # Load training data (already has descriptors as columns)
    df = pd.read_csv(cfg["data"])
    smiles = df["SMILES"].values
    y = df[cfg["target_col"]].values
    print(f"  Loaded {len(df)} molecules")

    # Load deployed model config
    model_dict = joblib.load(cfg["model"])
    features = model_dict["features"]
    best_params = model_dict["best_params"]
    log_target = model_dict.get("log_target", False)
    print(f"  Features: {len(features)}, log_target: {log_target}")
    print(f"  best_params: {best_params}")

    # Extract feature matrix
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing: {missing[:5]}...")
        features = [f for f in features if f in df.columns]
    X = df[features].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute Mahalanobis distance on deployed features
    print("  Computing Mahalanobis distances...")
    md = compute_mahalanobis(X)

    # Transform target if needed
    if log_target:
        y_train_target = np.log1p(y)
    else:
        y_train_target = y.copy()

    # 5-fold CV with OOF predictions
    print("  Running 5-fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    y_pred_oof = np.full(len(y), np.nan)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        xgb = XGBRegressor(**best_params, random_state=SEED, n_jobs=-1)
        xgb.fit(X[train_idx], y_train_target[train_idx])
        preds = xgb.predict(X[val_idx])
        if log_target:
            preds = np.expm1(preds)
        y_pred_oof[val_idx] = preds
        r2_fold = 1 - np.sum((y[val_idx] - preds)**2) / np.sum(
            (y[val_idx] - y[val_idx].mean())**2)
        print(f"    Fold {fold_idx}: R²={r2_fold:.4f}, "
              f"MAE={np.mean(np.abs(y[val_idx] - preds)):.4f}")

    abs_error = np.abs(y - y_pred_oof)
    overall_r2 = 1 - np.sum((y - y_pred_oof)**2) / np.sum((y - y.mean())**2)
    overall_mae = np.mean(abs_error)
    print(f"  Overall: R²={overall_r2:.4f}, MAE={overall_mae:.4f}")

    # Save
    out = pd.DataFrame({
        "SMILES": smiles,
        "y_true": y,
        "y_pred_oof": y_pred_oof,
        "abs_error": abs_error,
        "md": md,
    })
    out_path = cfg["out_dir"] / "molecule_level.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prop", choices=list(PROPERTIES.keys()),
                        help="Single property to generate (default: all)")
    args = parser.parse_args()

    if args.prop:
        generate_molecule_level(args.prop)
    else:
        for key in PROPERTIES:
            generate_molecule_level(key)
    print("\nDone.")
