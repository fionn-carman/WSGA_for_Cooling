#!/usr/bin/env python3
"""
5-fold XGBoost ensemble with RFE feature selection.

Combines:
- 5-fold cross-validation ensemble (better generalisation than single split)
- RFE with multiple repeats per fold (selects optimal feature subset)
- RandomizedSearchCV hyperparameter tuning on selected features
- StandardScaler preprocessing

For each outer fold:
  1. RFE (3 repeats) on training set -> select best feature subset
  2. RandomizedSearchCV on training set with selected features
  3. Evaluate on held-out test fold

Saves 5 models per target, compatible with load_fom1_direct_models().

Usage:
    python train_5fold_rfe.py --target Density_40C_g_cm^3
    python train_5fold_rfe.py --target Kinematic_Viscosity_40C --log_target
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# RFE with multiple repeats
# ============================================================

def run_rfe_repeat(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    step_size: int = 2,
    random_state: int = None,
) -> Tuple[int, List[str], float]:
    """
    Single RFE pass: iteratively remove features, pick count with lowest RMSE.

    Returns (best_n_features, best_feature_names, best_rmse).
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    current_features = list(range(len(feature_names)))
    best_rmse = float("inf")
    best_n = len(feature_names)
    best_feats = current_features.copy()

    while len(current_features) > 1:
        xgb = XGBRegressor(
            objective="reg:squarederror", n_jobs=-1,
            random_state=random_state, verbosity=0,
        )
        rfe = RFE(estimator=xgb, n_features_to_select=1, step=1)
        rfe.fit(X_train[:, current_features], y_train)

        n_to_keep = max(len(current_features) - step_size, 1)
        keep_mask = rfe.ranking_ <= n_to_keep
        kept = [f for f, k in zip(current_features, keep_mask) if k]

        xgb.fit(X_train[:, kept], y_train)
        y_pred = xgb.predict(X_val[:, kept])
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_n = len(kept)
            best_feats = kept.copy()

        current_features = kept

    return best_n, [feature_names[i] for i in best_feats], best_rmse


def select_features_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 3,
    step_size: int = 2,
) -> List[str]:
    """Run RFE with multiple repeats and return the best feature set."""
    best_rmse = float("inf")
    best_features = feature_names

    for i in range(n_repeats):
        n_feat, feats, rmse = run_rfe_repeat(
            X, y, feature_names, step_size=step_size, random_state=i
        )
        logger.info("    RFE repeat %d: %d features, RMSE=%.4f", i + 1, n_feat, rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_features = feats

    logger.info("    => Selected %d / %d features (RMSE=%.4f)",
                len(best_features), len(feature_names), best_rmse)
    return best_features


# ============================================================
# Main training
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train 5-fold XGBoost ensemble with RFE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=str, required=True,
                        help="Target column name (e.g. Density_40C_g_cm^3)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing *_cleaned.csv files")
    parser.add_argument("--outdir", type=str, default="./output_5fold",
                        help="Output directory")
    parser.add_argument("--log_target", action="store_true",
                        help="Apply log1p transform to target")
    parser.add_argument("--n_rfe_repeats", type=int, default=3,
                        help="Number of RFE repeats per fold")
    parser.add_argument("--rfe_step_size", type=int, default=2,
                        help="Features to drop per RFE iteration")
    parser.add_argument("--n_random_iter", type=int, default=1000,
                        help="RandomizedSearchCV iterations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data_dir) / f"{args.target}_cleaned.csv"
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return

    df = pd.read_csv(data_path)
    logger.info("Target: %s (%d samples)", args.target, len(df))

    descriptor_cols = [c for c in df.columns if c not in ("SMILES", args.target)]
    X = df[descriptor_cols].values.astype(np.float64)
    y = df[args.target].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if args.log_target:
        y = np.log1p(y)
        logger.info("Applying log1p transform")

    param_dist = {
        "n_estimators": [100, 200, 300, 500, 1000, 2000, 5000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5],
        "max_depth": [3, 4, 5, 6, 10, 20, 50, 100],
        "min_child_weight": [1, 2, 3, 4, 5, 6],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    out_dir = Path(args.outdir) / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X)):
        logger.info("=" * 50)
        logger.info("FOLD %d/4", fold_id)
        logger.info("=" * 50)
        t0 = time.time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Step 1: RFE feature selection on training set
        logger.info("  Step 1: RFE feature selection")
        selected_features = select_features_rfe(
            X_train, y_train, descriptor_cols,
            n_repeats=args.n_rfe_repeats,
            step_size=args.rfe_step_size,
        )
        feat_idx = [descriptor_cols.index(f) for f in selected_features]

        X_train_sel = X_train[:, feat_idx]
        X_test_sel = X_test[:, feat_idx]

        # Step 2: Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)
        X_test_s = scaler.transform(X_test_sel)

        # Step 3: RandomizedSearchCV
        logger.info("  Step 2: RandomizedSearchCV (%d iter)", args.n_random_iter)
        search = RandomizedSearchCV(
            XGBRegressor(random_state=args.seed, n_jobs=-1, verbosity=0),
            param_dist,
            n_iter=args.n_random_iter,
            cv=5,
            scoring="r2",
            random_state=args.seed + fold_id,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train_s, y_train)
        best_model = search.best_estimator_
        logger.info("  Best inner CV R2: %.4f", search.best_score_)

        # Step 4: Evaluate on test fold
        y_pred = best_model.predict(X_test_s)

        if args.log_target:
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred

        r2 = r2_score(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        elapsed = time.time() - t0

        logger.info("  => R2=%.4f  RMSE=%.4f  MAE=%.4f  (%d features, %.0fs)",
                    r2, rmse, mae, len(selected_features), elapsed)

        fold_metrics.append({
            "fold": fold_id,
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_features": len(selected_features),
            "training_time_s": float(elapsed),
            "best_inner_cv_r2": float(search.best_score_),
            "best_params": search.best_params_,
        })

        # Save fold model
        joblib.dump({
            "model": best_model,
            "scaler": scaler,
            "params": search.best_params_,
            "descriptor_columns": selected_features,
            "log_target": args.log_target,
        }, out_dir / f"xgboost_descriptors_fold{fold_id}_model.joblib")

        with open(out_dir / f"xgboost_descriptors_fold{fold_id}_metrics.json", "w") as f:
            json.dump(fold_metrics[-1], f, indent=2)

        # Save selected features for this fold
        with open(out_dir / f"fold{fold_id}_selected_features.txt", "w") as f:
            f.write("\n".join(selected_features))

    # Summary
    mean_r2 = np.mean([m["r2"] for m in fold_metrics])
    std_r2 = np.std([m["r2"] for m in fold_metrics])
    mean_rmse = np.mean([m["rmse"] for m in fold_metrics])
    mean_mae = np.mean([m["mae"] for m in fold_metrics])
    mean_nfeat = np.mean([m["n_features"] for m in fold_metrics])

    logger.info("=" * 50)
    logger.info("SUMMARY: %s", args.target)
    logger.info("  Mean R2:       %.4f (+-%.4f)", mean_r2, std_r2)
    logger.info("  Mean RMSE:     %.4f", mean_rmse)
    logger.info("  Mean MAE:      %.4f", mean_mae)
    logger.info("  Mean features: %.0f / %d", mean_nfeat, len(descriptor_cols))
    logger.info("=" * 50)

    summary = {
        "target": args.target,
        "log_target": args.log_target,
        "n_samples": len(df),
        "n_total_features": len(descriptor_cols),
        "mean_r2": float(mean_r2),
        "std_r2": float(std_r2),
        "mean_rmse": float(mean_rmse),
        "mean_mae": float(mean_mae),
        "mean_n_features": float(mean_nfeat),
        "fold_metrics": fold_metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
