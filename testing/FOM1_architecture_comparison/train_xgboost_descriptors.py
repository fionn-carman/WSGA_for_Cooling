#!/usr/bin/env python3
"""
XGBoost on RDKit descriptors only for FOM1 prediction.

Features: ~200 RDKit descriptors (no fingerprints).
Uses RandomizedSearchCV with 200 iterations and 5-fold inner CV.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost on RDKit descriptors for FOM1 prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"),
                        help="Directory containing prepared data")
    parser.add_argument("--target", type=str, default="FOM1_avg",
                        help="Target column to predict (FOM1_40, FOM1_100, or FOM1_avg)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold to train (0-4). If omitted, trains all.")
    parser.add_argument("--n_iter", type=int, default=200,
                        help="RandomizedSearchCV iterations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)

    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(data_dir / "fom1_dataset.csv")
    with open(data_dir / "fold_indices.json") as f:
        fold_indices = json.load(f)
    with open(data_dir / "descriptor_columns.json") as f:
        descriptor_columns = json.load(f)

    X = df[descriptor_columns].values
    y = df[args.target].values
    logger.info("Target: %s", args.target)
    logger.info("Feature matrix: %s (descriptors only)", X.shape)

    # HP search space
    param_dist = {
        "n_estimators": [100, 200, 500, 1000, 2000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 6, 8, 10],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    folds = range(5) if args.fold is None else [args.fold]

    for fold_id in folds:
        logger.info("=" * 50)
        logger.info("FOLD %d — XGBoost Descriptors", fold_id)
        logger.info("=" * 50)
        t0 = time.time()

        train_idx = fold_indices[str(fold_id)]["train"]
        test_idx = fold_indices[str(fold_id)]["test"]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # RandomizedSearchCV
        base_model = XGBRegressor(
            random_state=args.seed,
            n_jobs=-1,
            verbosity=0,
        )
        search = RandomizedSearchCV(
            base_model,
            param_dist,
            n_iter=args.n_iter,
            cv=5,
            scoring="r2",
            random_state=args.seed,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train_s, y_train)
        best_model = search.best_estimator_
        logger.info("Best params: %s", search.best_params_)
        logger.info("Best inner CV R²: %.4f", search.best_score_)

        # Predict on test
        y_pred = best_model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - t0

        logger.info("Test R²: %.4f, RMSE: %.4f, MAE: %.4f (%.1fs)", r2, rmse, mae, elapsed)

        # Save predictions
        pred_df = pd.DataFrame({
            "SMILES": df.iloc[test_idx]["SMILES"].values,
            "y_true": y_test,
            "y_pred": y_pred,
        })
        pred_df.to_csv(data_dir / f"xgboost_descriptors_fold{fold_id}_predictions.csv", index=False)

        # Save metrics
        metrics = {
            "model": "xgboost_descriptors",
            "target": args.target,
            "fold": fold_id,
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "training_time_s": float(elapsed),
            "best_params": search.best_params_,
            "best_inner_cv_r2": float(search.best_score_),
        }
        with open(data_dir / f"xgboost_descriptors_fold{fold_id}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save model (include descriptor_columns so feature names stay with the model)
        joblib.dump({"model": best_model, "scaler": scaler, "params": search.best_params_,
                     "descriptor_columns": descriptor_columns},
                     data_dir / f"xgboost_descriptors_fold{fold_id}_model.joblib")

    logger.info("XGBoost Descriptors training complete.")


if __name__ == "__main__":
    main()
