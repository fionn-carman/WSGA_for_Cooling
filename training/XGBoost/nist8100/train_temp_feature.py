#!/usr/bin/env python3
"""
Temperature-as-feature XGBoost training with 5-fold CV and RFE.

Instead of separate models per temperature, this trains ONE model per
property that takes (molecular descriptors, T) as input. Temperature is
an additional feature alongside RDKit descriptors.

Advantages:
- Single model learns the full T-dependence (better for interpolation)
- More training data per model (all temps pooled)
- Potentially better for nonlinear properties like viscosity

Architecture:
  Input: [descriptor_1, descriptor_2, ..., descriptor_N, T_K]
  Output: property value at that temperature

Uses the same 5-fold CV + RFE + RandomizedSearchCV pipeline as
train_5fold_rfe.py, but on long-format data with T as a feature.

The 5-fold split is done at the MOLECULE level (not row level) to
prevent data leakage — all temperatures of a molecule go into the
same fold.

Usage:
    python train_temp_feature.py --property density --data long/density_long.csv
    python train_temp_feature.py --property kv --data long/kv_long.csv --log_target
    python train_temp_feature.py --all --data-dir data_multitemp/long
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
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.feature_selection import RFE
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    GroupShuffleSplit,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from fold_utils import assign_folds, get_fold_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]
DESCRIPTOR_FNS = {desc[0]: desc[1] for desc in Descriptors._descList}


def ensure_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RDKit descriptors if not already present in the DataFrame."""
    present = [c for c in DESCRIPTOR_NAMES if c in df.columns]
    if len(present) >= 10:
        return df
    logger.info("  Descriptors missing (%d found) — computing from SMILES", len(present))
    rows = []
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({n: np.nan for n in DESCRIPTOR_NAMES})
        else:
            rows.append({n: DESCRIPTOR_FNS[n](mol) for n in DESCRIPTOR_NAMES})
    desc_df = pd.DataFrame(rows, index=df.index)
    new_cols = [c for c in desc_df.columns if c not in df.columns]
    df = pd.concat([df, desc_df[new_cols]], axis=1)
    logger.info("  Computed %d descriptors for %d molecules", len(new_cols), len(df))
    return df


# Properties that benefit from log-transform
LOG_TARGETS_DEFAULT = {"kv", "fom1"}

# All available properties
ALL_PROPERTIES = ["density", "kv", "tc", "cpsat", "beta", "fom1"]


# ============================================================
# RFE with multiple repeats (molecule-aware splitting)
# ============================================================


def run_rfe_repeat(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    step_size: int = 2,
    random_state: int = None,
) -> Tuple[int, List[str], float]:
    """Single RFE pass with molecule-grouped train/val split."""
    # Split by molecule groups (no leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    current_features = list(range(len(feature_names)))
    best_rmse = float("inf")
    best_n = len(feature_names)
    best_feats = current_features.copy()

    while len(current_features) > 1:
        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
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
    groups: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 3,
    step_size: int = 2,
) -> List[str]:
    """Run RFE with multiple repeats, using molecule-grouped splits."""
    best_rmse = float("inf")
    best_features = feature_names

    for i in range(n_repeats):
        n_feat, feats, rmse = run_rfe_repeat(
            X, y, groups, feature_names,
            step_size=step_size,
            random_state=i,
        )
        logger.info(
            "    RFE repeat %d: %d features, RMSE=%.4f", i + 1, n_feat, rmse
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_features = feats

    logger.info(
        "    => Selected %d / %d features (RMSE=%.4f)",
        len(best_features), len(feature_names), best_rmse,
    )
    return best_features


# ============================================================
# Training
# ============================================================


def train_property(
    prop: str,
    data_path: Path,
    outdir: Path,
    log_target: bool = False,
    n_rfe_repeats: int = 3,
    rfe_step_size: int = 2,
    n_random_iter: int = 1000,
    seed: int = 42,
):
    """Train temperature-as-feature model for one property."""

    df = pd.read_csv(data_path)
    logger.info("Property: %s (%d rows, %d molecules)", prop, len(df), df["SMILES"].nunique())

    # Failsafe: compute descriptors if missing
    df = ensure_descriptors(df)

    # Required columns
    if "T_K" not in df.columns:
        logger.error("Missing T_K column in %s", data_path)
        return
    if prop not in df.columns:
        logger.error("Missing property column '%s' in %s", prop, data_path)
        return

    # Feature columns: all RDKit descriptors + T_K
    meta_cols = {"SMILES", "T_C", "T_K", prop}
    descriptor_cols = [c for c in df.columns if c not in meta_cols]
    feature_cols = descriptor_cols + ["T_K"]

    # Prepare arrays
    X = df[feature_cols].values.astype(np.float64)
    y = df[prop].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Molecule groups for inner CV (RFE, RandomizedSearchCV)
    smiles_list = df["SMILES"].values
    unique_smiles = sorted(set(smiles_list))
    smiles_to_group = {s: i for i, s in enumerate(unique_smiles)}
    groups = np.array([smiles_to_group[s] for s in smiles_list])

    # Hash-based outer folds — consistent across properties
    mol_folds = assign_folds(smiles_list, n_folds=5)

    if log_target:
        y = np.log1p(y)
        logger.info("Applying log1p transform to target")

    param_dist = {
        "n_estimators": [100, 200, 300, 500, 1000, 2000, 5000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5],
        "max_depth": [3, 4, 5, 6, 10, 20, 50, 100],
        "min_child_weight": [1, 2, 3, 4, 5, 6],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    out_dir = outdir / f"{prop}_temp_feature"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5-fold CV at molecule level (hash-based for cross-property consistency)
    fold_metrics = []

    for fold_id in range(5):
        train_idx, test_idx = get_fold_indices(mol_folds, fold_id)
        # Resume check
        metrics_path = out_dir / f"fold{fold_id}_metrics.json"
        model_path = out_dir / f"fold{fold_id}_model.joblib"
        if metrics_path.exists() and model_path.exists():
            with open(metrics_path) as f:
                prev = json.load(f)
            fold_metrics.append(prev)
            logger.info(
                "FOLD %d/4 — already complete (R2=%.4f), skipping",
                fold_id, prev["r2"],
            )
            continue

        logger.info("=" * 50)
        logger.info("FOLD %d/4", fold_id)
        logger.info("=" * 50)
        t0 = time.time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        n_train_mols = len(set(groups_train))
        n_test_mols = len(set(groups[test_idx]))
        logger.info(
            "  Train: %d rows (%d molecules) | Test: %d rows (%d molecules)",
            len(X_train), n_train_mols, len(X_test), n_test_mols,
        )

        # Step 1: RFE feature selection (T_K is always kept)
        logger.info("  Step 1: RFE feature selection")
        selected_features = select_features_rfe(
            X_train, y_train, groups_train, feature_cols,
            n_repeats=n_rfe_repeats,
            step_size=rfe_step_size,
        )

        # Ensure T_K is always included
        if "T_K" not in selected_features:
            selected_features.append("T_K")
            logger.info("    (Added T_K — always required)")

        feat_idx = [feature_cols.index(f) for f in selected_features]

        X_train_sel = X_train[:, feat_idx]
        X_test_sel = X_test[:, feat_idx]

        # Step 2: Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)
        X_test_s = scaler.transform(X_test_sel)

        # Step 3: RandomizedSearchCV (using GroupKFold for inner CV)
        logger.info("  Step 2: RandomizedSearchCV (%d iter)", n_random_iter)
        inner_cv = GroupKFold(n_splits=5)
        search = RandomizedSearchCV(
            XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0),
            param_dist,
            n_iter=n_random_iter,
            cv=inner_cv,
            scoring="r2",
            random_state=seed + fold_id,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train_s, y_train, groups=groups_train)
        best_model = search.best_estimator_
        logger.info("  Best inner CV R2: %.4f", search.best_score_)

        # Step 4: Evaluate on test fold
        y_pred = best_model.predict(X_test_s)

        if log_target:
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred

        r2 = r2_score(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        elapsed = time.time() - t0

        logger.info(
            "  => R2=%.4f  RMSE=%.4f  MAE=%.4f  (%d features, %.0fs)",
            r2, rmse, mae, len(selected_features), elapsed,
        )

        # Per-temperature R2
        test_df = pd.DataFrame({
            "y_true": y_test_orig,
            "y_pred": y_pred_orig,
            "T_K": X_test[:, feature_cols.index("T_K")],
        })
        temp_r2 = {}
        for t_k, grp in test_df.groupby("T_K"):
            if len(grp) >= 5:
                tr2 = r2_score(grp["y_true"], grp["y_pred"])
                temp_r2[f"{round(t_k - 273.15)}C"] = round(tr2, 4)

        logger.info("  Per-temperature R2: %s", temp_r2)

        fold_metrics.append({
            "fold": fold_id,
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_features": len(selected_features),
            "training_time_s": float(elapsed),
            "best_inner_cv_r2": float(search.best_score_),
            "best_params": search.best_params_,
            "per_temperature_r2": temp_r2,
        })

        # Save fold model
        joblib.dump(
            {
                "model": best_model,
                "scaler": scaler,
                "params": search.best_params_,
                "descriptor_columns": selected_features,
                "log_target": log_target,
                "property": prop,
                "architecture": "temp_as_feature",
            },
            model_path,
        )

        with open(metrics_path, "w") as f:
            json.dump(fold_metrics[-1], f, indent=2)

        with open(out_dir / f"fold{fold_id}_selected_features.txt", "w") as f:
            f.write("\n".join(selected_features))

    # Summary
    mean_r2 = np.mean([m["r2"] for m in fold_metrics])
    std_r2 = np.std([m["r2"] for m in fold_metrics])
    mean_rmse = np.mean([m["rmse"] for m in fold_metrics])
    mean_mae = np.mean([m["mae"] for m in fold_metrics])
    mean_nfeat = np.mean([m["n_features"] for m in fold_metrics])

    logger.info("=" * 50)
    logger.info("SUMMARY: %s (temp-as-feature)", prop)
    logger.info("  Mean R2:       %.4f (+-%.4f)", mean_r2, std_r2)
    logger.info("  Mean RMSE:     %.4f", mean_rmse)
    logger.info("  Mean MAE:      %.4f", mean_mae)
    logger.info("  Mean features: %.0f / %d", mean_nfeat, len(feature_cols))
    logger.info("  Log target:    %s", log_target)

    # Aggregate per-temperature R2
    all_temps = set()
    for m in fold_metrics:
        all_temps.update(m.get("per_temperature_r2", {}).keys())
    if all_temps:
        logger.info("  Per-temperature R2 (mean across folds):")
        for t_label in sorted(all_temps, key=lambda x: int(x.rstrip("C"))):
            vals = [
                m["per_temperature_r2"][t_label]
                for m in fold_metrics
                if t_label in m.get("per_temperature_r2", {})
            ]
            if vals:
                logger.info("    %s: %.4f (+-%.4f)", t_label, np.mean(vals), np.std(vals))

    logger.info("=" * 50)

    summary = {
        "property": prop,
        "architecture": "temp_as_feature",
        "log_target": log_target,
        "n_samples": int(len(df)),
        "n_molecules": int(df["SMILES"].nunique()),
        "n_temperatures": int(df["T_C"].nunique()),
        "n_total_features": len(feature_cols),
        "mean_r2": float(mean_r2),
        "std_r2": float(std_r2),
        "mean_rmse": float(mean_rmse),
        "mean_mae": float(mean_mae),
        "mean_n_features": float(mean_nfeat),
        "fold_metrics": fold_metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Train temperature-as-feature XGBoost models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--property",
        type=str,
        choices=ALL_PROPERTIES,
        help="Property to train (or use --all)",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to long-format CSV for a single property",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all properties from --data-dir",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "nist_8100" / "long"),
        help="Directory containing long-format CSVs (used with --all)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./output_temp_feature",
        help="Output directory for models",
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Apply log1p transform (auto-applied for kv)",
    )
    parser.add_argument("--n_rfe_repeats", type=int, default=3)
    parser.add_argument("--rfe_step_size", type=int, default=2)
    parser.add_argument("--n_random_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)

    if args.all:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error("Data directory not found: %s", data_dir)
            return

        for prop in ALL_PROPERTIES:
            data_path = data_dir / f"{prop}_long.csv"
            if not data_path.exists():
                logger.warning("Skipping %s: %s not found", prop, data_path)
                continue

            log_target = args.log_target or prop in LOG_TARGETS_DEFAULT
            train_property(
                prop, data_path, outdir,
                log_target=log_target,
                n_rfe_repeats=args.n_rfe_repeats,
                rfe_step_size=args.rfe_step_size,
                n_random_iter=args.n_random_iter,
                seed=args.seed,
            )
    else:
        if not args.property:
            parser.error("Specify --property or use --all")
        if not args.data:
            parser.error("Specify --data when using --property")

        data_path = Path(args.data)
        if not data_path.exists():
            logger.error("Data file not found: %s", data_path)
            return

        log_target = args.log_target or args.property in LOG_TARGETS_DEFAULT
        train_property(
            args.property, data_path, outdir,
            log_target=log_target,
            n_rfe_repeats=args.n_rfe_repeats,
            rfe_step_size=args.rfe_step_size,
            n_random_iter=args.n_random_iter,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
