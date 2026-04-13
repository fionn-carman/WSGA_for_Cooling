#!/usr/bin/env python3
"""
R² vs OOD metric percentile — multi-metric, multi-property analysis.

For each of 3 OOD metrics (Mahalanobis, kNN, Tanimoto), assigns molecules to
decile bins, holds out each bin as test, trains XGBoost on the remaining 90%,
and evaluates.  Then zooms into the top decile with 4 sub-bins (2.5% each).

Usage:
    python r2_vs_ood_percentile.py --property fom1_40C
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Property configuration ───────────────────────────────────────────────
PROPERTIES = {
    "fom1_40C": {
        "csv": "nist_8100/fom1_cho_cleaned.csv",
        "target": "fom1_40C",
        "desc_csv": "nist_8100/descriptors_rdkit_mordred.csv",
        "model_dir": "fom1_40C",
        "log_transform": True,
    },
    "cpsat_40C": {
        "csv": "nist_8100/cpsat_cho_cleaned.csv",
        "target": "cpsat_40C_J_K_mol",
        "desc_csv": "nist_8100/descriptors_rdkit_mordred.csv",
        "model_dir": "cpsat_40C",
        "log_transform": False,
    },
    "beta_40C": {
        "csv": "nist_8100/beta_cho_cleaned.csv",
        "target": "beta_40C",
        "desc_csv": "nist_8100/descriptors_rdkit_mordred.csv",
        "model_dir": "beta_40C",
        "log_transform": False,
    },
    "density_40C": {
        "csv": "nist_8100/density_cho_cleaned.csv",
        "target": "density_40C_g_cm3",
        "desc_csv": "nist_8100/descriptors_rdkit_mordred.csv",
        "model_dir": "density_40C",
        "log_transform": False,
    },
    "viscosity_40C": {
        "csv": "nist_8100/viscosity_cho_cleaned.csv",
        "target": "kv_40C_cSt",
        "desc_csv": "nist_8100/descriptors_rdkit_mordred.csv",
        "model_dir": "viscosity_40C",
        "log_transform": True,
    },
    "tc_40C": {
        "csv": "nist_8100/tc_cho_cleaned.csv",
        "target": "tc_40C_W_mK",
        "desc_csv": "nist_8100/descriptors_rdkit_mordred.csv",
        "model_dir": "tc_40C",
        "log_transform": False,
    },
    "bp": {
        "csv": "constraints/BP-Measured_cleaned.csv",
        "target": "BP-Measured",
        "desc_csv": "constraints/descriptors_rdkit_mordred.csv",
        "model_dir": "bp",
        "log_transform": False,
    },
    "mp": {
        "csv": "constraints/MP-Measured_cleaned.csv",
        "target": "MP-Measured",
        "desc_csv": "constraints/descriptors_rdkit_mordred.csv",
        "model_dir": "mp",
        "log_transform": False,
    },
    "fp": {
        "csv": "constraints/flashpoint_cleaned.csv",
        "target": "flashpoint",
        "desc_csv": "constraints/descriptors_rdkit_mordred.csv",
        "model_dir": "fp",
        "log_transform": False,
    },
    "dc": {
        "csv": "constraints/DC_exp_cleaned.csv",
        "target": "DC_exp",
        "desc_csv": "constraints/descriptors_rdkit_mordred.csv",
        "model_dir": "dc",
        "log_transform": False,
    },
}


# ── Utility functions ────────────────────────────────────────────────────

def compute_mahalanobis(X: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from covariance centre (LedoitWolf shrinkage)."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
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


def compute_knn(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean Euclidean distance to k nearest neighbours (leave-one-out)."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(Xs)
    distances, _ = nn.kneighbors(Xs)
    return distances[:, 1:].mean(axis=1)  # skip self (column 0)


def compute_tanimoto(smiles_list: list) -> np.ndarray:
    """Mean Tanimoto distance (1 - mean similarity) to all other molecules (Morgan r=2, 2048 bits)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps.append(gen.GetFingerprint(mol))
            valid_idx.append(i)

    ts_dist = np.full(len(smiles_list), np.nan)
    for ii, i in enumerate(valid_idx):
        sims = DataStructs.BulkTanimotoSimilarity(fps[ii], fps)
        sims[ii] = np.nan  # exclude self
        ts_dist[i] = 1.0 - np.nanmean(sims)
        if (ii + 1) % 1000 == 0:
            logger.info("    Tanimoto: %d / %d", ii + 1, len(valid_idx))

    return ts_dist


def fast_feature_selection(X, y, feature_names, seed=42):
    """Variance → decorrelation → XGB importance → CV elbow."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    active_idx = list(range(X.shape[1]))
    X_scaled = StandardScaler().fit_transform(X)
    variances = np.var(X_scaled, axis=0)
    active_idx = [i for i, v in enumerate(variances) if v > 0.01]
    if not active_idx:
        active_idx = sorted(np.argsort(variances)[-10:].tolist())
    X_active = X[:, active_idx]

    if len(active_idx) > 1:
        corr = np.nan_to_num(np.corrcoef(X_active, rowvar=False), nan=0.0)
        tc = np.array(
            [
                abs(np.nan_to_num(np.corrcoef(X_active[:, i], y)[0, 1], nan=0.0))
                for i in range(X_active.shape[1])
            ]
        )
        drop = set()
        for i in range(len(active_idx)):
            if i in drop:
                continue
            for j in range(i + 1, len(active_idx)):
                if j in drop:
                    continue
                if abs(corr[i, j]) > 0.95:
                    if tc[i] >= tc[j]:
                        drop.add(j)
                    else:
                        drop.add(i)
                        break
        active_idx = [active_idx[k] for k in range(len(active_idx)) if k not in drop]
    X_active = X[:, active_idx]

    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_active, y)
    sorted_idx = np.argsort(xgb.feature_importances_)[::-1]
    N_cands = [n for n in [20, 40, 60, 80, 100, 120, 150, 200] if n <= len(active_idx)]
    if not N_cands or N_cands[-1] < len(active_idx):
        N_cands.append(len(active_idx))
    scores = []
    for n in N_cands:
        ev = XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        scores.append(
            np.mean(
                cross_val_score(ev, X_active[:, sorted_idx[:n]], y, cv=3, scoring="r2")
            )
        )
    best_n = N_cands[-1]
    best_score = max(scores)
    for n, s in zip(N_cands, scores):
        if s >= best_score - 0.005:
            best_n = n
            break
    return [feature_names[active_idx[i]] for i in sorted_idx[:best_n]]


def optuna_hp_tuning(X, y, n_trials=30, timeout=180, seed=42):
    """Optuna TPE HP tuning with MedianPruner. Inner 3-fold CV."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [500, 1000, 1500, 2000, 3000]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [4, 5, 6, 7, 8]),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 0.9]),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.9]
            ),
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=seed)
        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X)):
            model = XGBRegressor(
                objective="reg:squarederror",
                **params,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X[tr_idx], y[tr_idx])
            r2 = r2_score(y[val_idx], model.predict(X[val_idx]))
            fold_scores.append(r2)
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    n_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )
    logger.info(
        "      Optuna: %d trials (%d pruned), best R2=%.4f",
        len(study.trials),
        n_pruned,
        study.best_value,
    )
    return study.best_params


def train_and_evaluate(X_train, y_train, X_test, y_test_raw, desc_cols, log_transform):
    """Feature selection → Optuna → train → predict → metrics."""
    t0 = time.time()

    # Feature selection
    selected = fast_feature_selection(X_train, y_train, desc_cols, seed=SEED)
    fi = [desc_cols.index(f) for f in selected]
    logger.info("      %d features selected (%.0fs)", len(selected), time.time() - t0)

    # Optuna HP tuning
    best_params = optuna_hp_tuning(X_train[:, fi], y_train, n_trials=30, timeout=180)

    # Train final model
    model = XGBRegressor(
        **best_params, random_state=SEED, n_jobs=-1, verbosity=0
    )
    model.fit(X_train[:, fi], y_train)

    # Predict
    y_pred = model.predict(X_test[:, fi])
    if log_transform:
        y_pred = np.expm1(y_pred)

    # Metrics
    abs_err = np.abs(y_pred - y_test_raw)
    ss_res = np.sum((y_test_raw - y_pred) ** 2)
    ss_tot = np.sum((y_test_raw - y_test_raw.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = np.sqrt(np.mean((y_test_raw - y_pred) ** 2))
    mae = abs_err.mean()
    medae = np.median(abs_err)
    mean_bias = float((y_pred - y_test_raw).mean())

    elapsed = time.time() - t0
    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MedAE": float(medae),
        "mean_bias": mean_bias,
        "n_features": len(selected),
        "elapsed_s": elapsed,
    }, model, fi


def evaluate_only(model, fi, X_test, y_test_raw, log_transform):
    """Predict with existing model and compute metrics (for tail zoom)."""
    y_pred = model.predict(X_test[:, fi])
    if log_transform:
        y_pred = np.expm1(y_pred)
    abs_err = np.abs(y_pred - y_test_raw)
    ss_res = np.sum((y_test_raw - y_pred) ** 2)
    ss_tot = np.sum((y_test_raw - y_test_raw.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = np.sqrt(np.mean((y_test_raw - y_pred) ** 2))
    mae = abs_err.mean()
    medae = np.median(abs_err)
    mean_bias = float((y_pred - y_test_raw).mean())
    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MedAE": float(medae),
        "mean_bias": mean_bias,
    }


# ── Per-metric analysis ─────────────────────────────────────────────────

def run_metric(
    metric_name: str,
    metric_values: np.ndarray,
    X_all: np.ndarray,
    y_all: np.ndarray,
    y_all_raw: np.ndarray,
    desc_cols: list,
    log_transform: bool,
    out_dir: Path,
):
    """Coarse (10 decile) + tail (4 sub-bin) holdout analysis for one metric."""
    logger.info("  ── Metric: %s ──", metric_name)

    # Handle NaN metric values (shouldn't happen for MD/kNN, possible for TS)
    valid = ~np.isnan(metric_values)
    if not valid.all():
        logger.warning(
            "    %d NaN metric values — assigning to lowest decile",
            (~valid).sum(),
        )
        metric_values = metric_values.copy()
        metric_values[~valid] = np.nanmin(metric_values) - 1

    decile_labels = pd.qcut(metric_values, q=10, labels=False, duplicates="drop")
    n_actual_bins = int(decile_labels.max()) + 1
    if n_actual_bins < 10:
        logger.warning(
            "    pd.qcut produced %d bins (not 10) — ties in %s values",
            n_actual_bins, metric_name,
        )

    coarse_results = []
    d10_model = None
    d10_fi = None
    max_label = int(decile_labels.max())

    for decile in range(n_actual_bins):
        test_mask = decile_labels == decile
        train_mask = ~test_mask
        n_test = int(test_mask.sum())

        if n_test == 0:
            logger.warning("    Decile %d: 0 molecules, skipping", decile)
            continue

        metric_test = metric_values[test_mask]
        pct_lo = round(decile * 100 / n_actual_bins, 1)
        pct_hi = round((decile + 1) * 100 / n_actual_bins, 1)

        logger.info(
            "    Decile %d (%s-%s%%, n=%d)...", decile, pct_lo, pct_hi, n_test
        )

        metrics, model, fi = train_and_evaluate(
            X_all[train_mask],
            y_all[train_mask],
            X_all[test_mask],
            y_all_raw[test_mask],
            desc_cols,
            log_transform,
        )

        row = {
            "decile": decile,
            "pct_range": f"{pct_lo}-{pct_hi}",
            "n_test": n_test,
            "n_train": int(train_mask.sum()),
            "metric_mean": float(metric_test.mean()),
            "metric_median": float(np.median(metric_test)),
            "metric_min": float(metric_test.min()),
            "metric_max": float(metric_test.max()),
            **metrics,
        }
        coarse_results.append(row)

        logger.info(
            "      R²=%.3f  RMSE=%.3f  MAE=%.3f  (%.0fs)",
            metrics["R2"],
            metrics["RMSE"],
            metrics["MAE"],
            metrics["elapsed_s"],
        )

        if decile == max_label:
            d10_model = model
            d10_fi = fi

    coarse_df = pd.DataFrame(coarse_results)
    coarse_df.to_csv(out_dir / f"coarse_{metric_name}.csv", index=False)
    logger.info("    Saved coarse_%s.csv", metric_name)

    # ── Tail zoom: 4 sub-bins in top decile ──────────────────────────
    d10_mask = decile_labels == max_label
    d10_metric = metric_values[d10_mask]
    d10_X = X_all[d10_mask]
    d10_y_raw = y_all_raw[d10_mask]

    # Percentile boundaries from the FULL distribution
    p925 = np.percentile(metric_values, 92.5)
    p950 = np.percentile(metric_values, 95.0)
    p975 = np.percentile(metric_values, 97.5)

    sub_bin_labels = np.digitize(d10_metric, bins=[p925, p950, p975])
    tail_ranges = ["90.0-92.5", "92.5-95.0", "95.0-97.5", "97.5-100.0"]

    tail_results = []
    for sb in range(4):
        sb_mask = sub_bin_labels == sb
        n_sb = sb_mask.sum()
        if n_sb < 3:
            logger.warning("    Tail sub-bin %s: only %d molecules, skipping", tail_ranges[sb], n_sb)
            continue

        sb_metric = d10_metric[sb_mask]
        metrics = evaluate_only(
            d10_model, d10_fi, d10_X[sb_mask], d10_y_raw[sb_mask], log_transform
        )

        row = {
            "sub_bin": sb,
            "pct_range": tail_ranges[sb],
            "n_test": int(n_sb),
            "metric_mean": float(sb_metric.mean()),
            "metric_median": float(np.median(sb_metric)),
            "metric_min": float(sb_metric.min()),
            "metric_max": float(sb_metric.max()),
            **metrics,
        }
        tail_results.append(row)

        logger.info(
            "    Tail %s: n=%d  R²=%.3f  RMSE=%.3f",
            tail_ranges[sb],
            n_sb,
            metrics["R2"],
            metrics["RMSE"],
        )

    if tail_results:
        tail_df = pd.DataFrame(tail_results)
        tail_df.to_csv(out_dir / f"tail_{metric_name}.csv", index=False)
        logger.info("    Saved tail_%s.csv", metric_name)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", required=True, choices=list(PROPERTIES.keys()))
    parser.add_argument("--metrics", nargs="+", default=["md", "knn", "ts"],
                        choices=["md", "knn", "ts"],
                        help="Which OOD metrics to run (default: all three)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: r2_vs_ood_percentile/<property>)")
    args = parser.parse_args()

    prop_name = args.property
    cfg = PROPERTIES[prop_name]
    logger.info("=== Property: %s ===", prop_name)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = SCRIPT_DIR / "r2_vs_ood_percentile" / prop_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "training" / "data"
    model_dir = ROOT / "models"

    # Load property data
    prop_df = pd.read_csv(data_dir / cfg["csv"])
    prop_df = prop_df[prop_df[cfg["target"]].notna()].copy()
    logger.info("  Loaded %d molecules from %s", len(prop_df), cfg["csv"])

    # Load descriptors
    desc_df = pd.read_csv(data_dir / cfg["desc_csv"], low_memory=False)
    merged = prop_df[["SMILES", cfg["target"]]].merge(desc_df, on="SMILES", how="inner")
    logger.info("  Merged: %d molecules", len(merged))

    # Load deployed model features
    deployed = joblib.load(model_dir / cfg["model_dir"] / "model" / "xgb_model.joblib")
    dep_feats = deployed["selected_features"]
    logger.info("  Deployed model: %d features", len(dep_feats))

    # Prepare arrays
    desc_cols = [c for c in merged.columns if c not in ("SMILES", cfg["target"])]
    X_all = merged[desc_cols].values.astype(float)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    dep_fi = [desc_cols.index(f) for f in dep_feats if f in desc_cols]

    y_all_raw = merged[cfg["target"]].values.astype(float)
    if cfg["log_transform"]:
        y_all = np.log1p(y_all_raw)
    else:
        y_all = y_all_raw.copy()

    smiles_list = merged["SMILES"].tolist()

    # ── Compute OOD metrics ──────────────────────────────────────────
    logger.info("  Computing OOD metrics: %s ...", ", ".join(args.metrics))
    metric_map = {}

    if "md" in args.metrics:
        t0 = time.time()
        md_values = compute_mahalanobis(X_all[:, dep_fi])
        logger.info("    MD done (%.1fs): range [%.2f, %.2f]", time.time() - t0, md_values.min(), md_values.max())
        metric_map["md"] = md_values

    if "knn" in args.metrics:
        t0 = time.time()
        knn_values = compute_knn(X_all[:, dep_fi], k=5)
        logger.info("    kNN done (%.1fs): range [%.2f, %.2f]", time.time() - t0, knn_values.min(), knn_values.max())
        metric_map["knn"] = knn_values

    if "ts" in args.metrics:
        t0 = time.time()
        ts_values = compute_tanimoto(smiles_list)
        logger.info("    TS done (%.1fs): range [%.4f, %.4f]", time.time() - t0, np.nanmin(ts_values), np.nanmax(ts_values))
        metric_map["ts"] = ts_values

    # ── Run analysis per metric ──────────────────────────────────────
    for metric_name, metric_values in metric_map.items():
        run_metric(
            metric_name,
            metric_values,
            X_all,
            y_all,
            y_all_raw,
            desc_cols,
            cfg["log_transform"],
            out_dir,
        )

    # ── Save run metadata ────────────────────────────────────────────
    meta = {
        "property": prop_name,
        "n_molecules": len(merged),
        "n_deployed_features": len(dep_feats),
        "log_transform": cfg["log_transform"],
        "target_mean": float(y_all_raw.mean()),
        "target_std": float(y_all_raw.std()),
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=== Done: %s ===", prop_name)


if __name__ == "__main__":
    main()
