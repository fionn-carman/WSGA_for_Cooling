#!/usr/bin/env python3
"""
Quick comparison: DC model trained on original 584 CHO vs expanded 667 CHO.

Uses the same pipeline as train_single_temp.py (Mordred+RDKit descriptors,
hash-based 5-fold nested CV, fast feature selection, Optuna HP tuning)
but with reduced trials for fast local execution.

Usage:
    python compare_dc_cho.py              # default 5 trials
    python compare_dc_cho.py --n_trials 10
"""

import argparse
import builtins
import logging
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError("Install optuna: pip install optuna")

from fold_utils import assign_folds, get_fold_indices

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
DESCRIPTORS_PATH = DATA_DIR / "CHNO" / "descriptors_rdkit_mordred.csv"


# ============================================================
# Feature selection (from train_single_temp.py)
# ============================================================

def fast_feature_selection(X, y, feature_names, seed=42):
    n_start = X.shape[1]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    active_idx = list(range(n_start))

    # 1. Variance threshold
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    variances = np.var(X_scaled, axis=0)
    keep_mask = variances > 0.01
    active_idx = [i for i, k in enumerate(keep_mask) if k]

    if len(active_idx) == 0:
        top_var = np.argsort(variances)[-10:]
        active_idx = sorted(top_var.tolist())

    X_active = X[:, active_idx]
    n_after_var = len(active_idx)

    # 2. Correlation dedup
    if len(active_idx) > 1:
        corr_matrix = np.corrcoef(X_active, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        target_corr = np.array([
            abs(np.corrcoef(X_active[:, i], y)[0, 1])
            for i in range(X_active.shape[1])
        ])
        target_corr = np.nan_to_num(target_corr, nan=0.0)

        drop_set = set()
        for i in range(len(active_idx)):
            if i in drop_set:
                continue
            for j in range(i + 1, len(active_idx)):
                if j in drop_set:
                    continue
                if abs(corr_matrix[i, j]) > 0.95:
                    if target_corr[i] >= target_corr[j]:
                        drop_set.add(j)
                    else:
                        drop_set.add(i)
                        break

        keep_local = [k for k in range(len(active_idx)) if k not in drop_set]
        active_idx = [active_idx[k] for k in keep_local]

    X_active = X[:, active_idx]
    n_after_corr = len(active_idx)

    # 3. Importance-based top-N with elbow detection
    xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.1, max_depth=6,
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    xgb.fit(X_active, y)
    importances = xgb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    N_candidates = [n for n in [20, 40, 60, 80, 100, 120, 150, 200]
                    if n <= len(active_idx)]
    if not N_candidates or N_candidates[-1] < len(active_idx):
        N_candidates.append(len(active_idx))

    scores = []
    for n in N_candidates:
        top_n_idx = sorted_idx[:n]
        X_sub = X_active[:, top_n_idx]
        xgb_eval = XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
        cv_scores = cross_val_score(xgb_eval, X_sub, y, cv=3, scoring="r2")
        scores.append(np.mean(cv_scores))

    best_score = max(scores)
    best_n = N_candidates[-1]
    for n, s in zip(N_candidates, scores):
        if s >= best_score - 0.005:
            best_n = n
            break

    top_idx = sorted_idx[:best_n]
    selected_global_idx = [active_idx[i] for i in top_idx]
    selected_names = [feature_names[i] for i in selected_global_idx]

    logger.info("      Feature selection: %d -> %d (var) -> %d (corr) -> %d (elbow, R2=%.4f)",
                n_start, n_after_var, n_after_corr, best_n,
                scores[N_candidates.index(best_n)])
    return selected_names


# ============================================================
# Optuna HP tuning (from train_single_temp.py)
# ============================================================

def optuna_hp_tuning(X, y, n_trials=5, timeout=300, seed=42):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [100, 200, 300, 500, 1000, 2000]),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.5, log=True),
            "max_depth": trial.suggest_categorical(
                "max_depth", [3, 4, 5, 6, 10, 20]),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 6),
            "subsample": trial.suggest_categorical(
                "subsample", [0.7, 0.8, 0.9, 1.0]),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.7, 0.8, 0.9, 1.0]),
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X)):
            model = XGBRegressor(
                objective="reg:squarederror",
                **params, random_state=seed, n_jobs=-1, verbosity=0,
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
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=min(3, n_trials), n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    logger.info("      Optuna: %d trials (%d pruned), best R2=%.4f",
                len(study.trials), n_pruned, study.best_value)
    return study.best_params


# ============================================================
# Run nested CV for one dataset
# ============================================================

def run_nested_cv(smiles, y_raw, descriptors_df, label, n_trials, n_folds=5):
    logger.info("=" * 60)
    logger.info("Dataset: %s (%d molecules)", label, len(smiles))
    logger.info("=" * 60)

    # Join with descriptors
    prop_df = pd.DataFrame({"SMILES": smiles, "DC_exp": y_raw})
    merged = prop_df.merge(descriptors_df, on="SMILES", how="inner")
    logger.info("  After descriptor join: %d molecules", len(merged))

    smiles_arr = merged["SMILES"].values
    y = np.log1p(merged["DC_exp"].values)  # log-transform
    feature_cols = [c for c in merged.columns if c not in ("SMILES", "DC_exp")]
    X = merged[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = feature_cols

    # Hash-based fold assignment
    folds = assign_folds(smiles_arr, n_folds)
    all_preds = np.full(len(y), np.nan)

    fold_metrics = []
    t0 = time.time()

    for fold_i in range(n_folds):
        train_idx, test_idx = get_fold_indices(folds, fold_i)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info("  Fold %d: train=%d, test=%d", fold_i, len(train_idx), len(test_idx))

        # Feature selection (training only)
        selected = fast_feature_selection(X_train, y_train, feature_names, seed=fold_i)
        sel_idx = [feature_names.index(f) for f in selected]
        X_train_sel = X_train[:, sel_idx]
        X_test_sel = X_test[:, sel_idx]

        # HP tuning (training only)
        best_params = optuna_hp_tuning(X_train_sel, y_train,
                                        n_trials=n_trials, seed=fold_i)

        # Train final model
        model = XGBRegressor(
            objective="reg:squarederror",
            **best_params, random_state=fold_i, n_jobs=-1, verbosity=0,
        )
        model.fit(X_train_sel, y_train)
        preds_log = model.predict(X_test_sel)
        all_preds[test_idx] = preds_log

        # Fold metrics in real space
        y_test_real = np.expm1(y_test)
        preds_real = np.expm1(preds_log)
        r2 = r2_score(y_test_real, preds_real)
        rmse = math.sqrt(mean_squared_error(y_test_real, preds_real))
        mae = mean_absolute_error(y_test_real, preds_real)
        fold_metrics.append({"fold": fold_i, "r2": r2, "rmse": rmse, "mae": mae,
                             "n_features": len(selected), "params": best_params})
        logger.info("    Fold %d: R2=%.4f, RMSE=%.4f, MAE=%.4f (%d features)",
                    fold_i, r2, rmse, mae, len(selected))

    elapsed = time.time() - t0

    # Overall CV metrics (real space)
    y_real = np.expm1(y)
    preds_real = np.expm1(all_preds)
    cv_r2 = r2_score(y_real, preds_real)
    cv_rmse = math.sqrt(mean_squared_error(y_real, preds_real))
    cv_mae = mean_absolute_error(y_real, preds_real)

    logger.info("-" * 40)
    logger.info("  CV R2=%.4f, RMSE=%.4f, MAE=%.4f (%.1fs)", cv_r2, cv_rmse, cv_mae, elapsed)

    return {
        "label": label,
        "n_molecules": len(merged),
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "fold_metrics": fold_metrics,
        "elapsed": elapsed,
        "y_true_real": y_real,
        "y_pred_real": preds_real,
    }


def save_parity_plots(results_list, out_path):
    """Side-by-side parity plots for both datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)

    for ax, r in zip(axes, results_list):
        y_true = r["y_true_real"]
        y_pred = r["y_pred_real"]
        r2 = r2_score(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="none", s=20, c="#1f77b4")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = (hi - lo) * 0.03
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", lw=1, alpha=0.7)
        ax.set_xlabel("True DC")
        ax.set_ylabel("Predicted DC")
        ax.text(0.05, 0.95,
                f"N = {r['n_molecules']}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="grey", alpha=0.8))
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal")
        ax.set_title(r["label"], fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved parity plots to %s", out_path)


def save_value_comparison_plot(orig_df, exp_df, out_path):
    """Plot DC_exp values for overlapping molecules: original vs expanded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = orig_df.merge(exp_df, on="SMILES", suffixes=("_orig", "_exp"))
    y_orig = merged["DC_exp_orig"].values
    y_exp = merged["DC_exp_exp"].values

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(y_orig, y_exp, alpha=0.4, edgecolors="none", s=20, c="#d62728")
    lo = min(y_orig.min(), y_exp.min())
    hi = max(y_orig.max(), y_exp.max())
    margin = (hi - lo) * 0.03
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", lw=1, alpha=0.7)
    ax.set_xlabel("Original DC_exp")
    ax.set_ylabel("Expanded DC_exp")
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal")

    diff = abs(y_orig - y_exp)
    ax.text(0.05, 0.95,
            f"N overlap = {len(merged)}\nMean |diff| = {diff.mean():.3f}\n"
            f"Max |diff| = {diff.max():.3f}\n|diff| > 1.0: {(diff > 1.0).sum()}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved value comparison to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Compare DC CHO datasets")
    parser.add_argument("--n_trials", type=int, default=5,
                        help="Optuna trials per fold (default: 5)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip training, just plot value comparison")
    args = parser.parse_args()

    out_dir = SCRIPT_DIR / "output" / "dc_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    orig_path = DATA_DIR / "constraints" / "DC_exp_cleaned.csv"
    orig_df = pd.read_csv(orig_path, usecols=["SMILES", "DC_exp"])
    exp_path = DATA_DIR / "constraints" / "DC_exp_cleaned_667.csv"
    exp_df = pd.read_csv(exp_path)

    # Always save value comparison plot
    save_value_comparison_plot(orig_df, exp_df, out_dir / "dc_value_comparison.png")

    if args.plot_only:
        return

    logger.info("Loading descriptors from %s ...", DESCRIPTORS_PATH)
    desc_df = pd.read_csv(DESCRIPTORS_PATH)
    logger.info("  Descriptor cache: %d molecules x %d features",
                len(desc_df), len(desc_df.columns) - 1)
    logger.info("Original CHO: %d molecules", len(orig_df))
    logger.info("Expanded CHO: %d molecules", len(exp_df))

    # Run both
    results = []
    r1 = run_nested_cv(orig_df["SMILES"].values, orig_df["DC_exp"].values,
                       desc_df, "Original CHO (584)", args.n_trials)
    results.append(r1)

    r2 = run_nested_cv(exp_df["SMILES"].values, exp_df["DC_exp"].values,
                       desc_df, "Expanded CHO (667)", args.n_trials)
    results.append(r2)

    # Parity plots
    save_parity_plots(results, out_dir / "dc_parity_comparison.png")

    # Summary
    print("\n" + "=" * 65)
    print("DC Model Comparison: Original vs Expanded CHO")
    print("=" * 65)
    print(f"{'Dataset':<25} {'N':>5} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'Time':>7}")
    print("-" * 65)
    for r in results:
        print(f"{r['label']:<25} {r['n_molecules']:>5} {r['cv_r2']:>8.4f} "
              f"{r['cv_rmse']:>8.4f} {r['cv_mae']:>8.4f} {r['elapsed']:>6.0f}s")
    print("-" * 65)

    delta_r2 = results[1]["cv_r2"] - results[0]["cv_r2"]
    delta_rmse = results[1]["cv_rmse"] - results[0]["cv_rmse"]
    print(f"Delta (expanded - original): R2={delta_r2:+.4f}, RMSE={delta_rmse:+.4f}")


if __name__ == "__main__":
    main()
