#!/usr/bin/env python3
"""
Publication-quality MLP training pipeline for NIST 8100 single-temp models.

Mordred + RDKit descriptors, Optuna nested CV, permutation importance,
diagnostic plots.

Pipeline per property (40C):
  1. Load precomputed descriptors (descriptors_rdkit_mordred.csv)
  2. Hash-based 5-fold outer CV
  3. Per outer fold (on training data only):
     a. Fast feature selection (variance -> correlation -> importance top-N)
     b. Optuna HP tuning (100 trials, inner 5-fold CV)
     c. Scale X and y (StandardScaler)
     d. Train MLPRegressor with best params, predict on held-out test set
     e. Permutation importance on test set (n_repeats=10)
  4. Generate publication artifacts

Usage:
    python train_single_temp.py                      # all 6 properties
    python train_single_temp.py --properties density  # single property
    python train_single_temp.py --n_trials 20         # faster tuning
"""

import argparse
import json
import logging
import math
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNITS = {
    "density": "Density / g cm$^{-3}$",
    "viscosity": "Kinematic viscosity / cSt",
    "tc": "Thermal conductivity / W m$^{-1}$ K$^{-1}$",
    "cpsat": "$C_{p,sat}$ / J K$^{-1}$ mol$^{-1}$",
    "beta": "Thermal expansion coeff. / K$^{-1}$",
    "fom1": "FOM1",
}

MODELS = {
    "density":   {"csv": "density_cho_cleaned.csv",   "col_40": "density_40C_g_cm3",  "log": False},
    "viscosity": {"csv": "viscosity_cho_cleaned.csv", "col_40": "kv_40C_cSt",         "log": True},
    "tc":        {"csv": "tc_cho_cleaned.csv",        "col_40": "tc_40C_W_mK",        "log": False},
    "cpsat":     {"csv": "cpsat_cho_cleaned.csv",     "col_40": "cpsat_40C_J_K_mol",  "log": False},
    "beta":      {"csv": "beta_cho_cleaned.csv",      "col_40": "beta_40C",           "log": False},
    "fom1":      {"csv": "fom1_cho_cleaned.csv",      "col_40": "fom1_40C",           "log": True},
}

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "nist_8100"
OUTPUT_DIR = SCRIPT_DIR / "output" / "single_temp"
FIGURE_DIR = SCRIPT_DIR / "figures" / "single_temp"
DESCRIPTORS_PATH = DATA_DIR / "descriptors_rdkit_mordred.csv"

# MLP architecture candidates for Optuna
HIDDEN_LAYER_CHOICES = [
    (128,),
    (256,),
    (256, 128),
    (512, 256),
    (256, 128, 64),
]

BATCH_SIZE_CHOICES = [32, 64, 128, 256]
ACTIVATION_CHOICES = ["relu", "tanh"]


# ============================================================
# Feature selection (identical to XGBoost pipeline)
# ============================================================

def fast_feature_selection(X: np.ndarray, y: np.ndarray, feature_names: list,
                           seed: int = 42) -> list:
    """
    Fast feature selection (no leakage — runs on training data only):
      1. Variance threshold (drop near-constant after scaling, var < 0.01)
      2. Correlation dedup (|r| > 0.95 -> drop weaker target correlation)
      3. Importance-based top-N with elbow detection
    """
    n_start = X.shape[1]

    # Clean NaN/inf
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

    # Elbow: smallest N within 0.005 of best
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
# Optuna HP tuning (MLP-specific)
# ============================================================

def optuna_hp_tuning(X: np.ndarray, y: np.ndarray, n_trials: int = 100,
                     timeout: int = 600, seed: int = 42) -> dict:
    """Optuna TPE HP tuning with MedianPruner. Inner 5-fold CV with per-fold scaling."""

    def objective(trial):
        hidden_idx = trial.suggest_int("hidden_layer_idx", 0, len(HIDDEN_LAYER_CHOICES) - 1)
        hidden_layer_sizes = HIDDEN_LAYER_CHOICES[hidden_idx]

        params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True),
            "alpha": trial.suggest_float(
                "alpha", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", BATCH_SIZE_CHOICES),
            "activation": trial.suggest_categorical(
                "activation", ACTIVATION_CHOICES),
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X)):
            # Scale X per inner fold (no leakage)
            scaler_X = StandardScaler()
            X_tr = scaler_X.fit_transform(X[tr_idx])
            X_val = scaler_X.transform(X[val_idx])

            # Scale y per inner fold
            scaler_y = StandardScaler()
            y_tr = scaler_y.fit_transform(y[tr_idx].reshape(-1, 1)).ravel()
            y_val_raw = y[val_idx]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
                warnings.filterwarnings("ignore", category=FutureWarning)
                try:
                    from sklearn.exceptions import ConvergenceWarning
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                except ImportError:
                    pass

                model = MLPRegressor(
                    solver="adam",
                    early_stopping=True,
                    max_iter=1000,
                    n_iter_no_change=20,
                    validation_fraction=0.1,
                    random_state=seed,
                    verbose=False,
                    **params,
                )
                model.fit(X_tr, y_tr)

            preds_scaled = model.predict(X_val)
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
            r2 = r2_score(y_val_raw, preds)
            fold_scores.append(r2)

            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    logger.info("      Optuna: %d trials (%d pruned), best R2=%.4f",
                len(study.trials), n_pruned, study.best_value)

    # Convert hidden_layer_idx back to actual tuple
    best = study.best_params.copy()
    best["hidden_layer_sizes"] = HIDDEN_LAYER_CHOICES[best.pop("hidden_layer_idx")]
    return best


# ============================================================
# Plotting (CLAUDE.md style: no titles, no grids, frameon=False)
# ============================================================

def save_parity_plot(y_true, y_pred, out_dir, prop_name):
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    label = UNITS.get(prop_name, prop_name)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="none", s=20, c="#1f77b4")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.03
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", lw=1, alpha=0.7)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.text(0.05, 0.95,
            f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8))
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "parity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics_plot(y_true, y_pred, out_dir, prop_name):
    residuals = y_true - y_pred
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    label = UNITS.get(prop_name, prop_name)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)

    # Parity
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="none", s=20, c="#1f77b4")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.text(0.05, 0.95,
            f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8))

    # Residuals vs predicted
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.4, edgecolors="none", s=20, c="#1f77b4")
    ax.axhline(y=0, color="k", linestyle="--", lw=1, alpha=0.7)
    ax.set_xlabel(f"Predicted {label}")
    ax.set_ylabel("Residual")
    ax.text(0.05, 0.95,
            f"Mean = {residuals.mean():.4f}\nStd = {residuals.std():.4f}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8))

    # Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=40, edgecolor="black", alpha=0.7, color="#1f77b4")
    ax.axvline(x=0, color="k", linestyle="--", lw=1, alpha=0.7)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")

    # Q-Q
    ax = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("")
    ax.get_lines()[0].set(markersize=3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_permutation_importance_plot(imp_df, out_dir, max_display=20):
    """Bar chart of top permutation importances."""
    top = imp_df.head(max_display).copy()
    top = top.iloc[::-1]  # reverse for horizontal bar (top feature at top)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.barh(range(len(top)), top["permutation_importance"].values,
            color="#1f77b4", alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.set_xlabel("Mean permutation importance (decrease in $R^2$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance_bar.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Per-property training pipeline
# ============================================================

def train_property(prop_name: str, prop_config: dict, desc_df: pd.DataFrame,
                   n_trials: int, timeout: int) -> dict:
    """Full nested-CV pipeline for a single property at 40C."""
    t0_total = time.time()

    # --- Load property data ---
    csv_path = DATA_DIR / prop_config["csv"]
    target_col = prop_config["col_40"]
    log_transform = prop_config["log"]

    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df[target_col].notna()].copy()

    # --- Join with precomputed descriptors ---
    merged = df[["SMILES", target_col]].merge(desc_df, on="SMILES", how="inner")
    logger.info("  Loaded %d molecules (%d after descriptor join)",
                len(df), len(merged))

    smiles = merged["SMILES"].values
    y_raw = merged[target_col].values.astype(np.float64)
    desc_cols = [c for c in merged.columns if c not in ("SMILES", target_col)]
    X_raw = merged[desc_cols].values.astype(np.float64)

    if log_transform:
        y = np.log1p(y_raw)
    else:
        y = y_raw.copy()

    # --- Setup directories ---
    out_dir = OUTPUT_DIR / prop_name / "40C"
    fig_dir = FIGURE_DIR / prop_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Hash-based fold assignment ---
    folds = assign_folds(smiles, n_folds=5)

    # --- Per-fold training ---
    all_preds = np.full(len(y), np.nan)
    fold_metrics = []
    fold_selected_features = []
    fold_best_params = []
    fold_perm_imp_data = []  # (test_idx, importances_dict, selected_features)

    for fold_i in range(5):
        logger.info("    Fold %d:", fold_i)
        t0_fold = time.time()
        train_idx, test_idx = get_fold_indices(folds, fold_i)
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 3a. Feature selection (train only)
        selected = fast_feature_selection(X_train, y_train, desc_cols, seed=fold_i)
        feat_idx = [desc_cols.index(f) for f in selected]
        X_train_sel = np.nan_to_num(X_train[:, feat_idx], nan=0.0, posinf=0.0, neginf=0.0)
        X_test_sel = np.nan_to_num(X_test[:, feat_idx], nan=0.0, posinf=0.0, neginf=0.0)

        # 3b. Optuna HP tuning (train only, inner CV)
        best_params = optuna_hp_tuning(
            X_train_sel, y_train,
            n_trials=n_trials, timeout=timeout, seed=fold_i,
        )

        # 3c. Scale X and y
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_sel)
        X_test_scaled = scaler_X.transform(X_test_sel)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # 3d. Train + predict
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                from sklearn.exceptions import ConvergenceWarning
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            except ImportError:
                pass

            model = MLPRegressor(
                solver="adam",
                early_stopping=True,
                max_iter=1000,
                n_iter_no_change=20,
                validation_fraction=0.1,
                random_state=fold_i,
                verbose=False,
                **best_params,
            )
            model.fit(X_train_scaled, y_train_scaled)

        preds_scaled = model.predict(X_test_scaled)
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        all_preds[test_idx] = preds

        # 3e. Permutation importance on test set
        perm_result = permutation_importance(
            model, X_test_scaled,
            scaler_y.transform(y_test.reshape(-1, 1)).ravel(),
            n_repeats=10, random_state=fold_i, scoring="r2",
        )
        perm_imp_dict = dict(zip(selected, perm_result.importances_mean))
        fold_perm_imp_data.append((test_idx, perm_imp_dict, selected))

        fold_selected_features.append(selected)
        fold_best_params.append(best_params)

        # Per-fold metrics (real space)
        if log_transform:
            y_test_real = np.expm1(y_test)
            preds_real = np.expm1(preds)
        else:
            y_test_real = y_test
            preds_real = preds

        fold_r2 = r2_score(y_test_real, preds_real)
        fold_rmse = math.sqrt(mean_squared_error(y_test_real, preds_real))
        fold_mae = mean_absolute_error(y_test_real, preds_real)
        fold_time = time.time() - t0_fold

        # Serialise best_params (convert tuples to lists for JSON)
        best_params_json = best_params.copy()
        if "hidden_layer_sizes" in best_params_json:
            best_params_json["hidden_layer_sizes"] = list(best_params_json["hidden_layer_sizes"])

        fold_metrics.append({
            "fold": fold_i,
            "r2": round(float(fold_r2), 4),
            "rmse": round(float(fold_rmse), 4),
            "mae": round(float(fold_mae), 4),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_selected_features": len(selected),
            "best_params": best_params_json,
            "n_iter": int(model.n_iter_),
            "time_s": round(fold_time, 1),
        })

        # Save fold model
        joblib.dump({
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "selected_features": selected,
            "best_params": best_params,
            "log_transform": log_transform,
            "property": prop_name,
        }, out_dir / f"fold{fold_i}_model.joblib")

        logger.info("      R2=%.4f  RMSE=%.4f  (%d feats, %d/%d train/test, %d iters, %.1fs)",
                     fold_r2, fold_rmse, len(selected),
                     len(train_idx), len(test_idx), model.n_iter_, fold_time)

    # --- Aggregate predictions (real space) ---
    if log_transform:
        y_true_real = np.expm1(y)
        y_pred_real = np.expm1(all_preds)
    else:
        y_true_real = y.copy()
        y_pred_real = all_preds.copy()

    pred_df = pd.DataFrame({
        "SMILES": smiles,
        "fold": folds,
        "y_true": y_true_real,
        "y_pred": y_pred_real,
        "residual": y_true_real - y_pred_real,
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # --- Aggregate permutation importance (union of selected features) ---
    all_features_union = sorted(set().union(*fold_selected_features))
    n_union = len(all_features_union)

    # Average permutation importance across folds (0 for features not in that fold)
    perm_imp_agg = {f: [] for f in all_features_union}
    for _, perm_imp_dict, _ in fold_perm_imp_data:
        for f in all_features_union:
            perm_imp_agg[f].append(perm_imp_dict.get(f, 0.0))

    mean_perm_imp = {f: float(np.mean(vals)) for f, vals in perm_imp_agg.items()}

    perm_imp_df = pd.DataFrame({
        "feature": all_features_union,
        "permutation_importance": [mean_perm_imp[f] for f in all_features_union],
    }).sort_values("permutation_importance", ascending=False).reset_index(drop=True)
    perm_imp_df.to_csv(out_dir / "feature_importance.csv", index=False)

    # --- Full-data model ---
    logger.info("    Training full-data model...")
    # Feature selection on full data
    full_selected = fast_feature_selection(X_raw, y, desc_cols, seed=42)
    full_feat_idx = [desc_cols.index(f) for f in full_selected]
    X_full_sel = np.nan_to_num(X_raw[:, full_feat_idx], nan=0.0, posinf=0.0, neginf=0.0)

    # Use params from best-performing fold
    best_fold_idx = max(range(5), key=lambda i: fold_metrics[i]["r2"])
    full_params = fold_best_params[best_fold_idx]
    logger.info("    Using params from fold %d (R2=%.4f)", best_fold_idx,
                fold_metrics[best_fold_idx]["r2"])

    # Scale full data
    scaler_X_full = StandardScaler()
    X_full_scaled = scaler_X_full.fit_transform(X_full_sel)
    scaler_y_full = StandardScaler()
    y_full_scaled = scaler_y_full.fit_transform(y.reshape(-1, 1)).ravel()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            from sklearn.exceptions import ConvergenceWarning
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
        except ImportError:
            pass

        model_full = MLPRegressor(
            solver="adam",
            early_stopping=True,
            max_iter=1000,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
            **full_params,
        )
        model_full.fit(X_full_scaled, y_full_scaled)

    joblib.dump({
        "model": model_full,
        "scaler_X": scaler_X_full,
        "scaler_y": scaler_y_full,
        "selected_features": full_selected,
        "best_params": full_params,
        "log_transform": log_transform,
        "property": prop_name,
    }, out_dir / "model.joblib")

    # --- Plots ---
    valid = ~np.isnan(all_preds)
    save_parity_plot(y_true_real[valid], y_pred_real[valid], fig_dir, prop_name)
    save_diagnostics_plot(y_true_real[valid], y_pred_real[valid], fig_dir, prop_name)
    save_permutation_importance_plot(perm_imp_df, fig_dir)
    logger.info("    Plots saved to %s", fig_dir)

    # --- Overall CV metrics ---
    r2_cv = r2_score(y_true_real[valid], y_pred_real[valid])
    rmse_cv = math.sqrt(mean_squared_error(y_true_real[valid], y_pred_real[valid]))
    mae_cv = mean_absolute_error(y_true_real[valid], y_pred_real[valid])

    fold_r2s = [m["r2"] for m in fold_metrics]
    fold_rmses = [m["rmse"] for m in fold_metrics]
    fold_maes = [m["mae"] for m in fold_metrics]

    total_time = time.time() - t0_total

    # Serialise full_params for JSON
    full_params_json = full_params.copy()
    if "hidden_layer_sizes" in full_params_json:
        full_params_json["hidden_layer_sizes"] = list(full_params_json["hidden_layer_sizes"])

    summary = {
        "property": prop_name,
        "temperature": "40C",
        "r2": round(r2_cv, 4),
        "rmse": round(rmse_cv, 4),
        "mae": round(mae_cv, 4),
        "std_r2": round(float(np.std(fold_r2s)), 4),
        "std_rmse": round(float(np.std(fold_rmses)), 4),
        "std_mae": round(float(np.std(fold_maes)), 4),
        "n_molecules": int(valid.sum()),
        "n_raw_features": len(desc_cols),
        "n_union_features": n_union,
        "n_full_model_features": len(full_selected),
        "log_transform": log_transform,
        "hash_method": "md5_mod5",
        "best_fold_params_source": best_fold_idx,
        "full_model_params": full_params_json,
        "fold_metrics": fold_metrics,
        "time_s": round(total_time, 1),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train single-temp MLP models (Mordred + Optuna)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--properties", nargs="+", default=["all"],
        help="Properties to train (density viscosity tc cpsat beta fom1) or 'all'",
    )
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Optuna trials per fold")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Optuna timeout per fold (seconds)")
    args = parser.parse_args()

    # Resolve properties
    if args.properties == ["all"]:
        properties = list(MODELS.keys())
    else:
        properties = args.properties
        for p in properties:
            if p not in MODELS:
                parser.error(f"Unknown property: {p}. Valid: {list(MODELS.keys())}")

    # Load precomputed descriptors
    if not DESCRIPTORS_PATH.exists():
        logger.error("Precomputed descriptors not found: %s", DESCRIPTORS_PATH)
        logger.error("Run compute_descriptors.py first.")
        return

    logger.info("Loading precomputed descriptors from %s", DESCRIPTORS_PATH.name)
    t0 = time.time()
    desc_df = pd.read_csv(DESCRIPTORS_PATH)
    logger.info("  %d SMILES x %d descriptors (%.1fs)",
                len(desc_df), len(desc_df.columns) - 1, time.time() - t0)

    # Master fold assignments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_smiles = sorted(desc_df["SMILES"].unique())
    folds_master = assign_folds(all_smiles, n_folds=5)
    fold_df = pd.DataFrame({"SMILES": all_smiles, "fold": folds_master})
    fold_df.to_csv(OUTPUT_DIR / "fold_assignments.csv", index=False)
    logger.info("Fold assignments: %d SMILES, distribution: %s",
                len(fold_df), dict(zip(*np.unique(folds_master, return_counts=True))))

    # Train
    logger.info("=" * 70)
    logger.info("Training %d properties at 40C (%d Optuna trials/fold, %ds timeout)",
                len(properties), args.n_trials, args.timeout)
    logger.info("=" * 70)

    results = {}
    for prop_name in properties:
        logger.info("--- %s ---", prop_name.upper())
        summary = train_property(
            prop_name, MODELS[prop_name], desc_df,
            n_trials=args.n_trials, timeout=args.timeout,
        )
        if summary:
            results[f"{prop_name}_40C"] = summary
            logger.info("  => R2=%.4f +/- %.4f  RMSE=%.4f  N=%d  (%.1fs)",
                        summary["r2"], summary["std_r2"], summary["rmse"],
                        summary["n_molecules"], summary["time_s"])

    # Summary table
    print(f"\n{'=' * 80}")
    print("SINGLE-TEMP MLP MODELS (5-fold nested CV, Mordred+RDKit, Optuna)")
    print(f"{'=' * 80}")
    print(f"{'Model':<18} {'R2':>8} {'+-':>6} {'RMSE':>10} {'MAE':>10} "
          f"{'Feats':>6} {'N':>6} {'Time':>7}")
    print("-" * 80)
    for key, r in results.items():
        t_min = r["time_s"] / 60
        print(f"{key:<18} {r['r2']:>8.4f} {r['std_r2']:>6.4f} "
              f"{r['rmse']:>10.4f} {r['mae']:>10.4f} "
              f"{r['n_union_features']:>6} {r['n_molecules']:>6} {t_min:>6.1f}m")
    print(f"{'=' * 80}")

    # Save combined results
    with open(OUTPUT_DIR / "single_temp_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("All results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
