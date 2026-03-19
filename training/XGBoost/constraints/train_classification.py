#!/usr/bin/env python3
"""
Publication-quality XGBoost classification pipeline for biodegradability.

Mordred + RDKit descriptors, Optuna nested CV, SHAP analysis, diagnostic plots.
Same hash-based fold split and feature selection as the regression pipeline.

Pipeline:
  1. Load precomputed descriptors (descriptors_rdkit_mordred.csv)
  2. Hash-based 5-fold outer CV (stratified by class)
  3. Per outer fold (on training data only):
     a. Fast feature selection (variance -> correlation -> importance top-N)
     b. Optuna HP tuning (100 trials, inner stratified 5-fold CV, ROC-AUC)
     c. Train + predict on held-out test set
     d. SHAP values on test set
  4. Generate publication artifacts (ROC, PR, confusion matrix, SHAP)

Usage:
    python train_classification.py
    python train_classification.py --n_trials 20   # faster tuning
"""

import argparse
import builtins
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
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, log_loss,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
# SHAP monkey-patch: XGBoost base_score format bug ('[5.03E1]' as string)
# ---------------------------------------------------------------------------
_orig_float = builtins.float

def _safe_float(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        return _orig_float(x.strip("[]"))
    return _orig_float(x)

_real_builtins_float = builtins.float

def _patch_float():
    builtins.float = _safe_float

def _unpatch_float():
    builtins.float = _real_builtins_float

_original_XGBTreeModelLoader_init = shap.explainers._tree.XGBTreeModelLoader.__init__
def _patched_XGBTreeModelLoader_init(self, xgb_model):
    _patch_float()
    try:
        _original_XGBTreeModelLoader_init(self, xgb_model)
    finally:
        _unpatch_float()
shap.explainers._tree.XGBTreeModelLoader.__init__ = _patched_XGBTreeModelLoader_init

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
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "constraints"
OUTPUT_DIR = SCRIPT_DIR / "output" / "classification"
FIGURE_DIR = SCRIPT_DIR / "figures" / "classification"
DESCRIPTORS_PATH = DATA_DIR / "descriptors_rdkit_mordred.csv"

MODELS = {
    "biodegradability": {
        "csv": "biodegradability_cleaned.csv",
        "col": "Activity",
    },
}


# ============================================================
# Feature selection (adapted for classification — uses ROC-AUC)
# ============================================================

def fast_feature_selection(X: np.ndarray, y: np.ndarray, feature_names: list,
                           seed: int = 42) -> list:
    """
    Fast feature selection for classification (no leakage):
      1. Variance threshold (drop near-constant after scaling, var < 0.01)
      2. Correlation dedup (|r| > 0.95 -> drop weaker target correlation)
      3. Importance-based top-N with elbow detection (ROC-AUC)
    """
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

    # 3. Importance-based top-N with elbow detection (ROC-AUC)
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=6,
        random_state=seed, n_jobs=-1, verbosity=0,
        use_label_encoder=False, eval_metric="logloss",
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
        xgb_eval = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=seed, n_jobs=-1, verbosity=0,
            use_label_encoder=False, eval_metric="logloss",
        )
        cv_scores = cross_val_score(
            xgb_eval, X_sub, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
            scoring="roc_auc",
        )
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

    logger.info("      Feature selection: %d -> %d (var) -> %d (corr) -> %d (elbow, AUC=%.4f)",
                n_start, n_after_var, n_after_corr, best_n,
                scores[N_candidates.index(best_n)])

    return selected_names


# ============================================================
# Optuna HP tuning (classification — ROC-AUC)
# ============================================================

def optuna_hp_tuning(X: np.ndarray, y: np.ndarray, n_trials: int = 100,
                     timeout: int = 600, seed: int = 42) -> dict:
    """Optuna TPE HP tuning with MedianPruner. Inner stratified 5-fold CV."""

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
            "scale_pos_weight": trial.suggest_categorical(
                "scale_pos_weight", [1.0, 2.0, 3.0, 5.0]),
        }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **params, random_state=seed, n_jobs=-1, verbosity=0,
                use_label_encoder=False,
            )
            model.fit(X[tr_idx], y[tr_idx])
            y_proba = model.predict_proba(X[val_idx])[:, 1]
            auc = roc_auc_score(y[val_idx], y_proba)
            fold_scores.append(auc)

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
    logger.info("      Optuna: %d trials (%d pruned), best AUC=%.4f",
                len(study.trials), n_pruned, study.best_value)

    return study.best_params


# ============================================================
# Plotting (CLAUDE.md style: no titles, no grids, frameon=False)
# ============================================================

def save_roc_plot(y_true, y_proba, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot(fpr, tpr, lw=2, c="#1f77b4", label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics_plot(y_true, y_pred, y_proba, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)

    # ROC
    ax = axes[0, 0]
    ax.plot(fpr, tpr, lw=2, c="#1f77b4")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.text(0.95, 0.05, f"AUC = {auc:.4f}", transform=ax.transAxes,
            fontsize=9, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8))

    # Precision-Recall
    ax = axes[0, 1]
    ax.plot(rec, prec, lw=2, c="#1f77b4")
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1, alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.text(0.05, 0.05, f"PR-AUC = {pr_auc:.4f}", transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8))

    # Confusion matrix
    ax = axes[1, 0]
    ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Neg", "Pos"])
    ax.set_yticklabels(["Neg", "Pos"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Calibration
    ax = axes[1, 1]
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        ax.plot(prob_pred, prob_true, "o-", c="#1f77b4", lw=2)
    except ValueError:
        pass
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")

    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_shap_plots(shap_values, feature_names, X_display, out_dir,
                    max_display=20):
    explanation = shap.Explanation(
        values=shap_values,
        data=X_display,
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(10, 6), dpi=150)
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    fig = plt.figure(figsize=(10, 6), dpi=150)
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")


# ============================================================
# Per-property training pipeline
# ============================================================

def train_property(prop_name: str, prop_config: dict, desc_df: pd.DataFrame,
                   n_trials: int, timeout: int) -> dict:
    """Full nested-CV pipeline for a single classification target."""
    t0_total = time.time()

    csv_path = DATA_DIR / prop_config["csv"]
    target_col = prop_config["col"]

    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df[target_col].notna()].copy()

    merged = df[["SMILES", target_col]].merge(desc_df, on="SMILES", how="inner")
    logger.info("  Loaded %d molecules (%d after descriptor join)",
                len(df), len(merged))

    smiles = merged["SMILES"].values
    y = merged[target_col].values.astype(int)
    desc_cols = [c for c in merged.columns if c not in ("SMILES", target_col)]
    X_raw = merged[desc_cols].values.astype(np.float64)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    logger.info("  Class distribution: %d neg, %d pos (%.1f%% positive)",
                n_neg, n_pos, 100 * n_pos / len(y))

    out_dir = OUTPUT_DIR / prop_name
    fig_dir = FIGURE_DIR / prop_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    folds = assign_folds(smiles, n_folds=5)

    all_preds = np.full(len(y), np.nan)
    all_probas = np.full(len(y), np.nan)
    fold_metrics = []
    fold_selected_features = []
    fold_best_params = []
    fold_shap_data = []

    for fold_i in range(5):
        logger.info("    Fold %d:", fold_i)
        t0_fold = time.time()
        train_idx, test_idx = get_fold_indices(folds, fold_i)
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        selected = fast_feature_selection(X_train, y_train, desc_cols, seed=fold_i)
        feat_idx = [desc_cols.index(f) for f in selected]
        X_train_sel = np.nan_to_num(X_train[:, feat_idx], nan=0.0, posinf=0.0, neginf=0.0)
        X_test_sel = np.nan_to_num(X_test[:, feat_idx], nan=0.0, posinf=0.0, neginf=0.0)

        best_params = optuna_hp_tuning(
            X_train_sel, y_train,
            n_trials=n_trials, timeout=timeout, seed=fold_i,
        )

        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            **best_params, random_state=fold_i, n_jobs=-1, verbosity=0,
            use_label_encoder=False,
        )
        model.fit(X_train_sel, y_train)
        preds = model.predict(X_test_sel)
        probas = model.predict_proba(X_test_sel)[:, 1]
        all_preds[test_idx] = preds
        all_probas[test_idx] = probas

        # SHAP
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test_sel)

        scaler = StandardScaler()
        scaler.fit(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        fold_shap_data.append((test_idx, sv, X_test_scaled, selected))
        fold_selected_features.append(selected)
        fold_best_params.append(best_params)

        fold_auc = roc_auc_score(y_test, probas)
        fold_f1 = f1_score(y_test, preds)
        fold_acc = accuracy_score(y_test, preds)
        fold_time = time.time() - t0_fold

        fold_metrics.append({
            "fold": fold_i,
            "roc_auc": round(float(fold_auc), 4),
            "f1": round(float(fold_f1), 4),
            "accuracy": round(float(fold_acc), 4),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_selected_features": len(selected),
            "best_params": best_params,
            "time_s": round(fold_time, 1),
        })

        joblib.dump({
            "model": model,
            "selected_features": selected,
            "best_params": best_params,
            "property": prop_name,
            "task": "classification",
        }, out_dir / f"fold{fold_i}_model.joblib")

        logger.info("      AUC=%.4f  F1=%.4f  Acc=%.4f  (%d feats, %d/%d train/test, %.1fs)",
                     fold_auc, fold_f1, fold_acc, len(selected),
                     len(train_idx), len(test_idx), fold_time)

    # --- Aggregate predictions ---
    valid = ~np.isnan(all_probas)
    y_true = y[valid]
    y_pred = all_preds[valid].astype(int)
    y_proba = all_probas[valid]

    pred_df = pd.DataFrame({
        "SMILES": smiles,
        "fold": folds,
        "y_true": y,
        "y_pred": all_preds,
        "y_proba": all_probas,
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # --- Aggregate SHAP ---
    all_features_union = sorted(set().union(*fold_selected_features))
    n_union = len(all_features_union)
    union_index = {f: i for i, f in enumerate(all_features_union)}

    shap_full = np.zeros((len(y), n_union))
    X_display_full = np.zeros((len(y), n_union))

    for test_idx, sv, X_test_scaled, selected in fold_shap_data:
        for feat_i, feat_name in enumerate(selected):
            j = union_index[feat_name]
            shap_full[test_idx, j] = sv[:, feat_i]
            X_display_full[test_idx, j] = X_test_scaled[:, feat_i]

    np.savez(out_dir / "shap_data.npz",
             shap_values=shap_full,
             X_scaled=X_display_full,
             feature_names=np.array(all_features_union))

    mean_abs_shap = np.abs(shap_full).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        "feature": all_features_union,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_imp_df.to_csv(out_dir / "shap_importance.csv", index=False)

    # --- Full-data model ---
    logger.info("    Training full-data model...")
    full_selected = fast_feature_selection(X_raw, y, desc_cols, seed=42)
    full_feat_idx = [desc_cols.index(f) for f in full_selected]
    X_full_sel = np.nan_to_num(X_raw[:, full_feat_idx], nan=0.0, posinf=0.0, neginf=0.0)

    best_fold_idx = max(range(5), key=lambda i: fold_metrics[i]["roc_auc"])
    full_params = fold_best_params[best_fold_idx]
    logger.info("    Using params from fold %d (AUC=%.4f)", best_fold_idx,
                fold_metrics[best_fold_idx]["roc_auc"])

    model_full = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        **full_params, random_state=42, n_jobs=-1, verbosity=0,
        use_label_encoder=False,
    )
    model_full.fit(X_full_sel, y)

    joblib.dump({
        "model": model_full,
        "selected_features": full_selected,
        "best_params": full_params,
        "property": prop_name,
        "task": "classification",
    }, out_dir / "model.joblib")

    # --- Plots ---
    save_roc_plot(y_true, y_proba, fig_dir)
    save_diagnostics_plot(y_true, y_pred, y_proba, fig_dir)
    save_shap_plots(shap_full, all_features_union, X_display_full, fig_dir)
    logger.info("    Plots saved to %s", fig_dir)

    # --- Overall CV metrics ---
    auc_cv = roc_auc_score(y_true, y_proba)
    pr_auc_cv = average_precision_score(y_true, y_proba)
    f1_cv = f1_score(y_true, y_pred)
    acc_cv = accuracy_score(y_true, y_pred)
    bal_acc_cv = balanced_accuracy_score(y_true, y_pred)
    mcc_cv = matthews_corrcoef(y_true, y_pred)

    fold_aucs = [m["roc_auc"] for m in fold_metrics]
    fold_f1s = [m["f1"] for m in fold_metrics]

    total_time = time.time() - t0_total

    summary = {
        "property": prop_name,
        "task": "classification",
        "roc_auc": round(auc_cv, 4),
        "pr_auc": round(pr_auc_cv, 4),
        "f1": round(f1_cv, 4),
        "accuracy": round(acc_cv, 4),
        "balanced_accuracy": round(bal_acc_cv, 4),
        "mcc": round(mcc_cv, 4),
        "std_roc_auc": round(float(np.std(fold_aucs)), 4),
        "std_f1": round(float(np.std(fold_f1s)), 4),
        "n_molecules": int(valid.sum()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_raw_features": len(desc_cols),
        "n_union_features": n_union,
        "n_full_model_features": len(full_selected),
        "hash_method": "md5_mod5",
        "best_fold_params_source": best_fold_idx,
        "full_model_params": full_params,
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
        description="Train classification XGBoost models (Mordred + Optuna)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--properties", nargs="+", default=["all"],
        help="Properties to train (biodegradability) or 'all'",
    )
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Optuna trials per fold")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Optuna timeout per fold (seconds)")
    args = parser.parse_args()

    if args.properties == ["all"]:
        properties = list(MODELS.keys())
    else:
        properties = args.properties
        for p in properties:
            if p not in MODELS:
                parser.error(f"Unknown property: {p}. Valid: {list(MODELS.keys())}")

    if not DESCRIPTORS_PATH.exists():
        logger.error("Precomputed descriptors not found: %s", DESCRIPTORS_PATH)
        logger.error("Run compute_descriptors.py first.")
        return

    logger.info("Loading precomputed descriptors from %s", DESCRIPTORS_PATH.name)
    t0 = time.time()
    desc_df = pd.read_csv(DESCRIPTORS_PATH)
    logger.info("  %d SMILES x %d descriptors (%.1fs)",
                len(desc_df), len(desc_df.columns) - 1, time.time() - t0)

    logger.info("=" * 70)
    logger.info("Training %d classifiers (%d Optuna trials/fold, %ds timeout)",
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
            results[prop_name] = summary
            logger.info("  => AUC=%.4f +/- %.4f  F1=%.4f  N=%d  (%.1fs)",
                        summary["roc_auc"], summary["std_roc_auc"],
                        summary["f1"], summary["n_molecules"], summary["time_s"])

    # Summary
    print(f"\n{'=' * 80}")
    print("CLASSIFICATION MODELS (5-fold nested CV, Mordred+RDKit, Optuna)")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'AUC':>8} {'+-':>6} {'F1':>8} {'MCC':>8} "
          f"{'Feats':>6} {'N':>6} {'Time':>7}")
    print("-" * 80)
    for key, r in results.items():
        t_min = r["time_s"] / 60
        print(f"{key:<20} {r['roc_auc']:>8.4f} {r['std_roc_auc']:>6.4f} "
              f"{r['f1']:>8.4f} {r['mcc']:>8.4f} "
              f"{r['n_union_features']:>6} {r['n_molecules']:>6} {t_min:>6.1f}m")
    print(f"{'=' * 80}")

    with open(OUTPUT_DIR / "classification_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("All results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
