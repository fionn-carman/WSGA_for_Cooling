#!/usr/bin/env python3
"""
MLP biodegradability classifier, matching the XGBoost reference pipeline.

The architecture benchmark covers ten regression targets but never included the
biodegradability classifier, which was trained with XGBoost alone. This script
adds the MLP arm.

Protocol (identical to training/XGBoost/constraints/train_classification.py
except for the estimator):
  1. Load precomputed RDKit+Mordred descriptors
  2. Hash-based 5-fold outer CV (md5_mod5)
  3. Per outer fold, on training data only:
     a. Fast feature selection (variance -> correlation dedup -> importance top-N)
     b. Optuna TPE search, inner stratified 5-fold CV, ROC-AUC objective
     c. Fit on the outer training set, predict probabilities on the held-out fold
     d. Decision threshold picked on a held-out inner slice, never on the test fold
  4. Pool out-of-fold probabilities and report

Class imbalance (600 positive / 1288 negative) is handled by optional random
oversampling of the minority class, which Optuna turns on or off per fold.
MLPClassifier has no class_weight argument, so this is the equivalent of the
XGBoost pipeline's scale_pos_weight.

Usage:
    python train_biodeg.py                 # 100 trials per fold
    python train_biodeg.py --n_trials 20   # faster
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Install optuna: pip install optuna")

from fold_utils import assign_folds, get_fold_indices
import biodeg_common as bc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "classification" / "biodegradability"

# Same candidate architectures as the MLP regression pipeline
HIDDEN_LAYER_CHOICES = [(128,), (256,), (256, 128), (512, 256), (256, 128, 64)]
BATCH_SIZE_CHOICES = [32, 64, 128, 256]
ACTIVATION_CHOICES = ["relu", "tanh"]

MAX_FEATURES = 200


# ============================================================
# Feature selection (classification variant, ROC-AUC ranked)
# ============================================================

def fast_feature_selection(X, y, feature_names, seed=42):
    """Variance threshold -> correlation dedup -> importance top-N with elbow."""
    from xgboost import XGBClassifier

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Drop near-constant columns
    scaler = StandardScaler()
    with np.errstate(invalid="ignore", divide="ignore"):
        X_scaled = np.nan_to_num(scaler.fit_transform(X), nan=0.0)
    variances = np.var(X_scaled, axis=0)
    active = [i for i, v in enumerate(variances) if v > 0.01]
    if not active:
        active = list(np.argsort(variances)[-10:])

    # 2. Correlation dedup: of any |r| > 0.95 pair keep the one better
    #    correlated with the target
    Xa = X_scaled[:, active]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(Xa, rowvar=False))
        target_corr = np.abs(np.nan_to_num(
            np.array([np.corrcoef(Xa[:, j], y)[0, 1] for j in range(Xa.shape[1])])))
    drop = set()
    for i in range(len(active)):
        if i in drop:
            continue
        for j in range(i + 1, len(active)):
            if j in drop:
                continue
            if abs(corr[i, j]) > 0.95:
                drop.add(j if target_corr[j] <= target_corr[i] else i)
                if i in drop:
                    break
    active = [a for k, a in enumerate(active) if k not in drop]

    # 3. Importance ranking with a quick gradient-boosted model
    Xb = np.nan_to_num(X[:, active], nan=0.0, posinf=0.0, neginf=0.0)
    ranker = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=-1, verbosity=0, eval_metric="logloss",
    )
    ranker.fit(Xb, y)
    imp = ranker.feature_importances_
    order = np.argsort(imp)[::-1]

    # elbow: keep features up to 99% of cumulative importance, capped
    cum = np.cumsum(imp[order]) / max(imp.sum(), 1e-12)
    n_keep = int(np.searchsorted(cum, 0.99) + 1)
    n_keep = int(np.clip(n_keep, 20, min(MAX_FEATURES, len(active))))

    keep = [active[i] for i in order[:n_keep]]
    return [feature_names[i] for i in keep]


# ============================================================
# Optuna search
# ============================================================

def build_model(params, seed):
    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        learning_rate_init=params["learning_rate_init"],
        alpha=params["alpha"],
        batch_size=params["batch_size"],
        activation=params["activation"],
        solver="adam",
        early_stopping=True,
        max_iter=1000,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=seed,
        verbose=False,
    )


def optuna_hp_tuning(X, y, n_trials, seed):
    """Inner stratified 5-fold CV, maximising ROC-AUC."""

    def objective(trial):
        hidden_idx = trial.suggest_int("hidden_layer_idx", 0, len(HIDDEN_LAYER_CHOICES) - 1)
        params = {
            "hidden_layer_sizes": HIDDEN_LAYER_CHOICES[hidden_idx],
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES),
            "activation": trial.suggest_categorical("activation", ACTIVATION_CHOICES),
        }
        do_oversample = trial.suggest_categorical("oversample", [True, False])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = []
        for tr, va in skf.split(X, y):
            if do_oversample:
                tr = bc.oversample(np.asarray(tr), y, seed)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_va = scaler.transform(X[va])
            model = build_model(params, seed)
            model.fit(X_tr, y[tr])
            proba = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y[va], proba))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, catch=(ValueError, RuntimeError))

    best = dict(study.best_params)
    best["hidden_layer_sizes"] = HIDDEN_LAYER_CHOICES[best.pop("hidden_layer_idx")]
    return best, float(study.best_value)


# ============================================================
# Main pipeline
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    smiles, y = bc.load_biodeg(SCRIPT_DIR)
    desc_df = bc.load_descriptors(SCRIPT_DIR)

    merged = pd.DataFrame({"SMILES": smiles, bc.TARGET_COL: y}).merge(
        desc_df, on="SMILES", how="inner")
    smiles = merged["SMILES"].values
    y = merged[bc.TARGET_COL].values.astype(int)
    desc_cols = [c for c in merged.columns if c not in ("SMILES", bc.TARGET_COL)]
    X_raw = merged[desc_cols].values.astype(np.float64)

    logger.info("Loaded %d molecules (%d pos, %d neg), %d descriptors",
                len(y), int(y.sum()), int(len(y) - y.sum()), len(desc_cols))

    folds = assign_folds(smiles, n_folds=bc.N_FOLDS)
    all_proba = np.full(len(y), np.nan)
    fold_metrics, fold_params = [], []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_i in range(bc.N_FOLDS):
        t_fold = time.time()
        train_idx, test_idx = get_fold_indices(folds, fold_i)
        i_model, i_score = bc.inner_split(train_idx, fold_i)

        selected = fast_feature_selection(
            X_raw[train_idx], y[train_idx], desc_cols, seed=fold_i)
        feat_idx = [desc_cols.index(f) for f in selected]

        def prep(idx):
            return np.nan_to_num(X_raw[np.ix_(idx, feat_idx)],
                                 nan=0.0, posinf=0.0, neginf=0.0)

        best, inner_auc = optuna_hp_tuning(
            prep(i_model), y[i_model], n_trials=args.n_trials, seed=fold_i)
        logger.info("  fold %d: %d feats, inner AUC %.4f, %s",
                    fold_i, len(selected), inner_auc, best)

        # threshold on the held-out inner slice, using a model fit on i_model
        scaler = StandardScaler()
        fit_idx = bc.oversample(i_model, y, fold_i) if best.get("oversample") else i_model
        Xs = scaler.fit_transform(prep(fit_idx))
        thr_model = build_model(best, fold_i)
        thr_model.fit(Xs, y[fit_idx])
        score_proba = thr_model.predict_proba(scaler.transform(prep(i_score)))[:, 1]
        threshold = bc.best_f1_threshold(y[i_score], score_proba)

        # final model on the full outer training set
        scaler = StandardScaler()
        fit_idx = bc.oversample(train_idx, y, fold_i) if best.get("oversample") else train_idx
        X_tr = scaler.fit_transform(prep(fit_idx))
        model = build_model(best, fold_i)
        model.fit(X_tr, y[fit_idx])
        proba = model.predict_proba(scaler.transform(prep(test_idx)))[:, 1]
        all_proba[test_idx] = proba

        fm = bc.fold_scores(y[test_idx], proba, threshold)
        fm.update({
            "fold": fold_i,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_selected_features": len(selected),
            "inner_roc_auc": round(inner_auc, 4),
            "n_trials": args.n_trials,
            "time_s": round(time.time() - t_fold, 1),
        })
        fold_metrics.append(fm)

        params_record = dict(best)
        params_record["hidden_layer_sizes"] = list(best["hidden_layer_sizes"])
        params_record["selected_features"] = selected
        fold_params.append(params_record)

        joblib.dump({"model": model, "scaler": scaler,
                     "selected_features": selected, "threshold": threshold,
                     "best_params": params_record, "property": "biodegradability",
                     "task": "classification"},
                    OUTPUT_DIR / f"fold{fold_i}_model.joblib")

        logger.info("  fold %d  AUC=%.4f  F1=%.4f  F1_tuned=%.4f  (%.0fs)",
                    fold_i, fm["roc_auc"], fm["f1"], fm["f1_tuned"], fm["time_s"])

        # partial save after every fold so a walltime kill is not fatal
        bc.save_results(OUTPUT_DIR, "MLP", smiles, folds, y, all_proba,
                        fold_metrics, fold_params,
                        {"sampler": "TPE", "n_trials": args.n_trials,
                         "inner_cv": "stratified 5-fold", "objective": "ROC-AUC",
                         "threshold_selection": "max F1 on held-out inner 15%"},
                        time.time() - t0)

    summary = bc.save_results(
        OUTPUT_DIR, "MLP", smiles, folds, y, all_proba, fold_metrics, fold_params,
        {"sampler": "TPE", "n_trials": args.n_trials,
         "inner_cv": "stratified 5-fold", "objective": "ROC-AUC",
         "threshold_selection": "max F1 on held-out inner 15%"},
        time.time() - t0)

    logger.info("DONE  ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  F1_tuned=%.4f  MCC=%.4f",
                summary["roc_auc"], summary["pr_auc"], summary["f1"],
                summary["f1_tuned"], summary["mcc"])
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("fold_metrics", "best_params_per_fold")}, indent=2))


if __name__ == "__main__":
    main()
