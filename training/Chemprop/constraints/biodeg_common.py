"""
Shared pieces for the biodegradability classifier benchmark.

The XGBoost classifier in training/XGBoost/constraints/ is the reference. This
module reproduces its data loading, fold assignment and metric definitions so
that the MLP, D-MPNN and Uni-Mol results are directly comparable to it and to
each other. A copy of this file lives in each architecture's constraints/
directory, matching the existing convention for fold_utils.py.

Protocol notes:
  * Folds are the same hash-based md5_mod5 assignment used everywhere else, so
    every architecture sees identical train/test molecules and the comparison
    is paired fold by fold.
  * ROC-AUC is the headline metric because it is threshold-free. The XGBoost
    reference used scale_pos_weight and a fixed 0.5 threshold; the neural
    architectures have different (or no) native class-weighting, so reporting
    F1/accuracy at 0.5 alone would penalise them for a calibration difference
    rather than a representation difference. Both are therefore recorded: at
    the fixed 0.5 threshold, and at a threshold chosen on held-out inner data.
  * The tuned threshold is never chosen on the outer test fold.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

N_FOLDS = 5
TARGET_COL = "Activity"
CSV_NAME = "biodegradability_cleaned.csv"


def data_root(script_dir: Path) -> Path:
    """training/data/ — two levels up from an architecture's constraints/ dir."""
    return script_dir.parent.parent / "data"


def load_biodeg(script_dir: Path):
    """Return (smiles, y) for the biodegradability dataset."""
    csv_path = data_root(script_dir) / "constraints" / CSV_NAME
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df[TARGET_COL].notna()].copy()
    smiles = df["SMILES"].values
    y = df[TARGET_COL].values.astype(int)
    return smiles, y


def load_descriptors(script_dir: Path) -> pd.DataFrame:
    """Precomputed RDKit+Mordred descriptor table for the constraint datasets."""
    path = data_root(script_dir) / "constraints" / "descriptors_rdkit_mordred.csv"
    return pd.read_csv(path, low_memory=False)


def inner_split(train_idx, fold_i, frac_score=0.15):
    """
    Deterministic split of an outer training set into a modelling part and a
    held-out scoring part. The scoring part is used to pick the decision
    threshold, never to fit the model.
    """
    rng = np.random.default_rng(1000 + fold_i)
    perm = rng.permutation(len(train_idx))
    n_score = max(1, int(frac_score * len(train_idx)))
    i_score = train_idx[perm[:n_score]]
    i_model = train_idx[perm[n_score:]]
    return i_model, i_score


def best_f1_threshold(y_true, y_proba):
    """Threshold maximising F1 on the given (inner, held-out) data."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
              for t in grid]
    return float(grid[int(np.argmax(scores))])


def oversample(idx, y, seed):
    """Random oversampling of the minority class within a training index set."""
    rng = np.random.default_rng(seed)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return idx
    if len(pos) < len(neg):
        extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    else:
        extra = rng.choice(neg, size=len(pos) - len(neg), replace=True)
    out = np.concatenate([idx, extra])
    rng.shuffle(out)
    return out


def _threshold_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


def fold_scores(y_true, y_proba, threshold):
    """Per-fold metrics: AUCs plus threshold metrics at 0.5 and at `threshold`."""
    out = {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "threshold": round(float(threshold), 3),
    }
    out.update(_threshold_metrics(y_true, y_proba, 0.5))
    tuned = _threshold_metrics(y_true, y_proba, threshold)
    out.update({f"{k}_tuned": v for k, v in tuned.items()})
    return out


def save_results(out_dir: Path, model_name: str, smiles, folds, y, all_proba,
                 fold_metrics, fold_params, search_meta, elapsed):
    """
    Write predictions.csv and results.json. The results.json schema mirrors the
    XGBoost classifier's so the table generators can read either unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = ~np.isnan(all_proba)
    y_true = y[valid]
    y_proba = all_proba[valid]

    # Pooled threshold metrics use each fold's own inner-selected threshold,
    # applied to that fold's molecules — never a threshold fitted on the pool.
    thr_by_fold = {fm["fold"]: fm["threshold"] for fm in fold_metrics}
    pooled_thr = np.array([thr_by_fold.get(int(f), 0.5) for f in folds])[valid]
    y_pred_05 = (y_proba >= 0.5).astype(int)
    y_pred_tuned = (y_proba >= pooled_thr).astype(int)

    pd.DataFrame({
        "SMILES": smiles,
        "fold": folds,
        "y_true": y,
        "y_proba": all_proba,
        "y_pred_0.5": np.where(valid, (all_proba >= 0.5).astype(float), np.nan),
    }).to_csv(out_dir / "predictions.csv", index=False)

    def _std(key):
        vals = [fm[key] for fm in fold_metrics if key in fm]
        return round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0

    summary = {
        "property": "biodegradability",
        "task": "classification",
        "model": model_name,
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "f1": round(float(f1_score(y_true, y_pred_05, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred_05)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred_05)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred_05)), 4),
        "f1_tuned": round(float(f1_score(y_true, y_pred_tuned, zero_division=0)), 4),
        "accuracy_tuned": round(float(accuracy_score(y_true, y_pred_tuned)), 4),
        "balanced_accuracy_tuned": round(float(balanced_accuracy_score(y_true, y_pred_tuned)), 4),
        "mcc_tuned": round(float(matthews_corrcoef(y_true, y_pred_tuned)), 4),
        "std_roc_auc": _std("roc_auc"),
        "std_f1": _std("f1"),
        "n_molecules": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "n_folds_done": len(fold_metrics),
        "hash_method": "md5_mod5",
        "search": search_meta,
        "best_params_per_fold": fold_params,
        "fold_metrics": fold_metrics,
        "time_s": round(float(elapsed), 1),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
