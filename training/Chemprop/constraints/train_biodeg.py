#!/usr/bin/env python3
"""
Chemprop D-MPNN biodegradability classifier.

Adds the D-MPNN arm to the biodegradability comparison, which until now was
XGBoost only. The search protocol matches tune_single_temp.py (the tuned D-MPNN
regression run), and the fold assignment matches the XGBoost classifier, so
results are paired against both.

  for each of the 5 hash-based outer folds:
      outer-train -> inner_model (85%) / inner_score (15%)
      inner_model -> 80/10/10 train / val / rank split
      Optuna TPE over N_TRIALS configs, ranked by ROC-AUC on the rank split
      best config refit on the full outer-train with a 90/10 internal split
        for checkpointing, then used to predict the outer test fold
      decision threshold picked on inner_score, never on the test fold

Class imbalance (600 pos / 1288 neg) is handled by Chemprop's --class_balance
flag, which Optuna turns on or off per fold.

Usage:
    python train_biodeg.py --trials 20
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Chemprop v1 / sklearn >=1.6 compatibility patch (same as nist8100 scripts)
# ---------------------------------------------------------------------------
import sklearn.metrics as _skm
_orig_mse = _skm.mean_squared_error


def _patched_mse(y_true, y_pred, *, squared=True, **kwargs):
    mse = _orig_mse(y_true, y_pred, **kwargs)
    return mse if squared else mse ** 0.5


_skm.mean_squared_error = _patched_mse

import torch as _torch

_orig_torch_load = _torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


_torch.load = _patched_torch_load

import optuna

from fold_utils import assign_folds, get_fold_indices
import biodeg_common as bc

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "classification" / "biodegradability"
STUDY_DIR = SCRIPT_DIR / "optuna"

TRIAL_EPOCHS = 50
FINAL_EPOCHS = 200


# ============================================================
# Chemprop invocation
# ============================================================

def _write_csv(path, smiles, targets=None):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "target"])
        if targets is None:
            for smi in smiles:
                w.writerow([smi, ""])
        else:
            for smi, val in zip(smiles, targets):
                w.writerow([smi, int(val)])


def _chemprop_args(params, epochs, seed, save_dir, data_path):
    max_lr = params["max_lr"]
    gpu = ["--gpu", "0"] if _torch.cuda.is_available() else []
    args = [
        "chemprop_train",
        "--data_path", data_path,
        "--dataset_type", "classification",
        "--save_dir", save_dir,
        "--metric", "auc",
        "--epochs", str(epochs),
        "--batch_size", str(params["batch_size"]),
        "--init_lr", str(max_lr / 10),
        "--max_lr", str(max_lr),
        "--final_lr", str(max_lr / 10),
        "--hidden_size", str(params["hidden_size"]),
        "--depth", str(params["depth"]),
        "--ffn_num_layers", str(params["ffn_num_layers"]),
        "--ffn_hidden_size", str(params["ffn_hidden_size"]),
        "--dropout", str(params["dropout"]),
        "--aggregation", params["aggregation"],
        "--seed", str(seed),
        "--quiet",
    ] + gpu
    if params.get("class_balance"):
        args.append("--class_balance")
    return args


def score_config(params, tr_smi, tr_y, va_smi, va_y, sc_smi, sc_y, seed, epochs):
    """Train on tr, checkpoint on va, return ROC-AUC on sc."""
    from chemprop.train import chemprop_train

    tmp = tempfile.mkdtemp()
    try:
        train_csv = os.path.join(tmp, "train.csv")
        val_csv = os.path.join(tmp, "val.csv")
        score_csv = os.path.join(tmp, "score.csv")
        model_dir = os.path.join(tmp, "model")
        os.makedirs(model_dir, exist_ok=True)

        _write_csv(train_csv, tr_smi, tr_y)
        _write_csv(val_csv, va_smi, va_y)
        _write_csv(score_csv, sc_smi, sc_y)

        sys.argv = _chemprop_args(params, epochs, seed, model_dir, train_csv) + [
            "--separate_val_path", val_csv,
            "--separate_test_path", score_csv,
        ]
        chemprop_train()

        scores = pd.read_csv(os.path.join(model_dir, "test_scores.csv"))
        auc = float(scores.iloc[0]["Mean auc"])
        if not np.isfinite(auc):
            raise optuna.TrialPruned("chemprop returned a non-finite score")
        return auc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fit_and_predict(params, tr_smi, tr_y, te_smi, seed, epochs, save_dir):
    """Train on tr with an internal 90/10 split, return P(class=1) for te."""
    from chemprop.train import chemprop_train, chemprop_predict

    tmp = tempfile.mkdtemp()
    try:
        train_csv = os.path.join(tmp, "train.csv")
        test_csv = os.path.join(tmp, "test.csv")
        pred_csv = os.path.join(tmp, "preds.csv")
        os.makedirs(save_dir, exist_ok=True)

        _write_csv(train_csv, tr_smi, tr_y)
        _write_csv(test_csv, te_smi)

        sys.argv = _chemprop_args(params, epochs, seed, save_dir, train_csv) + [
            "--split_sizes", "0.9", "0.1", "0.0",
        ]
        chemprop_train()

        gpu = ["--gpu", "0"] if _torch.cuda.is_available() else []
        sys.argv = [
            "chemprop_predict",
            "--test_path", test_csv,
            "--checkpoint_dir", save_dir,
            "--preds_path", pred_csv,
        ] + gpu
        chemprop_predict()

        return pd.read_csv(pred_csv).iloc[:, 1].values.astype(np.float64)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Search space (same as the regression search, plus class_balance)
# ============================================================

def suggest(trial):
    return {
        "depth":           trial.suggest_int("depth", 3, 6),
        "hidden_size":     trial.suggest_categorical("hidden_size", [300, 600, 900]),
        "ffn_num_layers":  trial.suggest_int("ffn_num_layers", 2, 3),
        "ffn_hidden_size": trial.suggest_categorical("ffn_hidden_size", [300, 600]),
        "dropout":         trial.suggest_categorical("dropout", [0.0, 0.05, 0.1, 0.2, 0.3]),
        "max_lr":          trial.suggest_float("max_lr", 3e-4, 3e-3, log=True),
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128]),
        "aggregation":     trial.suggest_categorical("aggregation", ["mean", "sum", "norm"]),
        "class_balance":   trial.suggest_categorical("class_balance", [True, False]),
    }


DEFAULT_PARAMS = {
    "depth": 3, "hidden_size": 300, "ffn_num_layers": 2,
    "ffn_hidden_size": 300, "dropout": 0.0, "max_lr": 1e-3,
    "batch_size": 64, "aggregation": "mean", "class_balance": True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    smiles, y = bc.load_biodeg(SCRIPT_DIR)
    logger.info("Loaded %d molecules (%d pos, %d neg)",
                len(y), int(y.sum()), int(len(y) - y.sum()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    folds = assign_folds(smiles, n_folds=bc.N_FOLDS)
    all_proba = np.full(len(y), np.nan)
    fold_metrics, fold_params = [], []

    search_meta = {
        "sampler": "TPE", "n_trials": args.trials,
        "trial_epochs": TRIAL_EPOCHS, "final_epochs": FINAL_EPOCHS,
        "inner_split": "85/15 model/score, then 80/10/10 of the model part",
        "objective": "ROC-AUC on the inner ranking split",
        "threshold_selection": "max F1 on held-out inner 15%",
    }

    for fold_i in range(bc.N_FOLDS):
        t_fold = time.time()
        train_idx, test_idx = get_fold_indices(folds, fold_i)
        i_model, i_score = bc.inner_split(train_idx, fold_i)

        rng = np.random.default_rng(2000 + fold_i)
        perm = rng.permutation(len(i_model))
        n_in = len(i_model)
        n_tr, n_va = int(0.8 * n_in), int(0.1 * n_in)
        i_tr = i_model[perm[:n_tr]]
        i_va = i_model[perm[n_tr:n_tr + n_va]]
        i_rank = i_model[perm[n_tr + n_va:]]

        study = optuna.create_study(
            study_name=f"biodeg_fold{fold_i}",
            storage=f"sqlite:///{STUDY_DIR / 'biodegradability.db'}",
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=fold_i),
        )
        if not study.trials:
            study.enqueue_trial(DEFAULT_PARAMS)

        def objective(trial):
            p = suggest(trial)
            return score_config(p, smiles[i_tr], y[i_tr], smiles[i_va], y[i_va],
                                smiles[i_rank], y[i_rank], seed=fold_i,
                                epochs=TRIAL_EPOCHS)

        done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        if done < args.trials:
            study.optimize(objective, n_trials=args.trials - done,
                           catch=(RuntimeError, ValueError))

        best = dict(DEFAULT_PARAMS)
        best.update(study.best_params)
        logger.info("  fold %d best inner AUC %.4f  %s",
                    fold_i, study.best_value, best)

        # threshold from a model trained on i_model only
        thr_dir = str(OUTPUT_DIR / f"fold{fold_i}_thr")
        score_proba = fit_and_predict(best, smiles[i_model], y[i_model],
                                      smiles[i_score], seed=fold_i,
                                      epochs=FINAL_EPOCHS, save_dir=thr_dir)
        threshold = bc.best_f1_threshold(y[i_score], score_proba)
        shutil.rmtree(thr_dir, ignore_errors=True)

        proba = fit_and_predict(best, smiles[train_idx], y[train_idx],
                                smiles[test_idx], seed=fold_i,
                                epochs=FINAL_EPOCHS,
                                save_dir=str(OUTPUT_DIR / f"fold{fold_i}_model"))
        all_proba[test_idx] = proba

        fm = bc.fold_scores(y[test_idx], proba, threshold)
        fm.update({
            "fold": fold_i,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "inner_roc_auc": round(float(study.best_value), 4),
            "n_trials": len(study.trials),
            "time_s": round(time.time() - t_fold, 1),
        })
        fold_metrics.append(fm)
        fold_params.append(best)

        logger.info("  fold %d  AUC=%.4f  F1=%.4f  F1_tuned=%.4f  (%.0fs)",
                    fold_i, fm["roc_auc"], fm["f1"], fm["f1_tuned"], fm["time_s"])

        bc.save_results(OUTPUT_DIR, "Chemprop_DMPNN", smiles, folds, y, all_proba,
                        fold_metrics, fold_params, search_meta, time.time() - t0)

    summary = bc.save_results(OUTPUT_DIR, "Chemprop_DMPNN", smiles, folds, y,
                              all_proba, fold_metrics, fold_params, search_meta,
                              time.time() - t0)

    logger.info("DONE  ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  F1_tuned=%.4f  MCC=%.4f",
                summary["roc_auc"], summary["pr_auc"], summary["f1"],
                summary["f1_tuned"], summary["mcc"])
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("fold_metrics", "best_params_per_fold")}, indent=2))


if __name__ == "__main__":
    main()
