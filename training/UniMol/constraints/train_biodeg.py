#!/usr/bin/env python3
"""
Uni-Mol biodegradability classifier.

Adds the Uni-Mol arm to the biodegradability comparison, which until now was
XGBoost only. Fold assignment matches the XGBoost classifier exactly, so the
comparison is paired fold by fold.

  for each of the 5 hash-based outer folds:
      outer-train -> inner_model (85%) / inner_score (15%)
      Optuna TPE over N_TRIALS configs, each trained on a 85/15 split of
        inner_model and ranked by ROC-AUC on the held-out part
      best config refit on the full outer-train, then used to predict the
        outer test fold
      decision threshold picked on inner_score, never on the test fold

Note on protocol: the Uni-Mol *regression* models in the architecture benchmark
were run at published defaults with no search. This script does search, because
the XGBoost classifier it is being compared against was tuned with 100 Optuna
trials — leaving Uni-Mol untuned would repeat the asymmetry that made the
untuned D-MPNN regression numbers misleading. Pass --trials 0 to reproduce the
published-defaults behaviour instead.

Usage:
    python train_biodeg.py --trials 15
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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

DEFAULT_PARAMS = {
    "epochs": 50,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "warmup_ratio": 0.1,
}


def _as_proba(preds) -> np.ndarray:
    """Normalise whatever MolPredict returns into a 1-D P(class=1) array."""
    if isinstance(preds, dict):
        preds = list(preds.values())[0]
    arr = np.asarray(preds, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, 1] if arr.shape[1] == 2 else arr[:, 0]
    return np.clip(arr.ravel(), 0.0, 1.0)


def train_unimol_fold(tr_smi, tr_y, te_smi, params, save_path):
    """Train Uni-Mol on one split and return P(class=1) for te_smi."""
    from unimol_tools import MolTrain, MolPredict

    tmp = tempfile.mkdtemp()
    try:
        train_csv = os.path.join(tmp, "train.csv")
        pd.DataFrame({"SMILES": tr_smi,
                      "target": np.asarray(tr_y).astype(int)}).to_csv(
            train_csv, index=False)

        trainer = MolTrain(
            task="classification",
            model_name="unimolv1",
            epochs=int(params["epochs"]),
            learning_rate=float(params["learning_rate"]),
            batch_size=int(params["batch_size"]),
            warmup_ratio=float(params["warmup_ratio"]),
            early_stopping=20,
            metrics="auc",
            kfold=1,
            remove_hs=False,
            save_path=save_path,
            smiles_col="SMILES",
            target_cols=["target"],
        )
        trainer.fit(train_csv)

        predictor = MolPredict(load_model=save_path)
        return _as_proba(predictor.predict(data=list(te_smi)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suggest(trial):
    return {
        "epochs": trial.suggest_categorical("epochs", [30, 50, 80]),
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.03, 0.06, 0.1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15,
                    help="0 reproduces the published-defaults protocol")
    args = ap.parse_args()

    t0 = time.time()
    smiles, y = bc.load_biodeg(SCRIPT_DIR)
    logger.info("Loaded %d molecules (%d pos, %d neg)",
                len(y), int(y.sum()), int(len(y) - y.sum()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_DIR / "_scratch"

    folds = assign_folds(smiles, n_folds=bc.N_FOLDS)
    all_proba = np.full(len(y), np.nan)
    fold_metrics, fold_params = [], []

    search_meta = {
        "sampler": "TPE" if args.trials else "none (published defaults)",
        "n_trials": args.trials,
        "inner_split": "85/15 model/score, trials ranked on 15% of the model part",
        "objective": "ROC-AUC on the inner ranking split",
        "threshold_selection": "max F1 on held-out inner 15%",
    }

    for fold_i in range(bc.N_FOLDS):
        t_fold = time.time()
        train_idx, test_idx = get_fold_indices(folds, fold_i)
        i_model, i_score = bc.inner_split(train_idx, fold_i)

        if args.trials > 0:
            i_tr, i_rank = bc.inner_split(i_model, 100 + fold_i)

            def objective(trial):
                p = suggest(trial)
                path = str(scratch / f"f{fold_i}_t{trial.number}")
                try:
                    proba = train_unimol_fold(smiles[i_tr], y[i_tr],
                                              smiles[i_rank], p, path)
                    from sklearn.metrics import roc_auc_score
                    return float(roc_auc_score(y[i_rank], proba))
                finally:
                    shutil.rmtree(path, ignore_errors=True)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=fold_i))
            study.enqueue_trial(DEFAULT_PARAMS)
            study.optimize(objective, n_trials=args.trials,
                           catch=(RuntimeError, ValueError))
            best = dict(DEFAULT_PARAMS)
            best.update(study.best_params)
            inner_auc = float(study.best_value)
        else:
            best = dict(DEFAULT_PARAMS)
            inner_auc = float("nan")

        logger.info("  fold %d best inner AUC %.4f  %s", fold_i, inner_auc, best)

        # threshold from a model trained on i_model only
        thr_path = str(scratch / f"f{fold_i}_thr")
        score_proba = train_unimol_fold(smiles[i_model], y[i_model],
                                        smiles[i_score], best, thr_path)
        threshold = bc.best_f1_threshold(y[i_score], score_proba)
        shutil.rmtree(thr_path, ignore_errors=True)

        proba = train_unimol_fold(smiles[train_idx], y[train_idx],
                                  smiles[test_idx], best,
                                  str(OUTPUT_DIR / f"fold{fold_i}_model"))
        all_proba[test_idx] = proba

        fm = bc.fold_scores(y[test_idx], proba, threshold)
        fm.update({
            "fold": fold_i,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "inner_roc_auc": None if np.isnan(inner_auc) else round(inner_auc, 4),
            "n_trials": args.trials,
            "time_s": round(time.time() - t_fold, 1),
        })
        fold_metrics.append(fm)
        fold_params.append(best)

        logger.info("  fold %d  AUC=%.4f  F1=%.4f  F1_tuned=%.4f  (%.0fs)",
                    fold_i, fm["roc_auc"], fm["f1"], fm["f1_tuned"], fm["time_s"])

        bc.save_results(OUTPUT_DIR, "UniMol", smiles, folds, y, all_proba,
                        fold_metrics, fold_params, search_meta, time.time() - t0)

    shutil.rmtree(scratch, ignore_errors=True)

    summary = bc.save_results(OUTPUT_DIR, "UniMol", smiles, folds, y, all_proba,
                              fold_metrics, fold_params, search_meta,
                              time.time() - t0)

    logger.info("DONE  ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  F1_tuned=%.4f  MCC=%.4f",
                summary["roc_auc"], summary["pr_auc"], summary["f1"],
                summary["f1_tuned"], summary["mcc"])
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("fold_metrics", "best_params_per_fold")}, indent=2))


if __name__ == "__main__":
    main()
