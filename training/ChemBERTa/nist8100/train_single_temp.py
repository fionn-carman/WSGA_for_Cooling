#!/usr/bin/env python3
"""
Publication-quality ChemBERTa-2 training pipeline for NIST 8100 single-temp models.

Fine-tunes DeepChem/ChemBERTa-77M-MTR with Optuna hyperparameter tuning and
hash-based 5-fold outer CV. No descriptors needed — operates directly on SMILES.

Pipeline per property (40C):
  1. Load property CSV (SMILES + target column only)
  2. Apply log1p transform if needed (viscosity, fom1)
  3. Hash-based 5-fold assignment
  4. Per outer fold:
     a. Optuna tune (30 trials, inner 15% holdout, TPE + MedianPruner)
     b. StandardScaler on y_train
     c. Train final model (15% held out for early stopping)
     d. Predict on test fold -> inverse_scale -> expm1 if log -> metrics
     e. Save fold model state dict
  5. Aggregate OOF predictions -> predictions.csv
  6. Compute pooled metrics -> results.json
  7. Generate parity + diagnostics plots

Usage:
    python train_single_temp.py                              # all 6 properties
    python train_single_temp.py --properties density         # single property
    python train_single_temp.py --properties density --n_trials 2 --epochs 5  # quick test
"""

import argparse
import json
import logging
import math
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
from transformers import AutoTokenizer

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError("Install optuna: pip install optuna")

from fold_utils import assign_folds, get_fold_indices
from chemberta_model import (
    MODEL_NAME,
    ChemBERTaPredictor,
    gaussian_nll_loss,
    train_one_model,
    predict,
)

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


# ============================================================
# Optuna HP tuning
# ============================================================

def optuna_hp_tuning(
    smiles_train: list[str],
    y_train_scaled: np.ndarray,
    tokenizer,
    device: torch.device,
    n_trials: int = 30,
    timeout: int = 3600,
    epochs: int = 100,
    batch_size: int = 16,
    patience: int = 15,
    max_length: int = 128,
    seed: int = 42,
) -> dict:
    """Optuna TPE tuning with MedianPruner. Inner 15% holdout validation."""
    from sklearn.model_selection import train_test_split

    # Inner holdout split
    sm_tr, sm_vl, y_tr, y_vl = train_test_split(
        smiles_train, y_train_scaled,
        test_size=0.15, random_state=seed,
    )

    def objective(trial):
        lr_backbone = trial.suggest_float("lr_backbone", 1e-5, 3e-4, log=True)
        lr_head = trial.suggest_float("lr_head", 5e-4, 5e-3, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.2)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-4)

        try:
            model, best_val_loss = train_one_model(
                smiles_train=list(sm_tr),
                y_train=y_tr,
                smiles_val=list(sm_vl),
                y_val=y_vl,
                tokenizer=tokenizer,
                device=device,
                hidden_dim=hidden_dim,
                dropout=dropout,
                lr_backbone=lr_backbone,
                lr_head=lr_head,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                max_length=max_length,
                trial=trial,
                verbose=False,
            )
        finally:
            # Clean GPU memory between trials
            if "model" in dir():
                del model
            torch.cuda.empty_cache()

        return best_val_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    logger.info("      Optuna: %d trials (%d pruned), best val loss=%.4f",
                len(study.trials), n_pruned, study.best_value)

    return study.best_params


# ============================================================
# Per-property training pipeline
# ============================================================

def train_property(
    prop_name: str,
    prop_config: dict,
    tokenizer,
    device: torch.device,
    n_trials: int,
    timeout: int,
    epochs: int,
    batch_size: int,
    patience: int,
) -> dict:
    """Full 5-fold CV pipeline for a single property at 40C."""
    t0_total = time.time()

    # --- Load property data ---
    csv_path = DATA_DIR / prop_config["csv"]
    target_col = prop_config["col_40"]
    log_transform = prop_config["log"]

    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df[target_col].notna()].copy()

    smiles = df["SMILES"].values
    y_raw = df[target_col].values.astype(np.float64)
    logger.info("  Loaded %d molecules", len(df))

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
    fold_best_params = []

    for fold_i in range(5):
        logger.info("    Fold %d:", fold_i)
        t0_fold = time.time()
        train_idx, test_idx = get_fold_indices(folds, fold_i)

        smiles_train = smiles[train_idx].tolist()
        smiles_test = smiles[test_idx].tolist()
        y_train = y[train_idx]
        y_test = y[test_idx]

        # StandardScaler on y (fit on train)
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        # 4a. Optuna tuning
        logger.info("      Optuna tuning (%d trials)...", n_trials)
        best_params = optuna_hp_tuning(
            smiles_train=smiles_train,
            y_train_scaled=y_train_scaled,
            tokenizer=tokenizer,
            device=device,
            n_trials=n_trials,
            timeout=timeout,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=fold_i,
        )
        fold_best_params.append(best_params)
        logger.info("      Best params: %s", best_params)

        # 4b. Train final model on full training fold (15% held out for early stopping)
        from sklearn.model_selection import train_test_split
        sm_tr, sm_vl, y_tr, y_vl = train_test_split(
            smiles_train, y_train_scaled,
            test_size=0.15, random_state=fold_i + 100,
        )

        logger.info("      Training final model (train=%d, val=%d)...", len(sm_tr), len(sm_vl))
        model, best_val_loss = train_one_model(
            smiles_train=sm_tr,
            y_train=y_tr,
            smiles_val=sm_vl,
            y_val=y_vl,
            tokenizer=tokenizer,
            device=device,
            hidden_dim=best_params["hidden_dim"],
            dropout=best_params["dropout"],
            lr_backbone=best_params["lr_backbone"],
            lr_head=best_params["lr_head"],
            weight_decay=best_params["weight_decay"],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=True,
        )

        # 4c. Predict on test fold
        preds_scaled = predict(model, smiles_test, tokenizer, device, batch_size=batch_size)

        # Inverse scale
        preds_log = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        all_preds[test_idx] = preds_log

        # Save fold model
        torch.save(model.state_dict(), out_dir / f"fold{fold_i}_model.pt")

        # Clean up GPU
        del model
        torch.cuda.empty_cache()

        # Per-fold metrics (real space)
        if log_transform:
            y_test_real = np.expm1(y_test)
            preds_real = np.expm1(preds_log)
        else:
            y_test_real = y_test
            preds_real = preds_log

        fold_r2 = r2_score(y_test_real, preds_real)
        fold_rmse = math.sqrt(mean_squared_error(y_test_real, preds_real))
        fold_mae = mean_absolute_error(y_test_real, preds_real)
        fold_time = time.time() - t0_fold

        fold_metrics.append({
            "fold": fold_i,
            "r2": round(float(fold_r2), 4),
            "rmse": round(float(fold_rmse), 4),
            "mae": round(float(fold_mae), 4),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "best_params": best_params,
            "time_s": round(fold_time, 1),
        })

        logger.info("      R2=%.4f  RMSE=%.4f  MAE=%.4f  (%d/%d train/test, %.1fs)",
                     fold_r2, fold_rmse, fold_mae,
                     len(train_idx), len(test_idx), fold_time)

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

    # --- Plots ---
    valid = ~np.isnan(all_preds)
    save_parity_plot(y_true_real[valid], y_pred_real[valid], fig_dir, prop_name)
    save_diagnostics_plot(y_true_real[valid], y_pred_real[valid], fig_dir, prop_name)
    logger.info("    Plots saved to %s", fig_dir)

    # --- Overall CV metrics ---
    r2_cv = r2_score(y_true_real[valid], y_pred_real[valid])
    rmse_cv = math.sqrt(mean_squared_error(y_true_real[valid], y_pred_real[valid]))
    mae_cv = mean_absolute_error(y_true_real[valid], y_pred_real[valid])

    fold_r2s = [m["r2"] for m in fold_metrics]
    fold_rmses = [m["rmse"] for m in fold_metrics]
    fold_maes = [m["mae"] for m in fold_metrics]

    total_time = time.time() - t0_total

    summary = {
        "property": prop_name,
        "temperature": "40C",
        "model": "ChemBERTa-77M-MTR",
        "r2": round(r2_cv, 4),
        "rmse": round(rmse_cv, 4),
        "mae": round(mae_cv, 4),
        "std_r2": round(float(np.std(fold_r2s)), 4),
        "std_rmse": round(float(np.std(fold_rmses)), 4),
        "std_mae": round(float(np.std(fold_maes)), 4),
        "n_molecules": int(valid.sum()),
        "log_transform": log_transform,
        "hash_method": "md5_mod5",
        "config": {
            "model_name": MODEL_NAME,
            "n_trials": n_trials,
            "timeout": timeout,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "max_length": 128,
            "scheduler": "plateau",
            "loss": "gaussian_nll",
            "y_scaler": "StandardScaler",
        },
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
        description="Train single-temp ChemBERTa-2 models (Optuna-tuned)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--properties", nargs="+", default=["all"],
        help="Properties to train (density viscosity tc cpsat beta fom1) or 'all'",
    )
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Optuna trials per fold")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Optuna timeout per fold (seconds)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    args = parser.parse_args()

    # Resolve properties
    if args.properties == ["all"]:
        properties = list(MODELS.keys())
    else:
        properties = args.properties
        for p in properties:
            if p not in MODELS:
                parser.error(f"Unknown property: {p}. Valid: {list(MODELS.keys())}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Load tokenizer once (shared across all properties/folds)
    logger.info("Loading ChemBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Master fold assignments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    logger.info("=" * 70)
    logger.info("ChemBERTa-2: Training %d properties at 40C", len(properties))
    logger.info("  n_trials=%d, timeout=%ds, epochs=%d, batch_size=%d, patience=%d",
                args.n_trials, args.timeout, args.epochs, args.batch_size, args.patience)
    logger.info("=" * 70)

    results = {}
    for prop_name in properties:
        logger.info("--- %s ---", prop_name.upper())
        summary = train_property(
            prop_name, MODELS[prop_name],
            tokenizer=tokenizer,
            device=device,
            n_trials=args.n_trials,
            timeout=args.timeout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        if summary:
            results[f"{prop_name}_40C"] = summary
            logger.info("  => R2=%.4f +/- %.4f  RMSE=%.4f  N=%d  (%.1fs)",
                        summary["r2"], summary["std_r2"], summary["rmse"],
                        summary["n_molecules"], summary["time_s"])

    # Summary table
    print(f"\n{'=' * 80}")
    print("SINGLE-TEMP ChemBERTa-2 MODELS (5-fold hash-based CV, Optuna-tuned)")
    print(f"{'=' * 80}")
    print(f"{'Model':<18} {'R2':>8} {'+-':>6} {'RMSE':>10} {'MAE':>10} "
          f"{'N':>6} {'Time':>7}")
    print("-" * 80)
    for key, r in results.items():
        t_min = r["time_s"] / 60
        print(f"{key:<18} {r['r2']:>8.4f} {r['std_r2']:>6.4f} "
              f"{r['rmse']:>10.4f} {r['mae']:>10.4f} "
              f"{r['n_molecules']:>6} {t_min:>6.1f}m")
    print(f"{'=' * 80}")

    # Save combined results
    with open(OUTPUT_DIR / "single_temp_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("All results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
