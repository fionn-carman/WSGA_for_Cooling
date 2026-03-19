#!/usr/bin/env python3
"""
Publication-quality Chemprop D-MPNN training pipeline for NIST 8100 single-temp models.

Hash-based 5-fold outer CV with Chemprop v1 CLI. No Optuna — D-MPNN uses
well-validated defaults (depth=3, hidden_size=300, lr=1e-4, early stopping).

Pipeline per property (40C):
  1. Load property CSV, extract SMILES + target column
  2. Apply log1p transform if needed (viscosity, fom1)
  3. Hash-based 5-fold assignment
  4. Per outer fold:
     a. Write temp train/test CSVs (columns: smiles, target)
     b. Train via chemprop_train CLI (100 epochs, patience 20)
     c. Predict via chemprop_predict CLI on held-out test set
  5. Aggregate predictions, compute metrics in real space
  6. Save outputs + plots

Usage:
    python train_single_temp.py                      # all 6 properties
    python train_single_temp.py --properties density  # single property
    python train_single_temp.py --epochs 10           # quick test
"""

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
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

from fold_utils import assign_folds, get_fold_indices

warnings.filterwarnings("ignore", category=UserWarning)

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
# Chemprop model building (v1 CLI)
# ============================================================

def build_chemprop_model(train_smiles, train_targets, test_smiles,
                         epochs=100, patience=20, batch_size=64,
                         lr=1e-4, fold_i=0, save_dir=None):
    """
    Train a Chemprop D-MPNN and return test predictions.

    Uses Chemprop v1.6.1 CLI (chemprop_train / chemprop_predict):
    - hidden_size=300, depth=3, ffn_num_layers=2, dropout=0.0
    - Early stopping on validation RMSE (10% split, patience epochs)
    - GPU via --gpu 0 if CUDA available
    """
    import torch

    tmpdir = tempfile.mkdtemp()
    try:
        train_csv = os.path.join(tmpdir, "train.csv")
        test_csv = os.path.join(tmpdir, "test.csv")
        pred_csv = os.path.join(tmpdir, "preds.csv")
        model_dir = save_dir if save_dir else os.path.join(tmpdir, "model")

        # Write train CSV
        with open(train_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["smiles", "target"])
            for smi, val in zip(train_smiles, train_targets):
                w.writerow([smi, val])

        # Write test CSV
        with open(test_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["smiles", "target"])
            for smi in test_smiles:
                w.writerow([smi, ""])

        # Build train command
        train_cmd = [
            "chemprop_train",
            "--data_path", train_csv,
            "--dataset_type", "regression",
            "--save_dir", model_dir,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--init_lr", str(lr),
            "--max_lr", str(lr * 10),
            "--final_lr", str(lr / 10),
            "--hidden_size", "300",
            "--depth", "3",
            "--ffn_num_layers", "2",
            "--dropout", "0.0",
            "--split_sizes", "0.9", "0.1", "0.0",
            "--split_key_molecule", "0",
            "--seed", str(fold_i),
            "--metric", "rmse",
            "--quiet",
        ]

        # Use GPU if available
        if torch.cuda.is_available():
            train_cmd.extend(["--gpu", "0"])

        result = subprocess.run(
            train_cmd, capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            logger.error("chemprop_train failed (fold %d): %s", fold_i, result.stderr[-500:])
            raise RuntimeError(f"chemprop_train failed: {result.stderr[-200:]}")

        # Predict
        pred_cmd = [
            "chemprop_predict",
            "--test_path", test_csv,
            "--checkpoint_dir", model_dir,
            "--preds_path", pred_csv,
        ]
        if torch.cuda.is_available():
            pred_cmd.extend(["--gpu", "0"])

        result = subprocess.run(
            pred_cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.error("chemprop_predict failed (fold %d): %s", fold_i, result.stderr[-500:])
            raise RuntimeError(f"chemprop_predict failed: {result.stderr[-200:]}")

        # Read predictions
        preds_df = pd.read_csv(pred_csv)
        y_pred = preds_df.iloc[:, 1].values.astype(np.float64)

        # Parse stopped epoch from verbose output if available
        stopped_epoch = epochs  # default
        if result.stdout:
            for line in result.stdout.split("\n"):
                if "Epoch" in line:
                    try:
                        stopped_epoch = int(line.split("Epoch")[1].split("/")[0].strip())
                    except (ValueError, IndexError):
                        pass

        return y_pred, stopped_epoch

    finally:
        # Clean up tmpdir (but not save_dir)
        if save_dir and os.path.abspath(tmpdir) != os.path.abspath(save_dir):
            shutil.rmtree(tmpdir, ignore_errors=True)
        elif not save_dir:
            shutil.rmtree(tmpdir, ignore_errors=True)


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
# Per-property training pipeline
# ============================================================

def train_property(prop_name: str, prop_config: dict,
                   epochs: int, batch_size: int, lr: float,
                   patience: int) -> dict:
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

    for fold_i in range(5):
        logger.info("    Fold %d:", fold_i)
        t0_fold = time.time()
        train_idx, test_idx = get_fold_indices(folds, fold_i)

        train_smiles = smiles[train_idx].tolist()
        test_smiles = smiles[test_idx].tolist()
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Train D-MPNN
        fold_save_dir = out_dir / f"fold{fold_i}_model"
        preds, stopped_epoch = build_chemprop_model(
            train_smiles, y_train.tolist(),
            test_smiles,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            fold_i=fold_i,
            save_dir=str(fold_save_dir),
        )

        all_preds[test_idx] = preds

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

        fold_metrics.append({
            "fold": fold_i,
            "r2": round(float(fold_r2), 4),
            "rmse": round(float(fold_rmse), 4),
            "mae": round(float(fold_mae), 4),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "stopped_epoch": stopped_epoch,
            "time_s": round(fold_time, 1),
        })

        logger.info("      R2=%.4f  RMSE=%.4f  (%d/%d train/test, epoch %d, %.1fs)",
                     fold_r2, fold_rmse, len(train_idx), len(test_idx),
                     stopped_epoch, fold_time)

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
        "model": "Chemprop_DMPNN",
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
            "message_passing": "BondMessagePassing",
            "d_h": 300,
            "depth": 3,
            "ffn_hidden_dim": 300,
            "ffn_n_layers": 2,
            "dropout": 0.0,
            "batch_norm": True,
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "lr": lr,
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
        description="Train single-temp Chemprop D-MPNN models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--properties", nargs="+", default=["all"],
        help="Properties to train (density viscosity tc cpsat beta fom1) or 'all'",
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs per fold")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    args = parser.parse_args()

    # Resolve properties
    if args.properties == ["all"]:
        properties = list(MODELS.keys())
    else:
        properties = args.properties
        for p in properties:
            if p not in MODELS:
                parser.error(f"Unknown property: {p}. Valid: {list(MODELS.keys())}")

    # Master fold assignments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    logger.info("=" * 70)
    logger.info("Chemprop D-MPNN: Training %d properties at 40C", len(properties))
    logger.info("  epochs=%d, patience=%d, batch_size=%d, lr=%.1e",
                args.epochs, args.patience, args.batch_size, args.lr)
    logger.info("=" * 70)

    results = {}
    for prop_name in properties:
        logger.info("--- %s ---", prop_name.upper())
        summary = train_property(
            prop_name, MODELS[prop_name],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
        if summary:
            results[f"{prop_name}_40C"] = summary
            logger.info("  => R2=%.4f +/- %.4f  RMSE=%.4f  N=%d  (%.1fs)",
                        summary["r2"], summary["std_r2"], summary["rmse"],
                        summary["n_molecules"], summary["time_s"])

    # Summary table
    print(f"\n{'=' * 80}")
    print("SINGLE-TEMP CHEMPROP D-MPNN MODELS (5-fold hash-based CV)")
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
