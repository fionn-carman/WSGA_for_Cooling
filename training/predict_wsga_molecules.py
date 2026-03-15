#!/usr/bin/env python3
"""
Predict FOM1 for WSGA molecules using Ridge and MLP surrogates, then compare
to the existing XGBoost predictions already in the CSV.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FOM1_DATA_DIR = (
    Path(__file__).resolve().parent
    / "FOM1_architecture_comparison" / "results" / "fom1_40"
)
OUT_DIR = Path(__file__).resolve().parent / "surrogate_comparison"

WSGA_CSV = Path(
    "/Volumes/fc4018/projects/fionn2023/live/WSGA_for_Cooling/outputs"
    "/molprice_sweep/nocost_nonbio_seed7/all_evaluated_molecules.csv"
)

DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]
DESCRIPTOR_FUNCS = [desc[1] for desc in Descriptors._descList]


def calc_descriptors(mol):
    if mol is None:
        return [np.nan] * len(DESCRIPTOR_FUNCS)
    vals = []
    for func in DESCRIPTOR_FUNCS:
        try:
            vals.append(func(mol))
        except Exception:
            vals.append(np.nan)
    return vals


def compute_descriptors_batch(smiles_list, descriptor_columns):
    """Compute descriptors for a list of SMILES."""
    logger.info("Computing descriptors for %d molecules...", len(smiles_list))
    t0 = time.time()
    rows = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        rows.append(calc_descriptors(mol))
        if (i + 1) % 25000 == 0:
            logger.info("  %d / %d (%.0fs)", i + 1, len(smiles_list), time.time() - t0)

    desc_df = pd.DataFrame(rows, columns=DESCRIPTOR_NAMES)
    for c in descriptor_columns:
        if c not in desc_df.columns:
            desc_df[c] = 0.0

    X = desc_df[descriptor_columns].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    logger.info("Descriptors computed in %.0fs", time.time() - t0)
    return X


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load FOM1 training data ─────────────────────────────────────────────
    train_df = pd.read_csv(FOM1_DATA_DIR / "fom1_dataset.csv")
    with open(FOM1_DATA_DIR / "descriptor_columns.json") as f:
        descriptor_columns = json.load(f)

    ridge_path = OUT_DIR / "ridge_fom1avg.joblib"
    mlp_path = OUT_DIR / "mlp_fom1avg.joblib"
    scaler_path = OUT_DIR / "scaler_fom1avg.joblib"

    if ridge_path.exists() and mlp_path.exists() and scaler_path.exists():
        logger.info("Loading saved models...")
        scaler = joblib.load(scaler_path)
        ridge = joblib.load(ridge_path)
        mlp = joblib.load(mlp_path)
    else:
        X_train = train_df[descriptor_columns].values.astype(np.float32)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        y_train = train_df["FOM1_avg"].values.astype(np.float32)
        logger.info("Training set: %d molecules, %d descriptors", len(train_df), X_train.shape[1])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        logger.info("Training Ridge...")
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 50))
        ridge.fit(X_train_s, y_train)
        logger.info("Ridge done (alpha=%.4f)", ridge.alpha_)

        logger.info("Training MLP...")
        mlp = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", solver="adam", alpha=1e-5,
            batch_size=64, learning_rate_init=1e-3, max_iter=200,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=42,
        )
        mlp.fit(X_train_s, y_train)
        logger.info("MLP done (%d epochs)", mlp.n_iter_)

        joblib.dump(scaler, scaler_path)
        joblib.dump(ridge, ridge_path)
        joblib.dump(mlp, mlp_path)
        logger.info("Saved models to %s", OUT_DIR)

    # ── Load WSGA molecules: valid, MP<-30, top 1000 by XGBoost FOM1 ───────
    logger.info("Loading WSGA molecules...")
    wsga_df = pd.read_csv(WSGA_CSV)
    wsga_df = wsga_df[
        (wsga_df["is_valid"] == 1) & (wsga_df["MP-Measured"] < -30)
    ].drop_duplicates(subset=["SMILES"], keep="last")
    wsga_df = wsga_df.nlargest(1000, "FOM1_direct_avg").copy()
    logger.info("Top 1000 valid molecules (MP<-30): FOM1 range %.1f – %.1f",
                wsga_df["FOM1_direct_avg"].min(), wsga_df["FOM1_direct_avg"].max())

    # ── Compute descriptors and predict ─────────────────────────────────────
    X_wsga = compute_descriptors_batch(wsga_df["SMILES"].tolist(), descriptor_columns)
    X_wsga_s = scaler.transform(X_wsga)

    wsga_df["FOM1_Ridge"] = ridge.predict(X_wsga_s)
    wsga_df["FOM1_MLP"] = mlp.predict(X_wsga_s)

    # Existing XGBoost prediction from the WSGA run
    pred_cols = ["FOM1_direct_avg", "FOM1_Ridge", "FOM1_MLP"]
    col_labels = {"FOM1_direct_avg": "XGBoost", "FOM1_Ridge": "Ridge", "FOM1_MLP": "MLP"}

    # ── Stats ───────────────────────────────────────────────────────────────
    logger.info("\n=== Prediction statistics ===")
    for col in pred_cols:
        v = wsga_df[col].dropna()
        logger.info("%-18s mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
                     col_labels[col], v.mean(), v.std(), v.min(), v.max())

    corr = wsga_df[pred_cols].corr()
    corr.index = [col_labels[c] for c in corr.index]
    corr.columns = [col_labels[c] for c in corr.columns]
    logger.info("\n=== Correlation matrix ===\n%s", corr.to_string())

    # ── Top 30 by each model ────────────────────────────────────────────────
    for col in pred_cols:
        label = col_labels[col]
        top = wsga_df.nlargest(30, col)
        logger.info("\n=== Top 30 by %s ===", label)
        for i, (_, row) in enumerate(top.iterrows()):
            smi = row.get("CanonicalSMILES", row["SMILES"])
            logger.info("  %2d. %-45s  XGB=%.1f  Ridge=%.1f  MLP=%.1f",
                         i + 1, smi,
                         row["FOM1_direct_avg"], row["FOM1_Ridge"], row["FOM1_MLP"])

    # ── Save full predictions ───────────────────────────────────────────────
    out_cols = ["SMILES", "CanonicalSMILES", "FOM1_direct_avg", "FOM1_Ridge", "FOM1_MLP"]
    wsga_df[out_cols].to_csv(OUT_DIR / "wsga_all_predictions.csv", index=False)
    logger.info("\nSaved all predictions to %s", OUT_DIR / "wsga_all_predictions.csv")

    # ── Figures ─────────────────────────────────────────────────────────────
    # 1. Pairwise scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    pairs = [
        ("FOM1_direct_avg", "FOM1_Ridge"),
        ("FOM1_direct_avg", "FOM1_MLP"),
        ("FOM1_Ridge", "FOM1_MLP"),
    ]
    for ax, (cx, cy) in zip(axes, pairs):
        ax.scatter(wsga_df[cx], wsga_df[cy], alpha=0.03, s=3, c="grey")
        lims = [20, 120]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(col_labels[cx])
        ax.set_ylabel(col_labels[cy])
        r = np.corrcoef(wsga_df[cx].values, wsga_df[cy].values)[0, 1]
        ax.set_title(f"r = {r:.3f}")
        ax.set_aspect("equal")

    fig.suptitle("Pairwise Surrogate Predictions (189k WSGA molecules)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wsga_pairwise_predictions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "wsga_pairwise_predictions.png")

    # 2. Histogram overlay
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, color in [
        ("FOM1_direct_avg", "tab:red"),
        ("FOM1_Ridge", "tab:blue"),
        ("FOM1_MLP", "tab:green"),
    ]:
        ax.hist(wsga_df[col].dropna(), bins=100, alpha=0.5,
                label=col_labels[col], color=color)
    ax.set_xlabel("Predicted FOM1 (avg)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("FOM1 Prediction Distributions (189k valid WSGA molecules)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wsga_prediction_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "wsga_prediction_distributions.png")


if __name__ == "__main__":
    main()
