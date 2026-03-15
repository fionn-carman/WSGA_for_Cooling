#!/usr/bin/env python3
"""
Apply the delta_rho pipeline to top 1000 WSGA molecules.

Trains a delta_rho XGBoost model on the full training set, predicts delta_rho
for WSGA molecules, then reconstructs FOM1 using the improved beta and compares
to the existing pipeline and direct FOM1 predictions.
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
FOM1_DATA_DIR = (
    Path(__file__).resolve().parent
    / "FOM1_architecture_comparison" / "results" / "fom1_40"
)
OUT_DIR = Path(__file__).resolve().parent / "surrogate_comparison"

WSGA_CSV = Path(
    "/Volumes/fc4018/projects/fionn2023/live/WSGA_for_Cooling/outputs"
    "/molprice_sweep/nocost_nonbio_seed7/all_evaluated_molecules.csv"
)

dT = 60.0  # 100°C - 40°C

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
    logger.info("Computing descriptors for %d molecules...", len(smiles_list))
    t0 = time.time()
    rows = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        rows.append(calc_descriptors(mol))
    desc_df = pd.DataFrame(rows, columns=DESCRIPTOR_NAMES)
    for c in descriptor_columns:
        if c not in desc_df.columns:
            desc_df[c] = 0.0
    X = desc_df[descriptor_columns].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    logger.info("Descriptors computed in %.0fs", time.time() - t0)
    return X


def compute_fom1(rho_gcm3, Cp_JKmol, k, nu_cSt, beta, MW):
    """FOM1 = k * (beta * Cp * rho / (nu * k))^0.2813 (same as wsga_helper.py)."""
    Cp = Cp_JKmol / MW * 1000      # J/(K·mol) → J/(K·kg)
    nu = nu_cSt * 1e-6             # cSt → m²/s
    rho_kg = rho_gcm3 * 1000       # g/cm³ → kg/m³
    inner = np.clip(beta * Cp * rho_kg / (nu * k), 0, None)
    return k * np.power(inner, 0.2813)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load descriptor columns ─────────────────────────────────────────────
    with open(FOM1_DATA_DIR / "descriptor_columns.json") as f:
        descriptor_columns = json.load(f)

    # ── Train or load delta_rho model ───────────────────────────────────────
    model_path = OUT_DIR / "xgb_delta_rho.joblib"
    scaler_path = OUT_DIR / "scaler_delta_rho.joblib"

    if model_path.exists() and scaler_path.exists():
        logger.info("Loading saved delta_rho model...")
        xgb_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        # Load property data (same as beta_prediction_experiment.py)
        props = {}
        for prop_name in [
            "Density_40C_g_cm^3", "Density_100C_g_cm^3",
        ]:
            csv_path = DATA_DIR / f"{prop_name}_cleaned.csv"
            df = pd.read_csv(csv_path)
            props[prop_name] = df[["SMILES", prop_name]].drop_duplicates(
                subset=["SMILES"], keep="first")

        merged = props["Density_40C_g_cm^3"].merge(
            props["Density_100C_g_cm^3"], on="SMILES", how="inner")
        merged = merged.drop_duplicates(subset=["SMILES"], keep="first")

        delta_rho = (merged["Density_100C_g_cm^3"] - merged["Density_40C_g_cm^3"]).values
        logger.info("Delta_rho training set: %d molecules, mean=%.5f, std=%.5f",
                     len(merged), delta_rho.mean(), delta_rho.std())

        # Compute descriptors for training set
        desc_df = pd.read_csv(DATA_DIR / "Density_40C_g_cm^3_cleaned.csv")
        desc_df = desc_df.drop_duplicates(subset=["SMILES"], keep="first")
        desc_df = desc_df[desc_df["SMILES"].isin(merged["SMILES"])]
        desc_df = desc_df.set_index("SMILES").loc[merged["SMILES"]].reset_index()

        X_train = desc_df[descriptor_columns].values.astype(np.float32)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        logger.info("Training delta_rho XGBoost...")
        t0 = time.time()
        param_dist = {
            "n_estimators": [100, 200, 500, 1000, 2000],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 6, 8, 10],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }
        search = RandomizedSearchCV(
            XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            param_dist, n_iter=200, cv=5, scoring="r2",
            random_state=42, n_jobs=-1, verbose=0,
        )
        search.fit(X_train_s, delta_rho)
        xgb_model = search.best_estimator_
        logger.info("Trained in %.0fs, best CV R²=%.4f", time.time() - t0, search.best_score_)

        joblib.dump(xgb_model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info("Saved model to %s", model_path)

    # ── Load top 1000 WSGA molecules ────────────────────────────────────────
    logger.info("Loading WSGA molecules...")
    wsga = pd.read_csv(WSGA_CSV, low_memory=False)
    wsga = wsga[
        (wsga["is_valid"] == 1) & (wsga["MP-Measured"] < -30)
    ].drop_duplicates(subset=["SMILES"], keep="last")
    wsga = wsga.nlargest(1000, "FOM1_direct_avg").copy()
    logger.info("Top 1000 molecules loaded")

    # ── Compute descriptors and predict delta_rho ───────────────────────────
    X_wsga = compute_descriptors_batch(wsga["SMILES"].tolist(), descriptor_columns)
    X_wsga_s = scaler.transform(X_wsga)
    wsga["delta_rho_pred"] = xgb_model.predict(X_wsga_s)

    # ── Reconstruct FOM1 with delta_rho beta ────────────────────────────────
    # New beta from predicted delta_rho + predicted rho
    rho_40 = wsga["Density_40C_g_cm^3"].values
    rho_100 = wsga["Density_100C_g_cm^3"].values
    delta_rho_pred = wsga["delta_rho_pred"].values

    beta_40_deltarho = -(1 / rho_40) * (delta_rho_pred / dT)
    beta_100_deltarho = -(1 / rho_100) * (delta_rho_pred / dT)

    # Existing pipeline beta (from two density predictions)
    beta_40_pipeline = wsga["beta_40"].values
    beta_100_pipeline = wsga["beta_100"].values

    MW = wsga["MW"].values

    # FOM1 at 40°C — delta_rho pipeline
    fom1_40_dr = compute_fom1(
        rho_40,
        wsga["Heat_Capacity_Constant_Pressure_40C_J_K_Mol"].values,
        wsga["Thermal_Conductivity_40C"].values,
        wsga["Kinematic_Viscosity_40C"].values,
        beta_40_deltarho, MW)

    # FOM1 at 100°C — delta_rho pipeline
    fom1_100_dr = compute_fom1(
        rho_100,
        wsga["Heat_Capacity_Constant_Pressure_100C_J_K_Mol"].values,
        wsga["Thermal_Conductivity_100C"].values,
        wsga["Kinematic_Viscosity_100C"].values,
        beta_100_deltarho, MW)

    wsga["FOM1_40_deltarho"] = fom1_40_dr
    wsga["FOM1_100_deltarho"] = fom1_100_dr
    wsga["FOM1_avg_deltarho"] = (fom1_40_dr + fom1_100_dr) / 2

    # ── Compare all approaches ──────────────────────────────────────────────
    approaches = {
        "Pipeline (current)": ("FOM1_40", "FOM1_100", "FOM1_avg"),
        "Direct XGBoost": ("FOM1_40C_direct", "FOM1_100C_direct", "FOM1_direct_avg"),
        "Delta-rho pipeline": ("FOM1_40_deltarho", "FOM1_100_deltarho", "FOM1_avg_deltarho"),
    }

    logger.info("\n=== FOM1 prediction statistics (top 1000 molecules) ===")
    for name, (c40, c100, cavg) in approaches.items():
        v40 = wsga[c40].dropna()
        v100 = wsga[c100].dropna()
        vavg = wsga[cavg].dropna()
        logger.info("%-22s FOM1_40: %.1f±%.1f [%.1f–%.1f]  FOM1_100: %.1f±%.1f [%.1f–%.1f]  avg: %.1f±%.1f [%.1f–%.1f]",
                     name,
                     v40.mean(), v40.std(), v40.min(), v40.max(),
                     v100.mean(), v100.std(), v100.min(), v100.max(),
                     vavg.mean(), vavg.std(), vavg.min(), vavg.max())

    # ── Top 20 by each approach ─────────────────────────────────────────────
    for name, (c40, c100, cavg) in approaches.items():
        top = wsga.nlargest(20, cavg)
        logger.info("\n=== Top 20 by %s ===", name)
        logger.info("%-45s %7s %7s %7s %7s %7s", "SMILES", "Pipe", "Direct", "DeltaR", "40C_DR", "100_DR")
        for _, r in top.iterrows():
            smi = r.get("CanonicalSMILES", r["SMILES"])
            logger.info("%-45s %7.1f %7.1f %7.1f %7.1f %7.1f",
                         smi, r["FOM1_avg"], r["FOM1_direct_avg"],
                         r["FOM1_avg_deltarho"], r["FOM1_40_deltarho"], r["FOM1_100_deltarho"])

    # ── Pairwise scatter: 3 approaches ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    pairs = [
        ("FOM1_direct_avg", "FOM1_avg", "Direct XGBoost", "Pipeline"),
        ("FOM1_direct_avg", "FOM1_avg_deltarho", "Direct XGBoost", "Delta-rho pipeline"),
        ("FOM1_avg", "FOM1_avg_deltarho", "Pipeline", "Delta-rho pipeline"),
    ]
    for ax, (cx, cy, lx, ly) in zip(axes, pairs):
        ax.scatter(wsga[cx], wsga[cy], alpha=0.3, s=10, c="grey")
        lims = [30, 120]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        r = np.corrcoef(wsga[cx].dropna(), wsga[cy].dropna())[0, 1]
        ax.set_title(f"r = {r:.3f}")
        ax.set_aspect("equal")

    fig.suptitle("FOM1 Prediction Comparison (top 1000 WSGA molecules)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wsga_deltarho_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("\nSaved: %s", OUT_DIR / "wsga_deltarho_comparison.png")

    # ── Save results ────────────────────────────────────────────────────────
    out_cols = [
        "SMILES", "CanonicalSMILES",
        "FOM1_40", "FOM1_100", "FOM1_avg",
        "FOM1_40C_direct", "FOM1_100C_direct", "FOM1_direct_avg",
        "FOM1_40_deltarho", "FOM1_100_deltarho", "FOM1_avg_deltarho",
        "delta_rho_pred", "beta_40", "beta_100",
    ]
    wsga[out_cols].to_csv(OUT_DIR / "wsga_deltarho_predictions.csv", index=False)
    logger.info("Saved: %s", OUT_DIR / "wsga_deltarho_predictions.csv")


if __name__ == "__main__":
    main()
