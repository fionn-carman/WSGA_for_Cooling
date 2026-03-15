#!/usr/bin/env python3
"""
Experiment: Does predicting beta directly (instead of computing it from
two density predictions) improve pipeline FOM1 accuracy?

Approach:
  1. Load experimental density_40, density_100 for 806 molecules
  2. Compute true beta from experimental densities
  3. Train a direct XGBoost beta predictor (5-fold CV)
  4. For each fold, reconstruct pipeline FOM1 using:
     a) Experimental properties (ground truth)
     b) All predicted properties (current pipeline)
     c) Predicted properties BUT with direct-predicted beta replacing
        the beta computed from predicted densities
  5. Compare R² and MAE of pipeline FOM1 vs experimental FOM1

Also evaluates: direct prediction of (rho_100 - rho_40), i.e. the density
*difference*, which is the numerically sensitive quantity.
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
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
FOM1_DIR = Path(__file__).resolve().parent / "FOM1_architecture_comparison" / "results"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR = Path(__file__).resolve().parent / "beta_experiment"

dT = 60.0  # Temperature difference (100°C - 40°C)
g = 9.81   # Gravitational acceleration
L = 0.01   # Characteristic length (m) — same as in wsga_helper.py


def compute_beta(rho_40, rho_100):
    """Thermal expansion coefficient at 40°C (same as wsga_helper.py line 384)."""
    return -(1 / rho_40) * ((rho_100 - rho_40) / dT)


def load_property_data():
    """Load all 8 thermophysical properties + descriptors for 806 molecules."""
    props = {}
    for prop_name in [
        "Density_40C_g_cm^3",
        "Density_100C_g_cm^3",
        "Kinematic_Viscosity_40C",
        "Kinematic_Viscosity_100C",
        "Thermal_Conductivity_40C",
        "Thermal_Conductivity_100C",
        "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",
        "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",
    ]:
        csv_path = DATA_DIR / f"{prop_name}_cleaned.csv"
        df = pd.read_csv(csv_path)
        props[prop_name] = df[[
            "SMILES", prop_name
        ]].rename(columns={prop_name: prop_name})

    # Deduplicate each property (some _cleaned.csv have duplicate SMILES)
    for name in props:
        props[name] = props[name].drop_duplicates(subset=["SMILES"], keep="first")

    # Merge all on SMILES
    merged = props["Density_40C_g_cm^3"]
    for name, df in props.items():
        if name == "Density_40C_g_cm^3":
            continue
        merged = merged.merge(df, on="SMILES", how="inner")

    # Add molecular weight from descriptors
    desc_df = pd.read_csv(DATA_DIR / "Density_40C_g_cm^3_cleaned.csv")
    merged = merged.merge(desc_df[["SMILES", "MolWt"]], on="SMILES", how="left")

    # Add experimental FOM1
    fom1_df = pd.read_csv(
        Path(__file__).resolve().parent.parent
        / "BaselineFOM1Eval" / "output" / "baseline_fom1_results.csv"
    )
    merged = merged.merge(
        fom1_df[["SMILES", "FOM1_exp_40", "FOM1_exp_100", "FOM1_exp_avg"]],
        on="SMILES", how="inner"
    )

    merged = merged.drop_duplicates(subset=["SMILES"], keep="first")
    logger.info("Loaded %d molecules with all properties + FOM1", len(merged))
    return merged


def load_descriptors(smiles_list):
    """Load RDKit descriptor matrix for given SMILES."""
    desc_df = pd.read_csv(DATA_DIR / "Density_40C_g_cm^3_cleaned.csv")
    desc_df = desc_df.drop_duplicates(subset=["SMILES"], keep="first")
    desc_df = desc_df[desc_df["SMILES"].isin(smiles_list)]

    # Descriptor columns = everything except SMILES and target
    desc_cols = [c for c in desc_df.columns
                 if c not in ["SMILES", "Density_40C_g_cm^3"]]
    return desc_df, desc_cols


def train_xgb_cv(X, y, fold_indices, target_name, seed=42, n_iter=100):
    """Train XGBoost with 5-fold CV, return OOF predictions."""
    param_dist = {
        "n_estimators": [100, 200, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 6, 8],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    oof_preds = np.full(len(y), np.nan)
    models = []

    for fold_id in range(5):
        t0 = time.time()
        train_idx = fold_indices[str(fold_id)]["train"]
        test_idx = fold_indices[str(fold_id)]["test"]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        search = RandomizedSearchCV(
            XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0),
            param_dist, n_iter=n_iter, cv=5, scoring="r2",
            random_state=seed, n_jobs=-1, verbose=0,
        )
        search.fit(X_train_s, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test_s)
        oof_preds[test_idx] = y_pred

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - t0
        logger.info("  %s fold %d: R²=%.4f MAE=%.4f (%.0fs)",
                     target_name, fold_id, r2, mae, elapsed)
        models.append({"model": best_model, "scaler": scaler})

    overall_r2 = r2_score(y[~np.isnan(oof_preds)], oof_preds[~np.isnan(oof_preds)])
    overall_mae = mean_absolute_error(y[~np.isnan(oof_preds)], oof_preds[~np.isnan(oof_preds)])
    logger.info("  %s overall: R²=%.4f MAE=%.4f", target_name, overall_r2, overall_mae)

    return oof_preds, models


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = load_property_data()
    desc_df, desc_cols = load_descriptors(df["SMILES"].tolist())

    # Align descriptor rows with property data
    desc_df = desc_df.set_index("SMILES").loc[df["SMILES"]].reset_index()
    X = desc_df[desc_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    logger.info("Feature matrix: %s", X.shape)

    # Load fold indices (same folds as FOM1 training for fair comparison)
    fom1_data_dir = FOM1_DIR / "fom1_40"
    if (fom1_data_dir / "fold_indices.json").exists():
        with open(fom1_data_dir / "fold_indices.json") as f:
            fold_indices = json.load(f)
        logger.info("Using existing FOM1 fold indices")
    else:
        logger.info("Creating new 5-fold indices")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_indices = {}
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            fold_indices[str(i)] = {
                "train": train_idx.tolist(),
                "test": test_idx.tolist()
            }

    # ------------------------------------------------------------------
    # 2. Compute true beta from experimental densities
    # ------------------------------------------------------------------
    rho_40 = df["Density_40C_g_cm^3"].values
    rho_100 = df["Density_100C_g_cm^3"].values
    true_beta_40 = -(1 / rho_40) * ((rho_100 - rho_40) / dT)
    true_beta_100 = -(1 / rho_100) * ((rho_100 - rho_40) / dT)
    delta_rho = rho_100 - rho_40  # The numerically sensitive quantity

    logger.info("True beta_40: mean=%.6f, std=%.6f", true_beta_40.mean(), true_beta_40.std())
    logger.info("True delta_rho: mean=%.6f, std=%.6f", delta_rho.mean(), delta_rho.std())

    # ------------------------------------------------------------------
    # 3. Train direct predictors
    # ------------------------------------------------------------------
    logger.info("\n=== Training direct beta_40 predictor ===")
    pred_beta_40, _ = train_xgb_cv(X, true_beta_40, fold_indices, "beta_40")

    logger.info("\n=== Training direct beta_100 predictor ===")
    pred_beta_100, _ = train_xgb_cv(X, true_beta_100, fold_indices, "beta_100")

    logger.info("\n=== Training direct delta_rho predictor ===")
    pred_delta_rho, _ = train_xgb_cv(X, delta_rho, fold_indices, "delta_rho")

    # ------------------------------------------------------------------
    # 4. Get OOF property predictions from existing models
    #    (or train fresh ones for a fair comparison)
    # ------------------------------------------------------------------
    property_targets = {
        "Density_40C_g_cm^3": df["Density_40C_g_cm^3"].values,
        "Density_100C_g_cm^3": df["Density_100C_g_cm^3"].values,
        "Kinematic_Viscosity_40C": df["Kinematic_Viscosity_40C"].values,
        "Kinematic_Viscosity_100C": df["Kinematic_Viscosity_100C"].values,
        "Thermal_Conductivity_40C": df["Thermal_Conductivity_40C"].values,
        "Thermal_Conductivity_100C": df["Thermal_Conductivity_100C"].values,
        "Heat_Capacity_Constant_Pressure_40C_J_K_Mol": df["Heat_Capacity_Constant_Pressure_40C_J_K_Mol"].values,
        "Heat_Capacity_Constant_Pressure_100C_J_K_Mol": df["Heat_Capacity_Constant_Pressure_100C_J_K_Mol"].values,
    }

    oof_props = {}
    for prop_name, y in property_targets.items():
        logger.info("\n=== Training %s predictor ===", prop_name)
        oof_props[prop_name], _ = train_xgb_cv(X, y, fold_indices, prop_name)

    # ------------------------------------------------------------------
    # 5. Reconstruct FOM1 under different beta strategies
    # ------------------------------------------------------------------
    MW = df["MolWt"].values
    exp_fom1_40 = df["FOM1_exp_40"].values
    exp_fom1_100 = df["FOM1_exp_100"].values

    # Strategy A: Current pipeline (beta from predicted densities)
    pred_rho_40 = oof_props["Density_40C_g_cm^3"]
    pred_rho_100 = oof_props["Density_100C_g_cm^3"]
    pipeline_beta_40 = -(1 / pred_rho_40) * ((pred_rho_100 - pred_rho_40) / dT)
    pipeline_beta_100 = -(1 / pred_rho_100) * ((pred_rho_100 - pred_rho_40) / dT)

    # Strategy B: Direct beta prediction
    direct_beta_40 = pred_beta_40
    direct_beta_100 = pred_beta_100

    # Strategy C: Direct delta_rho → beta
    deltarho_beta_40 = -(1 / pred_rho_40) * (pred_delta_rho / dT)
    deltarho_beta_100 = -(1 / pred_rho_100) * (pred_delta_rho / dT)

    # Compute pipeline FOM1 at 40°C for each strategy
    strategies = {
        "Pipeline (current)": (pipeline_beta_40, pipeline_beta_100),
        "Direct beta": (direct_beta_40, direct_beta_100),
        "Direct delta_rho": (deltarho_beta_40, deltarho_beta_100),
    }

    results = {}
    for strat_name, (beta_40_vals, beta_100_vals) in strategies.items():
        # Build FOM1 at 40°C using the actual formula from wsga_helper.py:
        # FOM1 = k * (beta * Cp * rho / (nu * k))^0.2813
        rho = oof_props["Density_40C_g_cm^3"]
        Cp_mol = oof_props["Heat_Capacity_Constant_Pressure_40C_J_K_Mol"]
        k = oof_props["Thermal_Conductivity_40C"]
        nu_raw = oof_props["Kinematic_Viscosity_40C"]
        beta = beta_40_vals

        Cp = Cp_mol / MW * 1000   # J/(K·mol) → J/(K·kg)
        nu = nu_raw * 1e-6        # cSt → m²/s
        rho_kg = rho * 1000       # g/cm³ → kg/m³

        fom1_40 = k * np.power(
            np.clip(beta * Cp * rho_kg / (nu * k), 0, None), 0.2813)

        # FOM1 at 100°C
        rho = oof_props["Density_100C_g_cm^3"]
        Cp_mol = oof_props["Heat_Capacity_Constant_Pressure_100C_J_K_Mol"]
        k = oof_props["Thermal_Conductivity_100C"]
        nu_raw = oof_props["Kinematic_Viscosity_100C"]
        beta = beta_100_vals

        Cp = Cp_mol / MW * 1000
        nu = nu_raw * 1e-6
        rho_kg = rho * 1000

        fom1_100 = k * np.power(
            np.clip(beta * Cp * rho_kg / (nu * k), 0, None), 0.2813)

        fom1_avg = (fom1_40 + fom1_100) / 2

        # Metrics vs experimental
        valid = np.isfinite(fom1_avg) & np.isfinite(exp_fom1_40)
        r2_40 = r2_score(exp_fom1_40[valid], fom1_40[valid])
        r2_100 = r2_score(exp_fom1_100[valid], fom1_100[valid])
        r2_avg = r2_score(
            ((exp_fom1_40 + exp_fom1_100) / 2)[valid], fom1_avg[valid])
        mae_40 = mean_absolute_error(exp_fom1_40[valid], fom1_40[valid])
        mae_100 = mean_absolute_error(exp_fom1_100[valid], fom1_100[valid])
        mae_avg = mean_absolute_error(
            ((exp_fom1_40 + exp_fom1_100) / 2)[valid], fom1_avg[valid])

        results[strat_name] = {
            "fom1_40": fom1_40, "fom1_100": fom1_100, "fom1_avg": fom1_avg,
            "r2_40": r2_40, "r2_100": r2_100, "r2_avg": r2_avg,
            "mae_40": mae_40, "mae_100": mae_100, "mae_avg": mae_avg,
        }
        logger.info("\n%s:", strat_name)
        logger.info("  FOM1_40:  R²=%.4f  MAE=%.2f", r2_40, mae_40)
        logger.info("  FOM1_100: R²=%.4f  MAE=%.2f", r2_100, mae_100)
        logger.info("  FOM1_avg: R²=%.4f  MAE=%.2f", r2_avg, mae_avg)

    # ------------------------------------------------------------------
    # 6. Compare beta prediction accuracy
    # ------------------------------------------------------------------
    logger.info("\n=== Beta prediction accuracy ===")
    valid = np.isfinite(pred_beta_40) & np.isfinite(pipeline_beta_40)

    logger.info("Pipeline beta (from 2 densities): R²=%.4f MAE=%.6f",
                r2_score(true_beta_40[valid], pipeline_beta_40[valid]),
                mean_absolute_error(true_beta_40[valid], pipeline_beta_40[valid]))
    logger.info("Direct beta prediction:           R²=%.4f MAE=%.6f",
                r2_score(true_beta_40[valid], pred_beta_40[valid]),
                mean_absolute_error(true_beta_40[valid], pred_beta_40[valid]))
    logger.info("Delta-rho → beta:                 R²=%.4f MAE=%.6f",
                r2_score(true_beta_40[valid], deltarho_beta_40[valid]),
                mean_absolute_error(true_beta_40[valid], deltarho_beta_40[valid]))

    # ------------------------------------------------------------------
    # 7. Glyme-specific analysis
    # ------------------------------------------------------------------
    from rdkit import Chem
    n_ethers = []
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_ethers.append(0)
            continue
        # Count C-O-C ether linkages
        patt = Chem.MolFromSmarts("[#6]~[#8]~[#6]")
        n_ethers.append(len(mol.GetSubstructMatches(patt)))
    n_ethers = np.array(n_ethers)

    is_glyme = n_ethers >= 4
    logger.info("\n=== Glyme analysis (n_ether >= 4, n=%d) ===", is_glyme.sum())

    exp_avg = (exp_fom1_40 + exp_fom1_100) / 2
    for strat_name, res in results.items():
        fom1 = res["fom1_avg"]
        if is_glyme.sum() > 0 and np.isfinite(fom1[is_glyme]).all():
            bias = (fom1[is_glyme] - exp_avg[is_glyme]).mean()
            mae = mean_absolute_error(exp_avg[is_glyme], fom1[is_glyme])
            logger.info("  %s — glyme bias=%.2f, MAE=%.2f",
                        strat_name, bias, mae)

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    exp_avg = (exp_fom1_40 + exp_fom1_100) / 2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (strat_name, res) in zip(axes, results.items()):
        fom1 = res["fom1_avg"]
        valid = np.isfinite(fom1)

        ax.scatter(exp_avg[valid & ~is_glyme], fom1[valid & ~is_glyme],
                   alpha=0.3, s=15, c="grey", label="Other")
        ax.scatter(exp_avg[valid & is_glyme], fom1[valid & is_glyme],
                   alpha=0.8, s=40, c="red", marker="D", label="Glymes (n_ether≥4)",
                   edgecolors="black", linewidths=0.3)
        lims = [20, 120]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Experimental FOM1 (avg)")
        ax.set_ylabel("Pipeline FOM1 (avg)")
        ax.set_title(f"{strat_name}\nR²={res['r2_avg']:.4f}  MAE={res['mae_avg']:.2f}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    fig.suptitle("Pipeline FOM1: Effect of Beta Prediction Strategy", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beta_experiment_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("\nSaved: %s", OUT_DIR / "beta_experiment_parity.png")

    # Beta parity plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    beta_strategies = {
        "Pipeline (2 densities)": pipeline_beta_40,
        "Direct beta": pred_beta_40,
        "Direct delta_rho": deltarho_beta_40,
    }
    for ax, (name, pred_b) in zip(axes, beta_strategies.items()):
        valid = np.isfinite(pred_b)
        ax.scatter(true_beta_40[valid & ~is_glyme], pred_b[valid & ~is_glyme],
                   alpha=0.3, s=15, c="grey", label="Other")
        ax.scatter(true_beta_40[valid & is_glyme], pred_b[valid & is_glyme],
                   alpha=0.8, s=40, c="red", marker="D", label="Glymes",
                   edgecolors="black", linewidths=0.3)
        lims = [true_beta_40.min() * 0.9, true_beta_40.max() * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.5)
        r2 = r2_score(true_beta_40[valid], pred_b[valid])
        mae = mean_absolute_error(true_beta_40[valid], pred_b[valid])
        ax.set_title(f"{name}\nR²={r2:.4f}  MAE={mae:.6f}")
        ax.set_xlabel("True beta_40")
        ax.set_ylabel("Predicted beta_40")
        ax.legend(fontsize=8)

    fig.suptitle("Beta Prediction Accuracy", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beta_prediction_parity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "beta_prediction_parity.png")

    # ------------------------------------------------------------------
    # 9. Summary table
    # ------------------------------------------------------------------
    summary = []
    for strat_name, res in results.items():
        summary.append({
            "Strategy": strat_name,
            "FOM1_40 R²": f"{res['r2_40']:.4f}",
            "FOM1_40 MAE": f"{res['mae_40']:.2f}",
            "FOM1_100 R²": f"{res['r2_100']:.4f}",
            "FOM1_100 MAE": f"{res['mae_100']:.2f}",
            "FOM1_avg R²": f"{res['r2_avg']:.4f}",
            "FOM1_avg MAE": f"{res['mae_avg']:.2f}",
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "beta_experiment_results.csv", index=False)
    logger.info("\n%s", summary_df.to_string(index=False))
    logger.info("\nSaved: %s", OUT_DIR / "beta_experiment_results.csv")


if __name__ == "__main__":
    main()
