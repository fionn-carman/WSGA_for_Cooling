"""
Shared Test Set Training — XGBoost with Consistent Evaluation

Trains 10 XGBoost regression models (8 thermophysical properties + FOM1 at 40C
and 100C) using a single shared train/test split.  This allows direct comparison
of FOM1 predicted by a dedicated model vs FOM1 computed from separate property
predictions on the exact same held-out molecules.

Pipeline per target:
    1. RFE with multiple repeats for feature selection
    2. RandomizedSearchCV (10 000 iterations, 5-fold CV)
    3. Test-set evaluation with prediction intervals
    4. SHAP analysis + diagnostic plots

After all 10 models are trained the script computes FOM1 from the 8
thermophysical predictions and compares with the direct FOM1 predictions.

Execution modes
---------------
    # Train all 10 sequentially then compare
    python train_shared_test.py --data_dir ./data --outdir ./output_shared

    # Train a single target (for PBS array job index 0-9)
    python train_shared_test.py --data_dir ./data --outdir ./output_shared --target_index 0

    # Skip training, just run the FOM1 comparison (after all 10 are done)
    python train_shared_test.py --data_dir ./data --outdir ./output_shared --compare_only
"""

import os
import argparse
import warnings
import logging
import math
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Target configuration
# ============================================================

TARGET_CONFIG = [
    {"name": "Density_40C_g_cm^3",                             "log": False, "aux": None},
    {"name": "Density_100C_g_cm^3",                            "log": False, "aux": "Density_40C_g_cm^3"},
    {"name": "Kinematic_Viscosity_40C",                        "log": True,  "aux": None},
    {"name": "Kinematic_Viscosity_100C",                       "log": True,  "aux": None},
    {"name": "Thermal_Conductivity_40C",                       "log": False, "aux": None},
    {"name": "Thermal_Conductivity_100C",                      "log": False, "aux": None},
    {"name": "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",   "log": True,  "aux": None},
    {"name": "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",  "log": True,  "aux": None},
    {"name": "FOM1_exp_40",                                    "log": False, "aux": None},
    {"name": "FOM1_exp_100",                                   "log": False, "aux": None},
]

# Thermophysical property names used in the FOM1 formula
THERMO_TARGETS = [cfg["name"] for cfg in TARGET_CONFIG[:8]]


# ============================================================
# Shared split helpers
# ============================================================

def find_shared_smiles(data_dir):
    """Return sorted list of SMILES present in ALL 10 target cleaned CSVs."""
    smiles_sets = []
    for cfg in TARGET_CONFIG:
        csv_path = os.path.join(data_dir, f"{cfg['name']}_cleaned.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing cleaned CSV: {csv_path}")
        df = pd.read_csv(csv_path, usecols=["SMILES"])
        smiles_sets.append(set(df["SMILES"].dropna()))
        logger.info(f"  {cfg['name']}: {len(smiles_sets[-1])} molecules")

    shared = smiles_sets[0]
    for s in smiles_sets[1:]:
        shared = shared & s

    shared_list = sorted(shared)
    logger.info(f"  Shared intersection: {len(shared_list)} molecules")
    return shared_list


def create_shared_split(shared_smiles, test_size=0.1, random_state=42):
    """Create a single 90/10 split over SMILES strings."""
    train_smi, test_smi = train_test_split(
        shared_smiles, test_size=test_size, random_state=random_state
    )
    logger.info(f"  Train set: {len(train_smi)} molecules")
    logger.info(f"  Test set:  {len(test_smi)} molecules")
    return set(train_smi), set(test_smi)


def save_split(train_smiles, test_smiles, output_dir):
    """Persist the shared split for reproducibility."""
    split_dir = Path(output_dir) / "shared_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SMILES": sorted(train_smiles)}).to_csv(
        split_dir / "train_smiles.csv", index=False)
    pd.DataFrame({"SMILES": sorted(test_smiles)}).to_csv(
        split_dir / "test_smiles.csv", index=False)
    logger.info(f"  Saved split to {split_dir}")


def load_split(output_dir):
    """Load a previously saved shared split."""
    split_dir = Path(output_dir) / "shared_split"
    train_df = pd.read_csv(split_dir / "train_smiles.csv")
    test_df = pd.read_csv(split_dir / "test_smiles.csv")
    return set(train_df["SMILES"]), set(test_df["SMILES"])


# ============================================================
# Data loading
# ============================================================

def load_target_data(data_dir, target_name, shared_smiles):
    """Load a cleaned CSV, filter to shared SMILES, return DataFrame.

    FOM1 CSVs only have (SMILES, target) — descriptors are borrowed from a
    thermophysical CSV for those targets.
    """
    csv_path = os.path.join(data_dir, f"{target_name}_cleaned.csv")
    df = pd.read_csv(csv_path)

    # FOM1 CSVs only contain SMILES + target — need descriptors
    if target_name.startswith("FOM1"):
        # Borrow descriptors from Density_40C (guaranteed to be in shared set)
        desc_path = os.path.join(data_dir, "Density_40C_g_cm^3_cleaned.csv")
        df_desc = pd.read_csv(desc_path)
        # Keep only SMILES + descriptor columns (drop the Density target)
        desc_cols = [c for c in df_desc.columns if c != "Density_40C_g_cm^3"]
        df_desc = df_desc[desc_cols]
        df = df.merge(df_desc, on="SMILES", how="inner")

    # Filter to shared SMILES
    shared_set = set(shared_smiles) if not isinstance(shared_smiles, set) else shared_smiles
    df = df[df["SMILES"].isin(shared_set)].copy()

    df[target_name] = pd.to_numeric(df[target_name], errors="coerce")
    df = df.dropna(subset=[target_name])

    return df


def merge_auxiliary_feature(df, data_dir, aux_target, shared_smiles):
    """Add an auxiliary feature column from another target's cleaned CSV."""
    aux_path = os.path.join(data_dir, f"{aux_target}_cleaned.csv")
    df_aux = pd.read_csv(aux_path)
    df_aux = df_aux[["SMILES", aux_target]].copy()
    df_aux[aux_target] = pd.to_numeric(df_aux[aux_target], errors="coerce")
    df_aux = df_aux.dropna(subset=[aux_target])

    aux_col_name = f"AUX_{aux_target}"
    df_aux = df_aux.rename(columns={aux_target: aux_col_name})

    n_before = len(df)
    df = df.merge(df_aux, on="SMILES", how="inner")
    logger.info(f"  Auxiliary feature {aux_target}: {n_before} -> {len(df)} samples")
    return df, aux_col_name


# ============================================================
# RFE with multiple repeats (from train_regression.py)
# ============================================================

def run_rfe_repeat(X, y, feature_names, step_size=2, random_state=None):
    """One repeat of RFE with random 80/20 internal split."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    current_features = feature_names.copy()
    rmse_results = []
    feature_sets = {}

    while len(current_features) > 0:
        xgb = XGBRegressor(
            objective="reg:squarederror", n_jobs=-1,
            random_state=random_state, verbosity=0,
        )
        feature_indices = [feature_names.index(f) for f in current_features]
        rfe = RFE(estimator=xgb, n_features_to_select=1, step=1)
        rfe.fit(X_train[:, feature_indices], y_train)

        rankings = rfe.ranking_
        n_to_keep = max(len(current_features) - step_size, 1)
        keep_mask = rankings <= n_to_keep
        kept_features = [f for f, keep in zip(current_features, keep_mask) if keep]

        X_train_curr = X_train[:, [feature_names.index(f) for f in kept_features]]
        X_val_curr = X_val[:, [feature_names.index(f) for f in kept_features]]
        xgb.fit(X_train_curr, y_train)
        y_pred = xgb.predict(X_val_curr)

        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        rmse_results.append((len(kept_features), rmse))
        feature_sets[len(kept_features)] = kept_features

        current_features = kept_features
        if len(current_features) == 1:
            break

    df_rfe = pd.DataFrame(rmse_results, columns=["n_features", "rmse"])
    best_idx = df_rfe["rmse"].idxmin()
    best_n = int(df_rfe.loc[best_idx, "n_features"])
    return df_rfe, best_n, feature_sets[best_n]


def run_rfe_with_repeats(X, y, feature_names, n_repeats=3, step_size=2):
    """Run RFE with multiple random splits and pick overall best."""
    logger.info(f"  Running RFE with {n_repeats} repeats (step_size={step_size})...")

    rfe_results = []
    best_features_list = []

    for i in range(n_repeats):
        logger.info(f"    Repeat {i+1}/{n_repeats}")
        df_rfe, best_n, best_feats = run_rfe_repeat(
            X, y, feature_names, step_size=step_size, random_state=i,
        )
        rfe_results.append(df_rfe)
        best_features_list.append((best_n, best_feats))
        rmse_at_best = df_rfe.loc[df_rfe["n_features"] == best_n, "rmse"].values[0]
        logger.info(f"    -> {best_n} features, RMSE={rmse_at_best:.4f}")

    best_rmse = float("inf")
    overall_best_feats = None
    for idx, (best_n, best_feats) in enumerate(best_features_list):
        rmse_val = rfe_results[idx][rfe_results[idx]["n_features"] == best_n]["rmse"].values[0]
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            overall_best_feats = best_feats

    logger.info(f"  Overall best: {len(overall_best_feats)} features, RMSE={best_rmse:.4f}")
    return overall_best_feats, rfe_results


# ============================================================
# Hyperparameter optimisation
# ============================================================

def run_randomized_search(X_train, y_train, n_iter=10000):
    """RandomizedSearchCV with 5-fold CV."""
    logger.info(f"  RandomizedSearchCV: {n_iter} iterations ({n_iter * 5} fits)...")

    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 1000, 2000, 5000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5],
        "max_depth": [3, 4, 5, 6, 10, 20, 50, 100],
        "min_child_weight": [1, 2, 3, 4, 5, 6],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    base_model = XGBRegressor(
        objective="reg:squarederror", n_jobs=1,
        random_state=42, verbosity=0,
        tree_method="hist", max_bin=256,
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter, cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1, verbose=1, random_state=42,
        return_train_score=False, pre_dispatch="2*n_jobs",
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_rmse = -random_search.best_score_

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best 5-fold CV RMSE: {best_cv_rmse:.4f}")

    return best_model, best_params, best_cv_rmse, random_search


# ============================================================
# Prediction intervals
# ============================================================

def train_quantile_models(X_train, y_train, best_params):
    """Train quantile regression models for 80% prediction intervals."""
    logger.info("  Training quantile regression models...")
    gb_params = {
        "n_estimators": min(best_params.get("n_estimators", 500), 1000),
        "learning_rate": best_params.get("learning_rate", 0.1),
        "max_depth": best_params.get("max_depth", 5),
        "min_samples_leaf": best_params.get("min_child_weight", 1),
        "subsample": best_params.get("subsample", 0.8),
        "random_state": 42,
    }
    quantile_models = {}
    for q, name in [(0.1, "lower"), (0.5, "median"), (0.9, "upper")]:
        model = GradientBoostingRegressor(loss="quantile", alpha=q, **gb_params)
        model.fit(X_train, y_train)
        quantile_models[name] = model
    return quantile_models


# ============================================================
# Diagnostic plots + SHAP
# ============================================================

def save_diagnostic_plots(y_true, y_pred, output_dir, target_name,
                          prediction_intervals=None):
    """Save 4-panel diagnostic plots and plot data."""
    from scipy import stats as sp_stats

    residuals = y_true - y_pred
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Save raw data
    np.savez(output_dir / f"{target_name}_diagnostic_plot_data.npz",
             y_true=y_true, y_pred=y_pred, residuals=residuals,
             pi_lower=prediction_intervals["lower"] if prediction_intervals else np.array([]),
             pi_upper=prediction_intervals["upper"] if prediction_intervals else np.array([]))
    with open(output_dir / f"{target_name}_diagnostic_metrics.json", "w") as f:
        json.dump({"r2": r2, "rmse": rmse, "mae": mae,
                    "residual_mean": float(residuals.mean()),
                    "residual_std": float(residuals.std())}, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Parity
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none", s=30)
    if prediction_intervals is not None:
        idx_sort = np.argsort(y_true)
        ax.fill_between(y_true[idx_sort],
                        prediction_intervals["lower"][idx_sort],
                        prediction_intervals["upper"][idx_sort],
                        alpha=0.2, color="blue", label="80% PI")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect fit")
    ax.set_xlabel("True Values"); ax.set_ylabel("Predicted Values")
    ax.set_title("Parity Plot"); ax.legend()
    ax.text(0.05, 0.95, f"R\u00b2 = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Residuals
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="none", s=30)
    ax.axhline(y=0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Predicted Values"); ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")

    # Histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Residual"); ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")

    # Q-Q
    ax = axes[1, 1]
    sp_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")

    plt.suptitle(f"Diagnostic Plots: {target_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"{target_name}_diagnostics.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def save_shap_analysis(model, X, feature_names, output_dir, target_name):
    """SHAP beeswarm + bar plots."""
    if X.shape[0] > 1000:
        idx = np.random.choice(X.shape[0], 1000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    np.savez(output_dir / f"{target_name}_shap_data.npz",
             shap_values=shap_values, X_sample=X_sample,
             feature_names=np.array(feature_names, dtype=object))

    mean_abs = np.abs(shap_values).mean(axis=0)
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}) \
        .sort_values("mean_abs_shap", ascending=False) \
        .to_csv(output_dir / f"{target_name}_shap_importance.csv", index=False)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{target_name}_shap_beeswarm.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{target_name}_shap_bar.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def save_rfe_plot(rfe_results, best_n, output_dir, target_name):
    """RFE summary plot showing all repeats."""
    all_n = sorted(set(n for df in rfe_results for n in df["n_features"]))
    rmse_matrix = []
    for df in rfe_results:
        lookup = dict(zip(df["n_features"], df["rmse"]))
        rmse_matrix.append([lookup.get(n, np.nan) for n in all_n])
    rmse_matrix = np.array(rmse_matrix)
    mean_rmse = np.nanmean(rmse_matrix, axis=0)
    std_rmse = np.nanstd(rmse_matrix, axis=0)

    plt.figure(figsize=(8, 5))
    for row in rmse_matrix:
        plt.plot(all_n, row, color="grey", alpha=0.4)
    plt.errorbar(all_n, mean_rmse, yerr=std_rmse, fmt="-o", capsize=5,
                 color="blue", label="Mean +/- std")
    plt.axvline(x=best_n, color="r", linestyle="--",
                label=f"Selected: {best_n} features")
    plt.xlim(0)
    plt.xlabel("Number of Features")
    plt.ylabel("Validation RMSE")
    plt.title(f"RFE Feature Selection: {target_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{target_name}_rfe.png", dpi=300)
    plt.close()


# ============================================================
# Single-target training
# ============================================================

def train_single_target(df, target_name, log_target, aux_feature,
                        train_smiles, test_smiles, output_dir,
                        data_dir=None, shared_smiles=None,
                        n_random_iter=10000, n_rfe_repeats=3,
                        rfe_step_size=2):
    """Train one target using the shared split.

    Returns a DataFrame with columns: SMILES, y_true, y_pred
    (values are in original scale, i.e. un-logged if log_target was True).
    """
    logger.info("=" * 70)
    logger.info(f"TRAINING: {target_name}")
    logger.info("=" * 70)
    start_time = datetime.now()

    output_dir = Path(output_dir)
    rfe_dir = output_dir / "RFE"
    model_dir = output_dir / "model"
    plots_dir = output_dir / "plots"
    for d in [rfe_dir, model_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Handle auxiliary feature
    aux_col_name = None
    if aux_feature and data_dir:
        df, aux_col_name = merge_auxiliary_feature(
            df, data_dir, aux_feature, shared_smiles)

    # Apply log transform
    if log_target:
        if (df[target_name] <= 0).any():
            raise ValueError(f"Cannot log transform {target_name}: non-positive values")
        df[target_name] = np.log1p(df[target_name])
        logger.info("  Applied log1p transform")

    # Split using shared SMILES sets
    train_df = df[df["SMILES"].isin(train_smiles)].copy()
    test_df = df[df["SMILES"].isin(test_smiles)].copy()
    test_smiles_ordered = test_df["SMILES"].tolist()

    base_cols = ["SMILES", target_name]
    feature_names = [c for c in df.columns if c not in base_cols]
    X_train = train_df[feature_names].values
    y_train = train_df[target_name].values.astype(float)
    X_test = test_df[feature_names].values
    y_test = test_df[target_name].values.astype(float)

    logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}, Features: {len(feature_names)}")

    # --- Feature selection (RFE) ---
    selected_features, rfe_results = run_rfe_with_repeats(
        X_train, y_train, feature_names,
        n_repeats=n_rfe_repeats, step_size=rfe_step_size,
    )

    with open(rfe_dir / "selected_features.txt", "w") as f:
        for feat in selected_features:
            f.write(feat + "\n")
    for i, df_rfe in enumerate(rfe_results):
        df_rfe.to_csv(rfe_dir / f"rfe_repeat_{i+1}.csv", index=False)
    save_rfe_plot(rfe_results, len(selected_features), plots_dir, target_name)

    # Re-index to selected features
    X_train_sel = train_df[selected_features].values
    X_test_sel = test_df[selected_features].values

    # --- Hyperparameter search ---
    best_model, best_params, best_cv_rmse, random_search = run_randomized_search(
        X_train_sel, y_train, n_iter=n_random_iter,
    )

    cv_df = pd.DataFrame(random_search.cv_results_)
    cv_df.to_csv(output_dir / f"{target_name}_cv_results.csv", index=False)

    # --- Test evaluation ---
    y_test_pred = best_model.predict(X_test_sel)

    if log_target:
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_test_pred)
    else:
        y_test_orig = y_test
        y_pred_orig = y_test_pred

    test_rmse = math.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    test_r2 = r2_score(y_test_orig, y_pred_orig)
    test_mae = mean_absolute_error(y_test_orig, y_pred_orig)

    logger.info(f"  Test R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

    # --- Prediction intervals ---
    quantile_models = train_quantile_models(X_train_sel, y_train, best_params)
    pi_lower = quantile_models["lower"].predict(X_test_sel)
    pi_upper = quantile_models["upper"].predict(X_test_sel)
    if log_target:
        pi_lower = np.expm1(pi_lower)
        pi_upper = np.expm1(pi_upper)
    coverage = np.mean((y_test_orig >= pi_lower) & (y_test_orig <= pi_upper))
    logger.info(f"  80% PI coverage: {coverage*100:.1f}%")

    # --- Plots ---
    pi_dict = {"lower": pi_lower, "upper": pi_upper}
    save_diagnostic_plots(y_test_orig, y_pred_orig, plots_dir, target_name,
                          prediction_intervals=pi_dict)
    save_shap_analysis(best_model, train_df[selected_features].values,
                       selected_features, plots_dir, target_name)

    # --- Save model ---
    model_data = {
        "model": best_model,
        "quantile_models": quantile_models,
        "features": selected_features,
        "log_target": log_target,
        "best_params": best_params,
        "cv_rmse": best_cv_rmse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "pi_coverage": coverage,
        "training_date": datetime.now().isoformat(),
        "n_random_iter": n_random_iter,
        "n_rfe_repeats": n_rfe_repeats,
        "method": "shared_test_rfe_randomized",
        "auxiliary_feature": aux_feature,
        "auxiliary_feature_name": aux_col_name,
    }
    dump(model_data, model_dir / "xgb_model.joblib")

    # Human-readable summary
    with open(model_dir / "model_summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"XGBoost Shared-Test Model: {target_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log transform: {log_target}\n")
        f.write(f"Train samples: {len(train_df)}\n")
        f.write(f"Test samples:  {len(test_df)}\n")
        if aux_feature:
            f.write(f"Auxiliary: {aux_feature} (as {aux_col_name})\n")
        f.write(f"\nSelected features: {len(selected_features)}\n")
        f.write(f"5-fold CV RMSE: {best_cv_rmse:.4f}\n")
        f.write(f"Test R2:        {test_r2:.4f}\n")
        f.write(f"Test RMSE:      {test_rmse:.4f}\n")
        f.write(f"Test MAE:       {test_mae:.4f}\n")
        f.write(f"80% PI coverage: {coverage*100:.1f}%\n")
        f.write(f"\nBest hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nFeatures ({len(selected_features)}):\n")
        for feat in selected_features:
            f.write(f"  {feat}\n")

    # --- Save test predictions for FOM1 comparison ---
    pred_df = pd.DataFrame({
        "SMILES": test_smiles_ordered,
        "y_true": y_test_orig,
        "y_pred": y_pred_orig,
    })
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    logger.info(f"  Saved test predictions ({len(pred_df)} rows)")

    elapsed = datetime.now() - start_time
    logger.info(f"  Completed in {elapsed}")

    return pred_df


# ============================================================
# FOM1 comparison
# ============================================================

def compute_fom1_from_predictions(test_predictions):
    """Compute FOM1 at 40C and 100C from the 8 thermophysical predictions.

    Uses the exact formula from wsga_helper.py:
        beta_T = -(1/rho_T) * (rho_100 - rho_40) / dT
        Cp_T = HC_T / MW * 1000
        nu_T = KV_T * 1e-6
        FOM1_T = k_T * ((beta_T * Cp_T * rho_T * 1000) / (nu_T * k_T))**0.2813
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors as RDDescriptors

    # Start from SMILES present in ALL thermophysical predictions
    smiles_sets = []
    for target in THERMO_TARGETS:
        if target not in test_predictions:
            raise RuntimeError(f"Missing predictions for {target}")
        smiles_sets.append(set(test_predictions[target]["SMILES"]))

    common_smiles = sorted(smiles_sets[0].intersection(*smiles_sets[1:]))
    df = pd.DataFrame({"SMILES": common_smiles})

    # Merge each thermophysical prediction
    for target in THERMO_TARGETS:
        pred_df = test_predictions[target]
        df = df.merge(
            pred_df[["SMILES", "y_pred"]].rename(columns={"y_pred": target}),
            on="SMILES", how="inner",
        )

    # Molecular weight for Cp conversion
    df["MW"] = df["SMILES"].apply(
        lambda s: RDDescriptors.MolWt(Chem.MolFromSmiles(s))
    )

    dT = 60.0  # 40 -> 100C

    # Cp: J/(K*mol) -> J/(K*kg)
    df["Cp_40"] = df["Heat_Capacity_Constant_Pressure_40C_J_K_Mol"] / df["MW"] * 1000
    df["Cp_100"] = df["Heat_Capacity_Constant_Pressure_100C_J_K_Mol"] / df["MW"] * 1000

    # Kinematic viscosity: cSt -> m^2/s
    df["nu_40"] = df["Kinematic_Viscosity_40C"] * 1e-6
    df["nu_100"] = df["Kinematic_Viscosity_100C"] * 1e-6

    # Thermal expansion coefficient
    df["beta_40"] = -(1 / df["Density_40C_g_cm^3"]) * (
        (df["Density_100C_g_cm^3"] - df["Density_40C_g_cm^3"]) / dT
    )
    df["beta_100"] = -(1 / df["Density_100C_g_cm^3"]) * (
        (df["Density_100C_g_cm^3"] - df["Density_40C_g_cm^3"]) / dT
    )

    # FOM1
    df["FOM1_computed_40"] = (
        df["Thermal_Conductivity_40C"]
        * (
            (df["beta_40"] * df["Cp_40"] * df["Density_40C_g_cm^3"] * 1000)
            / (df["nu_40"] * df["Thermal_Conductivity_40C"])
        )
        ** 0.2813
    )
    df["FOM1_computed_100"] = (
        df["Thermal_Conductivity_100C"]
        * (
            (df["beta_100"] * df["Cp_100"] * df["Density_100C_g_cm^3"] * 1000)
            / (df["nu_100"] * df["Thermal_Conductivity_100C"])
        )
        ** 0.2813
    )

    logger.info(f"  Computed FOM1 for {len(df)} test molecules")
    return df


def compare_fom1(computed_df, test_predictions, output_dir):
    """Compare direct vs computed FOM1 on the shared test set.

    Produces metrics, comparison CSV, and parity plots.
    """
    output_dir = Path(output_dir)

    # Merge direct predictions
    df = computed_df.copy()
    for suffix, target_name in [("40", "FOM1_exp_40"), ("100", "FOM1_exp_100")]:
        if target_name not in test_predictions:
            logger.warning(f"  Missing direct predictions for {target_name}")
            continue
        pred_df = test_predictions[target_name]
        df = df.merge(
            pred_df[["SMILES", "y_true", "y_pred"]].rename(
                columns={"y_true": f"FOM1_true_{suffix}",
                          "y_pred": f"FOM1_direct_{suffix}"}),
            on="SMILES", how="inner",
        )

    df.to_csv(output_dir / "fom1_comparison.csv", index=False)
    logger.info(f"  Saved fom1_comparison.csv ({len(df)} molecules)")

    # Metrics
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("FOM1 COMPARISON: Direct Model vs Computed from Thermo")
    summary_lines.append("=" * 60)

    for temp in ["40", "100"]:
        true_col = f"FOM1_true_{temp}"
        direct_col = f"FOM1_direct_{temp}"
        computed_col = f"FOM1_computed_{temp}"

        if true_col not in df.columns:
            continue

        direct_r2 = r2_score(df[true_col], df[direct_col])
        direct_rmse = math.sqrt(mean_squared_error(df[true_col], df[direct_col]))
        direct_mae = mean_absolute_error(df[true_col], df[direct_col])

        computed_r2 = r2_score(df[true_col], df[computed_col])
        computed_rmse = math.sqrt(mean_squared_error(df[true_col], df[computed_col]))
        computed_mae = mean_absolute_error(df[true_col], df[computed_col])

        summary_lines.append(f"\nFOM1 at {temp}C (n={len(df)}):")
        summary_lines.append(f"  Direct model:     R2={direct_r2:.4f}  RMSE={direct_rmse:.4f}  MAE={direct_mae:.4f}")
        summary_lines.append(f"  From thermo preds: R2={computed_r2:.4f}  RMSE={computed_rmse:.4f}  MAE={computed_mae:.4f}")

        logger.info(f"  FOM1 at {temp}C:")
        logger.info(f"    Direct:   R2={direct_r2:.4f}, RMSE={direct_rmse:.4f}")
        logger.info(f"    Computed: R2={computed_r2:.4f}, RMSE={computed_rmse:.4f}")

    summary_text = "\n".join(summary_lines)
    with open(output_dir / "fom1_comparison_summary.txt", "w") as f:
        f.write(summary_text + "\n")

    # Parity plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, temp in enumerate(["40", "100"]):
        true_col = f"FOM1_true_{temp}"
        direct_col = f"FOM1_direct_{temp}"
        computed_col = f"FOM1_computed_{temp}"
        if true_col not in df.columns:
            continue

        ax = axes[ax_idx]
        ax.scatter(df[true_col], df[direct_col], alpha=0.6, s=40,
                   label="Direct model", color="#1f77b4")
        ax.scatter(df[true_col], df[computed_col], alpha=0.6, s=40,
                   marker="^", label="From thermo", color="#ff7f0e")

        lo = min(df[true_col].min(), df[direct_col].min(), df[computed_col].min())
        hi = max(df[true_col].max(), df[direct_col].max(), df[computed_col].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect fit")

        dr2 = r2_score(df[true_col], df[direct_col])
        cr2 = r2_score(df[true_col], df[computed_col])
        ax.set_xlabel("True FOM1", fontsize=12)
        ax.set_ylabel("Predicted FOM1", fontsize=12)
        ax.set_title(f"FOM1 at {temp}\u00b0C", fontsize=14)
        ax.legend(fontsize=10)
        ax.text(0.05, 0.95,
                f"Direct R\u00b2={dr2:.4f}\nComputed R\u00b2={cr2:.4f}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "fom1_comparison_parity.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    logger.info("  Saved fom1_comparison_parity.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Shared-test-set XGBoost training (8 thermo + 2 FOM1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing cleaned CSVs")
    parser.add_argument("--outdir", default="./output_shared",
                        help="Base output directory")
    parser.add_argument("--n_random_iter", type=int, default=10000)
    parser.add_argument("--n_rfe_repeats", type=int, default=3)
    parser.add_argument("--rfe_step_size", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--target_index", type=int, default=None,
                        help="Train only this target (0-9). If omitted, train all.")
    parser.add_argument("--compare_only", action="store_true",
                        help="Skip training — load saved predictions and run FOM1 comparison.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Compare-only mode
    # ------------------------------------------------------------------
    if args.compare_only:
        logger.info("=" * 70)
        logger.info("  COMPARE-ONLY MODE")
        logger.info("=" * 70)

        test_predictions = {}
        for cfg in TARGET_CONFIG:
            pred_path = outdir / cfg["name"] / "test_predictions.csv"
            if not pred_path.exists():
                logger.warning(f"  Missing {pred_path}")
                continue
            test_predictions[cfg["name"]] = pd.read_csv(pred_path)
            logger.info(f"  Loaded {cfg['name']}: {len(test_predictions[cfg['name']])} rows")

        # Check we have all 8 thermo targets
        missing = [t for t in THERMO_TARGETS if t not in test_predictions]
        if missing:
            logger.error(f"  Missing thermophysical predictions: {missing}")
            logger.error("  Cannot compute FOM1 from thermo predictions.")
            return

        computed_df = compute_fom1_from_predictions(test_predictions)
        compare_fom1(computed_df, test_predictions, outdir)
        return

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("  SHARED TEST SET TRAINING")
    logger.info("=" * 70)

    # Determine shared SMILES and create split
    split_path = outdir / "shared_split" / "train_smiles.csv"
    if split_path.exists():
        logger.info("  Loading existing shared split...")
        train_smiles, test_smiles = load_split(args.outdir)
        shared_smiles = sorted(train_smiles | test_smiles)
    else:
        logger.info("  Finding shared SMILES across all 10 targets...")
        shared_smiles = find_shared_smiles(args.data_dir)
        train_smiles, test_smiles = create_shared_split(
            shared_smiles, test_size=args.test_size)
        save_split(train_smiles, test_smiles, args.outdir)

    # Determine which targets to train
    if args.target_index is not None:
        if args.target_index < 0 or args.target_index >= len(TARGET_CONFIG):
            raise ValueError(f"target_index must be 0-{len(TARGET_CONFIG)-1}")
        targets_to_train = [TARGET_CONFIG[args.target_index]]
    else:
        targets_to_train = TARGET_CONFIG

    # Train each target
    test_predictions = {}
    for cfg in targets_to_train:
        target_name = cfg["name"]
        df = load_target_data(args.data_dir, target_name, shared_smiles)

        pred_df = train_single_target(
            df, target_name, cfg["log"], cfg["aux"],
            train_smiles, test_smiles,
            outdir / target_name,
            data_dir=args.data_dir,
            shared_smiles=shared_smiles,
            n_random_iter=args.n_random_iter,
            n_rfe_repeats=args.n_rfe_repeats,
            rfe_step_size=args.rfe_step_size,
        )
        test_predictions[target_name] = pred_df

    # If we trained all 10, run FOM1 comparison
    if args.target_index is None:
        missing = [t for t in THERMO_TARGETS if t not in test_predictions]
        if not missing:
            logger.info("\n" + "=" * 70)
            logger.info("  FOM1 COMPARISON")
            logger.info("=" * 70)
            computed_df = compute_fom1_from_predictions(test_predictions)
            compare_fom1(computed_df, test_predictions, outdir)
        else:
            logger.info("  Skipping FOM1 comparison (missing thermo targets)")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
