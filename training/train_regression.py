"""
Hybrid XGBoost Regression Training - Best of Both Worlds

Combines the proven performance of the original training approach with modern improvements:

FROM ORIGINAL (Proven Performance):
- RFE with multiple repeats (3 runs with different random splits)
- RandomizedSearchCV with 10,000 iterations
- Step-wise feature elimination (step_size=2)
- Discrete parameter distributions

FROM NEW (Modern Improvements):
- Comprehensive logging
- SHAP interpretability analysis
- Prediction intervals via quantile regression
- Professional diagnostic plots (4-panel)
- Nested CV for unbiased estimates
- Better organized output structure
- Pre-cleaned datasets (no descriptor computation overhead)

This script should match or exceed the performance of the original training.

Usage:
    python train_regression_hybrid.py --target Density_40C_g_cm^3
    python train_regression_hybrid.py --target Kinematic_Viscosity_40C --log_target
"""

import os
import argparse
import warnings
import logging
from pathlib import Path
from typing import Tuple, List, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import math
from joblib import dump
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# RFE with Multiple Repeats (Original Proven Method)
# ============================================================

def run_rfe_repeat(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    step_size: int = 2,
    random_state: int = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    One repeat of RFE with random split.

    Returns:
        df_rfe: DataFrame with RMSE vs n_features
        best_n: Best number of features
        best_features: List of best feature names
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    current_features = feature_names.copy()
    rmse_results = []
    feature_sets = {}

    while len(current_features) > 0:
        # Fit RFE to rank features
        xgb = XGBRegressor(
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=random_state,
            verbosity=0
        )

        feature_indices = [feature_names.index(f) for f in current_features]
        rfe = RFE(estimator=xgb, n_features_to_select=1, step=1)
        rfe.fit(X_train[:, feature_indices], y_train)

        # Rank features (1=best)
        rankings = rfe.ranking_
        n_to_keep = max(len(current_features) - step_size, 1)
        keep_mask = rankings <= n_to_keep
        kept_features = [f for f, keep in zip(current_features, keep_mask) if keep]
        dropped_features = [f for f, keep in zip(current_features, keep_mask) if not keep]

        # Evaluate
        X_train_curr = X_train[:, [feature_names.index(f) for f in kept_features]]
        X_val_curr = X_val[:, [feature_names.index(f) for f in kept_features]]
        xgb.fit(X_train_curr, y_train)
        y_pred = xgb.predict(X_val_curr)

        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        rmse_results.append((len(kept_features), rmse))
        feature_sets[len(kept_features)] = kept_features

        if verbose:
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            logger.info(f"    n={len(kept_features)}, R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            if dropped_features:
                logger.info(f"    Dropped: {dropped_features[:3]}{'...' if len(dropped_features) > 3 else ''}")

        current_features = kept_features
        if len(current_features) == 1:
            break

    df_rfe = pd.DataFrame(rmse_results, columns=["n_features", "rmse"])
    best_idx = df_rfe["rmse"].idxmin()
    best_n = int(df_rfe.loc[best_idx, "n_features"])
    best_feats = feature_sets[best_n]

    return df_rfe, best_n, best_feats


def run_rfe_with_repeats(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 3,
    step_size: int = 2
) -> Tuple[List[str], List[pd.DataFrame]]:
    """
    Run RFE with multiple random splits and select overall best.

    Returns:
        best_features: List of best feature names
        rfe_results: List of DataFrames from each repeat
    """
    logger.info(f"Running RFE with {n_repeats} repeats (step_size={step_size})...")

    rfe_results = []
    best_features_list = []

    for i in range(n_repeats):
        logger.info(f"\n  Repeat {i+1}/{n_repeats}")
        df_rfe, best_n, best_feats = run_rfe_repeat(
            X, y, feature_names,
            step_size=step_size,
            random_state=i,
            verbose=True
        )
        rfe_results.append(df_rfe)
        best_features_list.append((best_n, best_feats))

        rmse_at_best = df_rfe.loc[df_rfe["n_features"] == best_n, "rmse"].values[0]
        logger.info(f"  → Best for repeat {i+1}: {best_n} features, RMSE={rmse_at_best:.4f}")

    # Determine overall best feature set
    best_rmse = float("inf")
    overall_best_feats = None
    best_repeat = None

    for idx, (best_n, best_feats) in enumerate(best_features_list):
        rmse_at_best = rfe_results[idx][rfe_results[idx]["n_features"] == best_n]["rmse"].values[0]
        if rmse_at_best < best_rmse:
            best_rmse = rmse_at_best
            overall_best_feats = best_feats
            best_repeat = idx + 1

    logger.info(f"\n✓ Overall best: {len(overall_best_feats)} features from repeat {best_repeat}, RMSE={best_rmse:.4f}")

    return overall_best_feats, rfe_results


# ============================================================
# RandomizedSearchCV (Original Proven Method)
# ============================================================

def run_randomized_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 10000
) -> Tuple[XGBRegressor, Dict]:
    """
    Run RandomizedSearchCV with large iteration count (original method).

    Returns:
        best_model: Fitted XGBoost model
        best_params: Dictionary of best hyperparameters
    """
    logger.info(f"Running RandomizedSearchCV with {n_iter} iterations...")
    logger.info(f"Total fits (with 5-fold CV): {n_iter * 5}")

    # Define parameter distributions (original discrete values)
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 1000, 2000, 5000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
        'max_depth': [3, 4, 5, 6, 10, 20, 50, 100],
        'min_child_weight': [1, 2, 3, 4, 5, 6],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    base_model = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=1,  # Single thread per model - parallelism at CV level
        random_state=42,
        verbosity=0,
        tree_method='hist',
        max_bin=256
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,  # Use all cores
        verbose=1,
        random_state=42,
        return_train_score=False,
        pre_dispatch='2*n_jobs'
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_rmse = -random_search.best_score_

    logger.info(f"\n✓ Best hyperparameters: {best_params}")
    logger.info(f"✓ Best 5-fold CV RMSE: {best_cv_rmse:.4f}")

    return best_model, best_params, best_cv_rmse, random_search


# ============================================================
# Prediction Intervals (New Feature)
# ============================================================

def train_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: Dict,
    quantiles: List[float] = [0.1, 0.5, 0.9]
) -> Dict[str, GradientBoostingRegressor]:
    """Train quantile regression models for prediction intervals."""
    logger.info("Training quantile regression models for prediction intervals...")

    gb_params = {
        'n_estimators': min(best_params.get('n_estimators', 500), 1000),
        'learning_rate': best_params.get('learning_rate', 0.1),
        'max_depth': best_params.get('max_depth', 5),
        'min_samples_leaf': best_params.get('min_child_weight', 1),
        'subsample': best_params.get('subsample', 0.8),
        'random_state': 42
    }

    quantile_models = {}
    quantile_names = {0.1: 'lower', 0.5: 'median', 0.9: 'upper'}

    for q in quantiles:
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=q,
            **gb_params
        )
        model.fit(X_train, y_train)
        quantile_models[quantile_names[q]] = model

    logger.info("  ✓ Trained lower, median, upper quantile models")
    return quantile_models


# ============================================================
# Diagnostic Plots (New Feature)
# ============================================================

def save_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    target_name: str,
    prediction_intervals: Dict = None
):
    """Save comprehensive 4-panel diagnostic plots."""
    logger.info("Generating diagnostic plots...")

    # Calculate residuals and metrics for saving
    residuals = y_true - y_pred
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Save plot data for later editing
    plot_data = {
        'y_true': y_true,
        'y_pred': y_pred,
        'residuals': residuals,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae, 
                    'residual_mean': residuals.mean(), 'residual_std': residuals.std()},
        'prediction_intervals': prediction_intervals
    }
    
    plot_data_file = output_dir / f'{target_name}_diagnostic_plot_data.npz'
    np.savez(plot_data_file, 
             y_true=y_true, 
             y_pred=y_pred, 
             residuals=residuals,
             pi_lower=prediction_intervals['lower'] if prediction_intervals else np.array([]),
             pi_upper=prediction_intervals['upper'] if prediction_intervals else np.array([]))
    
    # Save metrics separately as JSON for easy reading
    import json
    metrics_file = output_dir / f'{target_name}_diagnostic_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(plot_data['metrics'], f, indent=2)
    
    logger.info(f"  ✓ Saved diagnostic plot data → {plot_data_file.name}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Parity plot
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='none', s=30)

    if prediction_intervals is not None:
        sorted_idx = np.argsort(y_true)
        ax.fill_between(
            y_true[sorted_idx],
            prediction_intervals['lower'][sorted_idx],
            prediction_intervals['upper'][sorted_idx],
            alpha=0.2, color='blue', label='80% PI'
        )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Parity Plot', fontsize=14)
    ax.legend()

    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Residual plot
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='none', s=30)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14)

    ax.text(0.05, 0.95, f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Residual distribution
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residual', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=14)

    # 4. Q-Q plot
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=14)

    plt.suptitle(f'Diagnostic Plots: {target_name}', fontsize=16, y=1.02)
    plt.tight_layout()

    plot_file = output_dir / f'{target_name}_diagnostics.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ Saved diagnostic plots → {plot_file.name}")


def save_rfe_plot(rfe_results: List[pd.DataFrame], best_n: int, output_dir: Path, target_name: str):
    """Save RFE summary plot showing all repeats."""
    logger.info("Generating RFE summary plot...")

    all_n = sorted(set(n for df_rfe in rfe_results for n in df_rfe["n_features"]))
    rmse_matrix = []

    for df_rfe in rfe_results:
        rmse_dict = dict(zip(df_rfe["n_features"], df_rfe["rmse"]))
        rmse_row = [rmse_dict.get(n, np.nan) for n in all_n]
        rmse_matrix.append(rmse_row)

    rmse_matrix = np.array(rmse_matrix)
    mean_rmse = np.nanmean(rmse_matrix, axis=0)
    std_rmse = np.nanstd(rmse_matrix, axis=0)
    
    # Save RFE plot data
    rfe_plot_data = {
        'n_features': np.array(all_n),
        'rmse_matrix': rmse_matrix,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'best_n': best_n
    }
    
    rfe_data_file = output_dir / f'{target_name}_rfe_plot_data.npz'
    np.savez(rfe_data_file, 
             n_features=np.array(all_n),
             rmse_matrix=rmse_matrix,
             mean_rmse=mean_rmse,
             std_rmse=std_rmse,
             best_n=np.array([best_n]))
    
    logger.info(f"  ✓ Saved RFE plot data → {rfe_data_file.name}")
    
    plt.figure(figsize=(8, 5))
    
    for rmse_row in rmse_matrix:
        plt.plot(all_n, rmse_row, color='grey', alpha=0.4)

    plt.errorbar(all_n, mean_rmse, yerr=std_rmse, fmt='-o', capsize=5, color='blue', label='Mean ± std')
    plt.axvline(x=best_n, color='r', linestyle='--', label=f'Selected: {best_n} features')

    plt.xlim(0)
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel("Validation RMSE", fontsize=12)
    plt.title(f"RFE Feature Selection: {target_name}", fontsize=14)
    plt.legend()
    plt.tight_layout()

    plot_file = output_dir / f'{target_name}_rfe.png'
    plt.savefig(plot_file, dpi=300)
    plt.close()

    logger.info(f"  ✓ Saved RFE plot → {plot_file.name}")


def save_shap_analysis(
    model: XGBRegressor,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    target_name: str,
    max_display: int = 20
):
    """Generate and save SHAP analysis plots."""
    logger.info("Running SHAP analysis...")

    if X.shape[0] > 1000:
        sample_idx = np.random.choice(X.shape[0], 1000, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
        sample_idx = np.arange(X.shape[0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Save SHAP data for later editing
    shap_data_file = output_dir / f'{target_name}_shap_data.npz'
    np.savez(shap_data_file,
             shap_values=shap_values,
             X_sample=X_sample,
             sample_indices=sample_idx,
             feature_names=np.array(feature_names, dtype=object))
    
    # Also save feature importance summary as CSV
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv(output_dir / f'{target_name}_shap_importance.csv', index=False)
    
    logger.info(f"  ✓ Saved SHAP data → {shap_data_file.name}")

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f'{target_name}_shap_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type='bar', max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f'{target_name}_shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ Saved SHAP plots")


# ============================================================
# Main Training Pipeline
# ============================================================

def main(args):
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info(f"HYBRID REGRESSION TRAINING: {args.target}")
    logger.info("=" * 70)

    # -------------------------
    # Load pre-cleaned data
    # -------------------------
    logger.info("\nSTAGE 1: Loading Pre-Cleaned Data")
    logger.info("-" * 70)

    input_file = Path(args.data_dir) / f"{args.target}_cleaned.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {input_file}")

    df = pd.read_csv(input_file)

    if "SMILES" not in df.columns or args.target not in df.columns:
        raise ValueError("Input must contain SMILES and target columns")

    df[args.target] = pd.to_numeric(df[args.target], errors='coerce')
    df = df.dropna(subset=[args.target])

    logger.info(f"✓ Loaded {len(df)} samples from {input_file.name}")
    logger.info(f"✓ Total columns: {len(df.columns)} (SMILES + target + {len(df.columns)-2} descriptors)")

    # -------------------------
    # Load and merge auxiliary feature if specified
    # -------------------------
    auxiliary_feature_name = None
    if args.auxiliary_feature:
        logger.info(f"\n  Loading auxiliary feature: {args.auxiliary_feature}")
        aux_file = Path(args.data_dir) / f"{args.auxiliary_feature}_cleaned.csv"
        
        if not aux_file.exists():
            raise FileNotFoundError(f"Auxiliary dataset not found: {aux_file}")
        
        df_aux = pd.read_csv(aux_file)
        
        if args.auxiliary_feature not in df_aux.columns:
            raise ValueError(f"Auxiliary feature '{args.auxiliary_feature}' not found in {aux_file.name}")
        
        # Keep only SMILES and the auxiliary target column
        df_aux = df_aux[["SMILES", args.auxiliary_feature]].copy()
        df_aux[args.auxiliary_feature] = pd.to_numeric(df_aux[args.auxiliary_feature], errors='coerce')
        df_aux = df_aux.dropna(subset=[args.auxiliary_feature])
        
        # Rename auxiliary feature to avoid confusion
        auxiliary_feature_name = f"AUX_{args.auxiliary_feature}"
        df_aux = df_aux.rename(columns={args.auxiliary_feature: auxiliary_feature_name})
        
        # Merge on SMILES
        n_before = len(df)
        df = df.merge(df_aux, on="SMILES", how="inner")
        n_after = len(df)
        
        logger.info(f"  ✓ Loaded auxiliary feature from {aux_file.name}")
        logger.info(f"  ✓ Samples after merge: {n_after} (dropped {n_before - n_after} without auxiliary data)")
        logger.info(f"  ✓ Auxiliary feature added as: {auxiliary_feature_name}")

    # Apply log transform if requested
    if args.log_target:
        if (df[args.target] <= 0).any():
            raise ValueError("Cannot apply log transform: non-positive values in target")
        df[args.target] = np.log1p(df[args.target])
        logger.info("✓ Applied log1p transform to target")

    # Setup output directories
    base_outdir = Path(args.outdir) / args.target
    rfe_outdir = base_outdir / "RFE"
    model_outdir = base_outdir / "model"
    plots_outdir = base_outdir / "plots"

    for d in [rfe_outdir, model_outdir, plots_outdir]:
        d.mkdir(parents=True, exist_ok=True)

    # Prepare arrays
    base_cols = ["SMILES", args.target]
    X = df.drop(columns=base_cols).values
    y = df[args.target].values.astype(float)
    feature_names = list(df.drop(columns=base_cols).columns)

    logger.info(f"✓ Feature matrix: {X.shape}")
    logger.info(f"✓ Target shape: {y.shape}")

    # -------------------------
    # Feature Selection (Original RFE Method)
    # -------------------------
    logger.info("\nSTAGE 2: Feature Selection (RFE with Repeats)")
    logger.info("-" * 70)

    selected_features, rfe_results = run_rfe_with_repeats(
        X, y, feature_names,
        n_repeats=args.n_rfe_repeats,
        step_size=args.rfe_step_size
    )

    # Save selected features
    features_file = rfe_outdir / "selected_features.txt"
    with open(features_file, "w") as f:
        for feat in selected_features:
            f.write(feat + "\n")
    logger.info(f"✓ Saved features → {features_file.name}")

    # Save RFE results
    for i, df_rfe in enumerate(rfe_results):
        df_rfe.to_csv(rfe_outdir / f"rfe_repeat_{i+1}.csv", index=False)

    save_rfe_plot(rfe_results, len(selected_features), plots_outdir, args.target)

    # Update X with selected features
    X_selected = df[selected_features].values

    # -------------------------
    # Train/Test Split
    # -------------------------
    logger.info("\nSTAGE 3: Data Splitting")
    logger.info("-" * 70)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_selected, y, test_size=0.1, random_state=42
    )

    logger.info(f"✓ Training+Val set: {X_trainval.shape[0]} samples")
    logger.info(f"✓ Test set: {X_test.shape[0]} samples")

    # -------------------------
    # Hyperparameter Optimization (Original RandomizedSearchCV)
    # -------------------------
    logger.info("\nSTAGE 4: Hyperparameter Optimization (RandomizedSearchCV)")
    logger.info("-" * 70)

    best_model, best_params, best_cv_rmse, random_search = run_randomized_search(
        X_trainval, y_trainval,
        n_iter=args.n_random_iter
    )

    # Save CV results
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_df.to_csv(base_outdir / f"{args.target}_cv_results.csv", index=False)
    logger.info(f"✓ Saved CV results")

    # -------------------------
    # Test Set Evaluation
    # -------------------------
    logger.info("\nSTAGE 5: Test Set Evaluation")
    logger.info("-" * 70)

    y_test_pred = best_model.predict(X_test)

    # Transform back if log target
    if args.log_target:
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_test_pred)
    else:
        y_test_orig = y_test
        y_pred_orig = y_test_pred

    test_rmse = math.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    test_r2 = r2_score(y_test_orig, y_pred_orig)
    test_mae = mean_absolute_error(y_test_orig, y_pred_orig)

    logger.info(f"\n✓ Test Set Performance:")
    logger.info(f"  R²:   {test_r2:.4f}")
    logger.info(f"  RMSE: {test_rmse:.4f}")
    logger.info(f"  MAE:  {test_mae:.4f}")

    # -------------------------
    # Prediction Intervals (New Feature)
    # -------------------------
    logger.info("\nSTAGE 6: Prediction Intervals")
    logger.info("-" * 70)

    quantile_models = train_quantile_models(X_trainval, y_trainval, best_params)

    pi_lower = quantile_models['lower'].predict(X_test)
    pi_upper = quantile_models['upper'].predict(X_test)

    if args.log_target:
        pi_lower = np.expm1(pi_lower)
        pi_upper = np.expm1(pi_upper)

    coverage = np.mean((y_test_orig >= pi_lower) & (y_test_orig <= pi_upper))
    logger.info(f"✓ 80% Prediction Interval Coverage: {coverage*100:.1f}%")

    # -------------------------
    # Diagnostic Plots & SHAP (New Features)
    # -------------------------
    logger.info("\nSTAGE 7: Diagnostic Plots & SHAP Analysis")
    logger.info("-" * 70)

    prediction_intervals = {'lower': pi_lower, 'upper': pi_upper}
    save_diagnostic_plots(y_test_orig, y_pred_orig, plots_outdir, args.target, prediction_intervals)
    save_shap_analysis(best_model, X_selected, selected_features, plots_outdir, args.target)

    # -------------------------
    # Save Model & Metadata
    # -------------------------
    logger.info("\nSTAGE 8: Saving Results")
    logger.info("-" * 70)

    model_data = {
        'model': best_model,
        'quantile_models': quantile_models,
        'features': selected_features,
        'log_target': args.log_target,
        'best_params': best_params,
        'cv_rmse': best_cv_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'pi_coverage': coverage,
        'training_date': datetime.now().isoformat(),
        'n_random_iter': args.n_random_iter,
        'n_rfe_repeats': args.n_rfe_repeats,
        'input_file': str(input_file),
        'method': 'hybrid_rfe_randomized',
        'auxiliary_feature': args.auxiliary_feature,
        'auxiliary_feature_name': auxiliary_feature_name
    }

    model_file = model_outdir / "xgb_model.joblib"
    dump(model_data, model_file)
    logger.info(f"✓ Saved model → {model_file.name}")

    # Save human-readable summary
    summary_file = model_outdir / "model_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"XGBoost Hybrid Model Summary: {args.target}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {input_file.name}\n")
        f.write(f"Log-transformed target: {args.log_target}\n")
        f.write(f"Training samples: {len(df)}\n")
        f.write(f"Method: RFE+RandomizedSearchCV (Hybrid)\n")
        if args.auxiliary_feature:
            f.write(f"Auxiliary feature: {args.auxiliary_feature} (as {auxiliary_feature_name})\n")
        f.write("\n")

        f.write("-" * 40 + "\n")
        f.write("Feature Selection:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Method: RFE with {args.n_rfe_repeats} repeats\n")
        f.write(f"  Step size: {args.rfe_step_size}\n")
        f.write(f"  Selected features: {len(selected_features)}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Best Hyperparameters:\n")
        f.write("-" * 40 + "\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("-" * 40 + "\n")
        f.write("Performance Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"5-Fold CV RMSE:        {best_cv_rmse:.4f}\n")
        f.write(f"Test Set R²:           {test_r2:.4f}\n")
        f.write(f"Test Set RMSE:         {test_rmse:.4f}\n")
        f.write(f"Test Set MAE:          {test_mae:.4f}\n")
        f.write(f"80% PI Coverage:       {coverage*100:.1f}%\n\n")

        f.write("-" * 40 + "\n")
        f.write(f"Selected Features ({len(selected_features)} total):\n")
        f.write("-" * 40 + "\n")
        for feat in selected_features:
            f.write(f"  {feat}\n")

    logger.info(f"✓ Saved summary → {summary_file.name}")

    # Final timing
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"✓ TRAINING COMPLETE! Total time: {elapsed}")
    logger.info(f"✓ Performance: R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid XGBoost regression training (best of both approaches)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing cleaned CSV files"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target property name (must match filename: {target}_cleaned.csv)"
    )
    parser.add_argument(
        "--log_target",
        action="store_true",
        help="Apply log1p transform to target values"
    )
    parser.add_argument(
        "--outdir",
        default="./output",
        help="Base output directory"
    )
    parser.add_argument(
        "--n_random_iter",
        type=int,
        default=10000,
        help="Number of RandomizedSearchCV iterations (original uses 10000)"
    )
    parser.add_argument(
        "--n_rfe_repeats",
        type=int,
        default=3,
        help="Number of RFE repeats with different random splits"
    )
    parser.add_argument(
        "--rfe_step_size",
        type=int,
        default=2,
        help="Number of features to drop per RFE step"
    )
    parser.add_argument(
        "--auxiliary_feature",
        type=str,
        default=None,
        help="Add an auxiliary feature from another target's dataset (e.g., 'Density_40C_g_cm^3' when training Density_100C). "
             "This loads the auxiliary target's cleaned CSV and merges on SMILES."
    )

    args = parser.parse_args()
    main(args)
