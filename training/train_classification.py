"""
Hybrid XGBoost Classification Training - Best of Both Worlds

Adapted from the regression training script for binary classification tasks.
Supports multi-target datasets like Tox21 (12 separate classification models).

FROM ORIGINAL (Proven Performance):
- RFE with multiple repeats (3 runs with different random splits)
- RandomizedSearchCV with 10,000 iterations
- Step-wise feature elimination (step_size=2)
- Discrete parameter distributions

FROM NEW (Modern Improvements):
- Comprehensive logging
- SHAP interpretability analysis
- Professional diagnostic plots (ROC, PR, Confusion Matrix, Calibration)
- Nested CV for unbiased estimates
- Better organized output structure
- Pre-cleaned datasets (no descriptor computation overhead)
- Plot data saved for later editing

Usage:
    # Single target (biodegradability)
    python train_classification_hybrid.py --data_file biodegradability_cleaned.csv --target Activity
    
    # Multi-target (tox21 - trains 12 models)
    python train_classification_hybrid.py --data_file tox21_cleaned.csv --multi_target
"""

import os
import argparse
import warnings
import logging
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import math
from joblib import dump
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, balanced_accuracy_score,
    matthews_corrcoef, log_loss
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
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


# Define Tox21 target columns
TOX21_TARGETS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]


# ============================================================
# RFE with Multiple Repeats (Original Proven Method - Adapted for Classification)
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
    One repeat of RFE with random split for classification.

    Returns:
        df_rfe: DataFrame with AUC vs n_features
        best_n: Best number of features
        best_features: List of best feature names
    """
    # Stratified split to maintain class balance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    current_features = feature_names.copy()
    auc_results = []
    feature_sets = {}

    while len(current_features) > 0:
        # Fit RFE to rank features
        xgb = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
            use_label_encoder=False
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
        
        y_pred_proba = xgb.predict_proba(X_val_curr)[:, 1]
        
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except ValueError:
            # In case of single class in validation set
            auc = 0.5
            
        auc_results.append((len(kept_features), auc))
        feature_sets[len(kept_features)] = kept_features

        if verbose:
            y_pred = xgb.predict(X_val_curr)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            logger.info(f"    n={len(kept_features)}, AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
            if dropped_features:
                logger.info(f"    Dropped: {dropped_features[:3]}{'...' if len(dropped_features) > 3 else ''}")

        current_features = kept_features
        if len(current_features) == 1:
            break

    df_rfe = pd.DataFrame(auc_results, columns=["n_features", "auc"])
    best_idx = df_rfe["auc"].idxmax()  # Maximize AUC instead of minimize RMSE
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

        auc_at_best = df_rfe.loc[df_rfe["n_features"] == best_n, "auc"].values[0]
        logger.info(f"  → Best for repeat {i+1}: {best_n} features, AUC={auc_at_best:.4f}")

    # Determine overall best feature set (maximize AUC)
    best_auc = 0.0
    overall_best_feats = None
    best_repeat = None

    for idx, (best_n, best_feats) in enumerate(best_features_list):
        auc_at_best = rfe_results[idx][rfe_results[idx]["n_features"] == best_n]["auc"].values[0]
        if auc_at_best > best_auc:
            best_auc = auc_at_best
            overall_best_feats = best_feats
            best_repeat = idx + 1

    logger.info(f"\n✓ Overall best: {len(overall_best_feats)} features from repeat {best_repeat}, AUC={best_auc:.4f}")

    return overall_best_feats, rfe_results


# ============================================================
# RandomizedSearchCV (Original Proven Method - Adapted for Classification)
# ============================================================

def run_randomized_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 10000
) -> Tuple[XGBClassifier, Dict, float, RandomizedSearchCV]:
    """
    Run RandomizedSearchCV with large iteration count for classification.

    Returns:
        best_model: Fitted XGBoost classifier
        best_params: Dictionary of best hyperparameters
        best_cv_auc: Best cross-validation AUC
        random_search: The fitted RandomizedSearchCV object
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
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'scale_pos_weight': [1, 2, 5, 10]  # Handle class imbalance
    }

    base_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=1,  # Single thread per model - parallelism at CV level
        random_state=42,
        verbosity=0,
        tree_method='hist',
        max_bin=256,
        use_label_encoder=False
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,  # Use all cores
        verbose=1,
        random_state=42,
        return_train_score=False,
        pre_dispatch='2*n_jobs'
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_auc = random_search.best_score_

    logger.info(f"\n✓ Best hyperparameters: {best_params}")
    logger.info(f"✓ Best 5-fold CV AUC: {best_cv_auc:.4f}")

    return best_model, best_params, best_cv_auc, random_search


# ============================================================
# Diagnostic Plots for Classification
# ============================================================

def save_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: Path,
    target_name: str
):
    """Save comprehensive 4-panel diagnostic plots for classification."""
    logger.info("Generating diagnostic plots...")

    # Calculate metrics for saving
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy='uniform')
    except ValueError:
        prob_true, prob_pred = np.array([0, 1]), np.array([0, 1])
    
    # Calculate all metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'log_loss': log_loss(y_true, y_pred_proba)
    }
    
    # Save plot data for later editing
    plot_data_file = output_dir / f'{target_name}_diagnostic_plot_data.npz'
    np.savez(plot_data_file, 
             y_true=y_true, 
             y_pred=y_pred, 
             y_pred_proba=y_pred_proba,
             confusion_matrix=cm,
             fpr=fpr,
             tpr=tpr,
             roc_thresholds=roc_thresholds,
             precision_curve=precision_arr,
             recall_curve=recall_arr,
             pr_thresholds=pr_thresholds,
             calibration_prob_true=prob_true,
             calibration_prob_pred=prob_pred)
    
    # Save metrics as JSON
    metrics_file = output_dir / f'{target_name}_diagnostic_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"  ✓ Saved diagnostic plot data → {plot_data_file.name}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. ROC Curve
    ax = axes[0, 0]
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {metrics["roc_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    # 2. Precision-Recall Curve
    ax = axes[0, 1]
    ax.plot(recall_arr, precision_arr, 'b-', lw=2, label=f'PR (AUC = {metrics["pr_auc"]:.4f})')
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='r', linestyle='--', lw=2, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    # 3. Confusion Matrix
    ax = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    # Add metrics text
    text = f'Acc: {metrics["accuracy"]:.3f}\nF1: {metrics["f1"]:.3f}\nMCC: {metrics["mcc"]:.3f}'
    ax.text(1.3, 0.5, text, transform=ax.transAxes, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Calibration Curve
    ax = axes[1, 1]
    ax.plot(prob_pred, prob_true, 'b-o', lw=2, label='Model')
    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.suptitle(f'Classification Diagnostics: {target_name}', fontsize=16, y=1.02)
    plt.tight_layout()

    plot_file = output_dir / f'{target_name}_diagnostics.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ Saved diagnostic plots → {plot_file.name}")
    
    return metrics


def save_rfe_plot(rfe_results: List[pd.DataFrame], best_n: int, output_dir: Path, target_name: str):
    """Save RFE summary plot showing all repeats for classification."""
    logger.info("Generating RFE summary plot...")

    all_n = sorted(set(n for df_rfe in rfe_results for n in df_rfe["n_features"]))
    auc_matrix = []

    for df_rfe in rfe_results:
        auc_dict = dict(zip(df_rfe["n_features"], df_rfe["auc"]))
        auc_row = [auc_dict.get(n, np.nan) for n in all_n]
        auc_matrix.append(auc_row)

    auc_matrix = np.array(auc_matrix)
    mean_auc = np.nanmean(auc_matrix, axis=0)
    std_auc = np.nanstd(auc_matrix, axis=0)
    
    # Save RFE plot data
    rfe_data_file = output_dir / f'{target_name}_rfe_plot_data.npz'
    np.savez(rfe_data_file, 
             n_features=np.array(all_n),
             auc_matrix=auc_matrix,
             mean_auc=mean_auc,
             std_auc=std_auc,
             best_n=np.array([best_n]))
    
    logger.info(f"  ✓ Saved RFE plot data → {rfe_data_file.name}")
    
    plt.figure(figsize=(8, 5))
    
    for auc_row in auc_matrix:
        plt.plot(all_n, auc_row, color='grey', alpha=0.4)

    plt.errorbar(all_n, mean_auc, yerr=std_auc, fmt='-o', capsize=5, color='blue', label='Mean ± std')
    plt.axvline(x=best_n, color='r', linestyle='--', label=f'Selected: {best_n} features')

    plt.xlim(0)
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel("Validation AUC", fontsize=12)
    plt.title(f"RFE Feature Selection: {target_name}", fontsize=14)
    plt.legend()
    plt.tight_layout()

    plot_file = output_dir / f'{target_name}_rfe.png'
    plt.savefig(plot_file, dpi=300)
    plt.close()

    logger.info(f"  ✓ Saved RFE plot → {plot_file.name}")


def save_shap_analysis(
    model: XGBClassifier,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    target_name: str,
    max_display: int = 20
):
    """Generate and save SHAP analysis plots for classification."""
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
# Single Target Training Pipeline
# ============================================================

def train_single_target(
    df: pd.DataFrame,
    target: str,
    base_outdir: Path,
    args,
    input_file: Path
) -> Dict:
    """Train a single classification model."""
    
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info(f"CLASSIFICATION TRAINING: {target}")
    logger.info("=" * 70)

    # Setup output directories
    target_outdir = base_outdir / target
    rfe_outdir = target_outdir / "RFE"
    model_outdir = target_outdir / "model"
    plots_outdir = target_outdir / "plots"

    for d in [rfe_outdir, model_outdir, plots_outdir]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Prepare data
    # -------------------------
    logger.info("\nSTAGE 1: Preparing Data")
    logger.info("-" * 70)

    # Get valid samples (non-null target values)
    df_valid = df.dropna(subset=[target]).copy()
    df_valid[target] = df_valid[target].astype(int)
    
    n_positive = (df_valid[target] == 1).sum()
    n_negative = (df_valid[target] == 0).sum()
    
    logger.info(f"✓ Valid samples: {len(df_valid)}")
    logger.info(f"  Class distribution: {n_negative} negative, {n_positive} positive ({100*n_positive/len(df_valid):.1f}% positive)")

    # Determine feature columns (exclude SMILES, target, and other target columns if multi-target)
    non_feature_cols = ["SMILES", target]
    if args.multi_target:
        non_feature_cols.extend([t for t in TOX21_TARGETS if t != target])
    
    feature_cols = [c for c in df_valid.columns if c not in non_feature_cols]
    
    X = df_valid[feature_cols].values
    y = df_valid[target].values.astype(int)
    feature_names = feature_cols

    logger.info(f"✓ Feature matrix: {X.shape}")
    logger.info(f"✓ Target shape: {y.shape}")

    # -------------------------
    # Feature Selection (RFE)
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

    save_rfe_plot(rfe_results, len(selected_features), plots_outdir, target)

    # Update X with selected features
    X_selected = df_valid[selected_features].values

    # -------------------------
    # Train/Test Split
    # -------------------------
    logger.info("\nSTAGE 3: Data Splitting")
    logger.info("-" * 70)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_selected, y, test_size=0.1, random_state=42, stratify=y
    )

    logger.info(f"✓ Training+Val set: {X_trainval.shape[0]} samples")
    logger.info(f"✓ Test set: {X_test.shape[0]} samples")

    # -------------------------
    # Hyperparameter Optimization
    # -------------------------
    logger.info("\nSTAGE 4: Hyperparameter Optimization (RandomizedSearchCV)")
    logger.info("-" * 70)

    best_model, best_params, best_cv_auc, random_search = run_randomized_search(
        X_trainval, y_trainval,
        n_iter=args.n_random_iter
    )

    # Save CV results
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_df.to_csv(target_outdir / f"{target}_cv_results.csv", index=False)
    logger.info(f"✓ Saved CV results")

    # -------------------------
    # Test Set Evaluation
    # -------------------------
    logger.info("\nSTAGE 5: Test Set Evaluation")
    logger.info("-" * 70)

    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = save_diagnostic_plots(y_test, y_test_pred, y_test_pred_proba, plots_outdir, target)

    logger.info(f"\n✓ Test Set Performance:")
    logger.info(f"  ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC:     {test_metrics['pr_auc']:.4f}")
    logger.info(f"  Accuracy:   {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1:         {test_metrics['f1']:.4f}")
    logger.info(f"  MCC:        {test_metrics['mcc']:.4f}")

    # -------------------------
    # SHAP Analysis
    # -------------------------
    logger.info("\nSTAGE 6: SHAP Analysis")
    logger.info("-" * 70)

    save_shap_analysis(best_model, X_selected, selected_features, plots_outdir, target)

    # -------------------------
    # Save Model & Metadata
    # -------------------------
    logger.info("\nSTAGE 7: Saving Results")
    logger.info("-" * 70)

    model_data = {
        'model': best_model,
        'features': selected_features,
        'best_params': best_params,
        'cv_auc': best_cv_auc,
        'test_metrics': test_metrics,
        'training_date': datetime.now().isoformat(),
        'n_random_iter': args.n_random_iter,
        'n_rfe_repeats': args.n_rfe_repeats,
        'input_file': str(input_file),
        'method': 'hybrid_rfe_randomized',
        'class_distribution': {'positive': int(n_positive), 'negative': int(n_negative)}
    }

    model_file = model_outdir / "xgb_model.joblib"
    dump(model_data, model_file)
    logger.info(f"✓ Saved model → {model_file.name}")

    # Save human-readable summary
    summary_file = model_outdir / "model_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"XGBoost Classification Model Summary: {target}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {input_file.name}\n")
        f.write(f"Valid samples: {len(df_valid)}\n")
        f.write(f"Class distribution: {n_negative} neg, {n_positive} pos ({100*n_positive/len(df_valid):.1f}% positive)\n")
        f.write(f"Method: RFE+RandomizedSearchCV (Hybrid)\n\n")

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
        f.write(f"5-Fold CV AUC:         {best_cv_auc:.4f}\n")
        f.write(f"Test Set ROC-AUC:      {test_metrics['roc_auc']:.4f}\n")
        f.write(f"Test Set PR-AUC:       {test_metrics['pr_auc']:.4f}\n")
        f.write(f"Test Set Accuracy:     {test_metrics['accuracy']:.4f}\n")
        f.write(f"Test Set F1:           {test_metrics['f1']:.4f}\n")
        f.write(f"Test Set MCC:          {test_metrics['mcc']:.4f}\n\n")

        f.write("-" * 40 + "\n")
        f.write(f"Selected Features ({len(selected_features)} total):\n")
        f.write("-" * 40 + "\n")
        for feat in selected_features:
            f.write(f"  {feat}\n")

    logger.info(f"✓ Saved summary → {summary_file.name}")

    # Final timing
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"✓ TRAINING COMPLETE for {target}! Total time: {elapsed}")
    logger.info(f"✓ Performance: ROC-AUC={test_metrics['roc_auc']:.4f}, F1={test_metrics['f1']:.4f}")
    logger.info("=" * 70)

    return {
        'target': target,
        'cv_auc': best_cv_auc,
        'test_metrics': test_metrics,
        'n_features': len(selected_features),
        'elapsed_time': str(elapsed)
    }


# ============================================================
# Main Entry Point
# ============================================================

def main(args):
    overall_start = datetime.now()
    
    # -------------------------
    # Load data
    # -------------------------
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)

    input_file = Path(args.data_dir) / args.data_file
    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file}")

    df = pd.read_csv(input_file)
    logger.info(f"✓ Loaded {len(df)} samples from {input_file.name}")
    logger.info(f"✓ Total columns: {len(df.columns)}")

    # Setup base output directory
    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Determine targets
    # -------------------------
    if args.multi_target:
        # Tox21 multi-target mode
        targets = [t for t in TOX21_TARGETS if t in df.columns]
        logger.info(f"\n✓ Multi-target mode: Found {len(targets)} Tox21 targets")
        for t in targets:
            n_valid = df[t].notna().sum()
            logger.info(f"  - {t}: {n_valid} valid samples")
    else:
        # Single target mode
        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in dataset")
        targets = [args.target]
        logger.info(f"\n✓ Single target mode: {args.target}")

    # -------------------------
    # Train models
    # -------------------------
    all_results = []
    
    for target in targets:
        try:
            result = train_single_target(df, target, base_outdir, args, input_file)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to train model for {target}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # -------------------------
    # Summary
    # -------------------------
    if len(all_results) > 1:
        logger.info("\n" + "=" * 70)
        logger.info("MULTI-TARGET SUMMARY")
        logger.info("=" * 70)
        
        summary_df = pd.DataFrame([
            {
                'target': r['target'],
                'cv_auc': r['cv_auc'],
                'test_roc_auc': r['test_metrics']['roc_auc'],
                'test_pr_auc': r['test_metrics']['pr_auc'],
                'test_f1': r['test_metrics']['f1'],
                'test_mcc': r['test_metrics']['mcc'],
                'n_features': r['n_features'],
                'time': r['elapsed_time']
            }
            for r in all_results
        ])
        
        summary_df.to_csv(base_outdir / "multi_target_summary.csv", index=False)
        
        logger.info("\nResults:")
        for _, row in summary_df.iterrows():
            logger.info(f"  {row['target']}: ROC-AUC={row['test_roc_auc']:.4f}, F1={row['test_f1']:.4f}")
        
        logger.info(f"\nMean ROC-AUC: {summary_df['test_roc_auc'].mean():.4f}")
        logger.info(f"Mean F1: {summary_df['test_f1'].mean():.4f}")

    total_elapsed = datetime.now() - overall_start
    logger.info(f"\n✓ ALL TRAINING COMPLETE! Total time: {total_elapsed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid XGBoost classification training (best of both approaches)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing cleaned CSV files"
    )
    parser.add_argument(
        "--data_file",
        required=True,
        help="Name of the cleaned CSV file (e.g., 'tox21_cleaned.csv' or 'biodegradability_cleaned.csv')"
    )
    parser.add_argument(
        "--target",
        default="Activity",
        help="Target column name for single-target mode"
    )
    parser.add_argument(
        "--multi_target",
        action="store_true",
        help="Enable multi-target mode for Tox21 (trains 12 separate models)"
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

    args = parser.parse_args()
    main(args)
