#!/usr/bin/env python3
"""
SHAP feature importance analysis for XGBoost (descriptors) FOM1 models.

Loads the trained XGBoost descriptor models from all 5 folds, computes SHAP
values on the held-out test set for each fold, and concatenates to give
SHAP values for every molecule in the dataset.

Usage:
    python shap_analysis.py                    # Both temperatures
    python shap_analysis.py --target FOM1_40   # Single temperature
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_DISPLAY = 20


def run_shap_for_target(target, base_dir, fig_base_dir):
    """Compute SHAP values and generate plots for one temperature target."""
    target_key = target.lower()
    data_dir = base_dir / target_key
    fig_dir = fig_base_dir / target_key
    fig_dir.mkdir(parents=True, exist_ok=True)

    temp_label = "40\u00b0C" if "40" in target else "100\u00b0C"

    # Load data
    logger.info("Loading data for %s...", target)
    df = pd.read_csv(data_dir / "fom1_dataset.csv")
    with open(data_dir / "descriptor_columns.json") as f:
        descriptor_columns = json.load(f)
    with open(data_dir / "fold_indices.json") as f:
        fold_indices = json.load(f)

    # The saved models may have been trained with a different descriptor set
    # than the regenerated descriptor_columns.json (e.g. different RDKit version
    # producing different constant-column filtering).  Detect the expected
    # feature count from the fold-0 scaler and truncate/match accordingly.
    probe_path = data_dir / "xgboost_descriptors_fold0_model.joblib"
    if probe_path.exists():
        probe = joblib.load(probe_path)
        expected_n = probe["scaler"].n_features_in_
        if expected_n != len(descriptor_columns):
            logger.warning(
                "descriptor_columns.json has %d columns but model expects %d — "
                "truncating to first %d columns",
                len(descriptor_columns), expected_n, expected_n,
            )
            descriptor_columns = descriptor_columns[:expected_n]

    X_all = df[descriptor_columns].values
    n_samples = X_all.shape[0]
    n_features = X_all.shape[1]
    logger.info("Dataset: %d molecules, %d descriptors", n_samples, n_features)

    # Allocate arrays for concatenated SHAP values
    shap_values_all = np.zeros((n_samples, n_features))
    X_scaled_all = np.zeros((n_samples, n_features))
    assigned = np.zeros(n_samples, dtype=bool)

    for fold_id in range(5):
        model_path = data_dir / f"xgboost_descriptors_fold{fold_id}_model.joblib"
        if not model_path.exists():
            logger.warning("Missing model: %s", model_path)
            continue

        logger.info("Fold %d: loading model and computing SHAP values...", fold_id)
        model_data = joblib.load(model_path)
        model = model_data["model"]
        scaler = model_data["scaler"]

        test_idx = fold_indices[str(fold_id)]["test"]
        X_test = X_all[test_idx]
        X_test_scaled = scaler.transform(X_test)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test_scaled)

        # Place into full arrays at the correct indices
        for i, idx in enumerate(test_idx):
            shap_values_all[idx] = sv[i]
            X_scaled_all[idx] = X_test_scaled[i]
            assigned[idx] = True

    if not assigned.all():
        n_missing = (~assigned).sum()
        logger.warning("%d molecules were not in any test fold", n_missing)
        # Keep only assigned molecules
        mask = assigned
        shap_values_all = shap_values_all[mask]
        X_scaled_all = X_scaled_all[mask]

    logger.info("SHAP values computed for %d molecules", shap_values_all.shape[0])

    # Save raw SHAP data
    np.savez(
        fig_dir / "shap_data.npz",
        shap_values=shap_values_all,
        X_scaled=X_scaled_all,
        feature_names=np.array(descriptor_columns, dtype=object),
    )
    logger.info("Saved: shap_data.npz")

    # Feature importance CSV
    mean_abs_shap = np.abs(shap_values_all).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": descriptor_columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(fig_dir / "shap_importance.csv", index=False)
    logger.info("Saved: shap_importance.csv")

    # Log top 10
    logger.info("Top 10 features by mean |SHAP| for %s:", target)
    for _, row in importance_df.head(10).iterrows():
        logger.info("  %-30s  %.4f", row["feature"], row["mean_abs_shap"])

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_all, X_scaled_all,
        feature_names=descriptor_columns,
        max_display=MAX_DISPLAY,
        show=False,
    )
    plt.title(f"SHAP Feature Importance — {target} ({temp_label})", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "shap_beeswarm.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved: shap_beeswarm.png")

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_all, X_scaled_all,
        feature_names=descriptor_columns,
        plot_type="bar",
        max_display=MAX_DISPLAY,
        show=False,
    )
    plt.title(f"Mean |SHAP| — {target} ({temp_label})", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "shap_bar.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved: shap_bar.png")

    return importance_df


def run_combined(fig_base_dir):
    """Cross-temperature SHAP importance comparison."""
    fig_dir = fig_base_dir / "combined"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_40 = fig_base_dir / "fom1_40" / "shap_importance.csv"
    csv_100 = fig_base_dir / "fom1_100" / "shap_importance.csv"

    if not csv_40.exists() or not csv_100.exists():
        logger.warning("Need both per-temperature SHAP results for combined plot. Skipping.")
        return

    df_40 = pd.read_csv(csv_40)
    df_100 = pd.read_csv(csv_100)

    merged = df_40.merge(df_100, on="feature", suffixes=("_40", "_100"))
    merged["avg_importance"] = (merged["mean_abs_shap_40"] + merged["mean_abs_shap_100"]) / 2
    merged = merged.sort_values("avg_importance", ascending=False)
    top = merged.head(15).sort_values("avg_importance", ascending=True)

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(top))
    bar_height = 0.35

    ax.barh(
        y_pos - bar_height / 2, top["mean_abs_shap_40"],
        bar_height, label="40\u00b0C", color="#1f77b4",
        edgecolor="black", linewidth=0.5, alpha=0.85,
    )
    ax.barh(
        y_pos + bar_height / 2, top["mean_abs_shap_100"],
        bar_height, label="100\u00b0C", color="#d62728",
        edgecolor="black", linewidth=0.5, alpha=0.85,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("SHAP Feature Importance — 40\u00b0C vs 100\u00b0C\n(XGBoost Descriptors)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    fig.savefig(fig_dir / "shap_cross_temperature.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "shap_cross_temperature.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: shap_cross_temperature.png")

    merged.to_csv(fig_dir / "shap_cross_temperature.csv", index=False)
    logger.info("Saved: shap_cross_temperature.csv")


def main():
    parser = argparse.ArgumentParser(
        description="SHAP analysis for XGBoost descriptor models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=str, default=None,
                        choices=["FOM1_40", "FOM1_100"],
                        help="Target to analyse. If omitted, runs both + combined.")
    parser.add_argument("--results_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--fig_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "figures"))
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    fig_base_dir = Path(args.fig_dir)

    if args.target is not None:
        run_shap_for_target(args.target, base_dir, fig_base_dir)
    else:
        for target in ["FOM1_40", "FOM1_100"]:
            run_shap_for_target(target, base_dir, fig_base_dir)
        run_combined(fig_base_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
