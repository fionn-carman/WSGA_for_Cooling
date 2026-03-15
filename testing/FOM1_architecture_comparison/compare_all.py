#!/usr/bin/env python3
"""
Compare all FOM1 architecture results.

Two modes:
  1. Per-temperature (--target FOM1_40 or FOM1_100):
     Loads predictions from results/<target_lower>/, computes pooled + per-fold
     metrics, generates figures to figures/<target_lower>/.

  2. Combined (no --target):
     Loads both architecture_comparison.csv files and produces a grouped bar
     chart comparing 40C vs 100C R² per architecture in figures/combined/.

Usage:
    python compare_all.py --target FOM1_40
    python compare_all.py --target FOM1_100
    python compare_all.py  # combined cross-temperature view
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model display names (in order)
MODEL_NAMES = {
    "xgboost": "XGBoost (desc+fp)",
    "mlp_descriptors": "MLP (descriptors)",
    "mlp_fingerprints": "MLP (fingerprints)",
    "mlp_combined": "MLP (desc+fp)",
    "chemberta": "ChemBERTa-2",
    "ridge_embeddings": "Ridge (embeddings)",
    "bilstm": "BiLSTM",
    "ensemble_mlp": "Deep Ensemble MLP",
    "xgboost_descriptors": "XGBoost (descriptors)",
    "xgboost_fingerprints": "XGBoost (fingerprints)",
}

COLORS = {
    "xgboost": "#1f77b4",
    "mlp_descriptors": "#ff7f0e",
    "mlp_fingerprints": "#2ca02c",
    "mlp_combined": "#d62728",
    "chemberta": "#9467bd",
    "ridge_embeddings": "#8c564b",
    "bilstm": "#e377c2",
    "ensemble_mlp": "#7f7f7f",
    "xgboost_descriptors": "#17becf",
    "xgboost_fingerprints": "#bcbd22",
}


def load_results(data_dir, model_name, n_folds=5):
    """Load predictions and metrics for a model across all folds."""
    all_preds = []
    all_metrics = []
    has_all_folds = True

    for fold_id in range(n_folds):
        pred_path = data_dir / f"{model_name}_fold{fold_id}_predictions.csv"
        metrics_path = data_dir / f"{model_name}_fold{fold_id}_metrics.json"

        if not pred_path.exists():
            logger.warning("Missing predictions: %s", pred_path)
            has_all_folds = False
            continue

        pred_df = pd.read_csv(pred_path)
        all_preds.append(pred_df)

        if metrics_path.exists():
            with open(metrics_path) as f:
                all_metrics.append(json.load(f))

    if not all_preds:
        return None, None, None

    # Pooled predictions
    pooled = pd.concat(all_preds, ignore_index=True)
    return pooled, all_metrics, has_all_folds


def run_per_target(target, base_dir, fig_base_dir):
    """Run comparison for a single temperature target."""
    target_key = target.lower()
    data_dir = base_dir / target_key
    fig_dir = fig_base_dir / target_key
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logger.error("Results directory not found: %s", data_dir)
        return

    # Collect results
    results = []
    model_data = {}

    for model_key, display_name in MODEL_NAMES.items():
        pooled, metrics_list, has_all = load_results(data_dir, model_key)
        if pooled is None:
            logger.warning("Skipping %s — no predictions found", model_key)
            continue

        y_true = pooled["y_true"].values
        y_pred = pooled["y_pred"].values

        # Pooled metrics
        pooled_r2 = r2_score(y_true, y_pred)
        pooled_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        pooled_mae = mean_absolute_error(y_true, y_pred)

        # Per-fold metrics
        fold_r2s = [m["r2"] for m in metrics_list]
        fold_rmses = [m["rmse"] for m in metrics_list]
        fold_maes = [m["mae"] for m in metrics_list]
        total_time = sum(m.get("training_time_s", 0) for m in metrics_list)

        results.append({
            "model": model_key,
            "display_name": display_name,
            "pooled_r2": pooled_r2,
            "pooled_rmse": pooled_rmse,
            "pooled_mae": pooled_mae,
            "mean_r2": np.mean(fold_r2s),
            "std_r2": np.std(fold_r2s),
            "mean_rmse": np.mean(fold_rmses),
            "std_rmse": np.std(fold_rmses),
            "mean_mae": np.mean(fold_maes),
            "std_mae": np.std(fold_maes),
            "training_time_s": total_time,
        })

        model_data[model_key] = {
            "pooled": pooled,
            "fold_r2s": fold_r2s,
            "display_name": display_name,
        }

    if not results:
        logger.error("No results found for %s. Exiting.", target)
        return

    # Sort by pooled R²
    results.sort(key=lambda x: x["pooled_r2"], reverse=True)

    # Save comparison CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(data_dir / "architecture_comparison.csv", index=False)
    logger.info("Saved: %s", data_dir / "architecture_comparison.csv")

    # Temperature label for titles
    temp_label = "40\u00b0C" if "40" in target else "100\u00b0C"

    # Print ranked table
    logger.info("")
    logger.info("=" * 90)
    logger.info("ARCHITECTURE COMPARISON — %s — Ranked by Pooled R²", target)
    logger.info("=" * 90)
    logger.info("%-25s %8s %8s %8s %12s %10s", "Model", "Pool R²", "Pool RMSE", "Pool MAE", "R² (mean±std)", "Time (s)")
    logger.info("-" * 90)
    for r in results:
        logger.info("%-25s %8.4f %8.2f %8.2f %6.4f±%.4f %10.0f",
                     r["display_name"], r["pooled_r2"], r["pooled_rmse"], r["pooled_mae"],
                     r["mean_r2"], r["std_r2"], r["training_time_s"])
    logger.info("=" * 90)
    logger.info("WINNER: %s (R² = %.4f)", results[0]["display_name"], results[0]["pooled_r2"])

    # ---- FIGURES ----
    plt.style.use("seaborn-v0_8-paper")

    # 1. Horizontal bar chart of R² per architecture
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_results = list(reversed(results))
    names = [r["display_name"] for r in sorted_results]
    r2_vals = [r["pooled_r2"] for r in sorted_results]
    r2_stds = [r["std_r2"] for r in sorted_results]
    colors = [COLORS.get(r["model"], "#333333") for r in sorted_results]

    bars = ax.barh(names, r2_vals, xerr=r2_stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xlabel("Pooled Test R\u00b2", fontsize=12)
    ax.set_title(f"{target} Prediction — Architecture Comparison ({temp_label})", fontsize=14)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(fig_dir / "architecture_comparison_r2.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "architecture_comparison_r2.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: architecture_comparison_r2.png")

    # 2. Parity plots (2×4 grid)
    n_models = len(model_data)
    ncols = 4
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # Get global axis limits
    all_true = np.concatenate([model_data[r["model"]]["pooled"]["y_true"].values for r in results if r["model"] in model_data])
    all_pred = np.concatenate([model_data[r["model"]]["pooled"]["y_pred"].values for r in results if r["model"] in model_data])
    vmin = min(all_true.min(), all_pred.min()) - 5
    vmax = max(all_true.max(), all_pred.max()) + 5

    for i, r in enumerate(results):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        if r["model"] not in model_data:
            ax.set_visible(False)
            continue
        pooled = model_data[r["model"]]["pooled"]
        ax.scatter(pooled["y_true"], pooled["y_pred"], alpha=0.5, s=15, c=COLORS.get(r["model"], "#333"))
        ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{r['display_name']}\nR\u00b2={r['pooled_r2']:.3f}", fontsize=10)
        ax.set_aspect("equal")

    # Hide empty subplots
    for i in range(len(results), nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].set_visible(False)

    plt.suptitle(f"{target} Parity Plots — True vs Predicted ({temp_label})", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "parity_plots.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "parity_plots.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: parity_plots.png")

    # 3. Residual distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        if r["model"] not in model_data:
            continue
        pooled = model_data[r["model"]]["pooled"]
        residuals = pooled["y_true"] - pooled["y_pred"]
        ax.hist(residuals, bins=30, alpha=0.4, label=f"{r['display_name']} (R\u00b2={r['pooled_r2']:.3f})",
                color=COLORS.get(r["model"], "#333"), edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Residual (True - Predicted)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Residual Distributions — {target} ({temp_label})", fontsize=14)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "residual_distributions.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "residual_distributions.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: residual_distributions.png")

    # 4. Training curves for NN architectures
    nn_models = ["mlp_descriptors", "mlp_fingerprints", "mlp_combined", "chemberta", "bilstm", "ensemble_mlp"]
    nn_available = [m for m in nn_models if (data_dir / f"{m}_fold0_history.json").exists()]

    if nn_available:
        ncols_tc = min(3, len(nn_available))
        nrows_tc = (len(nn_available) + ncols_tc - 1) // ncols_tc
        fig, axes = plt.subplots(nrows_tc, ncols_tc, figsize=(6 * ncols_tc, 4 * nrows_tc))
        if nrows_tc == 1 and ncols_tc == 1:
            axes = np.array([[axes]])
        elif nrows_tc == 1:
            axes = axes.reshape(1, -1)
        elif ncols_tc == 1:
            axes = axes.reshape(-1, 1)

        for i, model_key in enumerate(nn_available):
            row, col = i // ncols_tc, i % ncols_tc
            ax = axes[row, col]
            for fold_id in range(5):
                hist_path = data_dir / f"{model_key}_fold{fold_id}_history.json"
                if not hist_path.exists():
                    continue
                with open(hist_path) as f:
                    hist = json.load(f)
                ax.plot(hist["train_loss"], alpha=0.4, color="blue", linewidth=0.8)
                ax.plot(hist["val_loss"], alpha=0.4, color="red", linewidth=0.8)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(MODEL_NAMES.get(model_key, model_key), fontsize=10)
            ax.legend(["Train", "Val"], loc="upper right", fontsize=8)

        for i in range(len(nn_available), nrows_tc * ncols_tc):
            row, col = i // ncols_tc, i % ncols_tc
            axes[row, col].set_visible(False)

        plt.suptitle(f"Training Curves — {target} ({temp_label})", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(fig_dir / "training_curves.png", dpi=300, bbox_inches="tight")
        fig.savefig(fig_dir / "training_curves.pdf", bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: training_curves.png")

    logger.info("All %s figures generated.", target)


def run_combined(base_dir, fig_base_dir):
    """Cross-temperature grouped bar chart comparing 40C vs 100C R² per architecture."""
    fig_dir = fig_base_dir / "combined"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load both comparison CSVs
    csv_40 = base_dir / "fom1_40" / "architecture_comparison.csv"
    csv_100 = base_dir / "fom1_100" / "architecture_comparison.csv"

    if not csv_40.exists() or not csv_100.exists():
        missing = []
        if not csv_40.exists():
            missing.append(str(csv_40))
        if not csv_100.exists():
            missing.append(str(csv_100))
        logger.error("Missing comparison CSV(s): %s", ", ".join(missing))
        logger.error("Run per-temperature comparisons first:")
        logger.error("  python compare_all.py --target FOM1_40")
        logger.error("  python compare_all.py --target FOM1_100")
        return

    df_40 = pd.read_csv(csv_40)
    df_100 = pd.read_csv(csv_100)

    # Merge on model key
    merged = df_40.merge(df_100, on="model", suffixes=("_40", "_100"))

    # Sort by average R² across both temperatures
    merged["avg_r2"] = (merged["pooled_r2_40"] + merged["pooled_r2_100"]) / 2
    merged = merged.sort_values("avg_r2", ascending=True)

    plt.style.use("seaborn-v0_8-paper")

    # Grouped horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(merged))
    bar_height = 0.35

    bars_40 = ax.barh(
        y_pos - bar_height / 2, merged["pooled_r2_40"],
        bar_height, xerr=merged["std_r2_40"],
        label="40\u00b0C", color="#1f77b4", edgecolor="black",
        linewidth=0.5, capsize=3, alpha=0.85,
    )
    bars_100 = ax.barh(
        y_pos + bar_height / 2, merged["pooled_r2_100"],
        bar_height, xerr=merged["std_r2_100"],
        label="100\u00b0C", color="#d62728", edgecolor="black",
        linewidth=0.5, capsize=3, alpha=0.85,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["display_name_40"])
    ax.set_xlabel("Pooled Test R\u00b2", fontsize=12)
    ax.set_title("FOM1 Architecture Comparison — 40\u00b0C vs 100\u00b0C", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(fig_dir / "cross_temperature_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "cross_temperature_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: cross_temperature_comparison.png")

    # Also save combined CSV
    merged.to_csv(fig_dir / "cross_temperature_comparison.csv", index=False)
    logger.info("Saved: cross_temperature_comparison.csv")

    # Print combined table
    logger.info("")
    logger.info("=" * 100)
    logger.info("CROSS-TEMPERATURE COMPARISON — Ranked by Average R²")
    logger.info("=" * 100)
    logger.info("%-25s %10s %10s %10s", "Model", "R² (40\u00b0C)", "R² (100\u00b0C)", "R² (avg)")
    logger.info("-" * 100)
    for _, r in merged.sort_values("avg_r2", ascending=False).iterrows():
        logger.info("%-25s %10.4f %10.4f %10.4f",
                     r["display_name_40"], r["pooled_r2_40"], r["pooled_r2_100"], r["avg_r2"])
    logger.info("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Compare all FOM1 architectures (per-temperature or combined)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=str, default=None,
                        choices=["FOM1_40", "FOM1_100"],
                        help="Target to compare. If omitted, runs combined cross-temperature view.")
    parser.add_argument("--results_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"),
                        help="Base results directory")
    parser.add_argument("--fig_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "figures"),
                        help="Base figures directory")
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    fig_base_dir = Path(args.fig_dir)

    if args.target is not None:
        run_per_target(args.target, base_dir, fig_base_dir)
    else:
        run_combined(base_dir, fig_base_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
