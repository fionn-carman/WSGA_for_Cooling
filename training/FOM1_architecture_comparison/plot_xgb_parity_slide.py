#!/usr/bin/env python3
"""
Generate a clean side-by-side parity plot for XGBoost (descriptors)
at 40°C and 100°C — designed for a presentation slide showing
improvement over the old pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path


def load_pooled_predictions(results_dir, target_key):
    """Concatenate predictions from all 5 folds."""
    dfs = []
    for fold in range(5):
        path = results_dir / target_key / f"xgboost_descriptors_fold{fold}_predictions.csv"
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def main():
    base = Path(__file__).resolve().parent
    results_dir = base / "results"
    fig_dir = base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, target_key, temp_label in zip(
        axes, ["fom1_40", "fom1_100"], ["40°C", "100°C"]
    ):
        df = load_pooled_predictions(results_dir, target_key)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.45, s=18, c="#1f77b4",
                   edgecolors="none", zorder=2)

        # Perfect prediction line
        lo = min(y_true.min(), y_pred.min()) - 2
        hi = max(y_true.max(), y_pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.7, zorder=1)

        # Metrics box
        text = (f"R²   = {r2:.4f}\n"
                f"RMSE = {rmse:.4f}\n"
                f"MAE  = {mae:.4f}")
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=11, fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="grey", alpha=0.9))

        ax.set_title(temp_label, fontsize=16, fontweight="bold")
        ax.set_xlabel("FOM1 — Experimental", fontsize=12)
        ax.set_ylabel("FOM1 — Predicted", fontsize=12)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("XGBoost (Descriptors) — 5-Fold Cross-Validation",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ["png", "pdf"]:
        out = fig_dir / f"xgb_descriptors_parity.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved to {fig_dir}/xgb_descriptors_parity.png/.pdf")
    plt.close()


if __name__ == "__main__":
    main()
