#!/usr/bin/env python3
"""
Analyse 5-fold RFE training results.

Generates:
1. Comparison table (current single-model vs 5-fold ensemble)
2. Per-fold R2 bar charts for all properties
3. KV feature consistency analysis (Jaccard similarity + core features)
4. DC fold variance analysis
5. Feature count distribution across folds

Saves figures to ~/Desktop/WSGA_Notes/Figures/5fold_rfe_analysis/
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path.home() / "Desktop/WSGA_Notes/Figures/5fold_rfe_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(__file__).parent.parent / "training/output_5fold"

# Current single-model R2 values (from plan document)
CURRENT_R2 = {
    "Density_40C_g_cm^3": 0.804,
    "Density_100C_g_cm^3": 0.997,  # chained
    "Thermal_Conductivity_40C": 0.868,
    "Thermal_Conductivity_100C": 0.865,
    "Heat_Capacity_Constant_Pressure_40C_J_K_Mol": 0.984,
    "Heat_Capacity_Constant_Pressure_100C_J_K_Mol": 0.991,
    "Kinematic_Viscosity_40C": 0.735,
    "Kinematic_Viscosity_100C": 0.777,
    "BP-Measured": 0.972,
    "MP-Measured": 0.899,
    "flashpoint": 0.922,
    "DC_exp": 0.908,
}

# Short display names
SHORT_NAMES = {
    "Density_40C_g_cm^3": "Density 40°C",
    "Density_100C_g_cm^3": "Density 100°C",
    "Thermal_Conductivity_40C": "TC 40°C",
    "Thermal_Conductivity_100C": "TC 100°C",
    "Heat_Capacity_Constant_Pressure_40C_J_K_Mol": "Cp 40°C",
    "Heat_Capacity_Constant_Pressure_100C_J_K_Mol": "Cp 100°C",
    "Kinematic_Viscosity_40C": "KV 40°C",
    "Kinematic_Viscosity_100C": "KV 100°C",
    "BP-Measured": "Boiling Pt",
    "MP-Measured": "Melting Pt",
    "flashpoint": "Flash Pt",
    "DC_exp": "Dielectric",
}

PROPERTY_ORDER = [
    "Density_40C_g_cm^3", "Density_100C_g_cm^3",
    "Thermal_Conductivity_40C", "Thermal_Conductivity_100C",
    "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",
    "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",
    "Kinematic_Viscosity_40C", "Kinematic_Viscosity_100C",
    "BP-Measured", "MP-Measured", "flashpoint", "DC_exp",
]


def load_results():
    """Load all available 5-fold results."""
    results = {}
    for prop_dir in sorted(RESULTS_DIR.iterdir()):
        if not prop_dir.is_dir():
            continue
        prop = prop_dir.name

        # Try summary.json first (complete properties)
        summary_path = prop_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                results[prop] = json.load(f)
        else:
            # Load individual fold metrics
            fold_metrics = []
            for fold_file in sorted(prop_dir.glob("xgboost_descriptors_fold*_metrics.json")):
                with open(fold_file) as f:
                    fold_metrics.append(json.load(f))
            if fold_metrics:
                r2s = [m["r2"] for m in fold_metrics]
                results[prop] = {
                    "target": prop,
                    "mean_r2": np.mean(r2s),
                    "std_r2": np.std(r2s),
                    "mean_rmse": np.mean([m["rmse"] for m in fold_metrics]),
                    "mean_mae": np.mean([m["mae"] for m in fold_metrics]),
                    "mean_n_features": np.mean([m["n_features"] for m in fold_metrics]),
                    "fold_metrics": fold_metrics,
                    "n_folds_complete": len(fold_metrics),
                }
    return results


def load_features(prop):
    """Load selected features for each fold of a property."""
    prop_dir = RESULTS_DIR / prop
    features = {}
    for i in range(5):
        feat_file = prop_dir / f"fold{i}_selected_features.txt"
        if feat_file.exists():
            features[i] = set(feat_file.read_text().strip().split("\n"))
    return features


# ============================================================
# Figure 1: Comparison table as figure
# ============================================================

def plot_comparison_table(results):
    """Bar chart comparing current single-model R2 vs 5-fold mean R2."""
    props = [p for p in PROPERTY_ORDER if p in results]
    names = [SHORT_NAMES[p] for p in props]
    current = [CURRENT_R2.get(p, np.nan) for p in props]
    fold_mean = [results[p]["mean_r2"] for p in props]
    fold_std = [results[p]["std_r2"] for p in props]
    n_folds = [len(results[p]["fold_metrics"]) for p in props]

    x = np.arange(len(props))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, current, width, label="Current single model",
                   color="#4C72B0", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, fold_mean, width, yerr=fold_std,
                   label="5-fold mean ± std", color="#DD8452", alpha=0.85,
                   edgecolor="white", capsize=3)

    # Mark incomplete properties
    for i, nf in enumerate(n_folds):
        if nf < 5:
            ax.text(x[i] + width/2, fold_mean[i] + fold_std[i] + 0.01,
                    f"{nf}/5", ha="center", va="bottom", fontsize=8, color="red")

    # Mark Density 100C as chained
    density_100_idx = props.index("Density_100C_g_cm^3") if "Density_100C_g_cm^3" in props else None
    if density_100_idx is not None:
        ax.annotate("chained*", xy=(x[density_100_idx] - width/2, current[density_100_idx]),
                    xytext=(x[density_100_idx] - width/2, current[density_100_idx] + 0.015),
                    ha="center", fontsize=7, color="#4C72B0", fontweight="bold")

    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Current Single Model vs 5-Fold Ensemble R²", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_ylim(0.5, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_current_vs_5fold.png", dpi=150)
    plt.close()
    print(f"Saved: comparison_current_vs_5fold.png")


# ============================================================
# Figure 2: Per-fold R2 breakdown
# ============================================================

def plot_per_fold_r2(results):
    """Per-fold R2 for all properties — highlights variance."""
    props = [p for p in PROPERTY_ORDER if p in results]
    n = len(props)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=False)
    axes = axes.flatten()

    for i, prop in enumerate(props):
        ax = axes[i]
        fm = results[prop]["fold_metrics"]
        r2s = [m["r2"] for m in fm]
        folds = list(range(len(r2s)))
        mean_r2 = np.mean(r2s)

        colors = ["#DD8452" if r < mean_r2 - 0.05 else "#4C72B0" for r in r2s]
        ax.bar(folds, r2s, color=colors, edgecolor="white", alpha=0.85)
        ax.axhline(y=mean_r2, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axhline(y=CURRENT_R2.get(prop, np.nan), color="green", linestyle=":",
                   linewidth=1.2, alpha=0.7)

        ax.set_title(SHORT_NAMES[prop], fontsize=10, fontweight="bold")
        ax.set_xlabel("Fold", fontsize=8)
        ax.set_xticks(folds)
        r2_range = max(r2s) - min(r2s)
        y_min = min(r2s) - max(0.03, r2_range * 0.3)
        y_max = max(max(r2s), CURRENT_R2.get(prop, 0)) + 0.03
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("R²", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Annotate range
        ax.text(0.98, 0.02, f"range: {max(r2s)-min(r2s):.3f}\nstd: {np.std(r2s):.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # Remove unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Per-Fold R² Breakdown (red dashed = 5-fold mean, green dotted = current model)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_fold_r2_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: per_fold_r2_breakdown.png")


# ============================================================
# Figure 3: KV feature consistency (Jaccard similarity)
# ============================================================

def plot_kv_feature_analysis(results):
    """Analyse feature consistency for KV models across folds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, prop in enumerate(["Kinematic_Viscosity_40C", "Kinematic_Viscosity_100C"]):
        features = load_features(prop)
        if len(features) < 2:
            continue

        # Jaccard similarity matrix
        n_folds = len(features)
        jaccard = np.zeros((n_folds, n_folds))
        for i in range(n_folds):
            for j in range(n_folds):
                intersection = len(features[i] & features[j])
                union = len(features[i] | features[j])
                jaccard[i, j] = intersection / union if union > 0 else 0

        ax = axes[row, 0]
        im = ax.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(f"{SHORT_NAMES[prop]} — Feature Jaccard Similarity", fontsize=11, fontweight="bold")
        ax.set_xticks(range(n_folds))
        ax.set_yticks(range(n_folds))
        ax.set_xticklabels([f"Fold {i}" for i in range(n_folds)])
        ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
        for i in range(n_folds):
            for j in range(n_folds):
                ax.text(j, i, f"{jaccard[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if jaccard[i,j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Feature frequency histogram
        all_feats = Counter()
        for fold_feats in features.values():
            for f in fold_feats:
                all_feats[f] += 1

        ax2 = axes[row, 1]
        freq_counts = Counter(all_feats.values())
        bars_x = sorted(freq_counts.keys())
        bars_y = [freq_counts[x] for x in bars_x]
        colors = ["#DD8452" if x < 3 else "#55A868" if x >= 4 else "#4C72B0" for x in bars_x]
        ax2.bar(bars_x, bars_y, color=colors, edgecolor="white")
        ax2.set_xlabel("Number of folds feature appears in", fontsize=10)
        ax2.set_ylabel("Number of features", fontsize=10)
        ax2.set_title(f"{SHORT_NAMES[prop]} — Feature Selection Frequency", fontsize=11, fontweight="bold")
        ax2.set_xticks(bars_x)

        # Annotate core features (in 4+ folds)
        core = [f for f, c in all_feats.items() if c >= 4]
        core_text = "\n".join(sorted(core)[:15])
        if len(core) > 15:
            core_text += f"\n... +{len(core)-15} more"
        ax2.text(0.98, 0.98, f"Core features (≥4 folds):\n{core_text}",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
                 family="monospace")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "kv_feature_consistency.png", dpi=150)
    plt.close()
    print(f"Saved: kv_feature_consistency.png")

    # Print core features for console output
    for prop in ["Kinematic_Viscosity_40C", "Kinematic_Viscosity_100C"]:
        features = load_features(prop)
        all_feats = Counter()
        for fold_feats in features.values():
            for f in fold_feats:
                all_feats[f] += 1
        core = sorted([(f, c) for f, c in all_feats.items() if c >= 4], key=lambda x: -x[1])
        print(f"\n{SHORT_NAMES[prop]} core features (≥4/5 folds):")
        for f, c in core:
            print(f"  {c}/5  {f}")


# ============================================================
# Figure 4: Feature count and R2 correlation
# ============================================================

def plot_features_vs_r2(results):
    """Scatter: n_features vs R2 per fold, coloured by property."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.tab20
    props = [p for p in PROPERTY_ORDER if p in results]

    for i, prop in enumerate(props):
        fm = results[prop]["fold_metrics"]
        n_feats = [m["n_features"] for m in fm]
        r2s = [m["r2"] for m in fm]
        ax.scatter(n_feats, r2s, color=cmap(i / len(props)), label=SHORT_NAMES[prop],
                   s=50, alpha=0.8, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Number of RFE-selected features", fontsize=12)
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_title("Feature Count vs Test R² Across All Folds", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "features_vs_r2.png", dpi=150)
    plt.close()
    print(f"Saved: features_vs_r2.png")


# ============================================================
# Figure 5: Stability ranking (std/mean R2)
# ============================================================

def plot_stability_ranking(results):
    """Coefficient of variation of R2 — lower = more stable."""
    props = [p for p in PROPERTY_ORDER if p in results and len(results[p]["fold_metrics"]) == 5]
    names = [SHORT_NAMES[p] for p in props]
    cv = [results[p]["std_r2"] / results[p]["mean_r2"] for p in props]
    r2_range = [max(m["r2"] for m in results[p]["fold_metrics"]) -
                min(m["r2"] for m in results[p]["fold_metrics"]) for p in props]

    # Sort by CV
    order = np.argsort(cv)[::-1]
    names_sorted = [names[i] for i in order]
    cv_sorted = [cv[i] for i in order]
    range_sorted = [r2_range[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#C44E52" if c > 0.05 else "#DD8452" if c > 0.02 else "#55A868" for c in cv_sorted]
    ax1.barh(range(len(names_sorted)), cv_sorted, color=colors, edgecolor="white")
    ax1.set_yticks(range(len(names_sorted)))
    ax1.set_yticklabels(names_sorted, fontsize=10)
    ax1.set_xlabel("Coefficient of Variation (σ/μ of R²)", fontsize=11)
    ax1.set_title("Model Stability Ranking\n(higher = more unstable)", fontsize=12, fontweight="bold")
    ax1.axvline(x=0.05, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.grid(axis="x", alpha=0.3)

    colors2 = ["#C44E52" if r > 0.15 else "#DD8452" if r > 0.05 else "#55A868" for r in range_sorted]
    ax2.barh(range(len(names_sorted)), range_sorted, color=colors2, edgecolor="white")
    ax2.set_yticks(range(len(names_sorted)))
    ax2.set_yticklabels(names_sorted, fontsize=10)
    ax2.set_xlabel("R² Range (max - min across folds)", fontsize=11)
    ax2.set_title("R² Range Across Folds\n(higher = more variable)", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "stability_ranking.png", dpi=150)
    plt.close()
    print(f"Saved: stability_ranking.png")


# ============================================================
# Console summary table
# ============================================================

def print_summary_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 110)
    print(f"{'Property':<20} {'Current R²':>10} {'5-Fold R²':>10} {'± Std':>8} "
          f"{'Δ':>8} {'Range':>8} {'Folds':>6} {'Features':>10}")
    print("-" * 110)

    for prop in PROPERTY_ORDER:
        if prop not in results:
            continue
        r = results[prop]
        curr = CURRENT_R2.get(prop, float("nan"))
        mean = r["mean_r2"]
        std = r["std_r2"]
        delta = mean - curr
        fm = r["fold_metrics"]
        r2s = [m["r2"] for m in fm]
        r2_range = max(r2s) - min(r2s)
        n_folds = len(fm)
        mean_feats = np.mean([m["n_features"] for m in fm])

        status = "✓" if n_folds == 5 else f"{n_folds}/5"
        delta_str = f"{delta:+.3f}"

        print(f"{SHORT_NAMES[prop]:<20} {curr:>10.3f} {mean:>10.3f} {std:>8.3f} "
              f"{delta_str:>8} {r2_range:>8.3f} {status:>6} {mean_feats:>10.1f}")

    print("=" * 110)
    print("\nNotes:")
    print("  - Density 100°C current R²=0.997 uses Density_40C as auxiliary input (chained)")
    print("  - Standalone 5-fold R²=0.820 is the fair comparison")
    print("  - KV models show highest instability (R² range >0.2)")
    print("  - Negative Δ means 5-fold is lower than current (expected for honest CV)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = load_results()
    print(f"Loaded results for {len(results)} properties")

    print_summary_table(results)
    plot_comparison_table(results)
    plot_per_fold_r2(results)
    plot_kv_feature_analysis(results)
    plot_features_vs_r2(results)
    plot_stability_ranking(results)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
