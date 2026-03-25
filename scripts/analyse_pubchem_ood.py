#!/usr/bin/env python
"""
OOD pre-analysis: check what fraction of PubChem CHO molecules
fall within the descriptor space of the NIST 8100 training data.

Samples 5,000 SMILES from the PubChem CHO corpus, computes Mordred+RDKit
descriptors, then evaluates Mahalanobis distances against each of 10
property models' training data.

Usage:
    python scripts/analyse_pubchem_ood.py [--n_sample 5000] [--output_dir scripts/ood_analysis]
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

# Suppress RDKit and Mordred warnings for cleaner output
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src/ to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from descriptors import (
    calc_descriptors,
    calc_mordred_descriptors,
    descriptor_funcs,
    descriptor_names,
    mordred_descriptor_names,
    rdkit_prefixed_names,
)
from wsga_helper import compute_mahalanobis_batch, compute_mahalanobis_params

# ── Paths ──
ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "training", "data")
PUBCHEM_CSV = os.path.join(ROOT, "data", "pubchem_cho_5_30ha.csv")

# All 10 property models (thermo + constraints)
THERMO_MODELS = ["density_40C", "viscosity_40C", "tc_40C", "cpsat_40C",
                 "beta_40C", "fom1_40C"]
CONSTRAINT_MODELS = ["bp", "mp", "fp", "dc"]
ALL_MODELS = THERMO_MODELS + CONSTRAINT_MODELS


def load_model_features(model_name):
    """Load feature list from a model joblib without loading the full XGBoost model."""
    import joblib
    model_path = os.path.join(MODEL_DIR, model_name, "model", "xgb_model.joblib")
    if not os.path.exists(model_path):
        print(f"  Warning: model not found at {model_path}")
        return None
    data = joblib.load(model_path)
    if "selected_features" in data:
        return data["selected_features"]
    elif "features" in data:
        return data["features"]
    return None


def compute_descriptors_batch(smiles_list):
    """Compute RDKit + Mordred descriptors for a list of SMILES. Returns a DataFrame."""
    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rdkit_vals = calc_descriptors(mol, descriptor_funcs)
        mordred_vals = calc_mordred_descriptors(mol)
        row = {"SMILES": smi}
        for name, val in zip(rdkit_prefixed_names, rdkit_vals):
            row[name] = val
        for name, val in zip(mordred_descriptor_names, mordred_vals):
            row[name] = val
        # Also add unprefixed rdkit names (for legacy models)
        for name, val in zip(descriptor_names, rdkit_vals):
            row[name] = val
        records.append(row)
    return pd.DataFrame(records)


def run_analysis(n_sample, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Sample PubChem SMILES ──
    print(f"Loading PubChem CHO corpus from {PUBCHEM_CSV}...")
    pubchem_df = pd.read_csv(PUBCHEM_CSV)
    total = len(pubchem_df)
    print(f"  Total molecules: {total:,}")

    sample = pubchem_df.sample(n=min(n_sample, total), random_state=42)
    pubchem_smiles = sample["SMILES"].tolist()
    print(f"  Sampled {len(pubchem_smiles):,} molecules")

    # ── 2. Compute descriptors ──
    print("\nComputing descriptors (this may take a few minutes)...")
    desc_df = compute_descriptors_batch(pubchem_smiles)
    n_valid = len(desc_df)
    print(f"  Valid molecules with descriptors: {n_valid:,} / {len(pubchem_smiles):,}")

    # ── 3. Load Mahalanobis params for each model ──
    print("\nComputing Mahalanobis parameters per model...")
    mahal_params = {}
    for model_name in ALL_MODELS:
        features = load_model_features(model_name)
        if features is None:
            print(f"  Skipping {model_name} (no model found)")
            continue
        params = compute_mahalanobis_params(model_name, features, DATA_DIR)
        if params is not None:
            mahal_params[model_name] = params

    if not mahal_params:
        print("ERROR: No Mahalanobis params could be computed. Check model/data paths.")
        sys.exit(1)

    # ── 4. Compute Mahalanobis distances for PubChem sample ──
    print(f"\nComputing Mahalanobis distances for {n_valid:,} PubChem molecules...")
    for model_name, params in mahal_params.items():
        compute_mahalanobis_batch(desc_df, params, model_name)

    # Also compute for NIST 8100 baseline (using descriptors from training data)
    print("\nComputing NIST 8100 baseline distances...")
    nist_desc_csv = os.path.join(DATA_DIR, "nist_8100", "descriptors_rdkit_mordred.csv")
    nist_desc_df = pd.read_csv(nist_desc_csv)
    # Add unprefixed rdkit names by stripping prefix
    for col in nist_desc_df.columns:
        if col.startswith("rdkit_"):
            nist_desc_df[col.replace("rdkit_", "", 1)] = nist_desc_df[col]
    for model_name, params in mahal_params.items():
        compute_mahalanobis_batch(nist_desc_df, params, model_name)

    # ── 5. Report ──
    print("\n" + "=" * 70)
    print("OOD ANALYSIS: PubChem CHO vs NIST 8100 Training Data")
    print("=" * 70)

    model_names_done = list(mahal_params.keys())
    ood_cols = [f"OOD_{m}" for m in model_names_done]
    mahal_cols = [f"Mahal_{m}" for m in model_names_done]

    # Per-model OOD rates
    print(f"\n{'Model':<18} {'PubChem OOD%':>14} {'NIST OOD%':>12} {'PubChem med D':>15} {'NIST med D':>12}")
    print("-" * 73)
    results_rows = []
    for model_name in model_names_done:
        ood_col = f"OOD_{model_name}"
        mahal_col = f"Mahal_{model_name}"
        pub_ood_rate = desc_df[ood_col].mean() * 100
        nist_ood_rate = nist_desc_df[ood_col].mean() * 100
        pub_median_d = desc_df[mahal_col].median()
        nist_median_d = nist_desc_df[mahal_col].median()
        print(f"{model_name:<18} {pub_ood_rate:>13.1f}% {nist_ood_rate:>11.1f}% {pub_median_d:>15.1f} {nist_median_d:>12.1f}")
        results_rows.append({
            "model": model_name,
            "pubchem_ood_pct": round(pub_ood_rate, 1),
            "nist_ood_pct": round(nist_ood_rate, 1),
            "pubchem_median_mahal": round(pub_median_d, 1),
            "nist_median_mahal": round(nist_median_d, 1),
        })

    # Aggregate OOD_any / OOD_count
    desc_df["OOD_any"] = desc_df[ood_cols].max(axis=1)
    desc_df["OOD_count"] = desc_df[ood_cols].sum(axis=1)
    ood_any_rate = desc_df["OOD_any"].mean() * 100
    mean_ood_count = desc_df["OOD_count"].mean()

    print(f"\n{'OOD on ANY model':<18} {ood_any_rate:>13.1f}%")
    print(f"{'Mean OOD count':<18} {mean_ood_count:>13.2f} / {len(model_names_done)} models")

    # OOD_count distribution
    print(f"\nOOD count distribution (out of {len(model_names_done)} models):")
    for k in range(len(model_names_done) + 1):
        n = (desc_df["OOD_count"] == k).sum()
        pct = n / len(desc_df) * 100
        bar = "#" * int(pct / 2)
        print(f"  {k:>2} models OOD: {n:>5} ({pct:>5.1f}%) {bar}")

    # Save results CSV
    results_csv = os.path.join(output_dir, "ood_rates_per_model.csv")
    pd.DataFrame(results_rows).to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")

    # Save full descriptor+OOD DataFrame
    ood_summary = desc_df[["SMILES"] + mahal_cols + ood_cols + ["OOD_any", "OOD_count"]]
    ood_summary.to_csv(os.path.join(output_dir, "pubchem_ood_details.csv"), index=False)

    # ── 6. Visualise ──
    print("\nGenerating figures...")

    # Figure 1: Per-model Mahalanobis distance distributions
    n_models = len(model_names_done)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), dpi=150)
    axes = axes.flatten()

    for i, model_name in enumerate(model_names_done):
        ax = axes[i]
        mahal_col = f"Mahal_{model_name}"
        threshold = np.sqrt(mahal_params[model_name]["threshold_sq"])

        # Clip for visualisation
        clip_max = max(desc_df[mahal_col].quantile(0.99),
                       nist_desc_df[mahal_col].quantile(0.99),
                       threshold * 1.2)

        pub_vals = desc_df[mahal_col].clip(upper=clip_max).dropna()
        nist_vals = nist_desc_df[mahal_col].clip(upper=clip_max).dropna()

        bins = np.linspace(0, clip_max, 60)
        ax.hist(nist_vals, bins=bins, alpha=0.6, density=True, label="NIST 8100",
                color="#2196F3", edgecolor="white", linewidth=0.3)
        ax.hist(pub_vals, bins=bins, alpha=0.6, density=True, label="PubChem CHO",
                color="#FF5722", edgecolor="white", linewidth=0.3)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label="OOD threshold")

        pub_ood = desc_df[f"OOD_{model_name}"].mean() * 100
        ax.set_xlabel(f"Mahalanobis distance / {model_name}")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=7)
        ax.text(0.97, 0.95, f"OOD: {pub_ood:.0f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="bold")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig1_path = os.path.join(output_dir, "mahalanobis_distributions.png")
    fig.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig1_path}")

    # Figure 2: OOD count histogram
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=150)
    counts = [int((desc_df["OOD_count"] == k).sum()) for k in range(len(model_names_done) + 1)]
    bars = ax2.bar(range(len(counts)), counts, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax2.set_xlabel(f"Number of models flagged OOD (out of {len(model_names_done)})")
    ax2.set_ylabel("Number of molecules")
    ax2.set_xticks(range(len(counts)))
    # Add count labels on bars
    for bar, c in zip(bars, counts):
        if c > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(desc_df) * 0.005,
                     str(c), ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, "ood_count_histogram.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {fig2_path}")

    # ── Decision gate ──
    print("\n" + "=" * 70)
    if ood_any_rate < 30:
        print(f"DECISION: OOD_any = {ood_any_rate:.1f}% (<30%) → PROCEED with sweep as-is")
    elif ood_any_rate < 60:
        print(f"DECISION: OOD_any = {ood_any_rate:.1f}% (30-60%) → PROCEED but note OOD risk; "
              "post-hoc filter OOD molecules from analysis")
    else:
        print(f"DECISION: OOD_any = {ood_any_rate:.1f}% (>60%) → CONSIDER adding OOD soft penalty "
              "to reward, or pre-filtering corpus to in-domain subset")
    print("=" * 70)

    return ood_any_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PubChem CHO OOD pre-analysis")
    parser.add_argument("--n_sample", type=int, default=5000,
                        help="Number of PubChem molecules to sample (default: 5000)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: scripts/ood_analysis)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "ood_analysis")

    run_analysis(args.n_sample, args.output_dir)
