#!/usr/bin/env python3
"""
Diagnose RDKit descriptor mismatch between training data and current environment.

Compares descriptor values stored in fom1_dataset.csv (computed during data_preparation.py)
with freshly-computed descriptor values from the current RDKit installation.

If values differ significantly, the FOM1 direct models will produce wrong predictions
because the StandardScaler was calibrated to the training-time descriptor values.

Usage:
    conda activate mol-rl
    python diagnose_descriptor_mismatch.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

print(f"RDKit version: {rdkit.__version__}")
print(f"Number of descriptors in _descList: {len(Descriptors._descList)}")

# Load training data
data_dir = Path(__file__).resolve().parent / "results" / "fom1_40"
df = pd.read_csv(data_dir / "fom1_dataset.csv")
print(f"Training dataset: {len(df)} molecules")

# Load the model's descriptor columns (the 105 it was trained on)
import joblib
model_data = joblib.load(data_dir / "xgboost_descriptors_fold0_model.joblib")
if "descriptor_columns" in model_data:
    model_desc_cols = model_data["descriptor_columns"]
    print(f"Model descriptor columns (from joblib): {len(model_desc_cols)}")
else:
    with open(data_dir / "descriptor_columns.json") as f:
        model_desc_cols = json.load(f)
    n_expected = model_data["scaler"].n_features_in_
    model_desc_cols = model_desc_cols[:n_expected]
    print(f"Model descriptor columns (from JSON, truncated): {len(model_desc_cols)}")

# Check which model columns exist in current RDKit
current_desc_names = [d[0] for d in Descriptors._descList]
missing = [c for c in model_desc_cols if c not in current_desc_names]
if missing:
    print(f"\nWARNING: {len(missing)} model descriptors NOT in current RDKit: {missing}")
else:
    print(f"\nAll {len(model_desc_cols)} model descriptors exist in current RDKit")

# Compute fresh descriptors for first 50 molecules
desc_funcs = {d[0]: d[1] for d in Descriptors._descList}
n_check = min(50, len(df))
print(f"\nComparing descriptor values for first {n_check} molecules...")

mismatches = {}
for col in model_desc_cols:
    if col not in df.columns:
        print(f"  SKIP {col}: not in CSV")
        continue
    if col not in desc_funcs:
        print(f"  SKIP {col}: not in current RDKit")
        continue

    csv_vals = df[col].values[:n_check]
    fresh_vals = []
    for smi in df["SMILES"].values[:n_check]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fresh_vals.append(np.nan)
        else:
            try:
                fresh_vals.append(desc_funcs[col](mol))
            except Exception:
                fresh_vals.append(np.nan)
    fresh_vals = np.array(fresh_vals, dtype=float)

    # Replace NaN/Inf with 0 (same as training pipeline)
    csv_vals = np.nan_to_num(csv_vals, nan=0.0, posinf=0.0, neginf=0.0)
    fresh_vals = np.nan_to_num(fresh_vals, nan=0.0, posinf=0.0, neginf=0.0)

    # Compare
    if np.allclose(csv_vals, fresh_vals, rtol=1e-5, atol=1e-8):
        continue  # match
    else:
        max_diff = np.max(np.abs(csv_vals - fresh_vals))
        rel_diff = np.max(np.abs(csv_vals - fresh_vals) / (np.abs(csv_vals) + 1e-10))
        mismatches[col] = {
            "max_abs_diff": max_diff,
            "max_rel_diff": rel_diff,
            "csv_mean": np.mean(csv_vals),
            "fresh_mean": np.mean(fresh_vals),
            "csv_example": csv_vals[0],
            "fresh_example": fresh_vals[0],
        }

print(f"\n{'='*70}")
if not mismatches:
    print("ALL DESCRIPTORS MATCH — no RDKit version mismatch detected.")
    print("The issue may be elsewhere (model bias, feature engineering, etc.).")
else:
    print(f"MISMATCH DETECTED in {len(mismatches)}/{len(model_desc_cols)} descriptors!")
    print(f"{'='*70}")
    print(f"{'Descriptor':<30} {'Max Abs Diff':>12} {'CSV Mean':>12} {'Fresh Mean':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    for col, info in sorted(mismatches.items(), key=lambda x: -x[1]["max_abs_diff"]):
        print(f"{col:<30} {info['max_abs_diff']:>12.4f} {info['csv_mean']:>12.4f} {info['fresh_mean']:>12.4f}")

    print(f"\n{'='*70}")
    print("DIAGNOSIS: The descriptor values in fom1_dataset.csv were computed")
    print("with a different RDKit version than the current environment.")
    print("The StandardScaler in the model was calibrated to the old values,")
    print("so predictions at inference time will be incorrect.")
    print()
    print("FIX: Re-run data_preparation.py in the mol-rl environment, then")
    print("retrain the XGBoost descriptor models:")
    print()
    print("  conda activate mol-rl")
    print("  python data_preparation.py --target FOM1_40 --output_dir ./results")
    print("  python data_preparation.py --target FOM1_100 --output_dir ./results")
    print("  # Then retrain (submit run_all.sh or run manually for XGBoost descriptors)")
