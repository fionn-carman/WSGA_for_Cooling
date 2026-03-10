#!/usr/bin/env python3
"""
Debug: trace descriptor values through the full evaluate_molecules pipeline
for a known molecule (COCCOCCOCCOC) to find where corruption occurs.

Usage:
    conda activate mol-rl
    python debug_wsga_fom1.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

# Add src to path
src_dir = str(Path(__file__).resolve().parent.parent.parent / "src")
sys.path.insert(0, src_dir)

from descriptors import descriptor_funcs, descriptor_names, calc_descriptors

print(f"descriptor_names: {len(descriptor_names)}")
print(f"Unique descriptor_names: {len(set(descriptor_names))}")

# Check for duplicates in mol-rl's descriptor list
dupes = [n for i, n in enumerate(descriptor_names) if n in descriptor_names[:i]]
if dupes:
    print(f"*** DUPLICATE DESCRIPTOR NAMES: {dupes} ***")
    print("This would cause pd.DataFrame to create duplicate columns!")
    print("df[col_name] would return a DataFrame instead of a Series!")
else:
    print("No duplicate names")

# Load FOM1 direct model
base_dir = Path(__file__).resolve().parent / "results" / "fom1_40"
mdata = joblib.load(base_dir / "xgboost_descriptors_fold0_model.joblib")
model = mdata["model"]
scaler = mdata["scaler"]
desc_cols = mdata["descriptor_columns"]

print(f"\nModel expects {len(desc_cols)} features")

# Test molecule
smi = "COCCOCCOCCOC"
mol = Chem.MolFromSmiles(smi)
print(f"\nTest molecule: {smi}")

# Step 1: Compute descriptors (as descriptors.py does)
vals = calc_descriptors(mol, descriptor_funcs)
print(f"calc_descriptors returned {len(vals)} values")

# Step 2: Create DataFrame (as evaluate_molecules line 272)
desc_df = pd.DataFrame([vals], columns=descriptor_names)

# Check for duplicate columns in the DataFrame
if desc_df.columns.duplicated().any():
    dup_cols = desc_df.columns[desc_df.columns.duplicated()].tolist()
    print(f"\n*** DataFrame has DUPLICATE COLUMNS: {dup_cols} ***")
    print("Checking if any of these are in the model's descriptor_columns...")
    overlap = [c for c in dup_cols if c in desc_cols]
    if overlap:
        print(f"*** OVERLAP WITH MODEL COLUMNS: {overlap} ***")
        print("*** THIS IS THE BUG! When selecting these columns, pandas returns")
        print("    a DataFrame (multiple columns) instead of a Series (one column).")
        print("    The scaler then gets the wrong shape or wrong values. ***")
    else:
        print("No overlap with model columns (duplicates don't affect FOM1 model)")
else:
    print("No duplicate columns in DataFrame")

# Step 3: Numeric coercion (as evaluate_molecules lines 274-279)
for col in desc_df.columns:
    desc_df[col] = pd.to_numeric(desc_df[col], errors='coerce')
desc_df = desc_df.fillna(0)

# Step 4: Simulate the full df construction
df = pd.DataFrame({"SMILES": [smi]})
df = pd.concat([df, desc_df], axis=1)
print(f"DataFrame columns: {len(df.columns)}")

# Step 5: Select model columns (as evaluate_molecules line 427)
X_raw = df[desc_cols].copy()
print(f"X_raw shape: {X_raw.shape}")
print(f"X_raw dtypes unique: {X_raw.dtypes.unique()}")

# Check if any column returned a DataFrame instead of Series (due to duplicates)
for col in desc_cols:
    selected = df[col]
    if isinstance(selected, pd.DataFrame):
        print(f"*** {col}: df['{col}'] returns DataFrame with {selected.shape[1]} columns! ***")

X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
X_raw = X_raw.replace([np.inf, -np.inf], 0)

# Step 6: Scale and predict
X_scaled = scaler.transform(X_raw.values)
y_pred = model.predict(X_scaled)
print(f"\nPrediction through WSGA path: {y_pred[0]:.4f}")

# Compare with direct prediction from CSV
df_csv = pd.read_csv(base_dir / "fom1_dataset.csv")
match = df_csv[df_csv["SMILES"] == smi]
if len(match) > 0:
    X_csv = match[desc_cols].values
    X_csv = np.nan_to_num(X_csv, nan=0.0, posinf=0.0, neginf=0.0)
    X_csv_scaled = scaler.transform(X_csv)
    y_csv = model.predict(X_csv_scaled)
    print(f"Prediction from CSV values: {y_csv[0]:.4f}")
    print(f"True FOM1_40: {match['FOM1_40'].values[0]:.4f}")

    # Compare feature values
    wsga_vals = X_raw.values[0]
    csv_vals = X_csv[0]
    diffs = np.abs(wsga_vals - csv_vals)
    if np.max(diffs) > 1e-5:
        print(f"\nFeature value differences found!")
        for i in np.where(diffs > 1e-5)[0]:
            print(f"  {desc_cols[i]}: WSGA={wsga_vals[i]:.6f} CSV={csv_vals[i]:.6f} diff={diffs[i]:.6f}")
    else:
        print(f"\nAll feature values match (max diff: {np.max(diffs):.2e})")

# Also check: what if we add extra columns (simulating property predictions)
# and then re-select? Does it still work?
df["FAKE_PROPERTY"] = 999.0
X_raw2 = df[desc_cols].copy()
X_raw2 = X_raw2.apply(pd.to_numeric, errors='coerce').fillna(0)
X_raw2 = X_raw2.replace([np.inf, -np.inf], 0)
X_scaled2 = scaler.transform(X_raw2.values)
y_pred2 = model.predict(X_scaled2)
print(f"\nPrediction after adding extra columns: {y_pred2[0]:.4f} (should match above)")
