#!/usr/bin/env python3
"""
Validate FOM1 direct predictions through the exact WSGA descriptor pipeline.

Takes known molecules from the FOM1 training set, computes descriptors
via the same descriptors.py / wsga_helper.py code path used at inference,
and compares predictions against known FOM1 values.

This isolates whether the issue is:
  (a) Feature count mismatch (model expects 135 but gets fewer)
  (b) Descriptor value mismatch (same name, different value)
  (c) Model limitation (correct features but poor extrapolation)

Usage:
    conda activate mol-rl
    python validate_wsga_pipeline.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

# Import the EXACT descriptor computation used by WSGA
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from descriptors import descriptor_funcs, descriptor_names, calc_descriptors

print(f"descriptor_names count (from descriptors.py): {len(descriptor_names)}")
print(f"RDKit _descList count: {len(Descriptors._descList)}")
print(f"Match: {len(descriptor_names) == len(Descriptors._descList)}")

# Load FOM1 direct models (same as load_fom1_direct_models)
base_dir = Path(__file__).resolve().parent / "results"

for temp_key, target_col in [("fom1_40", "FOM1_40"), ("fom1_100", "FOM1_100")]:
    print(f"\n{'='*70}")
    print(f"  {temp_key.upper()}")
    print(f"{'='*70}")

    temp_dir = base_dir / temp_key
    df = pd.read_csv(temp_dir / "fom1_dataset.csv")

    # Load fold 0 model
    mdata = joblib.load(temp_dir / "xgboost_descriptors_fold0_model.joblib")
    model = mdata["model"]
    scaler = mdata["scaler"]
    desc_cols = mdata.get("descriptor_columns")

    if desc_cols is None:
        import json
        with open(temp_dir / "descriptor_columns.json") as f:
            desc_cols = json.load(f)
        desc_cols = desc_cols[:scaler.n_features_in_]

    print(f"Model expects {scaler.n_features_in_} features")
    print(f"desc_cols has {len(desc_cols)} entries")

    # Check which model columns exist in descriptor_names
    missing_from_wsga = [c for c in desc_cols if c not in descriptor_names]
    if missing_from_wsga:
        print(f"\n*** PROBLEM: {len(missing_from_wsga)} model columns NOT in "
              f"descriptor_names (WSGA won't compute them):")
        for c in missing_from_wsga:
            print(f"    {c}")
        print("*** This means df[desc_cols] will KeyError in evaluate_molecules! ***")
    else:
        print(f"All {len(desc_cols)} model columns exist in descriptor_names (WSGA computes them)")

    # Now simulate the EXACT WSGA pipeline for 20 training molecules
    n_test = 20
    test_smiles = df["SMILES"].values[:n_test]
    y_true = df[target_col].values[:n_test]

    # Step 1: Compute descriptors exactly as WSGA does
    mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]
    desc_data = [calc_descriptors(mol, descriptor_funcs) for mol in mols]
    desc_df = pd.DataFrame(desc_data, columns=descriptor_names)

    # Step 2: Numeric coercion (same as evaluate_molecules lines 274-279)
    for col in desc_df.columns:
        desc_df[col] = pd.to_numeric(desc_df[col], errors='coerce')
    desc_df = desc_df.fillna(0)

    # Step 3: Select model columns (same as evaluate_molecules line 427)
    try:
        X_raw = desc_df[desc_cols].copy()
    except KeyError as e:
        print(f"\n*** KeyError selecting model columns: {e} ***")
        print("This confirms the WSGA can't provide all features the model needs.")
        continue

    X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_raw = X_raw.replace([np.inf, -np.inf], 0)

    print(f"\nX_raw shape: {X_raw.shape} (should be ({n_test}, {len(desc_cols)}))")

    # Step 4: Scale and predict
    X_scaled = scaler.transform(X_raw.values)
    y_pred_wsga = model.predict(X_scaled)

    # Compare with direct CSV prediction (bypass WSGA)
    X_csv = df[desc_cols].values[:n_test]
    X_csv = np.nan_to_num(X_csv, nan=0.0, posinf=0.0, neginf=0.0)
    X_csv_scaled = scaler.transform(X_csv)
    y_pred_csv = model.predict(X_csv_scaled)

    print(f"\n  {'SMILES':<40} {'True':>8} {'WSGA':>8} {'CSV':>8} {'Diff':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for i in range(min(10, n_test)):
        smi = test_smiles[i][:38]
        diff = y_pred_wsga[i] - y_pred_csv[i]
        print(f"  {smi:<40} {y_true[i]:>8.2f} {y_pred_wsga[i]:>8.2f} "
              f"{y_pred_csv[i]:>8.2f} {diff:>+8.4f}")

    max_diff = np.max(np.abs(y_pred_wsga - y_pred_csv))
    print(f"\n  Max |WSGA_pred - CSV_pred|: {max_diff:.6f}")

    if max_diff > 0.01:
        print("  *** MISMATCH: WSGA pipeline produces different predictions! ***")
        # Find which features differ
        feat_diffs = np.max(np.abs(desc_df[desc_cols].values[:n_test] - X_csv), axis=0)
        diff_idx = np.where(feat_diffs > 1e-5)[0]
        if len(diff_idx) > 0:
            print(f"  Features with value differences ({len(diff_idx)}):")
            for i in diff_idx[:10]:
                wsga_val = desc_df[desc_cols[i]].values[0]
                csv_val = X_csv[0, i]
                print(f"    {desc_cols[i]}: WSGA={wsga_val:.6f} CSV={csv_val:.6f}")
    else:
        print("  WSGA pipeline produces identical predictions to direct CSV path.")
        print("  Model is working correctly through the WSGA code path.")
        print(f"\n  If predictions seem low for novel molecules, this is expected —")
        print(f"  the model was trained on FOM1 range [{df[target_col].min():.1f}, "
              f"{df[target_col].max():.1f}] (mean {df[target_col].mean():.1f})")
        print(f"  and may regress toward the mean for out-of-distribution inputs.")
