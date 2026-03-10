#!/usr/bin/env python3
"""
Diagnose the 105 vs 135 feature count mismatch in FOM1 direct models.

Checks every fold's scaler and model to see how many features they expect,
compares with descriptor_columns in the joblib and JSON, and tests predictions
on known training molecules to verify correctness.

Usage:
    conda activate mol-rl
    python diagnose_feature_count.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Current RDKit descriptor setup
DESC_NAMES = [d[0] for d in Descriptors._descList]
DESC_FUNCS = {d[0]: d[1] for d in Descriptors._descList}

base_dir = Path(__file__).resolve().parent / "results"

for temp_key in ["fom1_40", "fom1_100"]:
    print(f"\n{'='*70}")
    print(f"  {temp_key.upper()}")
    print(f"{'='*70}")

    temp_dir = base_dir / temp_key
    df = pd.read_csv(temp_dir / "fom1_dataset.csv")

    # Load descriptor_columns.json
    with open(temp_dir / "descriptor_columns.json") as f:
        json_cols = json.load(f)
    print(f"descriptor_columns.json: {len(json_cols)} columns")

    # Check each fold
    for fold_id in range(5):
        model_path = temp_dir / f"xgboost_descriptors_fold{fold_id}_model.joblib"
        if not model_path.exists():
            print(f"  Fold {fold_id}: MISSING")
            continue
        mdata = joblib.load(model_path)
        scaler = mdata["scaler"]
        model = mdata["model"]
        joblib_cols = mdata.get("descriptor_columns", None)

        n_scaler = scaler.n_features_in_
        n_model = model.n_features_in_
        n_joblib = len(joblib_cols) if joblib_cols else "N/A"

        match = "OK" if n_scaler == n_model == (len(joblib_cols) if joblib_cols else n_scaler) else "MISMATCH"
        print(f"  Fold {fold_id}: scaler={n_scaler}, model={n_model}, joblib_cols={n_joblib}  [{match}]")

    # Use fold 0 for prediction test
    mdata0 = joblib.load(temp_dir / "xgboost_descriptors_fold0_model.joblib")
    scaler0 = mdata0["scaler"]
    model0 = mdata0["model"]
    desc_cols = mdata0.get("descriptor_columns", json_cols)
    n_expected = scaler0.n_features_in_

    # If mismatch, show what's going on
    if len(desc_cols) != n_expected:
        print(f"\n  ** MISMATCH: joblib has {len(desc_cols)} descriptor_columns "
              f"but scaler expects {n_expected} features **")
        print(f"  First 5 joblib cols: {desc_cols[:5]}")
        print(f"  Last 5 joblib cols:  {desc_cols[-5:]}")

        # Check if truncating to n_expected fixes it
        truncated = desc_cols[:n_expected]
        print(f"\n  If we truncate to first {n_expected}: do column names exist in CSV?")
        missing_in_csv = [c for c in truncated if c not in df.columns]
        print(f"  Missing from CSV: {len(missing_in_csv)} -> {missing_in_csv[:5]}")

    # Load fold indices
    with open(temp_dir / "fold_indices.json") as f:
        fold_indices = json.load(f)

    # Test: predict on training data using values FROM THE CSV (as training did)
    target_col = "FOM1_40" if "40" in temp_key else "FOM1_100"
    test_idx = fold_indices["0"]["test"]
    y_true = df[target_col].values[test_idx]

    # Method A: use CSV descriptor values (as training did)
    if len(desc_cols) != n_expected:
        # Try with truncated columns
        use_cols = desc_cols[:n_expected]
    else:
        use_cols = desc_cols

    X_csv = df[use_cols].values[test_idx]
    X_csv = np.nan_to_num(X_csv, nan=0.0, posinf=0.0, neginf=0.0)
    X_csv_scaled = scaler0.transform(X_csv)
    y_pred_csv = model0.predict(X_csv_scaled)

    # Method B: compute fresh descriptors (as WSGA does)
    test_smiles = df["SMILES"].values[test_idx]
    fresh_data = []
    for smi in test_smiles:
        mol = Chem.MolFromSmiles(smi)
        row = []
        for col in use_cols:
            if mol is None or col not in DESC_FUNCS:
                row.append(0.0)
            else:
                try:
                    row.append(DESC_FUNCS[col](mol))
                except Exception:
                    row.append(0.0)
        fresh_data.append(row)
    X_fresh = np.array(fresh_data, dtype=float)
    X_fresh = np.nan_to_num(X_fresh, nan=0.0, posinf=0.0, neginf=0.0)
    X_fresh_scaled = scaler0.transform(X_fresh)
    y_pred_fresh = model0.predict(X_fresh_scaled)

    print(f"\n  Prediction test on fold-0 test set ({len(test_idx)} molecules):")
    print(f"  {'':>20} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'True ' + target_col:>20} {np.mean(y_true):>10.2f} {np.min(y_true):>10.2f} {np.max(y_true):>10.2f}")
    print(f"  {'Pred (CSV vals)':>20} {np.mean(y_pred_csv):>10.2f} {np.min(y_pred_csv):>10.2f} {np.max(y_pred_csv):>10.2f}")
    print(f"  {'Pred (fresh desc)':>20} {np.mean(y_pred_fresh):>10.2f} {np.min(y_pred_fresh):>10.2f} {np.max(y_pred_fresh):>10.2f}")

    # Check if CSV and fresh predictions differ
    max_diff = np.max(np.abs(y_pred_csv - y_pred_fresh))
    print(f"\n  Max |pred_csv - pred_fresh|: {max_diff:.6f}")
    if max_diff > 0.01:
        print(f"  ** PREDICTIONS DIFFER between CSV values and fresh descriptors! **")
        # Find which features differ
        feat_diffs = np.max(np.abs(X_csv - X_fresh), axis=0)
        diff_idx = np.where(feat_diffs > 1e-5)[0]
        print(f"  Features with different values ({len(diff_idx)}):")
        for i in diff_idx[:10]:
            print(f"    {use_cols[i]}: CSV={X_csv[0,i]:.6f} vs Fresh={X_fresh[0,i]:.6f}")
    else:
        print(f"  Predictions match — descriptors are consistent.")

    # R² on test set
    ss_res = np.sum((y_true - y_pred_fresh) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\n  R² (fold 0, fresh descriptors): {r2:.4f}")

    # Check for systematic bias
    residuals = y_pred_fresh - y_true
    print(f"  Mean residual (pred - true): {np.mean(residuals):+.4f}")
    print(f"  Median residual: {np.median(residuals):+.4f}")
