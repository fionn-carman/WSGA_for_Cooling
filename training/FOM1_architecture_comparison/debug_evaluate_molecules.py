#!/usr/bin/env python3
"""
Debug: call the REAL evaluate_molecules function on a known molecule
and compare FOM1 direct predictions with manual computation.

This will pinpoint whether the issue is inside evaluate_molecules
(e.g., descriptor corruption from thermo model predictions) or
something else in the WSGA pipeline.

Usage (on HPC):
    conda activate mol-rl
    cd WSGA_for_Cooling/training/FOM1_architecture_comparison
    python debug_evaluate_molecules.py
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
from wsga_helper import (
    evaluate_molecules,
    load_regression_models_with_aux,
    load_fom1_direct_models,
    load_tox21_models,
    load_biodeg_model,
)
from scscore.standalone_model_numpy import SCScorer

print(f"descriptor_names: {len(descriptor_names)}")
print(f"Unique descriptor_names: {len(set(descriptor_names))}")
dupes = [n for i, n in enumerate(descriptor_names) if n in descriptor_names[:i]]
if dupes:
    print(f"*** DUPLICATE descriptor_names: {dupes} ***")

# ============================================================
# 1. Load all models exactly as wsga.py does
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = str(PROJECT_DIR / "models")
FOM1_DIR = str(Path(__file__).resolve().parent / "results")

print("\n--- Loading models ---")

thermo_targets = [
    "BP-Measured", "DC_exp", "Density_100C_g_cm^3", "Density_40C_g_cm^3",
    "flashpoint", "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",
    "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",
    "Kinematic_Viscosity_40C", "Kinematic_Viscosity_100C",
    "MP-Measured", "Thermal_Conductivity_100C", "Thermal_Conductivity_40C"
]
thermo_models = load_regression_models_with_aux(thermo_targets, MODEL_DIR)

sc_model = SCScorer()
sc_model.restore(os.path.join(MODEL_DIR, "SCScorer/scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"))

tox21_models = load_tox21_models(os.path.join(MODEL_DIR, "tox21"))
biodeg_model = load_biodeg_model(os.path.join(MODEL_DIR, "biodegradability"))
fom1_direct_models = load_fom1_direct_models(FOM1_DIR)

# ============================================================
# 2. Test molecule
# ============================================================
smi = "COCCOCCOCCOC"
print(f"\n--- Test molecule: {smi} ---")

# ============================================================
# 3. Manual FOM1 prediction (bypass evaluate_molecules)
# ============================================================
mol = Chem.MolFromSmiles(smi)
vals = calc_descriptors(mol, descriptor_funcs)
desc_df = pd.DataFrame([vals], columns=descriptor_names)
for col in desc_df.columns:
    desc_df[col] = pd.to_numeric(desc_df[col], errors='coerce')
desc_df = desc_df.fillna(0)

for temp_key in ['fom1_40', 'fom1_100']:
    mdata = fom1_direct_models[temp_key]
    desc_cols = mdata['descriptor_columns']

    X_raw_manual = desc_df[desc_cols].copy()
    X_raw_manual = X_raw_manual.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_raw_manual = X_raw_manual.replace([np.inf, -np.inf], 0)

    preds = []
    for model, scaler in zip(mdata['models'], mdata['scalers']):
        X_scaled = scaler.transform(X_raw_manual.values)
        preds.append(model.predict(X_scaled)[0])
    manual_pred = np.mean(preds)
    print(f"  Manual {temp_key}: {manual_pred:.4f}")

# ============================================================
# 4. Call evaluate_molecules (the real function)
# ============================================================
test_df = pd.DataFrame({"SMILES": [smi]})
result_df = evaluate_molecules(
    test_df, thermo_models, sc_model, tox21_models, biodeg_model,
    drop_descriptors=True, fom1_direct_models=fom1_direct_models
)

print(f"\n--- evaluate_molecules results ---")
for col in ['FOM1_40', 'FOM1_100', 'FOM1_40C_direct', 'FOM1_100C_direct']:
    if col in result_df.columns:
        print(f"  {col}: {result_df[col].values[0]:.4f}")
    else:
        print(f"  {col}: NOT IN OUTPUT")

# ============================================================
# 5. Call evaluate_molecules WITHOUT dropping descriptors
#    so we can inspect the feature values
# ============================================================
test_df2 = pd.DataFrame({"SMILES": [smi]})
result_df2 = evaluate_molecules(
    test_df2, thermo_models, sc_model, tox21_models, biodeg_model,
    drop_descriptors=False, fom1_direct_models=fom1_direct_models
)

print(f"\n--- Descriptor values in evaluate_molecules DataFrame ---")
desc_cols_40 = fom1_direct_models['fom1_40']['descriptor_columns']

# Check if any descriptor columns are missing
missing = [c for c in desc_cols_40 if c not in result_df2.columns]
if missing:
    print(f"  *** MISSING from df: {missing} ***")

# Check for duplicate columns
dup_cols = result_df2.columns[result_df2.columns.duplicated()].tolist()
if dup_cols:
    print(f"  *** DUPLICATE COLUMNS: {dup_cols} ***")
    overlap = [c for c in dup_cols if c in desc_cols_40]
    if overlap:
        print(f"  *** OVERLAP WITH FOM1 MODEL COLS: {overlap} ***")

# Compare descriptor values from evaluate_molecules vs manual
print(f"\n--- Comparing descriptor values (evaluate_molecules vs manual) ---")
n_diff = 0
for col in desc_cols_40:
    if col not in result_df2.columns:
        continue
    selected = result_df2[col]
    if isinstance(selected, pd.DataFrame):
        print(f"  *** {col}: returns DataFrame ({selected.shape[1]} columns)! ***")
        n_diff += 1
        continue
    eval_val = float(selected.values[0])
    manual_val = float(desc_df[col].values[0]) if col in desc_df.columns else 0.0
    if abs(eval_val - manual_val) > 1e-5:
        print(f"  DIFF {col}: eval={eval_val:.6f} manual={manual_val:.6f}")
        n_diff += 1

if n_diff == 0:
    print("  All descriptor values match!")
else:
    print(f"  {n_diff} differences found!")

# ============================================================
# 6. Check X_raw values right before FOM1 prediction
#    Re-simulate the exact line 427 extraction
# ============================================================
print(f"\n--- Simulating line 427: X_raw = df[desc_cols].copy() ---")
X_raw_eval = result_df2[desc_cols_40].copy()
X_raw_eval = X_raw_eval.apply(pd.to_numeric, errors='coerce').fillna(0)
X_raw_eval = X_raw_eval.replace([np.inf, -np.inf], 0)

print(f"  X_raw_eval shape: {X_raw_eval.shape}")
print(f"  X_raw_manual shape: {X_raw_manual.shape}")

if X_raw_eval.shape == X_raw_manual.shape:
    diffs = np.abs(X_raw_eval.values - X_raw_manual.values)
    max_diff = np.max(diffs)
    print(f"  Max feature value difference: {max_diff:.2e}")
    if max_diff > 1e-5:
        diff_idx = np.where(diffs[0] > 1e-5)[0]
        print(f"  Features with differences ({len(diff_idx)}):")
        for i in diff_idx[:20]:
            print(f"    {desc_cols_40[i]}: eval={X_raw_eval.values[0,i]:.6f} manual={X_raw_manual.values[0,i]:.6f}")
else:
    print(f"  *** SHAPE MISMATCH: eval={X_raw_eval.shape} manual={X_raw_manual.shape} ***")

# ============================================================
# 7. Check the total column count and names in df
# ============================================================
print(f"\n--- DataFrame column analysis ---")
print(f"  Total columns in df: {len(result_df2.columns)}")
print(f"  Unique column names: {len(set(result_df2.columns))}")
desc_in_df = [c for c in descriptor_names if c in result_df2.columns]
print(f"  descriptor_names found in df: {len(desc_in_df)}/{len(descriptor_names)}")
