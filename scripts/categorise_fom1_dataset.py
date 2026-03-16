"""
Categorise FOM1 dataset molecules into 4 categories for the stability sweep.

Takes the 806-molecule FOM1 dataset (with pre-computed properties from
BaselineFOM1Eval), predicts SCScore and MolPrice, applies constraints
matching the GA's filters, and classifies each molecule.

Categories (2 axes):
  - Biodegradable vs no filter
  - Stable (passes STABILITY_SMARTS + n_ether<=1) vs no additional filter

Output: results/fom1_dataset_categories.csv

Usage:
    python categorise_fom1_dataset.py
    python categorise_fom1_dataset.py --baseline_csv ../BaselineFOM1Eval/output/baseline_fom1_results.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path for model imports
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.error')

from wsga_helper import (
    has_invalid_fragments,
    count_ether_linkages,
    count_ester_groups,
    load_fom1_direct_models,
    STABILITY_SMARTS,
)
from descriptors import descriptor_funcs, descriptor_names, calc_descriptors
from evaluation import get_scscore_cached
from SCScorer import SCScorer


def predict_scscore(smiles_list, model_dir):
    """Predict SCScore for a list of SMILES."""
    sc_model = SCScorer()
    sc_model.restore(os.path.join(
        model_dir, "SCScorer/scscore/models/full_reaxys_model_1024bool/"
        "model.ckpt-10654.as_numpy.json.gz"))
    return [get_scscore_cached(sc_model, smi) for smi in smiles_list]


def predict_molprice(smiles_list, molprice_path):
    """Predict MolPrice for a list of SMILES."""
    from molprice import MolPriceModel
    mp_model = MolPriceModel(molprice_path)
    return mp_model.predict_batch(smiles_list)


def predict_fom1_direct(smiles_list, fom1_model_dir):
    """Predict FOM1 (40C, 100C, avg) using direct XGBoost ensemble."""
    fom1_models = load_fom1_direct_models(fom1_model_dir)

    desc_data = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        desc_data.append(calc_descriptors(mol, descriptor_funcs))
    desc_df = pd.DataFrame(desc_data, columns=descriptor_names)
    for col in desc_df.columns:
        desc_df[col] = pd.to_numeric(desc_df[col], errors="coerce")
    desc_df = desc_df.fillna(0).replace([np.inf, -np.inf], 0)

    results = {}
    for temp_key, col_name in [("fom1_40", "FOM1_40C_pred"),
                                ("fom1_100", "FOM1_100C_pred")]:
        mdata = fom1_models[temp_key]
        desc_cols = mdata["descriptor_columns"]
        X_raw = desc_df[desc_cols].values
        preds_all = []
        for model, scaler in zip(mdata["models"], mdata["scalers"]):
            X_scaled = scaler.transform(X_raw)
            preds_all.append(model.predict(X_scaled))
        results[col_name] = np.mean(preds_all, axis=0)

    results["FOM1_pred_avg"] = (results["FOM1_40C_pred"] +
                                 results["FOM1_100C_pred"]) / 2
    return results


def check_stability(smiles):
    """Check if a molecule passes stability constraints.

    Returns True if the molecule is 'stable' (no alkenes, no polyethers,
    no aldehydes, no carboxylic acids, no carbonates, n_ether<=1,
    n_ester<=2, n_oxygen<=4).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    # Check STABILITY_SMARTS
    for name, pattern in STABILITY_SMARTS.items():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return False

    # Ether/ester/oxygen counts
    if count_ether_linkages(mol) > 1:
        return False
    if count_ester_groups(mol) > 2:
        return False
    n_oxygen = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    if n_oxygen > 4:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Categorise FOM1 dataset into 4 stability/biodeg categories")
    parser.add_argument("--baseline_csv", type=str,
                        default="../BaselineFOM1Eval/output/baseline_fom1_results.csv")
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--molprice_model", type=str,
                        default="../models/MolPrice/MP_Morgan_hybrid.pkl")
    parser.add_argument("--fom1_model_dir", type=str,
                        default="../training/FOM1_architecture_comparison/results")
    parser.add_argument("--output", type=str,
                        default="../results/fom1_dataset_categories.csv")
    # Constraint thresholds (matching sweep defaults)
    parser.add_argument("--mp_threshold", type=float, default=-30)
    parser.add_argument("--bp_threshold", type=float, default=100)
    parser.add_argument("--fp_threshold", type=float, default=373.15)
    parser.add_argument("--dc_threshold", type=float, default=7)
    parser.add_argument("--tox_threshold", type=float, default=3)
    parser.add_argument("--sc_threshold", type=float, default=3)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load baseline data
    # ------------------------------------------------------------------
    print(f"Loading baseline: {args.baseline_csv}")
    df = pd.read_csv(args.baseline_csv)
    n_total = len(df)
    print(f"  {n_total} molecules loaded")

    # ------------------------------------------------------------------
    # 2. Predict SCScore
    # ------------------------------------------------------------------
    print("Predicting SCScore...")
    df["SCScore"] = predict_scscore(df["SMILES"].tolist(), args.model_dir)

    # ------------------------------------------------------------------
    # 3. Predict MolPrice
    # ------------------------------------------------------------------
    if os.path.exists(args.molprice_model):
        print(f"Predicting MolPrice from {args.molprice_model}...")
        df["MolPrice"] = predict_molprice(df["SMILES"].tolist(),
                                           args.molprice_model)
    else:
        print(f"WARNING: MolPrice model not found at {args.molprice_model}")
        df["MolPrice"] = np.nan

    # ------------------------------------------------------------------
    # 4. Predict FOM1 (direct model) for apples-to-apples comparison
    # ------------------------------------------------------------------
    if os.path.isdir(args.fom1_model_dir):
        print(f"Predicting FOM1 (direct) from {args.fom1_model_dir}...")
        fom1_preds = predict_fom1_direct(df["SMILES"].tolist(),
                                          args.fom1_model_dir)
        for k, v in fom1_preds.items():
            df[k] = v
        print(f"  Predicted FOM1 for {len(df)} molecules")
    else:
        print(f"WARNING: FOM1 model dir not found: {args.fom1_model_dir}")

    # ------------------------------------------------------------------
    # 5. Apply base constraints (matching GA defaults)
    # ------------------------------------------------------------------
    print("\nApplying base constraints...")

    # Structural filter (banned fragments)
    df["has_banned_fragments"] = df["SMILES"].apply(has_invalid_fragments)
    n_banned = df["has_banned_fragments"].sum()
    print(f"  Banned fragments: {n_banned} molecules")

    # Property constraints
    base_mask = (
        (df["MP-Measured"] < args.mp_threshold) &
        (df["BP-Measured"] >= args.bp_threshold) &
        (df["flashpoint"] >= args.fp_threshold) &
        (df["DC_exp"] <= args.dc_threshold) &
        (df["Tox21_Score"] <= args.tox_threshold) &
        (df["SCScore"] <= args.sc_threshold) &
        (~df["has_banned_fragments"])
    )

    df["passes_base_constraints"] = base_mask
    n_pass = base_mask.sum()
    print(f"  Pass all base constraints: {n_pass}/{n_total}")

    # Per-constraint breakdown
    print(f"    MP < {args.mp_threshold}: "
          f"{(df['MP-Measured'] < args.mp_threshold).sum()}")
    print(f"    BP >= {args.bp_threshold}: "
          f"{(df['BP-Measured'] >= args.bp_threshold).sum()}")
    print(f"    FP >= {args.fp_threshold}: "
          f"{(df['flashpoint'] >= args.fp_threshold).sum()}")
    print(f"    DC <= {args.dc_threshold}: "
          f"{(df['DC_exp'] <= args.dc_threshold).sum()}")
    print(f"    Tox21 <= {args.tox_threshold}: "
          f"{(df['Tox21_Score'] <= args.tox_threshold).sum()}")
    print(f"    SCScore <= {args.sc_threshold}: "
          f"{(df['SCScore'] <= args.sc_threshold).sum()}")
    print(f"    No banned fragments: {(~df['has_banned_fragments']).sum()}")

    # ------------------------------------------------------------------
    # 6. Classify stability and biodegradability
    # ------------------------------------------------------------------
    print("\nClassifying stability...")
    df["is_stable"] = df["SMILES"].apply(check_stability)
    n_stable = df["is_stable"].sum()
    print(f"  Stable: {n_stable}/{n_total}")

    df["is_biodegradable"] = df["Biodegradable"].astype(bool)
    n_biodeg = df["is_biodegradable"].sum()
    print(f"  Biodegradable: {n_biodeg}/{n_total}")

    # ------------------------------------------------------------------
    # 7. Assign categories (for molecules passing base constraints)
    # ------------------------------------------------------------------
    def _assign_category(row):
        if not row["passes_base_constraints"]:
            return "excluded"
        bio = row["is_biodegradable"]
        stable = row["is_stable"]
        if bio and stable:
            return "bio_stable"
        elif bio and not stable:
            return "bio_unstable"
        elif not bio and stable:
            return "nonbio_stable"
        else:
            return "nonbio_unstable"

    df["category"] = df.apply(_assign_category, axis=1)

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  CATEGORY SUMMARY (molecules passing base constraints)")
    print(f"{'='*60}")

    cats = ["bio_stable", "bio_unstable", "nonbio_stable", "nonbio_unstable"]
    for cat in cats:
        sub = df[df["category"] == cat]
        n = len(sub)
        fom1_str = ""
        if n > 0 and "FOM1_exp_avg" in sub.columns:
            fom1_str = (f"  FOM1_exp: {sub['FOM1_exp_avg'].mean():.1f} "
                        f"(max {sub['FOM1_exp_avg'].max():.1f})")
        print(f"  {cat:<20} {n:>4} molecules{fom1_str}")

    excluded = len(df[df["category"] == "excluded"])
    print(f"  {'excluded':<20} {excluded:>4} molecules")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
