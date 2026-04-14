"""Compute diversity metrics for a single niching sweep run.

For each run directory, computes:
  - Unique BRICS fragments (valid molecules)
  - Unique BRICS fragments (global population)
  - Mean pairwise Tanimoto similarity (ECFP4, sampled) for valid and global
  - Functional group breakdown (ether/ester/carbonate)

Designed to run as a single PBS array task per run directory.

Usage:
    python compute_diversity_metrics.py <run_dir> [--output <output_csv>]
"""

import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, AllChem


def load_molecules(csv_path, mp_hard=-30):
    """Load all_evaluated_molecules.csv, return global and valid DataFrames."""
    df = pd.read_csv(csv_path, low_memory=False)
    smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in df.columns else "SMILES"

    global_unique = df.drop_duplicates(subset=[smiles_col])[smiles_col].dropna().values

    mask = pd.Series(True, index=df.index)
    if "is_valid" in df.columns:
        mask &= df["is_valid"] == 1
    if "FitnessScore" in df.columns:
        mask &= df["FitnessScore"] > 0
    if "mp" in df.columns:
        mask &= df["mp"] < mp_hard
    elif "MP-Measured" in df.columns:
        mask &= df["MP-Measured"] < mp_hard

    valid_unique = df[mask].drop_duplicates(subset=[smiles_col])[smiles_col].dropna().values

    return global_unique, valid_unique


def compute_brics(smiles_arr):
    """Return set of unique BRICS fragment SMILES."""
    frags = set()
    for smi in smiles_arr:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                frags.update(BRICS.BRICSDecompose(mol))
            except Exception:
                pass
    return frags


def compute_mean_tanimoto(smiles_arr, n_sample=2000, n_pairs=50000, seed=42):
    """Mean pairwise Tanimoto (ECFP4) from a random sample."""
    rng = np.random.default_rng(seed)
    if len(smiles_arr) > n_sample:
        smiles_arr = rng.choice(smiles_arr, n_sample, replace=False)

    fps = []
    for smi in smiles_arr:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    if len(fps) < 2:
        return np.nan

    n_pairs = min(n_pairs, len(fps) * (len(fps) - 1) // 2)
    sims = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(fps), 2, replace=False)
        sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float(np.mean(sims))


def compute_func_groups(smiles_arr):
    """Count ether/ester/carbonate functional groups."""
    pat_carbonate = Chem.MolFromSmarts("[OX2]C(=O)[OX2]")
    pat_ester = Chem.MolFromSmarts("[CX3](=O)[OX2]")
    pat_ether = Chem.MolFromSmarts("[OD2]([CX4])[CX4]")

    counts = {"carbonate": 0, "ester": 0, "ether": 0, "other": 0}
    for smi in smiles_arr:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            if mol.HasSubstructMatch(pat_carbonate):
                counts["carbonate"] += 1
            elif mol.HasSubstructMatch(pat_ester):
                counts["ester"] += 1
            elif mol.HasSubstructMatch(pat_ether):
                counts["ether"] += 1
            else:
                counts["other"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to a single run directory")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <run_dir>/diversity_metrics.csv)")
    args = parser.parse_args()

    csv_path = os.path.join(args.run_dir, "all_evaluated_molecules.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return

    dirname = os.path.basename(os.path.normpath(args.run_dir))
    print(f"Processing {dirname} ...")

    global_smi, valid_smi = load_molecules(csv_path)
    print(f"  Global unique: {len(global_smi)}, Valid unique: {len(valid_smi)}")

    # BRICS
    print("  Computing BRICS (valid) ...")
    brics_valid = compute_brics(valid_smi)
    print(f"    {len(brics_valid)} unique fragments")

    print("  Computing BRICS (global) ...")
    brics_global = compute_brics(global_smi)
    print(f"    {len(brics_global)} unique fragments")

    # Tanimoto
    print("  Computing Tanimoto (valid) ...")
    ts_valid = compute_mean_tanimoto(valid_smi)
    print(f"    Mean TS = {ts_valid:.4f}")

    print("  Computing Tanimoto (global) ...")
    ts_global = compute_mean_tanimoto(global_smi)
    print(f"    Mean TS = {ts_global:.4f}")

    # Functional groups (valid only)
    print("  Computing functional groups ...")
    fg = compute_func_groups(valid_smi)

    result = {
        "dir": dirname,
        "n_global": len(global_smi),
        "n_valid": len(valid_smi),
        "brics_valid": len(brics_valid),
        "brics_global": len(brics_global),
        "ts_valid": ts_valid,
        "ts_global": ts_global,
        "fg_carbonate": fg["carbonate"],
        "fg_ester": fg["ester"],
        "fg_ether": fg["ether"],
        "fg_other": fg["other"],
    }

    out_path = args.output or os.path.join(args.run_dir, "diversity_metrics.csv")
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
