"""Compute diversity metrics at three population levels for a single run.

For each run directory, computes at three levels (global, valid, top 25):
  - Mean pairwise Tanimoto similarity (Morgan fingerprint, 2048 bits, sampled)
  - Unique BRICS fragments
  - Unique Murcko generic scaffolds
  - Functional group breakdown (ether/ester/carbonate) [valid only]

The fingerprint radius used for the Tanimoto computation is configurable via
--fp_radius (default 2, matching the WSGA niching production radius).

Usage:
    python compute_diversity_levels.py <run_dir> [--fp_radius 2]
"""

import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def load_populations(csv_path, mp_hard=-30):
    """Return (global_smiles, valid_smiles, top25_smiles)."""
    df = pd.read_csv(csv_path, low_memory=False)
    smiles_col = "CanonicalSMILES" if "CanonicalSMILES" in df.columns else "SMILES"

    global_smi = df.drop_duplicates(subset=[smiles_col])[smiles_col].dropna().values

    mask = pd.Series(True, index=df.index)
    if "is_valid" in df.columns:
        mask &= df["is_valid"] == 1
    if "FitnessScore" in df.columns:
        mask &= df["FitnessScore"] > 0
    if "mp" in df.columns:
        mask &= df["mp"] < mp_hard
    elif "MP-Measured" in df.columns:
        mask &= df["MP-Measured"] < mp_hard

    valid_df = df[mask].drop_duplicates(subset=[smiles_col])
    fom1_col = next((c for c in ["fom1_40C", "FOM1_40", "FOM1_avg"] if c in valid_df.columns), None)
    if fom1_col:
        valid_df = valid_df.sort_values(fom1_col, ascending=False)
    valid_smi = valid_df[smiles_col].dropna().values
    top25_smi = valid_smi[:25]

    return global_smi, valid_smi, top25_smi


def mean_pairwise_ts(smiles_arr, fp_radius=2,
                     n_sample=3000, n_pairs=100000, seed=42):
    """Mean pairwise Tanimoto (Morgan, 2048 bits) from a random sample."""
    rng = np.random.default_rng(seed)
    arr = smiles_arr
    if len(arr) > n_sample:
        arr = rng.choice(arr, n_sample, replace=False)

    fps = []
    for smi in arr:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(
                mol, fp_radius, nBits=2048))

    if len(fps) < 2:
        return np.nan

    if len(fps) <= 50:
        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    else:
        actual_pairs = min(n_pairs, len(fps) * (len(fps) - 1) // 2)
        sims = []
        for _ in range(actual_pairs):
            i, j = rng.choice(len(fps), 2, replace=False)
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    return float(np.mean(sims))


def count_brics(smiles_arr):
    """Return number of unique BRICS fragments."""
    frags = set()
    for smi in smiles_arr:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                frags.update(BRICS.BRICSDecompose(mol))
            except Exception:
                pass
    return len(frags)


def count_murcko(smiles_arr):
    """Return number of unique Murcko generic scaffolds."""
    scaffolds = set()
    for smi in smiles_arr:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol))
            scaffolds.add(Chem.MolToSmiles(scaf))
        except Exception:
            pass
    return len(scaffolds)


def compute_func_groups(smiles_arr):
    """Count ether/ester/carbonate."""
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
    parser.add_argument("run_dir")
    parser.add_argument("--fp_radius", type=int, default=2,
                        help="Morgan fingerprint radius for TS "
                             "(default 2, matching WSGA niching production)")
    parser.add_argument("--output_name", type=str,
                        default="diversity_levels.csv",
                        help="Output CSV filename inside run_dir")
    args = parser.parse_args()

    csv_path = os.path.join(args.run_dir, "all_evaluated_molecules.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return

    dirname = os.path.basename(os.path.normpath(args.run_dir))
    print(f"Processing {dirname} (fp_radius={args.fp_radius}) ...")

    global_smi, valid_smi, top25_smi = load_populations(csv_path)
    print(f"  Global: {len(global_smi):,}  Valid: {len(valid_smi):,}  Top25: {len(top25_smi)}")

    result = {"dir": dirname, "fp_radius": args.fp_radius}

    for label, arr in [("top25", top25_smi), ("valid", valid_smi), ("global", global_smi)]:
        print(f"  [{label}] Tanimoto ...")
        result[f"ts_{label}"] = mean_pairwise_ts(arr, fp_radius=args.fp_radius)
        print(f"    TS = {result[f'ts_{label}']:.4f}")

        print(f"  [{label}] BRICS ...")
        result[f"brics_{label}"] = count_brics(arr)
        print(f"    {result[f'brics_{label}']} fragments")

        print(f"  [{label}] Murcko ...")
        result[f"murcko_{label}"] = count_murcko(arr)
        print(f"    {result[f'murcko_{label}']} scaffolds")

    result["n_global"] = len(global_smi)
    result["n_valid"] = len(valid_smi)

    # Functional groups (valid only)
    fg = compute_func_groups(valid_smi)
    for k, v in fg.items():
        result[f"fg_{k}"] = v

    out_path = os.path.join(args.run_dir, args.output_name)
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
