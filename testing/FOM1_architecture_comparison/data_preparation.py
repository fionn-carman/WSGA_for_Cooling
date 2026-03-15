#!/usr/bin/env python3
"""
Data preparation for FOM1 temperature-specific architecture comparison.

Loads FOM1 training data, computes RDKit descriptors and Morgan fingerprints,
creates standardised 5-fold stratified splits, and saves all outputs.

Requires --target to specify which FOM1 target to stratify folds on.
All architecture scripts depend on the outputs of this script.

Usage:
    python data_preparation.py --target FOM1_40 --output_dir ./results
    python data_preparation.py --target FOM1_100 --output_dir ./results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# RDKit descriptor definitions
DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]
DESCRIPTOR_FUNCS = [desc[1] for desc in Descriptors._descList]


def calc_descriptors(mol):
    """Compute all RDKit descriptors for a single Mol object."""
    if mol is None:
        return [np.nan] * len(DESCRIPTOR_FUNCS)
    vals = []
    for func in DESCRIPTOR_FUNCS:
        try:
            v = func(mol)
            vals.append(v)
        except Exception:
            vals.append(np.nan)
    return vals


def compute_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan fingerprints for a list of SMILES."""
    fpgen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            fp = fpgen.GetFingerprint(mol)
            arr = np.zeros(n_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    return np.array(fps)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FOM1 dataset for temperature-specific architecture comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Directory containing FOM1 CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Base output directory (target subdirectory created automatically)",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["FOM1_40", "FOM1_100"],
        required=True,
        help="Target column to stratify folds on: FOM1_40 or FOM1_100",
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp_radius", type=int, default=2, help="Morgan FP radius")
    parser.add_argument("--fp_bits", type=int, default=2048, help="Morgan FP bits")
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)

    # Output to target-specific subdirectory
    target_key = args.target.lower()  # "fom1_40" or "fom1_100"
    output_dir = Path(args.output_dir) / target_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load FOM1 data ----
    logger.info("Loading FOM1 data from %s", data_dir)
    fom40_path = data_dir / "FOM1_exp_40_cleaned.csv"
    fom100_path = data_dir / "FOM1_exp_100_cleaned.csv"

    if not fom40_path.exists():
        logger.error("FOM1_exp_40_cleaned.csv not found at %s", fom40_path)
        sys.exit(1)
    if not fom100_path.exists():
        logger.error("FOM1_exp_100_cleaned.csv not found at %s", fom100_path)
        sys.exit(1)

    fom40 = pd.read_csv(fom40_path)
    fom100 = pd.read_csv(fom100_path)
    logger.info("FOM1_40: %d molecules, FOM1_100: %d molecules", len(fom40), len(fom100))

    # ---- 2. Merge on SMILES ----
    df = fom40.merge(fom100, on="SMILES", how="inner")
    df.columns = ["SMILES", "FOM1_40", "FOM1_100"]
    df["FOM1_avg"] = (df["FOM1_40"] + df["FOM1_100"]) / 2
    logger.info("After merge: %d molecules with both FOM1_40 and FOM1_100", len(df))

    # Remove duplicates
    df = df.drop_duplicates(subset="SMILES").reset_index(drop=True)
    logger.info("After dedup: %d molecules", len(df))

    # ---- 3. Compute RDKit descriptors ----
    logger.info("Computing RDKit descriptors (%d descriptors)...", len(DESCRIPTOR_NAMES))
    mols = [Chem.MolFromSmiles(smi) for smi in df["SMILES"]]
    n_failed = sum(1 for m in mols if m is None)
    if n_failed > 0:
        logger.warning("%d SMILES failed to parse", n_failed)

    desc_data = [calc_descriptors(mol) for mol in mols]
    desc_df = pd.DataFrame(desc_data, columns=DESCRIPTOR_NAMES)

    # Coerce to numeric, replace NaN/Inf with 0
    for col in desc_df.columns:
        desc_df[col] = pd.to_numeric(desc_df[col], errors="coerce")
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Drop constant columns
    constant_cols = [col for col in desc_df.columns if desc_df[col].nunique() <= 1]
    if constant_cols:
        logger.info("Dropping %d constant descriptor columns: %s", len(constant_cols), constant_cols[:5])
        desc_df = desc_df.drop(columns=constant_cols)

    descriptor_columns = list(desc_df.columns)
    logger.info("Final descriptor count: %d", len(descriptor_columns))

    # ---- 4. Compute Morgan fingerprints ----
    logger.info("Computing Morgan fingerprints (radius=%d, bits=%d)...", args.fp_radius, args.fp_bits)
    fps = compute_fingerprints(df["SMILES"].tolist(), radius=args.fp_radius, n_bits=args.fp_bits)
    logger.info("Fingerprint array shape: %s", fps.shape)

    # ---- 5. Create stratified 5-fold splits (stratified on selected target) ----
    logger.info("Creating %d-fold stratified splits on %s...", args.n_folds, args.target)
    bins = pd.qcut(df[args.target], q=5, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_indices = {}
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(df, bins)):
        fold_indices[str(fold_id)] = {
            "train": train_idx.tolist(),
            "test": test_idx.tolist(),
        }
        logger.info(
            "  Fold %d: train=%d, test=%d",
            fold_id, len(train_idx), len(test_idx),
        )

    # ---- 6. Save outputs ----
    # Dataset CSV (SMILES + FOM1 targets + descriptors)
    dataset = pd.concat([df[["SMILES", "FOM1_40", "FOM1_100", "FOM1_avg"]], desc_df], axis=1)
    dataset_path = output_dir / "fom1_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    logger.info("Saved dataset: %s (%d rows, %d cols)", dataset_path, len(dataset), len(dataset.columns))

    # Fold indices
    fold_path = output_dir / "fold_indices.json"
    with open(fold_path, "w") as f:
        json.dump(fold_indices, f)
    logger.info("Saved fold indices: %s", fold_path)

    # Fingerprints
    fp_path = output_dir / "fingerprints.npy"
    np.save(fp_path, fps)
    logger.info("Saved fingerprints: %s (shape %s)", fp_path, fps.shape)

    # Descriptor column names
    desc_cols_path = output_dir / "descriptor_columns.json"
    with open(desc_cols_path, "w") as f:
        json.dump(descriptor_columns, f)
    logger.info("Saved descriptor columns: %s (%d names)", desc_cols_path, len(descriptor_columns))

    # ---- 7. Print summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY (target: %s)", args.target)
    logger.info("=" * 60)
    logger.info("  Molecules: %d", len(df))
    logger.info("  Descriptors: %d", len(descriptor_columns))
    logger.info("  Fingerprint dim: %d", fps.shape[1])
    logger.info("")
    logger.info("  %s distribution:", args.target)
    logger.info("    Mean:   %.2f", df[args.target].mean())
    logger.info("    Std:    %.2f", df[args.target].std())
    logger.info("    Min:    %.2f", df[args.target].min())
    logger.info("    25%%:    %.2f", df[args.target].quantile(0.25))
    logger.info("    50%%:    %.2f", df[args.target].quantile(0.50))
    logger.info("    75%%:    %.2f", df[args.target].quantile(0.75))
    logger.info("    Max:    %.2f", df[args.target].max())
    logger.info("")
    logger.info("  FOM1_40 range: [%.2f, %.2f]", df["FOM1_40"].min(), df["FOM1_40"].max())
    logger.info("  FOM1_100 range: [%.2f, %.2f]", df["FOM1_100"].min(), df["FOM1_100"].max())
    logger.info("  Output directory: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
