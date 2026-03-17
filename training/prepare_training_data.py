#!/usr/bin/env python3
"""
Data preparation for model retraining.

Handles three data cleaning tasks:
  1. Flash point: canonicalize SMILES + median dedup + recompute descriptors
  2. DC: canonicalize SMILES + dedup + recompute descriptors
  3. FOM1: compute RDKit descriptors (CSVs currently have only SMILES + target)
  4. BP/MP: canonicalize + dedup (for future use)

Usage:
    python prepare_training_data.py --mode all
    python prepare_training_data.py --mode flashpoint
    python prepare_training_data.py --mode fom1
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# RDKit descriptor definitions (same as data_preparation.py)
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


def canonicalize_smiles(smi):
    """Canonicalize a SMILES string via RDKit. Returns None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def backup_file(src: Path, backup_dir: Path):
    """Copy file to backup directory if it exists."""
    if src.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        dst = backup_dir / src.name
        shutil.copy2(src, dst)
        logger.info("  Backed up %s → %s", src.name, dst)


def compute_descriptor_df(smiles_series: pd.Series) -> pd.DataFrame:
    """Compute all RDKit descriptors for a Series of SMILES strings."""
    logger.info("  Computing %d RDKit descriptors for %d molecules...",
                len(DESCRIPTOR_NAMES), len(smiles_series))
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_series]
    n_failed = sum(1 for m in mols if m is None)
    if n_failed > 0:
        logger.warning("  %d SMILES failed to parse", n_failed)

    desc_data = [calc_descriptors(mol) for mol in mols]
    desc_df = pd.DataFrame(desc_data, columns=DESCRIPTOR_NAMES)

    # Clean: coerce to numeric, replace inf → NaN → 0
    for col in desc_df.columns:
        desc_df[col] = pd.to_numeric(desc_df[col], errors="coerce")
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return desc_df


def canonicalize_dedup_recompute(csv_path: Path, target_col: str, backup_dir: Path,
                                  agg_func: str = "median"):
    """
    Canonicalize SMILES, deduplicate (aggregating target), recompute descriptors.

    Used for flashpoint, DC, BP, MP datasets.
    """
    logger.info("Processing %s (target: %s)", csv_path.name, target_col)
    backup_file(csv_path, backup_dir)

    df = pd.read_csv(csv_path)
    n_before = len(df)
    n_cols_before = len(df.columns)
    logger.info("  Loaded: %d rows, %d columns", n_before, n_cols_before)

    # Canonicalize SMILES
    df["SMILES"] = df["SMILES"].apply(canonicalize_smiles)
    n_invalid = df["SMILES"].isna().sum()
    if n_invalid > 0:
        logger.warning("  Dropping %d rows with invalid SMILES", n_invalid)
        df = df.dropna(subset=["SMILES"])

    # Ensure target is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Deduplicate: group by canonical SMILES, take median (or mean) of target
    n_unique = df["SMILES"].nunique()
    n_dupes = len(df) - n_unique
    logger.info("  Found %d unique SMILES (%d duplicates)", n_unique, n_dupes)

    df_dedup = df.groupby("SMILES", as_index=False).agg({target_col: agg_func})
    logger.info("  After dedup: %d rows (agg=%s)", len(df_dedup), agg_func)

    # Recompute descriptors
    desc_df = compute_descriptor_df(df_dedup["SMILES"])

    # Combine: SMILES + target + descriptors
    result = pd.concat([
        df_dedup[["SMILES", target_col]].reset_index(drop=True),
        desc_df.reset_index(drop=True)
    ], axis=1)

    # Save
    result.to_csv(csv_path, index=False)
    logger.info("  Saved: %d rows, %d columns → %s", len(result), len(result.columns), csv_path.name)
    logger.info("  Summary: %d → %d rows", n_before, len(result))

    # Target stats
    logger.info("  %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                target_col, result[target_col].mean(), result[target_col].std(),
                result[target_col].min(), result[target_col].max())
    return result


def compute_fom1_descriptors(csv_path: Path, target_col: str, backup_dir: Path):
    """
    Compute RDKit descriptors for FOM1 CSVs (which only have SMILES + target).
    Also canonicalize SMILES and dedup.
    """
    logger.info("Processing %s (target: %s)", csv_path.name, target_col)
    backup_file(csv_path, backup_dir)

    df = pd.read_csv(csv_path)
    n_before = len(df)
    logger.info("  Loaded: %d rows, %d columns", n_before, len(df.columns))

    # Canonicalize SMILES
    df["SMILES"] = df["SMILES"].apply(canonicalize_smiles)
    n_invalid = df["SMILES"].isna().sum()
    if n_invalid > 0:
        logger.warning("  Dropping %d rows with invalid SMILES", n_invalid)
        df = df.dropna(subset=["SMILES"])

    # Ensure target is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Dedup (median aggregation)
    n_unique = df["SMILES"].nunique()
    n_dupes = len(df) - n_unique
    if n_dupes > 0:
        logger.info("  Found %d duplicate SMILES, taking median", n_dupes)
        df = df.groupby("SMILES", as_index=False).agg({target_col: "median"})

    # Compute descriptors
    desc_df = compute_descriptor_df(df["SMILES"])

    # Combine: SMILES + target + descriptors
    result = pd.concat([
        df[["SMILES", target_col]].reset_index(drop=True),
        desc_df.reset_index(drop=True)
    ], axis=1)

    # Save
    result.to_csv(csv_path, index=False)
    logger.info("  Saved: %d rows, %d columns → %s", len(result), len(result.columns), csv_path.name)
    logger.info("  Summary: %d → %d rows, 2 → %d columns", n_before, len(result), len(result.columns))

    # Target stats
    logger.info("  %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                target_col, result[target_col].mean(), result[target_col].std(),
                result[target_col].min(), result[target_col].max())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data: canonicalize, dedup, compute descriptors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["flashpoint", "dc", "fom1", "bp", "mp", "all"],
        default="all",
        help="Which dataset(s) to process",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing cleaned CSV files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    backup_dir = data_dir / "pre_retrain_backup"

    modes = [args.mode] if args.mode != "all" else ["flashpoint", "dc", "fom1"]

    logger.info("=" * 60)
    logger.info("TRAINING DATA PREPARATION")
    logger.info("Modes: %s", modes)
    logger.info("Data dir: %s", data_dir)
    logger.info("Backup dir: %s", backup_dir)
    logger.info("=" * 60)

    for mode in modes:
        logger.info("")
        logger.info("-" * 60)

        if mode == "flashpoint":
            canonicalize_dedup_recompute(
                data_dir / "flashpoint_cleaned.csv",
                "flashpoint",
                backup_dir,
            )

        elif mode == "dc":
            canonicalize_dedup_recompute(
                data_dir / "DC_exp_cleaned.csv",
                "DC_exp",
                backup_dir,
            )

        elif mode == "fom1":
            compute_fom1_descriptors(
                data_dir / "FOM1_exp_40_cleaned.csv",
                "FOM1_exp_40",
                backup_dir,
            )
            compute_fom1_descriptors(
                data_dir / "FOM1_exp_100_cleaned.csv",
                "FOM1_exp_100",
                backup_dir,
            )

        elif mode == "bp":
            canonicalize_dedup_recompute(
                data_dir / "BP-Measured_cleaned.csv",
                "BP-Measured",
                backup_dir,
            )

        elif mode == "mp":
            canonicalize_dedup_recompute(
                data_dir / "MP-Measured_cleaned.csv",
                "MP-Measured",
                backup_dir,
            )

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
