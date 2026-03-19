#!/usr/bin/env python3
"""
Precompute Mordred 2D + RDKit descriptors for all unique SMILES across
the constraint property datasets (DC, MP, BP, FP).

Produces: training/data/constraints/descriptors_rdkit_mordred.csv

Dependencies: mordred (pip install mordred), rdkit

Usage:
    python compute_descriptors.py
    python compute_descriptors.py --force   # recompute even if cache exists
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

try:
    from mordred import Calculator, descriptors as mordred_descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "constraints"
OUTPUT_PATH = DATA_DIR / "descriptors_rdkit_mordred.csv"

CSV_FILES = [
    "DC_exp_cleaned.csv",
    "MP-Measured_cleaned.csv",
    "BP-Measured_cleaned.csv",
    "flashpoint_cleaned.csv",
    "biodegradability_cleaned.csv",
]

RDKIT_NAMES = [desc[0] for desc in Descriptors._descList]
RDKIT_FNS = {desc[0]: desc[1] for desc in Descriptors._descList}


def collect_unique_smiles() -> list:
    """Gather unique SMILES from all property CSVs."""
    all_smiles = set()
    for csv_name in CSV_FILES:
        csv_path = DATA_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["SMILES"])
            n = len(df["SMILES"].unique())
            all_smiles.update(df["SMILES"].unique())
            logger.info("  %s: %d unique SMILES", csv_name, n)
        else:
            logger.warning("  %s not found, skipping", csv_name)
    smiles_list = sorted(all_smiles)
    logger.info("Total unique SMILES: %d", len(smiles_list))
    return smiles_list


def compute_rdkit(smiles_list: list) -> pd.DataFrame:
    """Compute all RDKit descriptors."""
    logger.info("Computing RDKit descriptors for %d molecules...", len(smiles_list))
    t0 = time.time()
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({n: np.nan for n in RDKIT_NAMES})
        else:
            row = {}
            for n in RDKIT_NAMES:
                try:
                    row[n] = RDKIT_FNS[n](mol)
                except Exception:
                    row[n] = np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.columns = [f"rdkit_{c}" for c in df.columns]
    elapsed = time.time() - t0
    logger.info("  RDKit: %d descriptors, %.1fs (%.2f ms/mol)",
                len(df.columns), elapsed, 1000 * elapsed / len(smiles_list))
    return df


def compute_mordred(smiles_list: list) -> pd.DataFrame:
    """Compute Mordred 2D descriptors."""
    if not MORDRED_AVAILABLE:
        raise ImportError(
            "mordred not installed. Install with: pip install mordred"
        )

    logger.info("Computing Mordred 2D descriptors for %d molecules...", len(smiles_list))
    calc = Calculator(mordred_descriptors, ignore_3D=True)

    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles("")  # placeholder — produces NaN
        mols.append(mol)

    t0 = time.time()
    result = calc.pandas(mols)
    elapsed = time.time() - t0
    logger.info("  Mordred: %d descriptors, %.1fs (%.2f ms/mol)",
                len(result.columns), elapsed, 1000 * elapsed / len(smiles_list))

    # Convert to numeric, coerce errors to NaN
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result.columns = [f"mordred_{c}" for c in result.columns]
    return result


def clean_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Drop high-NaN and zero-variance columns."""
    desc_cols = [c for c in df.columns if c != "SMILES"]
    n_start = len(desc_cols)

    # Ensure all numeric
    for col in desc_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop columns with >50% NaN
    nan_frac = df[desc_cols].isnull().mean()
    high_nan = nan_frac[nan_frac > 0.5].index.tolist()
    if high_nan:
        df = df.drop(columns=high_nan)
        logger.info("  Dropped %d columns with >50%% NaN", len(high_nan))

    # Drop zero-variance columns
    desc_cols = [c for c in df.columns if c != "SMILES"]
    variances = df[desc_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        df = df.drop(columns=zero_var)
        logger.info("  Dropped %d zero-variance columns", len(zero_var))

    n_end = len(df.columns) - 1
    logger.info("  Cleaning: %d -> %d descriptor columns", n_start, n_end)
    return df


def main():
    parser = argparse.ArgumentParser(description="Precompute descriptors")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cache exists")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        logger.info("Descriptors already cached at %s", OUTPUT_PATH)
        logger.info("Use --force to recompute.")
        df = pd.read_csv(OUTPUT_PATH, nrows=0)
        logger.info("Cached: %d descriptor columns", len(df.columns) - 1)
        return

    logger.info("=" * 60)
    logger.info("Precomputing descriptors (RDKit + Mordred 2D)")
    logger.info("=" * 60)
    t0_total = time.time()

    smiles_list = collect_unique_smiles()

    rdkit_df = compute_rdkit(smiles_list)
    mordred_df = compute_mordred(smiles_list)

    combined = pd.DataFrame({"SMILES": smiles_list})
    combined = pd.concat([combined, rdkit_df, mordred_df], axis=1)
    logger.info("Combined: %d SMILES x %d descriptors (before cleaning)",
                len(combined), len(combined.columns) - 1)

    combined = clean_descriptors(combined)

    combined.to_csv(OUTPUT_PATH, index=False)
    elapsed = time.time() - t0_total
    logger.info("Saved to %s", OUTPUT_PATH)
    logger.info("Final: %d SMILES x %d descriptors (%.1fs total)",
                len(combined), len(combined.columns) - 1, elapsed)

    # NaN stats
    desc_cols = [c for c in combined.columns if c != "SMILES"]
    total_cells = len(combined) * len(desc_cols)
    nan_cells = int(combined[desc_cols].isnull().sum().sum())
    logger.info("NaN cells: %d / %d (%.2f%%)",
                nan_cells, total_cells,
                100 * nan_cells / total_cells if total_cells > 0 else 0)


if __name__ == "__main__":
    main()
