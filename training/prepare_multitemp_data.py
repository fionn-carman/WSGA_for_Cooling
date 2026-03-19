#!/usr/bin/env python3
"""
Prepare multi-temperature training datasets from WTT extraction data.

Generates two formats from the multi-temperature extraction CSV:

1. **Wide format** (existing architecture):
   One row per molecule, columns for each (property × temperature).
   Used by separate per-temperature XGBoost models (train_5fold_rfe.py).

2. **Long format** (new temperature-as-feature architecture):
   One row per (molecule × temperature), with T as a column alongside
   RDKit descriptors. Used by unified per-property models (train_temp_feature.py).

Both formats compute RDKit descriptors and apply quality filters.

Usage:
    python prepare_multitemp_data.py
    python prepare_multitemp_data.py --input ../NIST_WTT_Collection/wtt_multitemp_collection.csv
    python prepare_multitemp_data.py --wide-only     # skip long format
    python prepare_multitemp_data.py --long-only     # skip wide format
"""

import argparse
import logging
import sys
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

# RDKit descriptors
DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]
DESCRIPTOR_FUNCS = [desc[1] for desc in Descriptors._descList]

# Properties and their units (must match config._TEMP_PROPS)
PROPERTIES = [
    ("density", "g_cm3"),
    ("kv", "cSt"),
    ("tc", "W_mK"),
    ("cpsat", "J_K_mol"),
]

# Temperature grid (must match config.TEMP_GRID_C)
TEMP_GRID_C = list(range(0, 110, 10))

# Sanity ranges for filtering
SANITY_RANGES = {
    "density": (0.4, 2.5),
    "kv": (0.05, 50000),
    "tc": (0.03, 1.0),
    "cpsat": (20, 2000),
}

# Properties to log-transform during training
LOG_TARGETS = {"kv"}


def calc_descriptors(mol):
    """Compute all RDKit descriptors for a single Mol object."""
    if mol is None:
        return [np.nan] * len(DESCRIPTOR_FUNCS)
    vals = []
    for func in DESCRIPTOR_FUNCS:
        try:
            vals.append(func(mol))
        except Exception:
            vals.append(np.nan)
    return vals


def load_and_validate(csv_path: str) -> pd.DataFrame:
    """Load multi-temp extraction CSV and validate."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    # Must have SMILES
    has_smiles = df["SMILES"].str.len() > 0
    logger.info("  With SMILES: %d / %d", has_smiles.sum(), len(df))
    df = df[has_smiles].copy()

    # Canonicalize SMILES
    def canon(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    df["SMILES"] = df["SMILES"].apply(canon)
    df = df[df["SMILES"].notna()].copy()
    logger.info("  Valid SMILES: %d", len(df))

    # Deduplicate by canonical SMILES
    before = len(df)
    df = df.drop_duplicates(subset=["SMILES"], keep="first")
    logger.info("  After dedup: %d (removed %d duplicates)", len(df), before - len(df))

    return df


def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Add RDKit descriptor columns."""
    logger.info("Computing RDKit descriptors for %d molecules...", len(df))

    mols = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
    desc_data = [calc_descriptors(mol) for mol in mols]
    desc_df = pd.DataFrame(desc_data, columns=DESCRIPTOR_NAMES, index=df.index)

    # Replace inf with NaN
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are all NaN or constant
    n_before = desc_df.shape[1]
    desc_df = desc_df.dropna(axis=1, how="all")
    nunique = desc_df.nunique()
    desc_df = desc_df.loc[:, nunique > 1]
    logger.info(
        "  Descriptors: %d → %d (dropped %d constant/NaN columns)",
        n_before, desc_df.shape[1], n_before - desc_df.shape[1],
    )

    return pd.concat([df, desc_df], axis=1)


def generate_wide_format(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, int]:
    """Generate wide-format CSVs: one per (property × temperature).

    Each file has columns: SMILES, <target>, <descriptor_1>, ..., <descriptor_N>
    Compatible with existing train_5fold_rfe.py pipeline.
    """
    logger.info("\n" + "=" * 60)
    logger.info("WIDE FORMAT: Separate CSVs per property × temperature")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    descriptor_cols = [c for c in df.columns if c in DESCRIPTOR_NAMES]

    for prop, unit in PROPERTIES:
        lo, hi = SANITY_RANGES.get(prop, (None, None))

        for t_c in TEMP_GRID_C:
            col_name = f"{prop}_{t_c}C_{unit}"
            if col_name not in df.columns:
                continue

            # Get rows with valid property values
            vals = pd.to_numeric(df[col_name], errors="coerce")
            valid = vals.notna()
            if lo is not None:
                valid &= vals >= lo
            if hi is not None:
                valid &= vals <= hi

            subset = df[valid].copy()
            if len(subset) < 10:
                continue

            # Build output: SMILES + target + descriptors
            target_name = col_name
            out_df = subset[["SMILES"]].copy()
            out_df[target_name] = vals[valid].values

            for desc_col in descriptor_cols:
                out_df[desc_col] = pd.to_numeric(
                    subset[desc_col], errors="coerce"
                ).values

            # Drop rows with NaN descriptors
            out_df = out_df.dropna(subset=descriptor_cols)

            out_path = output_dir / f"{col_name}_cleaned.csv"
            out_df.to_csv(out_path, index=False)
            counts[col_name] = len(out_df)

            logger.info("  %s: %d molecules → %s", col_name, len(out_df), out_path.name)

    logger.info("Generated %d wide-format files", len(counts))
    return counts


def generate_long_format(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, int]:
    """Generate long-format CSVs: one per property, with T as a feature.

    Each file has columns: SMILES, T_K, T_C, <target>, <descriptor_1>, ...
    One row per (molecule × temperature). For train_temp_feature.py.
    """
    logger.info("\n" + "=" * 60)
    logger.info("LONG FORMAT: Single CSV per property with T as feature")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    descriptor_cols = [c for c in df.columns if c in DESCRIPTOR_NAMES]

    for prop, unit in PROPERTIES:
        lo, hi = SANITY_RANGES.get(prop, (None, None))

        rows = []
        for _, mol_row in df.iterrows():
            smiles = mol_row["SMILES"]
            desc_vals = {c: mol_row[c] for c in descriptor_cols}

            for t_c in TEMP_GRID_C:
                col_name = f"{prop}_{t_c}C_{unit}"
                if col_name not in df.columns:
                    continue

                val_str = mol_row.get(col_name, "")
                try:
                    val = float(val_str) if val_str else None
                except (ValueError, TypeError):
                    val = None

                if val is None:
                    continue
                if lo is not None and val < lo:
                    continue
                if hi is not None and val > hi:
                    continue

                row = {
                    "SMILES": smiles,
                    "T_C": t_c,
                    "T_K": round(t_c + 273.15, 2),
                }
                row[prop] = val
                row.update(desc_vals)
                rows.append(row)

        if not rows:
            continue

        long_df = pd.DataFrame(rows)

        # Convert descriptor columns to numeric
        for col in descriptor_cols:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

        # Drop rows with NaN descriptors
        long_df = long_df.dropna(subset=descriptor_cols)

        out_path = output_dir / f"{prop}_long.csv"
        long_df.to_csv(out_path, index=False)
        counts[prop] = len(long_df)

        n_molecules = long_df["SMILES"].nunique()
        n_temps = long_df["T_C"].nunique()
        logger.info(
            "  %s: %d rows (%d molecules × up to %d temps) → %s",
            prop, len(long_df), n_molecules, n_temps, out_path.name,
        )

    logger.info("Generated %d long-format files", len(counts))
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-temperature training datasets"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../NIST_WTT_Collection/wtt_multitemp_collection.csv",
        help="Multi-temperature extraction CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_multitemp",
        help="Output directory for training CSVs",
    )
    parser.add_argument("--wide-only", action="store_true", help="Only generate wide format")
    parser.add_argument("--long-only", action="store_true", help="Only generate long format")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input CSV not found: %s", input_path)
        logger.info(
            "Run the multi-temperature extractor first:\n"
            "  python wtt_extractor.py --multitemp --proxy socks5://localhost:1081"
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Load and validate
    df = load_and_validate(str(input_path))
    if df.empty:
        logger.error("No valid molecules after filtering")
        sys.exit(1)

    # Compute descriptors
    df = compute_descriptors(df)

    # Generate formats
    if not args.long_only:
        wide_counts = generate_wide_format(df, output_dir / "wide")

    if not args.wide_only:
        long_counts = generate_long_format(df, output_dir / "long")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Input molecules: %d", len(df))
    if not args.long_only:
        logger.info("Wide-format files: %d", len(wide_counts))
        for target, n in sorted(wide_counts.items()):
            logger.info("  %s: %d", target, n)
    if not args.wide_only:
        logger.info("Long-format files: %d", len(long_counts))
        for prop, n in sorted(long_counts.items()):
            logger.info("  %s: %d rows", prop, n)


if __name__ == "__main__":
    main()
