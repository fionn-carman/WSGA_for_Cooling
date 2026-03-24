#!/usr/bin/env python3
"""
Prepare CHNO constraint datasets from external paper sources + WTT.

Flash Point: Sun et al. 2020 (integrated_dataset_public.csv)
Boiling Point: Zang et al. 2017 (ci6b00625_si_002.xlsx, sheet 'BP') + WTT bp_K
Melting Point: Zang et al. 2017 (ci6b00625_si_002.xlsx, sheet 'MP') + WTT mp_K
Dielectric Constant: CRC handbook (crc_dc_resolved.csv)

All datasets: filter CHNO, MW <= 500, canonicalize, deduplicate.
Output: training/data/CHNO/constraints/{property}_cleaned.csv
"""

import argparse
import logging
import time
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

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "CHNO" / "constraints"
WTT_PATH = SCRIPT_DIR.parent / "data" / "nist_8100" / "wtt_multitemp_merged.csv"

# External data sources (default paths)
FP_SOURCE = Path.home() / "Downloads" / "9275210 (1)" / "integrated_dataset_public.csv"
BPMP_SOURCE = Path.home() / "Downloads" / "ci6b00625_si_002 (1).xlsx"
DC_SOURCE = SCRIPT_DIR.parent.parent / "data_collection" / "crc_dielectric" / "crc_dc_resolved.csv"

ALLOWED_ATOMS = {"C", "H", "N", "O"}
MAX_MW = 500


def filter_chno_smiles(smiles_list: list) -> dict:
    """
    Filter and canonicalize SMILES to CHNO organic (contains C, atoms in CHNO, MW <= 500).
    Returns dict mapping original SMILES -> canonical SMILES.
    """
    mapping = {}
    for smi in smiles_list:
        smi = str(smi).strip()
        if not smi or smi in ("nan", "None", ""):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        atoms = {a.GetSymbol() for a in mol.GetAtoms()}
        if "C" not in atoms:
            continue
        if not atoms.issubset(ALLOWED_ATOMS):
            continue
        if Descriptors.ExactMolWt(mol) > MAX_MW:
            continue
        mapping[smi] = Chem.MolToSmiles(mol)
    return mapping


def deduplicate_median(df: pd.DataFrame, smiles_col: str, target_col: str,
                       max_std: float = None) -> pd.DataFrame:
    """Deduplicate by canonical SMILES, taking median of target. Optionally drop high-std."""
    grouped = df.groupby(smiles_col)[target_col]
    medians = grouped.median()
    stds = grouped.std().fillna(0)
    counts = grouped.count()

    result = pd.DataFrame({
        smiles_col: medians.index,
        target_col: medians.values,
        "_std": stds.values,
        "_count": counts.values,
    })

    if max_std is not None:
        n_before = len(result)
        # Only filter molecules with multiple measurements and high std
        keep = (result["_count"] == 1) | (result["_std"] <= max_std)
        result = result[keep]
        logger.info("  Std filter (max=%.1f): %d -> %d", max_std, n_before, len(result))

    result = result.drop(columns=["_std", "_count"])
    return result


def count_n_containing(smiles_series: pd.Series) -> int:
    """Count molecules containing nitrogen."""
    count = 0
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(str(smi))
        if mol and "N" in {a.GetSymbol() for a in mol.GetAtoms()}:
            count += 1
    return count


# ============================================================
# Flash Point
# ============================================================

def prepare_flashpoint(fp_source: Path) -> pd.DataFrame:
    """Prepare flash point dataset from Sun et al. 2020."""
    logger.info("--- Flash Point (Sun et al. 2020) ---")

    if not fp_source.exists():
        logger.error("Flash point source not found: %s", fp_source)
        return None

    df = pd.read_csv(fp_source)
    logger.info("  Loaded %d records", len(df))

    # Filter CHNO
    smiles_map = filter_chno_smiles(df["smiles"].tolist())
    df = df[df["smiles"].isin(smiles_map)].copy()
    df["SMILES"] = df["smiles"].map(smiles_map)
    logger.info("  After CHNO filter: %d", len(df))

    # Deduplicate (median, drop if std > 5 K)
    result = deduplicate_median(df, "SMILES", "flashpoint", max_std=5.0)
    logger.info("  After dedup: %d molecules", len(result))

    n_with_n = count_n_containing(result["SMILES"])
    logger.info("  N-containing: %d", n_with_n)

    return result


# ============================================================
# Boiling Point
# ============================================================

def prepare_boiling_point(bpmp_source: Path, wtt_path: Path) -> pd.DataFrame:
    """Prepare boiling point dataset from Zang et al. 2017 + WTT."""
    logger.info("--- Boiling Point (Zang et al. 2017 + WTT) ---")

    records = []

    # Source 1: Zang et al. Excel
    if bpmp_source.exists():
        try:
            bp_df = pd.read_excel(bpmp_source, sheet_name="BP")
            logger.info("  Zang BP sheet: %d records, columns: %s",
                       len(bp_df), list(bp_df.columns))

            # Find SMILES and BP columns
            smi_col = None
            bp_col = None
            for col in bp_df.columns:
                col_lower = str(col).lower()
                if "smiles" in col_lower:
                    smi_col = col
                elif "bp" in col_lower or "boiling" in col_lower:
                    bp_col = col

            if smi_col and bp_col:
                smiles_map = filter_chno_smiles(bp_df[smi_col].tolist())
                for _, row in bp_df.iterrows():
                    smi = str(row[smi_col]).strip()
                    if smi in smiles_map:
                        val = row[bp_col]
                        if pd.notna(val):
                            records.append({
                                "SMILES": smiles_map[smi],
                                "BP-Measured": float(val),
                                "source": "zang2017",
                            })
                logger.info("  Zang BP: %d valid records", len(records))
            else:
                logger.warning("  Could not identify SMILES/BP columns in Zang sheet")
        except Exception as e:
            logger.warning("  Error reading Zang BP: %s", e)
    else:
        logger.warning("  Zang Excel not found: %s", bpmp_source)

    # Source 2: WTT bp_K
    n_zang = len(records)
    if wtt_path.exists():
        wtt = pd.read_csv(wtt_path, usecols=["SMILES", "bp_K"], low_memory=False)
        wtt = wtt.dropna(subset=["bp_K"])
        smiles_map = filter_chno_smiles(wtt["SMILES"].tolist())
        for _, row in wtt.iterrows():
            smi = str(row["SMILES"]).strip()
            if smi in smiles_map:
                bp_C = float(row["bp_K"]) - 273.15
                records.append({
                    "SMILES": smiles_map[smi],
                    "BP-Measured": bp_C,
                    "source": "wtt",
                })
        logger.info("  WTT BP: %d records", len(records) - n_zang)

    if not records:
        logger.error("  No BP data collected")
        return None

    df = pd.DataFrame(records)
    logger.info("  Total BP records: %d", len(df))

    # Deduplicate by SMILES (median)
    result = deduplicate_median(df, "SMILES", "BP-Measured")
    logger.info("  After dedup: %d molecules", len(result))

    n_with_n = count_n_containing(result["SMILES"])
    logger.info("  N-containing: %d", n_with_n)

    return result


# ============================================================
# Melting Point
# ============================================================

def prepare_melting_point(bpmp_source: Path, wtt_path: Path) -> pd.DataFrame:
    """Prepare melting point dataset from Zang et al. 2017 + WTT."""
    logger.info("--- Melting Point (Zang et al. 2017 + WTT) ---")

    records = []

    # Source 1: Zang et al. Excel
    if bpmp_source.exists():
        try:
            mp_df = pd.read_excel(bpmp_source, sheet_name="MP")
            logger.info("  Zang MP sheet: %d records, columns: %s",
                       len(mp_df), list(mp_df.columns))

            smi_col = None
            mp_col = None
            for col in mp_df.columns:
                col_lower = str(col).lower()
                if "smiles" in col_lower:
                    smi_col = col
                elif "mp" in col_lower or "melting" in col_lower:
                    mp_col = col

            if smi_col and mp_col:
                smiles_map = filter_chno_smiles(mp_df[smi_col].tolist())
                for _, row in mp_df.iterrows():
                    smi = str(row[smi_col]).strip()
                    if smi in smiles_map:
                        val = row[mp_col]
                        if pd.notna(val):
                            records.append({
                                "SMILES": smiles_map[smi],
                                "MP-Measured": float(val),
                                "source": "zang2017",
                            })
                logger.info("  Zang MP: %d valid records", len(records))
            else:
                logger.warning("  Could not identify SMILES/MP columns in Zang sheet")
        except Exception as e:
            logger.warning("  Error reading Zang MP: %s", e)
    else:
        logger.warning("  Zang Excel not found: %s", bpmp_source)

    # Source 2: WTT mp_K
    n_zang = len(records)
    if wtt_path.exists():
        wtt = pd.read_csv(wtt_path, usecols=["SMILES", "mp_K"], low_memory=False)
        wtt = wtt.dropna(subset=["mp_K"])
        smiles_map = filter_chno_smiles(wtt["SMILES"].tolist())
        for _, row in wtt.iterrows():
            smi = str(row["SMILES"]).strip()
            if smi in smiles_map:
                mp_C = float(row["mp_K"]) - 273.15
                records.append({
                    "SMILES": smiles_map[smi],
                    "MP-Measured": mp_C,
                    "source": "wtt",
                })
        logger.info("  WTT MP: %d records", len(records) - n_zang)

    if not records:
        logger.error("  No MP data collected")
        return None

    df = pd.DataFrame(records)
    logger.info("  Total MP records: %d", len(df))

    result = deduplicate_median(df, "SMILES", "MP-Measured")
    logger.info("  After dedup: %d molecules", len(result))

    n_with_n = count_n_containing(result["SMILES"])
    logger.info("  N-containing: %d", n_with_n)

    return result


# ============================================================
# Dielectric Constant
# ============================================================

def prepare_dielectric(dc_source: Path) -> pd.DataFrame:
    """Prepare dielectric constant dataset from CRC resolved data."""
    logger.info("--- Dielectric Constant (CRC Handbook) ---")

    if not dc_source.exists():
        logger.error("DC source not found: %s", dc_source)
        return None

    df = pd.read_csv(dc_source)
    logger.info("  Loaded %d records", len(df))

    # Filter valid SMILES + CHNO
    smiles_map = filter_chno_smiles(df["SMILES"].dropna().tolist())
    df = df[df["SMILES"].isin(smiles_map)].copy()
    df["SMILES"] = df["SMILES"].map(smiles_map)
    logger.info("  After CHNO filter: %d", len(df))

    # Need DC_25C
    df = df.dropna(subset=["DC_25C"])

    # Sanity check
    df = df[(df["DC_25C"] >= 0.5) & (df["DC_25C"] <= 200)].copy()
    logger.info("  After DC range filter: %d", len(df))

    # Rename and deduplicate
    result = df[["SMILES", "DC_25C"]].copy()
    result = result.rename(columns={"DC_25C": "DC_exp"})
    result = deduplicate_median(result, "SMILES", "DC_exp")
    logger.info("  After dedup: %d molecules", len(result))

    n_with_n = count_n_containing(result["SMILES"])
    logger.info("  N-containing: %d", n_with_n)

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare CHNO constraint datasets")
    parser.add_argument("--fp_source", default=str(FP_SOURCE))
    parser.add_argument("--bpmp_source", default=str(BPMP_SOURCE))
    parser.add_argument("--dc_source", default=str(DC_SOURCE))
    parser.add_argument("--wtt_path", default=str(WTT_PATH))
    args = parser.parse_args()

    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Flash Point
    fp_df = prepare_flashpoint(Path(args.fp_source))
    if fp_df is not None:
        fp_df.to_csv(DATA_DIR / "flashpoint_cleaned.csv", index=False)
        logger.info("  Saved: %d molecules -> flashpoint_cleaned.csv", len(fp_df))

    # Boiling Point
    bp_df = prepare_boiling_point(Path(args.bpmp_source), Path(args.wtt_path))
    if bp_df is not None:
        bp_df.to_csv(DATA_DIR / "BP-Measured_cleaned.csv", index=False)
        logger.info("  Saved: %d molecules -> BP-Measured_cleaned.csv", len(bp_df))

    # Melting Point
    mp_df = prepare_melting_point(Path(args.bpmp_source), Path(args.wtt_path))
    if mp_df is not None:
        mp_df.to_csv(DATA_DIR / "MP-Measured_cleaned.csv", index=False)
        logger.info("  Saved: %d molecules -> MP-Measured_cleaned.csv", len(mp_df))

    # Dielectric Constant
    dc_df = prepare_dielectric(Path(args.dc_source))
    if dc_df is not None:
        dc_df.to_csv(DATA_DIR / "DC_exp_cleaned.csv", index=False)
        logger.info("  Saved: %d molecules -> DC_exp_cleaned.csv", len(dc_df))

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("CHNO constraint preparation complete (%.1fs)", elapsed)
    logger.info("=" * 60)

    # Summary
    for name, path in [
        ("flashpoint", DATA_DIR / "flashpoint_cleaned.csv"),
        ("BP-Measured", DATA_DIR / "BP-Measured_cleaned.csv"),
        ("MP-Measured", DATA_DIR / "MP-Measured_cleaned.csv"),
        ("DC_exp", DATA_DIR / "DC_exp_cleaned.csv"),
    ]:
        if path.exists():
            tmp = pd.read_csv(path)
            logger.info("  %s: %d molecules, columns: %s",
                        name, len(tmp), list(tmp.columns))


if __name__ == "__main__":
    main()
