#!/usr/bin/env python3
"""
Build final merged DC dataset from resolved LB data + existing CHNO DC_exp_cleaned.csv.

Steps:
1. Load resolved LB data
2. Filter: CHNO-only organic (contains C, only C/H/N/O), MW <= 500
3. Sanity check DC values (0.5-200 range)
4. Deduplicate against existing DC_exp_cleaned.csv by canonical SMILES
5. Merge (SMILES + DC_exp only — no descriptors)

Usage:
    python build_lb_dataset.py [--input lb_dc_resolved.csv] [--existing PATH] [--output PATH]
                               [--no-merge] [--report PATH]
"""

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT = SCRIPT_DIR / "lb_dc_resolved.csv"
DEFAULT_EXISTING = REPO_DIR / "training" / "data" / "CHNO" / "constraints" / "DC_exp_cleaned.csv"
DEFAULT_OUTPUT = REPO_DIR / "training" / "data" / "CHNO" / "constraints" / "DC_exp_cleaned.csv"
DEFAULT_REPORT = SCRIPT_DIR / "merge_report.txt"

ALLOWED_ATOMS = {"C", "H", "N", "O"}
MAX_MW = 500
DC_MIN = 0.5
DC_MAX = 200.0


def filter_chno_organic(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to CHNO-only organic compounds with MW <= MAX_MW."""
    valid_rows = []
    reasons = {"no_smiles": 0, "invalid_smiles": 0, "non_chno": 0, "high_mw": 0, "no_carbon": 0}

    for idx, row in df.iterrows():
        smi = str(row.get("SMILES", "")).strip()
        if not smi or smi == "nan" or smi == "None":
            reasons["no_smiles"] += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            reasons["invalid_smiles"] += 1
            continue

        atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
        if "C" not in atoms:
            reasons["no_carbon"] += 1
            continue
        if not atoms.issubset(ALLOWED_ATOMS):
            reasons["non_chno"] += 1
            continue

        mw = Descriptors.ExactMolWt(mol)
        if mw > MAX_MW:
            reasons["high_mw"] += 1
            continue

        # Canonicalise SMILES
        df.at[idx, "SMILES"] = Chem.MolToSmiles(mol)
        valid_rows.append(idx)

    log.info("CHNO filter: %d -> %d (rejected: %s)", len(df), len(valid_rows), reasons)
    return df.loc[valid_rows].copy()


def sanity_check_dc(df: pd.DataFrame) -> pd.DataFrame:
    """Remove entries with suspicious DC values."""
    n_before = len(df)

    # Remove no DC
    has_dc = df["DC_25C"].notna()
    df = df[has_dc].copy()

    # Range check
    valid_range = (df["DC_25C"] >= DC_MIN) & (df["DC_25C"] <= DC_MAX)
    out_of_range = df[~valid_range]
    if len(out_of_range) > 0:
        log.warning("DC values outside [%.1f, %.1f] range (%d):", DC_MIN, DC_MAX, len(out_of_range))
        for _, row in out_of_range.iterrows():
            log.warning("  %s: DC=%.4f", row["name"], row["DC_25C"])

    df = df[valid_range].copy()
    log.info("Sanity check: %d -> %d", n_before, len(df))
    return df


def deduplicate_and_merge(
    lb_df: pd.DataFrame,
    existing_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Deduplicate LB against existing data, merge, return stats."""
    existing_smiles = set(existing_df["SMILES"].unique())
    lb_unique = lb_df.drop_duplicates(subset="SMILES")

    # Split into overlap and new
    is_new = ~lb_unique["SMILES"].isin(existing_smiles)
    new_lb = lb_unique[is_new].copy()
    overlap_lb = lb_unique[~is_new].copy()

    stats = {
        "existing_count": len(existing_df),
        "lb_total": len(lb_df),
        "lb_unique_smiles": len(lb_unique),
        "lb_new": len(new_lb),
        "lb_overlap": len(overlap_lb),
    }

    log.info("Deduplication: %d LB unique SMILES, %d new, %d overlap",
             stats["lb_unique_smiles"], stats["lb_new"], stats["lb_overlap"])

    if len(new_lb) == 0:
        log.info("No new compounds to add")
        stats["final_count"] = len(existing_df)
        return existing_df, stats

    # Build new rows (SMILES + DC_exp only)
    new_rows = pd.DataFrame({
        "SMILES": new_lb["SMILES"].values,
        "DC_exp": new_lb["DC_25C"].values,
    })

    # Merge
    merged = pd.concat([existing_df, new_rows], ignore_index=True)

    # Final dedup
    n_before = len(merged)
    merged = merged.drop_duplicates(subset="SMILES", keep="first")
    if len(merged) < n_before:
        log.warning("Removed %d duplicate SMILES after merge", n_before - len(merged))

    stats["final_count"] = len(merged)
    log.info("Final dataset: %d molecules", stats["final_count"])

    return merged, stats


def write_report(stats: dict, report_path: str):
    """Write merge statistics report."""
    lines = [
        "LB Dielectric Constant Dataset Merge Report",
        "=" * 50,
        "",
        f"Existing CHNO dataset:  {stats['existing_count']} molecules",
        f"LB total extracted:     {stats['lb_total']} records",
        f"LB unique SMILES:       {stats['lb_unique_smiles']}",
        f"LB new (not in existing): {stats['lb_new']}",
        f"LB overlap:             {stats['lb_overlap']}",
        f"Final dataset:          {stats.get('final_count', 'N/A')} molecules",
        "",
    ]
    if stats.get("final_count"):
        growth = stats["final_count"] / stats["existing_count"]
        lines.append(
            f"Growth: {stats['existing_count']} -> {stats['final_count']} ({growth:.2f}x)"
        )
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Report saved to %s", report_path)


def main():
    parser = argparse.ArgumentParser(description="Build merged CHNO DC dataset")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Resolved LB CSV")
    parser.add_argument("--existing", default=str(DEFAULT_EXISTING), help="Existing DC dataset")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output merged CSV")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Merge report path")
    parser.add_argument("--no-merge", action="store_true",
                        help="Don't merge, just process LB data")
    args = parser.parse_args()

    if not Path(args.input).exists():
        log.error("Input not found: %s", args.input)
        return

    df = pd.read_csv(args.input)
    log.info("Loaded %d records from %s", len(df), args.input)

    # Step 1: Filter to CHNO organic
    df = filter_chno_organic(df)

    # Step 2: Sanity check DC values
    df = sanity_check_dc(df)

    log.info("LB compounds ready for merge: %d", len(df))

    if args.no_merge:
        result = pd.DataFrame({"SMILES": df["SMILES"].values, "DC_exp": df["DC_25C"].values})
        result.to_csv(args.output, index=False)
        log.info("Saved LB-only dataset to %s (%d molecules)", args.output, len(result))
        return

    # Step 3: Load existing and merge
    if not Path(args.existing).exists():
        log.error("Existing dataset not found: %s", args.existing)
        return

    existing_df = pd.read_csv(args.existing)
    log.info("Existing dataset: %d molecules, %d columns",
             len(existing_df), len(existing_df.columns))

    # Backup existing
    backup_path = Path(args.existing).with_name("DC_exp_cleaned_pre_lb.csv")
    if not backup_path.exists():
        shutil.copy2(args.existing, backup_path)
        log.info("Backed up existing dataset to %s", backup_path)

    merged_df, stats = deduplicate_and_merge(df, existing_df)

    # Verify column count
    expected_cols = len(existing_df.columns)
    actual_cols = len(merged_df.columns)
    if actual_cols != expected_cols:
        log.warning("Column count mismatch: expected %d, got %d", expected_cols, actual_cols)

    # Verify no duplicate SMILES
    n_unique = merged_df["SMILES"].nunique()
    if n_unique != len(merged_df):
        log.warning("Duplicate SMILES: %d rows but %d unique", len(merged_df), n_unique)

    merged_df.to_csv(args.output, index=False)
    log.info("Saved merged dataset to %s", args.output)

    write_report(stats, args.report)


if __name__ == "__main__":
    main()
