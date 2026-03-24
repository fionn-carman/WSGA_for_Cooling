#!/usr/bin/env python3
"""
Compute dielectric constant at 25C (298.15 K) from LB raw data.

Temperature standardisation logic (no polynomial fits in LB):
1. Direct: any measurement within +/-5 K of 298.15 K -> use median
2. Interpolated: closest bracketing pair (gap <= 30 K) -> linear interpolation
3. No 25C value available

Pools all T/epsilon measurements per compound across all references.

Usage:
    python compute_dc_25C.py [--input lb_dc_raw.csv] [--output lb_dc_25C.csv]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "lb_dc_raw.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "lb_dc_25C.csv"

T_TARGET = 298.15  # 25 degrees C in Kelvin
T_DIRECT_TOL = 5.0  # K tolerance for direct use
T_INTERP_GAP = 30.0  # Max gap for linear interpolation


def compute_dc_for_compound(group: pd.DataFrame) -> dict:
    """Compute DC_25C for a single compound from pooled T/epsilon data.

    Returns dict with DC_25C, dc_method, n_measurements, T_range_K.
    """
    temps = group["T_K"].values
    epsilons = group["epsilon"].values
    n = len(temps)

    result = {
        "DC_25C": None,
        "dc_method": "no_25C_available",
        "n_measurements": n,
        "T_range_K": f"{temps.min():.1f}-{temps.max():.1f}" if n > 0 else "",
    }

    # Tier 1: Direct — measurements within ±T_DIRECT_TOL of target
    direct_mask = np.abs(temps - T_TARGET) <= T_DIRECT_TOL
    if direct_mask.any():
        direct_eps = epsilons[direct_mask]
        result["DC_25C"] = float(np.median(direct_eps))
        result["dc_method"] = "direct"
        return result

    # Tier 2: Linear interpolation — closest bracketing pair
    below_mask = temps < T_TARGET
    above_mask = temps > T_TARGET
    if below_mask.any() and above_mask.any():
        # Find closest T below and above target
        below_temps = temps[below_mask]
        below_eps = epsilons[below_mask]
        above_temps = temps[above_mask]
        above_eps = epsilons[above_mask]

        # Closest below: highest T that's below target
        idx_low = np.argmax(below_temps)
        t_low = below_temps[idx_low]
        eps_low = below_eps[idx_low]

        # Closest above: lowest T that's above target
        idx_high = np.argmin(above_temps)
        t_high = above_temps[idx_high]
        eps_high = above_eps[idx_high]

        gap = t_high - t_low
        if gap <= T_INTERP_GAP and gap > 0:
            # Linear interpolation
            frac = (T_TARGET - t_low) / gap
            dc_25c = eps_low + frac * (eps_high - eps_low)
            result["DC_25C"] = float(dc_25c)
            result["dc_method"] = "interpolated"
            return result

    return result


def standardise_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Group by compound, compute DC_25C for each."""
    # Group by compound number (unique identifier)
    groups = df.groupby("compound_no")

    results = []
    method_counts = {"direct": 0, "interpolated": 0, "no_25C_available": 0}

    for compound_no, group in groups:
        # Get compound metadata from first row
        first = group.iloc[0]
        dc_result = compute_dc_for_compound(group)

        results.append({
            "compound_no": compound_no,
            "mol_formula": first["mol_formula"],
            "name": first["name"],
            "cas_number": first["cas_number"],
            **dc_result,
        })
        method_counts[dc_result["dc_method"]] += 1

    log.info("Temperature standardisation methods: %s", method_counts)
    return pd.DataFrame(results)


def validate_dc_values(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and remove suspicious DC_25C values."""
    valid_mask = df["DC_25C"].notna()
    n_before = valid_mask.sum()

    # Remove negative values
    neg_mask = valid_mask & (df["DC_25C"] < 0)
    if neg_mask.any():
        log.warning("Negative DC_25C values found (%d), setting to NaN", neg_mask.sum())
        df.loc[neg_mask, "DC_25C"] = None
        df.loc[neg_mask, "dc_method"] = "no_25C_available"

    # Flag very high values
    high_mask = valid_mask & (df["DC_25C"] > 200)
    if high_mask.any():
        log.warning("Very high DC_25C values (%d):", high_mask.sum())
        for _, row in df[high_mask].iterrows():
            log.warning("  %s: DC_25C=%.2f", row["name"], row["DC_25C"])

    n_after = df["DC_25C"].notna().sum()
    if n_before != n_after:
        log.info("Validation removed %d values", n_before - n_after)

    return df


def main():
    parser = argparse.ArgumentParser(description="Compute DC at 25C from LB raw data")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV")
    args = parser.parse_args()

    if not Path(args.input).exists():
        log.error("Input not found: %s", args.input)
        return

    df = pd.read_csv(args.input)
    log.info("Loaded %d records from %s", len(df), args.input)
    log.info("Unique compounds: %d", df["compound_no"].nunique())

    result_df = standardise_temperature(df)
    result_df = validate_dc_values(result_df)

    # Summary
    has_dc = result_df["DC_25C"].notna()
    log.info("Compounds with DC at 25C: %d / %d (%.1f%%)",
             has_dc.sum(), len(result_df), 100 * has_dc.sum() / len(result_df))

    if has_dc.any():
        log.info("DC_25C range: %.4f - %.4f",
                 result_df.loc[has_dc, "DC_25C"].min(),
                 result_df.loc[has_dc, "DC_25C"].max())

        # Method breakdown
        log.info("Method breakdown:")
        for method, count in result_df["dc_method"].value_counts().items():
            log.info("  %s: %d", method, count)

    # Spot check known values
    for name_substr, expected in [("methanol", 33), ("2-propanol", 19)]:
        matches = result_df[result_df["name"].str.contains(name_substr, case=False, na=False)]
        if len(matches) > 0:
            row = matches.iloc[0]
            if pd.notna(row["DC_25C"]):
                log.info("Spot check: %s -> DC_25C=%.2f (expected ~%d)",
                         row["name"], row["DC_25C"], expected)

    result_df.to_csv(args.output, index=False)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
