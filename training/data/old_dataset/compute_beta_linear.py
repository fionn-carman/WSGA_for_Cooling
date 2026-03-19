#!/usr/bin/env python3
"""
Compute thermal expansion coefficient (beta) at 40C from linear approximation
using density at 40C and 100C.

    beta_40 = -(1/rho_40) * (rho_100 - rho_40) / (100 - 40)

Produces: beta_40C_linear_cleaned.csv with columns [SMILES, beta_40C_linear]
Only includes molecules present in both density datasets.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

d40 = pd.read_csv(DATA_DIR / "Density_40C_g_cm^3_cleaned.csv",
                   usecols=["SMILES", "Density_40C_g_cm^3"])
d100 = pd.read_csv(DATA_DIR / "Density_100C_g_cm^3_cleaned.csv",
                    usecols=["SMILES", "Density_100C_g_cm^3"])

merged = d40.merge(d100, on="SMILES", how="inner")
print(f"Molecules with both 40C and 100C density: {len(merged)}")

dT = 100.0 - 40.0  # K (same as °C difference)
merged["beta_40C_linear"] = -(1.0 / merged["Density_40C_g_cm^3"]) * \
    (merged["Density_100C_g_cm^3"] - merged["Density_40C_g_cm^3"]) / dT

# Drop any NaN/inf
before = len(merged)
merged = merged[merged["beta_40C_linear"].notna()].copy()
merged = merged[merged["beta_40C_linear"] != float("inf")].copy()
merged = merged[merged["beta_40C_linear"] != float("-inf")].copy()
print(f"After cleaning: {len(merged)} ({before - len(merged)} dropped)")

# Summary stats
beta = merged["beta_40C_linear"]
print(f"Beta range: {beta.min():.6f} to {beta.max():.6f} K^-1")
print(f"Beta mean: {beta.mean():.6f}, median: {beta.median():.6f}")

out = merged[["SMILES", "beta_40C_linear"]]
out_path = DATA_DIR / "beta_40C_linear_cleaned.csv"
out.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
