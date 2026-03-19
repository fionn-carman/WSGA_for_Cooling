"""
Compute beta (thermal expansion coefficient) at all temperature points.

For each molecule, fits quadratic ρ(T) = a + bT + cT² to density data,
then evaluates β(T) = -(1/ρ(T)) × dρ/dT at each temperature.

Input:  training/data/nist_8100/density_cho_cleaned.csv
Output: training/data/nist_8100/beta_all_temps.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "nist_8100"
TEMPS_C = list(range(0, 110, 10))  # 0, 10, 20, ..., 100


def main():
    df = pd.read_csv(DATA_DIR / "density_cho_cleaned.csv")
    print(f"Loaded density data: {len(df)} molecules")

    density_cols = [f"density_{t}C_g_cm3" for t in TEMPS_C]

    results = []
    for _, row in df.iterrows():
        smi = row["SMILES"]
        name = row["name"]

        # Extract non-NaN density values
        temps = []
        densities = []
        for t, col in zip(TEMPS_C, density_cols):
            val = row[col]
            if pd.notna(val):
                temps.append(t)
                densities.append(val)

        n_points = len(temps)
        result = {"SMILES": smi, "name": name, "n_points": n_points}

        if n_points < 3:
            for t in TEMPS_C:
                result[f"beta_{t}C"] = np.nan
            result["fit_r2"] = np.nan
            result["flag"] = "too_few_points"
            results.append(result)
            continue

        # Fit quadratic: ρ(T) = a + bT + cT²
        temps_arr = np.array(temps, dtype=float)
        dens_arr = np.array(densities, dtype=float)
        coeffs = np.polyfit(temps_arr, dens_arr, 2)  # [c, b, a]
        c_coeff, b_coeff, a_coeff = coeffs

        # R²
        predicted = np.polyval(coeffs, temps_arr)
        ss_res = np.sum((dens_arr - predicted) ** 2)
        ss_tot = np.sum((dens_arr - np.mean(dens_arr)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Evaluate beta at each temperature
        flags = []
        if r2 < 0.99:
            flags.append("low_r2")
        if n_points < 6:
            flags.append("few_points")

        # Density drop check
        rho_0 = a_coeff
        rho_100 = a_coeff + b_coeff * 100 + c_coeff * 100**2
        if rho_0 > 0 and rho_100 / rho_0 < 0.5:
            flags.append("steep_drop")

        for t in TEMPS_C:
            rho_t = a_coeff + b_coeff * t + c_coeff * t**2
            drho_dT = b_coeff + 2 * c_coeff * t
            if rho_t > 0:
                beta_t = -(1.0 / rho_t) * drho_dT
            else:
                beta_t = np.nan
            result[f"beta_{t}C"] = beta_t

        # Flag any negative or extreme betas
        beta_vals = [result[f"beta_{t}C"] for t in TEMPS_C
                     if pd.notna(result[f"beta_{t}C"])]
        if any(b < 0 for b in beta_vals):
            flags.append("negative_beta")
        if any(b > 0.005 for b in beta_vals):
            flags.append("high_beta")

        result["fit_r2"] = r2
        result["flag"] = "|".join(flags) if flags else ""
        results.append(result)

    beta_df = pd.DataFrame(results)

    # Reorder columns
    col_order = (["SMILES", "name"] +
                 [f"beta_{t}C" for t in TEMPS_C] +
                 ["fit_r2", "n_points", "flag"])
    beta_df = beta_df[col_order]

    out_path = DATA_DIR / "beta_all_temps.csv"
    beta_df.to_csv(out_path, index=False)

    # Summary
    valid_40 = beta_df["beta_40C"].notna() & (beta_df["beta_40C"] > 0)
    print(f"Beta computed: {valid_40.sum()}/{len(beta_df)} molecules (valid at 40C)")
    print(f"Beta_40C range: {beta_df.loc[valid_40, 'beta_40C'].min():.6f} – "
          f"{beta_df.loc[valid_40, 'beta_40C'].max():.6f} K⁻¹")
    print(f"Beta_40C median: {beta_df.loc[valid_40, 'beta_40C'].median():.6f} K⁻¹")

    flagged = beta_df[beta_df["flag"] != ""]
    print(f"Flagged: {len(flagged)} molecules")
    for flag in ["low_r2", "negative_beta", "high_beta", "few_points",
                 "steep_drop", "too_few_points"]:
        n = flagged["flag"].str.contains(flag).sum() if len(flagged) > 0 else 0
        if n > 0:
            print(f"  {flag}: {n}")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
