#!/usr/bin/env python3
"""
Compute improved beta (thermal expansion coefficient) and FOM1 from NIST data.

Step 1: Fit quadratic to density(T) for each molecule → β(40°C)
Step 2: Compute improved FOM1 at 40°C from component properties + improved beta
Step 3: Parity plot comparing old FOM1 vs improved FOM1

Usage:
    python compute_improved_fom1.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUT_DIR = Path("pipeline_comparison")
TEMPS_C = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors._descList]


def compute_beta():
    """Step 1: Compute β(40°C) from quadratic fit to density(T) for each molecule."""
    logger.info("=" * 60)
    logger.info("STEP 1: Computing beta from density(T) quadratic fits")
    logger.info("=" * 60)

    df = pd.read_csv(DATA_DIR / "density_cho_cleaned.csv")
    logger.info("Loaded density data: %d molecules", len(df))

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
        if n_points < 3:
            results.append({
                "SMILES": smi, "name": name, "beta_40C": np.nan,
                "rho_40C_fit": np.nan, "fit_r2": np.nan,
                "n_points": n_points, "flag": "too_few_points",
            })
            continue

        # Fit quadratic: ρ(T) = a + bT + cT²
        temps_arr = np.array(temps, dtype=float)
        dens_arr = np.array(densities, dtype=float)
        coeffs = np.polyfit(temps_arr, dens_arr, 2)  # [c, b, a] (highest power first)
        c_coeff, b_coeff, a_coeff = coeffs

        # R² of fit
        predicted = np.polyval(coeffs, temps_arr)
        ss_res = np.sum((dens_arr - predicted) ** 2)
        ss_tot = np.sum((dens_arr - np.mean(dens_arr)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # β(40°C) = -(1/ρ(40)) × dρ/dT|_{T=40}
        # dρ/dT = b + 2cT
        rho_40 = a_coeff + b_coeff * 40 + c_coeff * 40**2
        drho_dT_40 = b_coeff + 2 * c_coeff * 40
        beta_40 = -(1.0 / rho_40) * drho_dT_40

        # Flagging
        flags = []
        if r2 < 0.99:
            flags.append("low_r2")
        if beta_40 < 0:
            flags.append("negative_beta")
        if beta_40 > 0.003:
            flags.append("high_beta")
        if n_points < 6:
            flags.append("few_points")
        # Density drop check (approaching critical point)
        rho_0 = a_coeff  # ρ(0°C)
        rho_100 = a_coeff + b_coeff * 100 + c_coeff * 100**2
        if rho_0 > 0 and rho_100 / rho_0 < 0.5:
            flags.append("steep_drop")

        flag_str = "|".join(flags) if flags else ""

        results.append({
            "SMILES": smi,
            "name": name,
            "beta_40C": beta_40,
            "rho_40C_fit": rho_40,
            "fit_r2": r2,
            "n_points": n_points,
            "flag": flag_str,
        })

    beta_df = pd.DataFrame(results)
    beta_df.to_csv(DATA_DIR / "beta_40C.csv", index=False)

    # Summary
    valid = beta_df["beta_40C"].notna()
    logger.info("Beta computed: %d/%d molecules", valid.sum(), len(beta_df))
    logger.info("Beta range: %.6f – %.6f K⁻¹",
                beta_df.loc[valid, "beta_40C"].min(),
                beta_df.loc[valid, "beta_40C"].max())
    logger.info("Median beta: %.6f K⁻¹", beta_df.loc[valid, "beta_40C"].median())

    flagged = beta_df[beta_df["flag"] != ""]
    logger.info("Flagged molecules: %d", len(flagged))
    for flag in ["low_r2", "negative_beta", "high_beta", "few_points",
                 "steep_drop", "too_few_points"]:
        n = flagged["flag"].str.contains(flag).sum() if len(flagged) > 0 else 0
        if n > 0:
            logger.info("  %s: %d", flag, n)

    logger.info("Saved: %s", DATA_DIR / "beta_40C.csv")
    return beta_df


def compute_improved_fom1(beta_df):
    """Step 2: Compute improved FOM1 at 40°C from component properties."""
    logger.info("=" * 60)
    logger.info("STEP 2: Computing improved FOM1 at 40°C")
    logger.info("=" * 60)

    # Load component datasets — keep SMILES, name, 40°C value, and descriptors
    density_df = pd.read_csv(DATA_DIR / "density_cho_cleaned.csv")
    viscosity_df = pd.read_csv(DATA_DIR / "viscosity_cho_cleaned.csv")
    tc_df = pd.read_csv(DATA_DIR / "tc_cho_cleaned.csv")
    cpsat_df = pd.read_csv(DATA_DIR / "cpsat_cho_cleaned.csv")

    logger.info("Dataset sizes: density=%d, viscosity=%d, tc=%d, cpsat=%d",
                len(density_df), len(viscosity_df), len(tc_df), len(cpsat_df))

    # Beta: keep all computable (non-NaN), positive betas
    beta_valid = beta_df[
        beta_df["beta_40C"].notna() & (beta_df["beta_40C"] > 0)
    ][["SMILES", "beta_40C"]].copy()
    logger.info("Valid beta molecules: %d", len(beta_valid))

    # Descriptor columns (from density CSV — identical across all CSVs)
    desc_cols = [c for c in DESCRIPTOR_NAMES if c in density_df.columns]

    # Build intersection: merge all 4 components + beta on SMILES
    # Keep descriptors from density (they're the same across all CSVs)
    keep_density = ["SMILES", "name", "density_40C_g_cm3"] + desc_cols
    keep_visc = ["SMILES", "kv_40C_cSt"]
    keep_tc = ["SMILES", "tc_40C_W_mK"]
    keep_cp = ["SMILES", "cpsat_40C_J_K_mol"]

    merged = density_df[keep_density].merge(
        viscosity_df[keep_visc], on="SMILES"
    ).merge(
        tc_df[keep_tc], on="SMILES"
    ).merge(
        cpsat_df[keep_cp], on="SMILES"
    ).merge(
        beta_valid, on="SMILES"
    )

    # Drop rows with any NaN in required columns
    required = ["density_40C_g_cm3", "kv_40C_cSt", "tc_40C_W_mK",
                "cpsat_40C_J_K_mol", "beta_40C"]
    merged = merged.dropna(subset=required).reset_index(drop=True)
    logger.info("Intersection molecules (all 4 components + valid beta): %d", len(merged))

    # Compute FOM1
    # FOM1 = k × ((β × Cp_specific × ρ_SI) / (ν_SI × k))^0.2813
    rho = merged["density_40C_g_cm3"].values          # g/cm³
    kv = merged["kv_40C_cSt"].values                  # cSt
    k = merged["tc_40C_W_mK"].values                  # W/mK
    cp_molar = merged["cpsat_40C_J_K_mol"].values     # J/(K·mol)
    beta = merged["beta_40C"].values                   # K⁻¹
    mw = merged["MolWt"].values                        # g/mol (from descriptors)

    # Unit conversions (all SI)
    cp_specific = cp_molar / mw * 1000  # J/(K·mol) / (g/mol) × 1000 g/kg = J/(kg·K)
    rho_SI = rho * 1000                 # kg/m³
    nu_SI = kv * 1e-6                   # m²/s

    inner = (beta * cp_specific * rho_SI) / (nu_SI * k)
    fom1_improved = k * np.power(inner, 0.2813)

    merged["FOM1_improved_40C"] = fom1_improved

    # Reorder columns: SMILES, name, targets, then descriptors
    front_cols = ["SMILES", "name", "FOM1_improved_40C",
                  "density_40C_g_cm3", "kv_40C_cSt", "tc_40C_W_mK",
                  "cpsat_40C_J_K_mol", "beta_40C"]
    other_cols = [c for c in merged.columns if c not in front_cols]
    merged = merged[front_cols + other_cols]

    merged.to_csv(DATA_DIR / "FOM1_improved_40C.csv", index=False)

    logger.info("FOM1 improved range: %.1f – %.1f", fom1_improved.min(), fom1_improved.max())
    logger.info("FOM1 improved median: %.1f", np.median(fom1_improved))
    logger.info("Saved: %s", DATA_DIR / "FOM1_improved_40C.csv")

    return merged


def is_glyme(smi):
    """Detect glymes: ≥3 ether linkages (OD2 bonded to 2 carbons), no carbonyl."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    # Exclude molecules with any C=O
    carbonyl = Chem.MolFromSmarts("[#6]=[#8]")
    if mol.HasSubstructMatch(carbonyl):
        return False
    # Count ether oxygens: O with degree 2, bonded to 2 carbons
    ether = Chem.MolFromSmarts("[OD2]([#6])[#6]")
    matches = mol.GetSubstructMatches(ether)
    return len(matches) >= 3


def plot_old_vs_improved(fom1_improved_df):
    """Step 3: Parity plot comparing old FOM1 vs improved FOM1."""
    logger.info("=" * 60)
    logger.info("STEP 3: Old vs improved FOM1 parity plot")
    logger.info("=" * 60)

    # Load old FOM1
    old_fom1_df = pd.read_csv(DATA_DIR / "FOM1_cho_cleaned.csv")[["SMILES", "FOM1_40C"]]

    # Merge with improved
    comp = fom1_improved_df[["SMILES", "FOM1_improved_40C"]].merge(
        old_fom1_df, on="SMILES"
    ).dropna()
    logger.info("Molecules with both old and improved FOM1: %d", len(comp))

    old = comp["FOM1_40C"].values
    improved = comp["FOM1_improved_40C"].values

    r2 = r2_score(old, improved)
    mae = np.mean(np.abs(old - improved))
    logger.info("Old vs Improved: R²=%.4f, MAE=%.2f", r2, mae)

    # Detect glymes
    glyme_mask = comp["SMILES"].apply(is_glyme).values
    n_glymes = glyme_mask.sum()
    logger.info("Glymes detected: %d", n_glymes)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.scatter(old[~glyme_mask], improved[~glyme_mask],
               alpha=0.3, s=10, color="steelblue", label=f"Other ({(~glyme_mask).sum()})")
    if n_glymes > 0:
        ax.scatter(old[glyme_mask], improved[glyme_mask],
                   alpha=0.8, s=40, color="crimson", marker="D",
                   label=f"Glymes ({n_glymes})")

    # 1:1 line
    lims = [min(old.min(), improved.min()), max(old.max(), improved.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Old FOM1 (40°C)")
    ax.set_ylabel("Improved FOM1 (40°C)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(frameon=False)
    ax.set_aspect("equal")

    # Annotation
    ax.text(0.05, 0.95, f"R² = {r2:.4f}\nMAE = {mae:.1f}\nn = {len(comp)}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "old_vs_improved_fom1.png")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "old_vs_improved_fom1.png")

    # If glymes present, log their differences
    if n_glymes > 0:
        glyme_rows = comp[glyme_mask].copy()
        glyme_rows["diff"] = glyme_rows["FOM1_improved_40C"] - glyme_rows["FOM1_40C"]
        glyme_rows["pct_diff"] = 100 * glyme_rows["diff"] / glyme_rows["FOM1_40C"]
        logger.info("Glyme FOM1 differences (improved - old):")
        for _, row in glyme_rows.iterrows():
            logger.info("  %s: old=%.1f, improved=%.1f, diff=%+.1f (%.1f%%)",
                        row["SMILES"], row["FOM1_40C"], row["FOM1_improved_40C"],
                        row["diff"], row["pct_diff"])


def main():
    beta_df = compute_beta()
    fom1_df = compute_improved_fom1(beta_df)
    plot_old_vs_improved(fom1_df)
    logger.info("Done.")


if __name__ == "__main__":
    main()
