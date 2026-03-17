#!/usr/bin/env python3
"""
Trim the cleaned NIST WTT dataset to final columns and generate
parity plots comparing training data vs NIST WTT for overlapping molecules.

Demonstrates that the old training Cp = ideal gas Cp (not saturation Cp),
and that the old FOM1 values are therefore also computed from IG Cp.
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_CSV = os.path.join(SCRIPT_DIR, "nist_wtt_cleaned.csv")
TRIMMED_CSV = os.path.join(SCRIPT_DIR, "nist_wtt_trimmed.csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
TRAINING_DIR = os.path.join(SCRIPT_DIR, "..", "..", "training", "data")

# Egheosa's original dataset — used to identify experimental viscosity molecules
EGHEOSA_CSV_URL = (
    "https://raw.githubusercontent.com/EmicoBinsfinder/MLForTribology/main/"
    "GeneticAlgoRuns/CyclicMoleculeBenchmarkingDataset_NISTWEBBOOK.csv"
)
EGHEOSA_CSV_LOCAL = os.path.join(SCRIPT_DIR, "egheosa_benchmarking.csv")

os.makedirs(FIG_DIR, exist_ok=True)

# Keys whose parity comparison should exclude experimental-viscosity molecules
# (viscosity is experimental → carries forward into FOM1 calculation)
EXCLUDE_EXP_VISC_KEYS = {
    "visc_40", "visc_100",
    "fom1_ig_40", "fom1_ig_100",
    "fom1_sat_40", "fom1_sat_100",
}

# ── Column mapping: (training_csv, training_target_col, nist_col, label, unit) ──
PROPERTY_MAP = {
    # Group A: standard properties (should match well)
    "density_40": (
        "Density_40C_g_cm^3_cleaned.csv", "Density_40C_g_cm^3",
        "density_40C_g_cm3", "Density 40 \u00b0C", "g cm\u207b\u00b3",
    ),
    "density_100": (
        "Density_100C_g_cm^3_cleaned.csv", "Density_100C_g_cm^3",
        "density_100C_g_cm3", "Density 100 \u00b0C", "g cm\u207b\u00b3",
    ),
    "visc_40": (
        "Kinematic_Viscosity_40C_cleaned.csv", "Kinematic_Viscosity_40C",
        "kv_40C_cSt", "Viscosity 40 \u00b0C", "cSt",
    ),
    "visc_100": (
        "Kinematic_Viscosity_100C_cleaned.csv", "Kinematic_Viscosity_100C",
        "kv_100C_cSt", "Viscosity 100 \u00b0C", "cSt",
    ),
    "tc_40": (
        "Thermal_Conductivity_40C_cleaned.csv", "Thermal_Conductivity_40C",
        "tc_40C_W_mK", "Thermal Conductivity 40 \u00b0C", "W m\u207b\u00b9 K\u207b\u00b9",
    ),
    "tc_100": (
        "Thermal_Conductivity_100C_cleaned.csv", "Thermal_Conductivity_100C",
        "tc_100C_W_mK", "Thermal Conductivity 100 \u00b0C", "W m\u207b\u00b9 K\u207b\u00b9",
    ),
    # Group A: Cp IG alignment (proves training Cp = IG Cp)
    "cp_ig_40": (
        "Heat_Capacity_Constant_Pressure_40C_J_K_Mol_cleaned.csv",
        "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",
        "cp_40C_J_K_mol", "Cp 40 \u00b0C (Training vs IG)", "J K\u207b\u00b9 mol\u207b\u00b9",
    ),
    "cp_ig_100": (
        "Heat_Capacity_Constant_Pressure_100C_J_K_Mol_cleaned.csv",
        "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",
        "cp_100C_J_K_mol", "Cp 100 \u00b0C (Training vs IG)", "J K\u207b\u00b9 mol\u207b\u00b9",
    ),
    # Group A: FOM1 IG alignment
    "fom1_ig_40": (
        "FOM1_exp_40_cleaned.csv", "FOM1_exp_40",
        "FOM1_ig_40", "FOM1 40 \u00b0C (Training vs IG)", "W m\u207b\u00b9 K\u207b\u00b9",
    ),
    "fom1_ig_100": (
        "FOM1_exp_100_cleaned.csv", "FOM1_exp_100",
        "FOM1_ig_100", "FOM1 100 \u00b0C (Training vs IG)", "W m\u207b\u00b9 K\u207b\u00b9",
    ),
    # Group B: Sat comparison (should NOT match)
    "cp_sat_40": (
        "Heat_Capacity_Constant_Pressure_40C_J_K_Mol_cleaned.csv",
        "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",
        "cpsat_40C_J_K_mol", "Cp 40 \u00b0C (Training vs Sat)", "J K\u207b\u00b9 mol\u207b\u00b9",
    ),
    "cp_sat_100": (
        "Heat_Capacity_Constant_Pressure_100C_J_K_Mol_cleaned.csv",
        "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",
        "cpsat_100C_J_K_mol", "Cp 100 \u00b0C (Training vs Sat)", "J K\u207b\u00b9 mol\u207b\u00b9",
    ),
    "fom1_sat_40": (
        "FOM1_exp_40_cleaned.csv", "FOM1_exp_40",
        "FOM1_sat_40", "FOM1 40 \u00b0C (Training vs Sat)", "W m\u207b\u00b9 K\u207b\u00b9",
    ),
    "fom1_sat_100": (
        "FOM1_exp_100_cleaned.csv", "FOM1_exp_100",
        "FOM1_sat_100", "FOM1 100 \u00b0C (Training vs Sat)", "W m\u207b\u00b9 K\u207b\u00b9",
    ),
}

# Ordered keys for the 7x2 grid (left=40C, right=100C)
PLOT_ORDER = [
    ("density_40",  "density_100"),   # row 1
    ("visc_40",     "visc_100"),      # row 2
    ("tc_40",       "tc_100"),        # row 3
    ("cp_ig_40",    "cp_ig_100"),     # row 4
    ("fom1_ig_40",  "fom1_ig_100"),   # row 5
    ("cp_sat_40",   "cp_sat_100"),    # row 6
    ("fom1_sat_40", "fom1_sat_100"),  # row 7
]


def canonicalize(smi):
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_training(csv_name, target_col):
    """Load training CSV, return DataFrame with canonical SMILES and target."""
    path = os.path.join(TRAINING_DIR, csv_name)
    df = pd.read_csv(path, usecols=["SMILES", target_col])
    df["canon_smi"] = df["SMILES"].apply(canonicalize)
    df = df.dropna(subset=["canon_smi", target_col])
    # Deduplicate on canonical SMILES (take first)
    df = df.drop_duplicates(subset="canon_smi", keep="first")
    return df[["canon_smi", target_col]].rename(columns={target_col: "training_val"})


def load_experimental_visc_smiles():
    """Load Egheosa's dataset and return set of canonical SMILES where
    viscosity method is 'Experimental' (not NIST equation-based)."""
    import urllib.request
    if not os.path.exists(EGHEOSA_CSV_LOCAL):
        print("  Downloading Egheosa benchmarking dataset...")
        urllib.request.urlretrieve(EGHEOSA_CSV_URL, EGHEOSA_CSV_LOCAL)
    df = pd.read_csv(EGHEOSA_CSV_LOCAL)
    exp_mask = df["Viscosity Prediction"] == "Experimental"
    exp_smiles = set()
    for smi in df.loc[exp_mask, "SMILES"]:
        csmi = canonicalize(str(smi))
        if csmi is not None:
            exp_smiles.add(csmi)
    print(f"  {len(exp_smiles)} molecules with experimental viscosity (excluded from visc/FOM1 parity)")
    return exp_smiles


def main():
    # ── Step 1: Trim ─────────────────────────────────────────────────────────
    print("Loading cleaned NIST WTT dataset...")
    nist = pd.read_csv(CLEANED_CSV)
    print(f"  Loaded: {len(nist)} rows, {len(nist.columns)} columns")

    keep_cols = [
        "SMILES", "name",
        "density_40C_g_cm3", "density_100C_g_cm3",
        "kv_40C_cSt", "kv_100C_cSt",
        "tc_40C_W_mK", "tc_100C_W_mK",
        "cpsat_40C_J_K_mol", "cpsat_100C_J_K_mol",
        "cp_40C_J_K_mol", "cp_100C_J_K_mol",
        "FOM1_sat_40", "FOM1_sat_100",
        "FOM1_ig_40", "FOM1_ig_100",
    ]
    trimmed = nist[keep_cols].copy()
    trimmed.to_csv(TRIMMED_CSV, index=False)
    print(f"  Saved trimmed: {TRIMMED_CSV} ({len(trimmed)} rows, {len(trimmed.columns)} columns)")

    # ── Step 2: Canonicalize NIST SMILES ─────────────────────────────────────
    print("Canonicalizing NIST SMILES...")
    nist["canon_smi"] = nist["SMILES"].apply(canonicalize)
    nist = nist.dropna(subset=["canon_smi"])
    nist = nist.drop_duplicates(subset="canon_smi", keep="first")
    print(f"  {len(nist)} unique valid NIST molecules")

    # ── Step 3: Load experimental viscosity exclusion list ──────────────────
    exp_visc_smiles = load_experimental_visc_smiles()

    # ── Step 4: Parity plots ─────────────────────────────────────────────────
    print("Generating parity plot grid...")
    fig, axes = plt.subplots(7, 2, figsize=(10, 28), dpi=150)

    # Group labels
    group_a_label = "Group A: Alignment check (should match)"
    group_b_label = "Group B: Sat discrepancy (should NOT match)"

    for row_idx, (key_left, key_right) in enumerate(PLOT_ORDER):
        for col_idx, key in enumerate([key_left, key_right]):
            ax = axes[row_idx, col_idx]
            csv_name, target_col, nist_col, label, unit = PROPERTY_MAP[key]

            # Load training data
            train_df = load_training(csv_name, target_col)

            # Merge on canonical SMILES
            nist_sub = nist[["canon_smi", nist_col]].dropna(subset=[nist_col])
            merged = train_df.merge(nist_sub, on="canon_smi", how="inner")

            # Exclude experimental-viscosity molecules from visc & FOM1 panels
            if key in EXCLUDE_EXP_VISC_KEYS:
                n_before = len(merged)
                merged = merged[~merged["canon_smi"].isin(exp_visc_smiles)]
                n_excluded = n_before - len(merged)
                if n_excluded > 0:
                    print(f"    {key}: excluded {n_excluded} experimental-visc molecules ({n_before} → {len(merged)})")

            x = merged["training_val"].values
            y = merged[nist_col].values

            if len(merged) < 3:
                ax.text(0.5, 0.5, f"N = {len(merged)}\n(insufficient)",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel(f"Training / {unit}")
                ax.set_ylabel(f"NIST WTT / {unit}")
                continue

            r2 = r2_score(x, y)

            # Colour: blue for Group A (rows 0-4), red for Group B (rows 5-6)
            color = "#2166ac" if row_idx < 5 else "#b2182b"

            ax.scatter(x, y, s=12, alpha=0.5, c=color, edgecolors="none")

            # 1:1 line
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    "k--", lw=0.8, alpha=0.6)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)

            ax.set_aspect("equal")
            ax.set_xlabel(f"Training / {unit}")
            ax.set_ylabel(f"NIST WTT / {unit}")
            ax.text(0.05, 0.92, f"R\u00b2 = {r2:.4f}\nN = {len(merged)}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8,
                              edgecolor="none"))

            # Row label on left column only
            if col_idx == 0:
                ax.set_ylabel(f"{label}\nNIST WTT / {unit}")

            # Add group separator label
            if row_idx == 0 and col_idx == 0:
                ax.annotate(group_a_label, xy=(0, 1.15),
                            xycoords="axes fraction", fontsize=10,
                            fontweight="bold", color="#2166ac")
            if row_idx == 5 and col_idx == 0:
                ax.annotate(group_b_label, xy=(0, 1.15),
                            xycoords="axes fraction", fontsize=10,
                            fontweight="bold", color="#b2182b")

    plt.tight_layout(h_pad=2.5)
    fig_path = os.path.join(FIG_DIR, "parity_grid.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
