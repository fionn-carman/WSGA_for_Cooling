#!/usr/bin/env python3
"""
Prepare CHNO thermophysical datasets from WTT multitemp merged data.

From wtt_multitemp_merged.csv:
1. Filter: valid SMILES, contains C, atoms in {C,H,N,O}, MW <= 500
2. Canonicalize SMILES, deduplicate (first occurrence)
3. Viscosity clip: remove molecules where kv_0C_cSt > 60
4. Outlier removal: negative density, negative cpsat, kv > 1000 at any temp
5. Per-property cleaned CSVs
6. Beta from density quadratic fit (R^2 > 0.99, positive beta, >=3 points)
7. FOM1 from component properties

Output: training/data/CHNO/{property}_chno_cleaned.csv
"""

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
DATA_DIR = SCRIPT_DIR.parent / "data" / "CHNO"
WTT_PATH = SCRIPT_DIR.parent / "data" / "nist_8100" / "wtt_multitemp_merged.csv"

ALLOWED_ATOMS = {"C", "H", "N", "O"}
MAX_MW = 500
TEMPS = list(range(0, 110, 10))  # 0, 10, 20, ..., 100
TEMP_LABELS = [f"{t}C" for t in TEMPS]

DENSITY_COLS = [f"density_{t}C_g_cm3" for t in TEMPS]
VISCOSITY_COLS = [f"kv_{t}C_cSt" for t in TEMPS]
TC_COLS = [f"tc_{t}C_W_mK" for t in TEMPS]
CPSAT_COLS = [f"cpsat_{t}C_J_K_mol" for t in TEMPS]

# RDKit descriptors (unprefixed, matching existing CHO format)
RDKIT_NAMES = [d[0] for d in Descriptors._descList]
RDKIT_FNS = {d[0]: d[1] for d in Descriptors._descList}


def filter_chno(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to CHNO organic molecules with MW <= 500."""
    valid_rows = []
    canonical_smiles = {}
    mw_from_rdkit = {}
    reasons = {
        "no_smiles": 0, "invalid_smiles": 0, "no_carbon": 0,
        "non_chno": 0, "high_mw": 0,
    }

    for idx, row in df.iterrows():
        smi = str(row.get("SMILES", "")).strip()
        if not smi or smi in ("nan", "None", ""):
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

        can_smi = Chem.MolToSmiles(mol)
        canonical_smiles[idx] = can_smi
        mw_from_rdkit[idx] = mw
        valid_rows.append(idx)

    df = df.loc[valid_rows].copy()
    df["SMILES"] = df.index.map(canonical_smiles)
    df["MW"] = df.index.map(mw_from_rdkit)

    logger.info("CHNO filter: %d -> %d (rejected: %s)",
                len(valid_rows) + sum(reasons.values()), len(valid_rows), reasons)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by canonical SMILES (keep first occurrence)."""
    n_before = len(df)
    df = df.drop_duplicates(subset="SMILES", keep="first")
    logger.info("Deduplication: %d -> %d", n_before, len(df))
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outlier molecules based on physical plausibility."""
    n_before = len(df)
    mask = pd.Series(True, index=df.index)

    # Negative density at any temp
    for col in DENSITY_COLS:
        if col in df.columns:
            bad = df[col].notna() & (df[col] < 0)
            mask &= ~bad

    # Negative cpsat at any temp
    for col in CPSAT_COLS:
        if col in df.columns:
            bad = df[col].notna() & (df[col] < 0)
            mask &= ~bad

    # Viscosity > 1000 cSt at any temp
    for col in VISCOSITY_COLS:
        if col in df.columns:
            bad = df[col].notna() & (df[col] > 1000)
            mask &= ~bad

    df = df[mask].copy()
    logger.info("Outlier removal: %d -> %d", n_before, len(df))
    return df


def viscosity_clip(df: pd.DataFrame) -> pd.DataFrame:
    """Remove molecules where kv_0C_cSt > 60."""
    n_before = len(df)
    col = "kv_0C_cSt"
    if col in df.columns:
        keep = df[col].isna() | (df[col] <= 60)
        df = df[keep].copy()
    logger.info("Viscosity clip (kv_0C > 60): %d -> %d", n_before, len(df))
    return df


def compute_rdkit_descriptors(smiles_list: list) -> pd.DataFrame:
    """Compute all RDKit descriptors (unprefixed, matching existing format)."""
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
    elapsed = time.time() - t0
    logger.info("  RDKit: %d descriptors, %.1fs", len(df.columns), elapsed)
    return df


def save_property_csv(base_df: pd.DataFrame, prop_cols: list,
                      desc_df: pd.DataFrame, output_path: Path,
                      prop_name: str, extra_cols: list = None):
    """Save per-property cleaned CSV with inline RDKit descriptors."""
    # Filter to molecules that have at least one non-NaN value for this property
    has_data = base_df[prop_cols].notna().any(axis=1)
    subset = base_df[has_data].copy()

    # Build output
    out_cols = ["SMILES", "name"] + prop_cols
    if extra_cols:
        out_cols += extra_cols
    out_df = subset[out_cols].copy()

    # Add descriptors by SMILES join
    smiles_to_idx = {s: i for i, s in enumerate(base_df["SMILES"].values)}
    desc_rows = []
    for smi in out_df["SMILES"].values:
        if smi in smiles_to_idx:
            desc_rows.append(desc_df.iloc[smiles_to_idx[smi]])
        else:
            desc_rows.append(pd.Series({n: np.nan for n in RDKIT_NAMES}))
    desc_block = pd.DataFrame(desc_rows).reset_index(drop=True)
    out_df = out_df.reset_index(drop=True)
    out_df = pd.concat([out_df, desc_block], axis=1)

    out_df.to_csv(output_path, index=False)
    n_with_n = sum(1 for s in out_df["SMILES"] if "N" in
                   {a.GetSymbol() for a in Chem.MolFromSmiles(s).GetAtoms()}
                   if Chem.MolFromSmiles(s) is not None)
    logger.info("  %s: %d molecules (%d with N) -> %s",
                prop_name, len(out_df), n_with_n, output_path.name)
    return out_df


def compute_beta(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute thermal expansion coefficient beta from density quadratic fit.

    rho(T) = a + b*T + c*T^2
    beta(T) = -(1/rho) * drho/dT = -(b + 2*c*T) / rho(T)

    Filters: R^2 > 0.99, positive beta at all temps with data, >= 3 density points.
    """
    logger.info("Computing beta from density quadratic fits...")
    temps_C = np.array(TEMPS, dtype=float)

    results = []
    n_skipped_points = 0
    n_skipped_r2 = 0
    n_skipped_negative = 0

    for _, row in base_df.iterrows():
        # Get density values at each temp
        densities = np.array([row.get(col, np.nan) for col in DENSITY_COLS], dtype=float)
        valid = ~np.isnan(densities)
        n_points = int(valid.sum())

        if n_points < 3:
            n_skipped_points += 1
            continue

        T_valid = temps_C[valid]
        rho_valid = densities[valid]

        # Quadratic fit: rho = c2*T^2 + c1*T + c0
        coeffs = np.polyfit(T_valid, rho_valid, 2)
        rho_pred = np.polyval(coeffs, T_valid)

        # R^2
        ss_res = np.sum((rho_valid - rho_pred) ** 2)
        ss_tot = np.sum((rho_valid - rho_valid.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        if r2 < 0.99:
            n_skipped_r2 += 1
            continue

        # Compute beta at each temp
        # drho/dT = coeffs[1] + 2*coeffs[0]*T  (polyfit returns [c2, c1, c0])
        beta_vals = {}
        all_positive = True
        for t in TEMPS:
            rho_t = np.polyval(coeffs, t)
            drho_dt = coeffs[1] + 2 * coeffs[0] * t
            if rho_t > 0:
                beta_t = -drho_dt / rho_t
                if beta_t <= 0:
                    all_positive = False
                    break
                beta_vals[f"beta_{t}C"] = beta_t
            else:
                beta_vals[f"beta_{t}C"] = np.nan

        if not all_positive:
            n_skipped_negative += 1
            continue

        result = {
            "SMILES": row["SMILES"],
            "name": row["name"],
        }
        result.update(beta_vals)
        result["fit_r2"] = r2
        result["n_points"] = n_points
        result["flag"] = ""
        results.append(result)

    logger.info("  Beta: %d molecules (skipped: %d <3pts, %d low R2, %d negative beta)",
                len(results), n_skipped_points, n_skipped_r2, n_skipped_negative)

    return pd.DataFrame(results)


def compute_fom1(base_df: pd.DataFrame, beta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FOM1 at all temperatures.

    FOM1 = k * (beta * Cp_mass * rho / (nu * k))^0.2813

    Requires density, viscosity, tc, cpsat, beta all present.
    Units: k=W/(m*K), beta=1/K, Cp_mass=J/(kg*K), rho=kg/m^3, nu=m^2/s.
    """
    logger.info("Computing FOM1...")

    # Merge beta into base
    beta_cols = ["SMILES"] + [f"beta_{t}C" for t in TEMPS]
    merged = base_df.merge(beta_df[beta_cols], on="SMILES", how="inner")

    results = []
    for _, row in merged.iterrows():
        fom1_vals = {}
        any_valid = False
        mw = row.get("MW", np.nan)
        if np.isnan(mw) or mw <= 0:
            continue

        for t in TEMPS:
            density = row.get(f"density_{t}C_g_cm3", np.nan)
            kv = row.get(f"kv_{t}C_cSt", np.nan)
            tc = row.get(f"tc_{t}C_W_mK", np.nan)
            cpsat = row.get(f"cpsat_{t}C_J_K_mol", np.nan)
            beta = row.get(f"beta_{t}C", np.nan)

            if any(np.isnan(v) for v in [density, kv, tc, cpsat, beta]):
                fom1_vals[f"fom1_{t}C"] = np.nan
                continue
            if any(v <= 0 for v in [density, kv, tc, cpsat, beta]):
                fom1_vals[f"fom1_{t}C"] = np.nan
                continue

            # Unit conversions
            rho_kg_m3 = density * 1000
            nu_m2_s = kv * 1e-6
            cp_mass = cpsat / mw * 1000  # J/(K*mol) / (g/mol) * 1000 = J/(K*kg)

            inner = (beta * cp_mass * rho_kg_m3) / (nu_m2_s * tc)
            if inner <= 0:
                fom1_vals[f"fom1_{t}C"] = np.nan
                continue

            fom1 = tc * (inner ** 0.2813)
            fom1_vals[f"fom1_{t}C"] = fom1
            any_valid = True

        if any_valid:
            result = {"SMILES": row["SMILES"], "name": row["name"]}
            result.update(fom1_vals)
            results.append(result)

    logger.info("  FOM1: %d molecules", len(results))
    return pd.DataFrame(results)


def count_n_containing(smiles_series: pd.Series) -> int:
    """Count molecules containing nitrogen."""
    count = 0
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol and "N" in {a.GetSymbol() for a in mol.GetAtoms()}:
            count += 1
    return count


def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load WTT data ---
    logger.info("Loading WTT multitemp merged data from %s", WTT_PATH)
    df = pd.read_csv(WTT_PATH, low_memory=False)
    logger.info("  Loaded %d molecules", len(df))

    # --- Step 1: Filter CHNO ---
    df = filter_chno(df)

    # --- Step 2: Rename wtt_name -> name ---
    df["name"] = df["wtt_name"]

    # --- Step 3: Deduplicate ---
    df = deduplicate(df)

    # --- Step 4: Outlier removal ---
    df = remove_outliers(df)

    logger.info("After filtering: %d CHNO molecules", len(df))
    n_with_n = count_n_containing(df["SMILES"])
    logger.info("  N-containing molecules: %d", n_with_n)

    # --- Step 5: Compute RDKit descriptors ---
    desc_df = compute_rdkit_descriptors(df["SMILES"].tolist())

    # --- Step 6: Per-property CSVs ---
    logger.info("=" * 60)
    logger.info("Generating per-property cleaned CSVs")
    logger.info("=" * 60)

    # Density (no viscosity clip needed)
    save_property_csv(df, DENSITY_COLS, desc_df,
                      DATA_DIR / "density_chno_cleaned.csv", "density")

    # Viscosity (with clip)
    df_visc = viscosity_clip(df.copy())
    save_property_csv(df_visc, VISCOSITY_COLS, desc_df,
                      DATA_DIR / "viscosity_chno_cleaned.csv", "viscosity")

    # Thermal conductivity
    save_property_csv(df, TC_COLS, desc_df,
                      DATA_DIR / "tc_chno_cleaned.csv", "tc")

    # Cpsat
    save_property_csv(df, CPSAT_COLS, desc_df,
                      DATA_DIR / "cpsat_chno_cleaned.csv", "cpsat")

    # --- Step 7: Beta from density quadratic fit ---
    beta_df = compute_beta(df)
    beta_cols = [f"beta_{t}C" for t in TEMPS]
    beta_extra = ["fit_r2", "n_points", "flag"]

    # Save beta CSV with descriptors
    beta_with_desc = beta_df.copy()
    smiles_to_idx = {s: i for i, s in enumerate(df["SMILES"].values)}
    desc_rows = []
    for smi in beta_with_desc["SMILES"].values:
        if smi in smiles_to_idx:
            desc_rows.append(desc_df.iloc[smiles_to_idx[smi]])
        else:
            desc_rows.append(pd.Series({n: np.nan for n in RDKIT_NAMES}))
    desc_block = pd.DataFrame(desc_rows).reset_index(drop=True)
    beta_with_desc = beta_with_desc.reset_index(drop=True)
    beta_with_desc = pd.concat([beta_with_desc, desc_block], axis=1)
    beta_with_desc.to_csv(DATA_DIR / "beta_chno_cleaned.csv", index=False)
    n_with_n_beta = count_n_containing(beta_df["SMILES"])
    logger.info("  beta: %d molecules (%d with N) -> beta_chno_cleaned.csv",
                len(beta_df), n_with_n_beta)

    # --- Step 8: FOM1 from components ---
    fom1_df = compute_fom1(df, beta_df)
    fom1_cols = [f"fom1_{t}C" for t in TEMPS]

    # Save FOM1 CSV with descriptors
    fom1_with_desc = fom1_df.copy()
    desc_rows = []
    for smi in fom1_with_desc["SMILES"].values:
        if smi in smiles_to_idx:
            desc_rows.append(desc_df.iloc[smiles_to_idx[smi]])
        else:
            desc_rows.append(pd.Series({n: np.nan for n in RDKIT_NAMES}))
    desc_block = pd.DataFrame(desc_rows).reset_index(drop=True)
    fom1_with_desc = fom1_with_desc.reset_index(drop=True)
    fom1_with_desc = pd.concat([fom1_with_desc, desc_block], axis=1)
    fom1_with_desc.to_csv(DATA_DIR / "fom1_chno_cleaned.csv", index=False)
    n_with_n_fom1 = count_n_containing(fom1_df["SMILES"])
    logger.info("  fom1: %d molecules (%d with N) -> fom1_chno_cleaned.csv",
                len(fom1_df), n_with_n_fom1)

    # --- Summary ---
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("CHNO data preparation complete (%.1fs)", elapsed)
    logger.info("=" * 60)

    # Verification summary
    for name, path in [
        ("density", DATA_DIR / "density_chno_cleaned.csv"),
        ("viscosity", DATA_DIR / "viscosity_chno_cleaned.csv"),
        ("tc", DATA_DIR / "tc_chno_cleaned.csv"),
        ("cpsat", DATA_DIR / "cpsat_chno_cleaned.csv"),
        ("beta", DATA_DIR / "beta_chno_cleaned.csv"),
        ("fom1", DATA_DIR / "fom1_chno_cleaned.csv"),
    ]:
        if path.exists():
            tmp = pd.read_csv(path, usecols=["SMILES"])
            logger.info("  %s: %d molecules", name, len(tmp))

    # FOM1 sanity checks
    if (DATA_DIR / "fom1_chno_cleaned.csv").exists():
        fom1_check = pd.read_csv(DATA_DIR / "fom1_chno_cleaned.csv")
        fom1_40 = fom1_check["fom1_40C"].dropna()
        logger.info("  FOM1@40C range: %.1f - %.1f (mean=%.1f)",
                    fom1_40.min(), fom1_40.max(), fom1_40.mean())
        if (fom1_40 < 0).any():
            logger.warning("  WARNING: %d negative FOM1 values!", (fom1_40 < 0).sum())


if __name__ == "__main__":
    main()
