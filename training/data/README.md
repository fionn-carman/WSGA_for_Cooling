# Training Data

All datasets used for training XGBoost property models in the WSGA cooling fluid design pipeline.

## Directory Structure

```
data/
  constraints/      Safety/constraint property datasets (~850-3300 molecules)
  nist_8100/        NIST thermophysical datasets (8,100 CHO molecules, multi-temperature)
  old_dataset/      Original ~850-molecule hydrocarbon dataset (superseded by nist_8100)
```

## Data Sources

### constraints/

Curated from experimental literature and public databases. Each molecule has a
measured safety/constraint property plus 217 RDKit molecular descriptors.
SMILES are canonicalised and deduplicated (median aggregation for regression,
majority vote for classification). These datasets have not changed between the
old and new training pipelines — the same constraint models are used throughout.

### nist_8100/

Collected from the NIST WebBook and NIST Thermodynamics Research Center (TRC)
via the Web Thermo Tables (WTT) interface. Restricted to CHO-only molecules
(carbon, hydrogen, oxygen) that are liquid at relevant temperatures. Solids and
molecules with density above 1.5 g/cm3 or below 0.4 g/cm3 were removed.
Each property file contains experimental values at 11 temperatures (0-100C in
10C steps) plus 217 RDKit descriptors. Beta was computed from a quadratic fit
to density(T) and FOM1 was computed from component properties using the
corrected formula with quadratic beta and saturated liquid Cp.

### old_dataset/

The original hydrocarbon dataset (~850 molecules) used in early model training.
Sourced from literature compilations and experimental databases. Superseded by
nist_8100 for thermophysical models but retained for reference and comparison.
Heat capacity and FOM1 files use saturated liquid Cp (not ideal gas).

---

## Constraint Datasets

All files: `constraints/<name>_cleaned.csv` (219 columns: SMILES + target + 217 descriptors)

| File | Target | Molecules | Min | Max | Mean | Std | Unit |
|------|--------|-----------|-----|-----|------|-----|------|
| BP-Measured_cleaned.csv | BP-Measured | 2,504 | -88.6 | 637.0 | 195.6 | 85.6 | degC |
| MP-Measured_cleaned.csv | MP-Measured | 2,823 | -187.6 | 385.0 | 51.0 | 103.7 | degC |
| flashpoint_cleaned.csv | flashpoint | 3,302 | 85.2 | 721.4 | 364.9 | 69.9 | K |
| DC_exp_cleaned.csv | DC_exp | 529 | 1.45 | 74.0 | 7.24 | 8.26 | - |
| biodegradability_cleaned.csv | Activity | 1,888 | 0 | 1 | 0.32 | 0.47 | binary |

Notes:
- Flash point is in Kelvin (model predicts in K; subtract 273.15 for degC)
- Biodegradability is binary classification (1 = biodegradable, 0 = not)
- Dielectric constant is dimensionless

## NIST 8,100 Thermophysical Datasets

All files: `nist_8100/<name>_cho_cleaned.csv`

Multi-temperature datasets with values at 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 degC.

| File | Property | Molecules | Columns | Unit |
|------|----------|-----------|---------|------|
| density_cho_cleaned.csv | Liquid density | 8,082 | 230 | g/cm3 |
| viscosity_cho_cleaned.csv | Kinematic viscosity | 8,082 | 230 | cSt |
| tc_cho_cleaned.csv | Thermal conductivity | 8,062 | 230 | W/mK |
| cpsat_cho_cleaned.csv | Saturated liquid Cp | 7,377 | 230 | J/K/mol |
| beta_cho_cleaned.csv | Thermal expansion coeff | 8,082 | 16 | 1/K |
| fom1_cho_cleaned.csv | Figure of merit (FOM1) | 7,376 | 230 | W/mK |

### Property Ranges (40C and 100C)

| Property | 40C Min | 40C Max | 40C Mean | 100C Min | 100C Max | 100C Mean |
|----------|---------|---------|----------|----------|----------|-----------|
| Density / g cm-3 | 0.57 | 1.48 | 0.89 | 0.48 | 1.45 | 0.84 |
| Viscosity / cSt | 0.09 | 22.6 | 3.97 | 0.09 | 11.7 | 1.33 |
| Thermal conductivity / W mK-1 | 0.029 | 0.395 | 0.138 | 0.027 | 0.381 | 0.128 |
| Cp_sat / J K-1 mol-1 | 58 | 951 | 351 | 120 | 1,010 | 386 |
| Beta / K-1 | 0.0000 | 0.0022 | 0.0009 | -0.0026 | 0.0032 | 0.0010 |
| FOM1 / W mK-1 | 14.6 | 186.9 | 74.5 | 17.9 | 201.8 | 93.7 |

Notes:
- Beta is computed from quadratic fit to density(T): rho(T) = a + bT + cT2, beta = -(1/rho) drho/dT
- FOM1 = k * (beta * Cp * rho / (nu * k))^0.2813, computed at all 11 temperatures
- Cp is molar saturated liquid heat capacity (NOT ideal gas)
- Beta has 16 columns (SMILES, name, 11 temps, fit_r2, n_points, flag) — no RDKit descriptors
- 18 molecules with negative beta at any temperature removed from all datasets

## Old Dataset (Reference)

Files in `old_dataset/`. Superseded by nist_8100 but kept for comparison.

| File | Property | Molecules | Columns |
|------|----------|-----------|---------|
| processed_full_hydrocarbon_dataset.csv | Raw multi-property | 1,162 | 230 |
| Density_40C_g_cm^3_cleaned.csv | Density 40C | 833 | 131 |
| Density_100C_g_cm^3_cleaned.csv | Density 100C | 832 | 131 |
| Kinematic_Viscosity_40C_cleaned.csv | Viscosity 40C | 831 | 131 |
| Kinematic_Viscosity_100C_cleaned.csv | Viscosity 100C | 831 | 131 |
| Thermal_Conductivity_40C_cleaned.csv | TC 40C | 843 | 131 |
| Thermal_Conductivity_100C_cleaned.csv | TC 100C | 843 | 131 |
| Heat_Capacity_Constant_Pressure_40C_J_K_Mol_cleaned.csv | Cp_sat 40C | 661 | 131 |
| Heat_Capacity_Constant_Pressure_100C_J_K_Mol_cleaned.csv | Cp_sat 100C | 660 | 131 |
| FOM1_exp_40_cleaned.csv | FOM1 40C | 638 | 219 |
| FOM1_exp_100_cleaned.csv | FOM1 100C | 638 | 219 |
| training_reference.csv | SMILES reference list | 5,997 | 1 |

Notes:
- Old dataset has 131 columns (fewer RDKit descriptors from older RDKit version)
- FOM1 files have 219 columns (recomputed with current RDKit)
- processed_full_hydrocarbon_dataset.csv is the raw source; per-property files are filtered subsets
