"""
Validate, compare, and clean the NIST WTT master dataset.

Step 1: MW cross-check against discovery_compounds.csv
Step 2: Compare overlapping molecules with training data (parity plots)
Step 3: Clean out problematic molecules (salts, charged, radicals, etc.)
"""

import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(SCRIPT_DIR, 'nist_wtt_master.csv')
DISCOVERY_CSV = os.path.join(SCRIPT_DIR, '..', 'discovery_compounds.csv')
TRAINING_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'training', 'data')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Mapping: master column -> (training CSV filename, training property column)
PROPERTY_MAP = {
    'density_40C_g_cm3':  ('Density_40C_g_cm^3_cleaned.csv',                     'Density_40C_g_cm^3'),
    'density_100C_g_cm3': ('Density_100C_g_cm^3_cleaned.csv',                    'Density_100C_g_cm^3'),
    'kv_40C_cSt':         ('Kinematic_Viscosity_40C_cleaned.csv',                 'Kinematic_Viscosity_40C'),
    'kv_100C_cSt':        ('Kinematic_Viscosity_100C_cleaned.csv',                'Kinematic_Viscosity_100C'),
    'tc_40C_W_mK':        ('Thermal_Conductivity_40C_cleaned.csv',                'Thermal_Conductivity_40C'),
    'tc_100C_W_mK':       ('Thermal_Conductivity_100C_cleaned.csv',               'Thermal_Conductivity_100C'),
    'cp_40C_J_K_mol':     ('Heat_Capacity_Constant_Pressure_40C_J_K_Mol_cleaned.csv', 'Heat_Capacity_Constant_Pressure_40C_J_K_Mol'),
    'cp_100C_J_K_mol':    ('Heat_Capacity_Constant_Pressure_100C_J_K_Mol_cleaned.csv', 'Heat_Capacity_Constant_Pressure_100C_J_K_Mol'),
    'bp_K':               ('BP-Measured_cleaned.csv',                              'BP-Measured'),
    'mp_K':               ('MP-Measured_cleaned.csv',                              'MP-Measured'),
    'fp_K':               ('flashpoint_cleaned.csv',                               'flashpoint'),
}

# Unit conversions: training data uses °C for BP/MP, master uses K
# Apply conversion to training values before comparison
UNIT_CONVERT = {
    'bp_K': lambda x: x + 273.15,   # training °C -> K
    'mp_K': lambda x: x + 273.15,   # training °C -> K
}


def canonicalise(smi):
    """Return canonical SMILES or None if invalid."""
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ============================================================
# Load master dataset
# ============================================================
print('=' * 70)
print('Loading datasets...')
print('=' * 70)

master = pd.read_csv(MASTER_CSV)
print(f'Master dataset: {len(master)} rows')

# Add canonical SMILES
master['canon_smi'] = master['SMILES'].apply(canonicalise)
n_valid = master['canon_smi'].notna().sum()
n_invalid = master['canon_smi'].isna().sum()
print(f'  Valid SMILES: {n_valid}')
print(f'  Invalid/missing SMILES: {n_invalid}')


# ============================================================
# STEP 1: MW Cross-Check
# ============================================================
print('\n' + '=' * 70)
print('STEP 1: MW Cross-Check against discovery_compounds.csv')
print('=' * 70)

discovery = pd.read_csv(DISCOVERY_CSV)
print(f'Discovery compounds: {len(discovery)} rows')

# Normalise names for joining
master['name_lower'] = master['name'].str.strip().str.lower()
discovery['name_lower'] = discovery['name'].str.strip().str.lower()

# Join on normalised name
merged = master.merge(
    discovery[['name_lower', 'mw', 'formula']],
    on='name_lower',
    how='left',
    suffixes=('', '_nist')
)

n_matched = merged['mw'].notna().sum()
n_unmatched = merged['mw'].isna().sum()
print(f'Name matches with discovery: {n_matched} / {len(master)}')
print(f'Unmatched (no name in discovery): {n_unmatched}')

# Compare MolWt (from SMILES) vs mw (from NIST formula)
matched = merged[merged['mw'].notna()].copy()
matched['mw_diff'] = (matched['MolWt'] - matched['mw']).abs()
matched['mw_pct_diff'] = matched['mw_diff'] / matched['mw'] * 100

# Flag mismatches > 0.5 Da (allows for rounding in formula-based MW)
MW_THRESHOLD = 0.5
mismatches = matched[matched['mw_diff'] > MW_THRESHOLD].copy()

print(f'\nMW agreement (threshold = {MW_THRESHOLD} Da):')
print(f'  Matching: {len(matched) - len(mismatches)}')
print(f'  Mismatched: {len(mismatches)}')

if len(mismatches) > 0:
    print(f'\n  MW Mismatches (|MolWt - NIST MW| > {MW_THRESHOLD} Da):')
    print(f'  {"Name":<45} {"SMILES MW":>10} {"NIST MW":>10} {"Diff":>8} {"Formula":<15}')
    print(f'  {"-"*45} {"-"*10} {"-"*10} {"-"*8} {"-"*15}')
    for _, row in mismatches.sort_values('mw_diff', ascending=False).head(30).iterrows():
        print(f'  {str(row["name"])[:45]:<45} {row["MolWt"]:>10.2f} {row["mw"]:>10.2f} {row["mw_diff"]:>8.2f} {str(row.get("formula", "")):<15}')

# Check unmatched molecules are valid CHO
unmatched_rows = merged[merged['mw'].isna() & merged['canon_smi'].notna()].copy()
non_cho_unmatched = []
for idx, row in unmatched_rows.iterrows():
    mol = Chem.MolFromSmiles(str(row['SMILES']))
    if mol is not None:
        atoms = set(a.GetSymbol() for a in mol.GetAtoms())
        if not atoms.issubset({'C', 'H', 'O'}):
            non_cho_unmatched.append((row['name'], row['SMILES'], atoms))

if non_cho_unmatched:
    print(f'\n  WARNING: {len(non_cho_unmatched)} unmatched molecules contain non-CHO atoms:')
    for name, smi, atoms in non_cho_unmatched[:10]:
        print(f'    {name}: {smi} -> atoms: {atoms}')
else:
    print(f'\n  All {len(unmatched_rows)} unmatched molecules are valid CHO compounds.')


# ============================================================
# STEP 2: Dataset Comparison
# ============================================================
print('\n' + '=' * 70)
print('STEP 2: Dataset Comparison (Master vs Training Data)')
print('=' * 70)

# Get master canonical SMILES set (only valid ones)
master_valid = master[master['canon_smi'].notna()].copy()

# Load training data and compare
comparison_results = {}
training_smiles_sets = {}

for master_col, (train_file, train_prop_col) in PROPERTY_MAP.items():
    train_path = os.path.join(TRAINING_DIR, train_file)
    if not os.path.exists(train_path):
        print(f'  WARNING: {train_file} not found, skipping.')
        continue

    train_df = pd.read_csv(train_path, usecols=['SMILES', train_prop_col])
    train_df['canon_smi'] = train_df['SMILES'].apply(canonicalise)
    train_df = train_df[train_df['canon_smi'].notna()].copy()

    # Sets for comparison
    master_set = set(master_valid[master_valid[master_col].notna()]['canon_smi'])
    train_set = set(train_df['canon_smi'])

    overlap = master_set & train_set
    master_only = master_set - train_set
    train_only = train_set - master_set

    training_smiles_sets[master_col] = train_set

    comparison_results[master_col] = {
        'train_file': train_file,
        'master_count': len(master_set),
        'train_count': len(train_set),
        'overlap': len(overlap),
        'master_only': len(master_only),
        'train_only': len(train_only),
    }

# Summary table
print(f'\n  {"Property":<22} {"Master":>7} {"Train":>7} {"Overlap":>8} {"New(M)":>7} {"Only(T)":>8}')
print(f'  {"-"*22} {"-"*7} {"-"*7} {"-"*8} {"-"*7} {"-"*8}')
for prop, res in comparison_results.items():
    print(f'  {prop:<22} {res["master_count"]:>7} {res["train_count"]:>7} '
          f'{res["overlap"]:>8} {res["master_only"]:>7} {res["train_only"]:>8}')


# ============================================================
# Parity plots for overlapping molecules
# ============================================================
print('\n  Generating parity plots...')

for master_col, (train_file, train_prop_col) in PROPERTY_MAP.items():
    train_path = os.path.join(TRAINING_DIR, train_file)
    if not os.path.exists(train_path):
        continue

    train_df = pd.read_csv(train_path, usecols=['SMILES', train_prop_col])
    train_df['canon_smi'] = train_df['SMILES'].apply(canonicalise)
    train_df = train_df[train_df['canon_smi'].notna()].drop_duplicates('canon_smi')

    master_prop = master_valid[master_valid[master_col].notna()][['canon_smi', master_col]].drop_duplicates('canon_smi')

    # Merge on canonical SMILES
    parity = master_prop.merge(train_df[['canon_smi', train_prop_col]], on='canon_smi', how='inner')

    if len(parity) < 3:
        print(f'    {master_col}: only {len(parity)} overlapping points, skipping parity plot.')
        continue

    x = parity[train_prop_col].values.copy()
    y = parity[master_col].values.copy()

    # Convert training units if needed (e.g. °C -> K for BP/MP)
    if master_col in UNIT_CONVERT:
        x = UNIT_CONVERT[master_col](x)

    # Compute stats
    abs_diff = np.abs(x - y)
    pct_diff = abs_diff / (np.abs(x) + 1e-10) * 100
    mae = np.mean(abs_diff)
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    n_outliers = np.sum(pct_diff > 10)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    sc = ax.scatter(x, y, c=abs_diff, cmap='RdYlGn_r', alpha=0.7, s=30, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='|Difference|')

    # 1:1 line
    all_vals = np.concatenate([x, y])
    lims = [np.min(all_vals) * 0.95, np.max(all_vals) * 1.05]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    unit_note = ' [converted to K]' if master_col in UNIT_CONVERT else ''
    ax.set_xlabel(f'Training ({train_prop_col}){unit_note}')
    ax.set_ylabel(f'Master ({master_col})')
    ax.legend([f'N={len(parity)}, R²={r2:.3f}\nMAE={mae:.3f}, Outliers(>10%)={n_outliers}'],
              frameon=False, loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'parity_{master_col}.png'))
    plt.close(fig)
    print(f'    {master_col}: N={len(parity)}, R²={r2:.3f}, MAE={mae:.3f}, outliers(>10%)={n_outliers}')


# ============================================================
# Property range comparison
# ============================================================
print('\n  Property range comparison:')
print(f'  {"Property":<22} {"Master min":>10} {"Master max":>10} {"Master mean":>11} '
      f'{"Train min":>10} {"Train max":>10} {"Train mean":>11}')
print(f'  {"-"*22} {"-"*10} {"-"*10} {"-"*11} {"-"*10} {"-"*10} {"-"*11}')

for master_col, (train_file, train_prop_col) in PROPERTY_MAP.items():
    train_path = os.path.join(TRAINING_DIR, train_file)
    if not os.path.exists(train_path):
        continue

    train_df = pd.read_csv(train_path, usecols=['SMILES', train_prop_col])
    m_vals = master_valid[master_col].dropna()
    t_vals = train_df[train_prop_col].dropna()
    # Convert training units if needed
    if master_col in UNIT_CONVERT:
        t_vals = UNIT_CONVERT[master_col](t_vals)

    if len(m_vals) > 0 and len(t_vals) > 0:
        print(f'  {master_col:<22} {m_vals.min():>10.2f} {m_vals.max():>10.2f} {m_vals.mean():>11.2f} '
              f'{t_vals.min():>10.2f} {t_vals.max():>10.2f} {t_vals.mean():>11.2f}')


# ============================================================
# MW distribution comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Master MW
master_mw = master_valid['MolWt'].dropna()
ax.hist(master_mw, bins=60, alpha=0.5, label=f'Master (N={len(master_mw)})', density=True, color='steelblue')

# Training: combine all unique SMILES across training sets
all_train_smiles = set()
for _, (train_file, _) in PROPERTY_MAP.items():
    train_path = os.path.join(TRAINING_DIR, train_file)
    if os.path.exists(train_path):
        df = pd.read_csv(train_path, usecols=['SMILES'])
        for s in df['SMILES']:
            cs = canonicalise(s)
            if cs:
                all_train_smiles.add(cs)

train_mw = []
for smi in all_train_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        train_mw.append(Descriptors.MolWt(mol))
train_mw = np.array(train_mw)

ax.hist(train_mw, bins=60, alpha=0.5, label=f'Training (N={len(train_mw)})', density=True, color='coral')
ax.set_xlabel('Molecular Weight / Da')
ax.set_ylabel('Density')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'mw_distribution_comparison.png'))
plt.close(fig)
print(f'\n  MW distribution plot saved.')


# ============================================================
# STEP 3: Data Cleaning
# ============================================================
print('\n' + '=' * 70)
print('STEP 3: Data Cleaning')
print('=' * 70)

df = master.copy()
n_start = len(df)
cleaning_log = []  # (name, SMILES, reason)


def remove_rows(df, mask, reason):
    """Remove rows matching mask, log them, return filtered df."""
    removed = df[mask]
    for _, row in removed.iterrows():
        cleaning_log.append({
            'name': row.get('name', ''),
            'SMILES': row.get('SMILES', ''),
            'reason': reason,
        })
    n_removed = mask.sum()
    df = df[~mask].copy()
    print(f'  {reason:<35} removed {n_removed:>5} rows  ({len(df):>5} remaining)')
    return df


# Filter 1: No SMILES
mask = df['SMILES'].isna() | (df['SMILES'].astype(str).str.strip() == '')
df = remove_rows(df, mask, 'No SMILES')

# Filter 2: Invalid SMILES (RDKit can't parse)
mask = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(str(s)) is None)
df = remove_rows(df, mask, 'Invalid SMILES')

# Filter 3: Disconnected (salts, mixtures)
mask = df['SMILES'].str.contains('.', regex=False, na=False)
df = remove_rows(df, mask, 'Disconnected (contains ".")')

# Filter 4: Charged species
def has_charge(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return False
    return Chem.GetFormalCharge(mol) != 0

mask = df['SMILES'].apply(has_charge)
df = remove_rows(df, mask, 'Charged species')

# Filter 5: Radicals
def has_radicals(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return False
    return any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms())

mask = df['SMILES'].apply(has_radicals)
df = remove_rows(df, mask, 'Radicals')

# Filter 6: Non-CHO atoms
def is_non_cho(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return False
    atoms = set(a.GetSymbol() for a in mol.GetAtoms())
    return not atoms.issubset({'C', 'H', 'O'})

mask = df['SMILES'].apply(is_non_cho)
df = remove_rows(df, mask, 'Non-CHO atoms')

# Filter 7: MW too low
mask = df['MolWt'] < 50
df = remove_rows(df, mask, 'MW < 50')

# Filter 8: Extreme viscosity
mask = df['kv_40C_cSt'].notna() & (df['kv_40C_cSt'] > 200)
df = remove_rows(df, mask, 'kv_40C > 200 cSt')

# Filter 9: Extreme density
mask = df['density_40C_g_cm3'].notna() & (
    (df['density_40C_g_cm3'] < 0.5) | (df['density_40C_g_cm3'] > 2.0)
)
df = remove_rows(df, mask, 'Density outside [0.5, 2.0]')

# Filter 10: Negative/zero thermal conductivity
mask = df['tc_40C_W_mK'].notna() & (df['tc_40C_W_mK'] <= 0)
df = remove_rows(df, mask, 'tc_40C <= 0')

n_end = len(df)
n_removed_total = n_start - n_end
print(f'\n  Total: {n_removed_total} rows removed ({n_start} -> {n_end})')

# Recompute canonical SMILES for cleaned df
df['canon_smi'] = df['SMILES'].apply(canonicalise)

# Save cleaned dataset
output_cols = [c for c in master.columns if c not in ['name_lower', 'canon_smi']]
df[output_cols].to_csv(os.path.join(SCRIPT_DIR, 'nist_wtt_cleaned.csv'), index=False)
print(f'\n  Saved: nist_wtt_cleaned.csv ({len(df)} rows)')

# Save cleaning log
log_df = pd.DataFrame(cleaning_log)
log_df.to_csv(os.path.join(SCRIPT_DIR, 'cleaning_log.csv'), index=False)
print(f'  Saved: cleaning_log.csv ({len(log_df)} entries)')


# ============================================================
# STEP 4: Final Report
# ============================================================
print('\n' + '=' * 70)
print('STEP 4: Final Report')
print('=' * 70)

n_unique_smi = df['canon_smi'].nunique()
print(f'\n  Cleaned dataset: {len(df)} rows, {n_unique_smi} unique molecules')

# Property completeness
prop_cols = [
    'density_40C_g_cm3', 'density_100C_g_cm3',
    'kv_40C_cSt', 'kv_100C_cSt',
    'tc_40C_W_mK', 'tc_100C_W_mK',
    'cp_40C_J_K_mol', 'cp_100C_J_K_mol',
    'mp_K', 'bp_K', 'fp_K',
]
print(f'\n  Property completeness:')
print(f'  {"Property":<25} {"Non-null":>8} {"Coverage":>8}')
print(f'  {"-"*25} {"-"*8} {"-"*8}')
for col in prop_cols:
    n = df[col].notna().sum()
    pct = n / len(df) * 100 if len(df) > 0 else 0
    print(f'  {col:<25} {n:>8} {pct:>7.1f}%')

# Count molecules with all 8 thermophysical props at 40C
thermo_40 = ['density_40C_g_cm3', 'kv_40C_cSt', 'tc_40C_W_mK', 'cp_40C_J_K_mol']
thermo_100 = ['density_100C_g_cm3', 'kv_100C_cSt', 'tc_100C_W_mK', 'cp_100C_J_K_mol']
n_all_40 = df[thermo_40].notna().all(axis=1).sum()
n_all_100 = df[thermo_100].notna().all(axis=1).sum()
n_all_both = df[thermo_40 + thermo_100].notna().all(axis=1).sum()
print(f'\n  Molecules with all 4 thermo props at 40C:  {n_all_40}')
print(f'  Molecules with all 4 thermo props at 100C: {n_all_100}')
print(f'  Molecules with all 8 thermo props:         {n_all_both}')

# Removal summary by reason
print(f'\n  Removal summary:')
reason_counts = log_df['reason'].value_counts()
for reason, count in reason_counts.items():
    print(f'    {reason:<35} {count:>5}')

print(f'\n  Done.')
