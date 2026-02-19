#!/usr/bin/env python
"""
Compute Mahalanobis distance for evaluated GA molecules against each model's
training set in RFE-selected descriptor space.

For each of the 12 regression models, this script:
  1. Loads the RFE-selected feature list
  2. Loads the training data (which already contains pre-computed descriptors)
  3. Computes the training-set mean and covariance in the RFE feature space
  4. Computes Mahalanobis distance for every evaluated molecule
  5. Flags molecules as OOD if D_M^2 > chi^2(p, 0.975)

Output: augmented CSV with Mahal_{model} and OOD_{model} columns plus
        OOD_any (1 if ANY model flags OOD) and OOD_count.

Usage:
    python compute_mahalanobis.py --csv ../outputs/.../all_evaluated_molecules.csv
    python compute_mahalanobis.py --csv ../outputs/.../all_evaluated_molecules.csv \\
        --models_dir ../models --training_dir ../training/data
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    print("ERROR: RDKit required.  Install: conda install -c conda-forge rdkit")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add src/ to path for descriptor imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from descriptors import calc_descriptors, descriptor_names, descriptor_funcs


# ======================================================================
# Constants
# ======================================================================

REGRESSION_TARGETS = [
    "BP-Measured", "DC_exp", "Density_100C_g_cm^3", "Density_40C_g_cm^3",
    "flashpoint", "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",
    "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",
    "Kinematic_Viscosity_40C", "Kinematic_Viscosity_100C",
    "MP-Measured", "Thermal_Conductivity_100C", "Thermal_Conductivity_40C",
]


# ======================================================================
# Descriptor computation
# ======================================================================

def compute_descriptors_for_smiles(smiles_list):
    """Compute all RDKit descriptors for a list of SMILES strings.

    Returns a DataFrame with descriptor_names as columns.
    Rows where RDKit fails to parse the molecule will have NaN values.
    """
    records = []
    for smi in tqdm(smiles_list, desc="Computing descriptors"):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is not None:
            vals = calc_descriptors(mol, descriptor_funcs)
            records.append(vals)
        else:
            records.append([np.nan] * len(descriptor_names))

    return pd.DataFrame(records, columns=descriptor_names)


# ======================================================================
# Mahalanobis distance
# ======================================================================

def load_rfe_features(models_dir, target):
    """Load the RFE-selected feature names for a model."""
    rfe_path = os.path.join(models_dir, target, "RFE", "selected_features.txt")
    if not os.path.exists(rfe_path):
        return None
    with open(rfe_path) as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def compute_mahal_for_model(eval_desc_df, training_dir, models_dir,
                            target, threshold_pct=0.975,
                            pca_variance=0.95, max_feat_ratio=20):
    """Compute Mahalanobis distance for evaluated molecules against one model.

    Pipeline:
      1. Load RFE features and training data
      2. Standardise (z-score) using training statistics
      3. PCA to retain components explaining pca_variance of training variance
         (avoids ill-conditioned covariance from too many correlated features)
      4. Ledoit-Wolf covariance in PCA space
      5. Mahalanobis distance in PCA space
      6. OOD threshold from empirical 97.5th percentile of training D_M^2

    Parameters
    ----------
    eval_desc_df : DataFrame
        Evaluated molecules with all RDKit descriptors as columns.
    training_dir : str
        Path to directory containing {target}_cleaned.csv files.
    models_dir : str
        Path to models directory (for RFE feature lists).
    target : str
        Model/property name.
    threshold_pct : float
        Percentile for OOD threshold (default 0.975).
    pca_variance : float
        Fraction of training variance to retain in PCA (default 0.95).
    max_feat_ratio : int
        Maximum ratio of n_train / n_components to ensure stable covariance
        (default 20). If PCA gives more components, truncate.

    Returns
    -------
    mahal : ndarray (n_molecules,)
        Mahalanobis distance (D_M, not squared).
    ood : ndarray (n_molecules,)
        1 = out-of-domain, 0 = in-domain.
    n_features : int
        Number of PCA components used.
    """
    # Load RFE features
    rfe_features = load_rfe_features(models_dir, target)
    if rfe_features is None:
        print(f"  WARNING: No RFE features for {target}, skipping")
        n = len(eval_desc_df)
        return np.full(n, np.nan), np.full(n, -1, dtype=int), 0

    # Load training data
    train_path = os.path.join(training_dir, f"{target}_cleaned.csv")
    if not os.path.exists(train_path):
        print(f"  WARNING: No training data for {target} at {train_path}, skipping")
        n = len(eval_desc_df)
        return np.full(n, np.nan), np.full(n, -1, dtype=int), 0

    train_df = pd.read_csv(train_path)

    # Check which features are available in both training and eval data
    available = [f for f in rfe_features
                 if f in train_df.columns and f in eval_desc_df.columns]
    if len(available) < len(rfe_features):
        missing = set(rfe_features) - set(available)
        print(f"  WARNING: {target}: {len(missing)} features missing: {missing}")
    if len(available) < 2:
        print(f"  WARNING: {target}: too few features ({len(available)}), skipping")
        n = len(eval_desc_df)
        return np.full(n, np.nan), np.full(n, -1, dtype=int), 0

    # Extract descriptor matrices
    X_train = train_df[available].values.astype(float)
    X_eval = eval_desc_df[available].values.astype(float)

    # Drop training rows with NaN
    train_valid = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[train_valid]
    n_train, n_raw_feat = X_train.shape

    # Step 1: Standardise using training statistics
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)

    # Step 2: PCA — reduce to components explaining pca_variance
    max_components = min(n_raw_feat, n_train - 1)
    pca = PCA(n_components=min(max_components, n_raw_feat)).fit(X_train_sc)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_pca = int(np.searchsorted(cum_var, pca_variance) + 1)
    n_pca = min(n_pca, max_components)

    # Cap components so n_train / n_components >= max_feat_ratio
    max_by_ratio = max(2, n_train // max_feat_ratio)
    if n_pca > max_by_ratio:
        n_pca = max_by_ratio

    var_retained = cum_var[n_pca - 1] if n_pca <= len(cum_var) else cum_var[-1]

    print(f"    RFE features: {n_raw_feat}, PCA components: {n_pca} "
          f"({100*var_retained:.1f}% variance), n_train: {n_train}")

    # Project into PCA space (truncated)
    X_train_pca = X_train_sc @ pca.components_[:n_pca].T
    n_feat = n_pca

    # Step 3: Ledoit-Wolf covariance in PCA space
    lw = LedoitWolf().fit(X_train_pca)
    mu = lw.location_
    cov_inv = lw.precision_
    shrinkage = lw.shrinkage_

    print(f"    Ledoit-Wolf shrinkage: {shrinkage:.4f}")

    # Step 4: Empirical threshold from training D_M^2
    diff_train = X_train_pca - mu
    train_mahal_sq = np.sum(diff_train @ cov_inv * diff_train, axis=1)
    train_mahal_sq = np.maximum(train_mahal_sq, 0)

    threshold_sq = np.percentile(train_mahal_sq, 100 * threshold_pct)

    train_median_dm = np.sqrt(np.median(train_mahal_sq))
    train_p975_dm = np.sqrt(threshold_sq)
    print(f"    Training D_M: median={train_median_dm:.2f}, "
          f"p97.5={train_p975_dm:.2f}")

    # Step 5: Compute Mahalanobis distance for evaluated molecules
    n_eval = len(X_eval)
    mahal = np.full(n_eval, np.nan)
    ood = np.zeros(n_eval, dtype=int)

    eval_valid = ~np.isnan(X_eval).any(axis=1)

    if eval_valid.sum() > 0:
        X_valid = X_eval[eval_valid]
        # Apply same standardisation and PCA projection
        X_valid_sc = scaler.transform(X_valid)
        X_valid_pca = X_valid_sc @ pca.components_[:n_pca].T

        diff = X_valid_pca - mu
        mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)
        mahal_sq = np.maximum(mahal_sq, 0)

        mahal[eval_valid] = np.sqrt(mahal_sq)
        ood[eval_valid] = (mahal_sq > threshold_sq).astype(int)

    # Molecules with NaN descriptors → flag as OOD
    ood[~eval_valid] = 1

    return mahal, ood, n_feat


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute Mahalanobis distance for evaluated GA molecules"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to all_evaluated_molecules.csv")
    parser.add_argument("--models_dir", type=str, default=None,
                        help="Path to models/ directory (default: ../models)")
    parser.add_argument("--training_dir", type=str, default=None,
                        help="Path to training data directory (default: ../training/data)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: {csv_dir}/all_evaluated_molecules_mahal.csv)")
    parser.add_argument("--threshold", type=float, default=0.975,
                        help="Chi-squared percentile for OOD threshold (default: 0.975)")
    args = parser.parse_args()

    # Resolve defaults
    if args.models_dir is None:
        args.models_dir = os.path.join(REPO_ROOT, "models")
    if args.training_dir is None:
        args.training_dir = os.path.join(REPO_ROOT, "training", "data")
    if args.out is None:
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        args.out = os.path.join(csv_dir, "all_evaluated_molecules_mahal.csv")

    # Validate inputs
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.")
        sys.exit(1)
    if not os.path.isdir(args.models_dir):
        print(f"ERROR: Models directory not found: {args.models_dir}")
        sys.exit(1)
    if not os.path.isdir(args.training_dir):
        print(f"ERROR: Training data directory not found: {args.training_dir}")
        sys.exit(1)

    # ── Step 1: Load evaluated molecules ──
    print("=" * 60)
    print("  STEP 1: Load evaluated molecules")
    print("=" * 60)

    eval_df = pd.read_csv(args.csv)
    # Normalise SMILES column
    if "CanonicalSMILES" in eval_df.columns and "SMILES" not in eval_df.columns:
        eval_df.rename(columns={"CanonicalSMILES": "SMILES"}, inplace=True)

    smiles_col = "SMILES"
    if smiles_col not in eval_df.columns:
        print(f"ERROR: No SMILES column found in {args.csv}")
        sys.exit(1)

    smiles_list = eval_df[smiles_col].tolist()
    n_mols = len(smiles_list)
    print(f"  Loaded {n_mols} molecules from {args.csv}")

    # ── Step 2: Compute RDKit descriptors ──
    print(f"\n{'='*60}")
    print("  STEP 2: Compute RDKit descriptors")
    print("=" * 60)

    desc_df = compute_descriptors_for_smiles(smiles_list)
    n_valid = desc_df.notna().all(axis=1).sum()
    print(f"  Computed {len(descriptor_names)} descriptors for {n_mols} molecules")
    print(f"  Valid (no NaN): {n_valid} / {n_mols}")

    # ── Step 3: Compute Mahalanobis distance per model ──
    print(f"\n{'='*60}")
    print("  STEP 3: Compute Mahalanobis distance per model")
    print("=" * 60)

    ood_columns = []

    for target in REGRESSION_TARGETS:
        print(f"\n  --- {target} ---")

        mahal, ood, n_feat = compute_mahal_for_model(
            desc_df, args.training_dir, args.models_dir,
            target, threshold_pct=args.threshold,
        )

        # Sanitise column name (replace special chars)
        col_safe = target.replace("^", "").replace("/", "_")
        mahal_col = f"Mahal_{col_safe}"
        ood_col = f"OOD_{col_safe}"

        eval_df[mahal_col] = mahal
        eval_df[ood_col] = ood
        ood_columns.append(ood_col)

        n_ood = (ood == 1).sum()
        n_id = (ood == 0).sum()
        median_d = np.nanmedian(mahal)
        print(f"    In-domain: {n_id}, OOD: {n_ood} ({100*n_ood/n_mols:.1f}%)")
        print(f"    Eval median D_M: {median_d:.2f}")

    # ── Step 4: Summary columns ──
    print(f"\n{'='*60}")
    print("  STEP 4: Summary columns")
    print("=" * 60)

    ood_matrix = eval_df[ood_columns].values
    # Treat -1 (skipped models) as 0 for the summary
    ood_matrix_clean = np.where(ood_matrix < 0, 0, ood_matrix)

    eval_df["OOD_any"] = (ood_matrix_clean.max(axis=1) > 0).astype(int)
    eval_df["OOD_count"] = (ood_matrix_clean > 0).sum(axis=1)

    n_any_ood = eval_df["OOD_any"].sum()
    print(f"  OOD for ANY model: {n_any_ood} / {n_mols} ({100*n_any_ood/n_mols:.1f}%)")
    print(f"  OOD count distribution:")
    for cnt in range(0, 13):
        n = (eval_df["OOD_count"] == cnt).sum()
        if n > 0:
            print(f"    {cnt} models: {n} molecules ({100*n/n_mols:.1f}%)")

    # ── Step 5: Save ──
    print(f"\n{'='*60}")
    print("  STEP 5: Save augmented CSV")
    print("=" * 60)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    eval_df.to_csv(args.out, index=False)
    print(f"  Saved: {os.path.abspath(args.out)}")
    print(f"  Columns added: {len(ood_columns) * 2 + 2} "
          f"(12 Mahal + 12 OOD + OOD_any + OOD_count)")
    print("\nDone.")


if __name__ == "__main__":
    main()
