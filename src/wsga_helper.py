import os
import pandas as pd
import pickle
import random
from rdkit import Chem
from rdkit.Chem import Draw, rdchem, MACCSkeys, Descriptors
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import joblib
import time

from mutations import mutate
from descriptors import descriptor_funcs, descriptor_names, calc_descriptors
from evaluation import load_models, get_scscore_cached, strict_canonicalize_smiles


def is_biodegradable(smiles, biodeg_model):
    """
    Predict biodegradability using the trained classification model.
    
    Works with both old-style pickle models (dict with 'model' and 'features')
    and new joblib models from the hybrid training script.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    try:
        desc_values = [func(mol) for func in descriptor_funcs]
    except Exception:
        return False

    if any(pd.isna(desc_values)) or any(val is None for val in desc_values):
        return False

    base_features = pd.DataFrame([desc_values], columns=descriptor_names)

    # Check if model has specific features (new style from hybrid training)
    if hasattr(biodeg_model, 'get') and callable(biodeg_model.get):
        # It's a dict from joblib (new style)
        model = biodeg_model['model']
        features = biodeg_model.get('features', descriptor_names)
        
        # Select only the features the model was trained on
        try:
            X = base_features[features]
        except KeyError:
            # Fall back to all features if some are missing
            X = base_features
    else:
        # Old style - model is the classifier directly
        model = biodeg_model
        
        # Add interaction terms for old-style models
        if 'MolWt' in base_features.columns and 'MolLogP' in base_features.columns:
            base_features["MolWt_x_MolLogP"] = base_features["MolWt"] * base_features["MolLogP"]
        if 'TPSA' in base_features.columns and 'NumRotatableBonds' in base_features.columns:
            base_features["TPSA_x_NumRotatableBonds"] = base_features["TPSA"] * base_features["NumRotatableBonds"]
        if 'MolLogP' in base_features.columns and 'RingCount' in base_features.columns:
            base_features["MolLogP_x_RingCount"] = base_features["MolLogP"] * base_features["RingCount"]
        X = base_features

    try:
        prediction = model.predict(X)[0]
        return prediction == 1
    except Exception as e:
        print("Biodeg prediction error:", e)
        return False


def get_size_dependent_weights(n_heavy, min_ha, max_ha):
    """
    Calculate mutation weights based on molecule size.
    
    Returns a dictionary of mutation -> weight
    
    Logic:
    - size_ratio = 0.0 means molecule is at MIN size (favor growing)
    - size_ratio = 1.0 means molecule is at MAX size (favor shrinking)
    - size_ratio = 0.5 means molecule is mid-sized (balanced)
    """
    # Normalize size to [0, 1] range
    size_ratio = (n_heavy - min_ha) / (max_ha - min_ha) if max_ha > min_ha else 0.5
    size_ratio = max(0.0, min(1.0, size_ratio))  # Clamp to [0, 1]
    
    # Define base weights and how they scale with size
    # Format: (base_weight, size_scaling)
    # size_scaling > 0 means weight INCREASES with size (shrinking mutations)
    # size_scaling < 0 means weight DECREASES with size (growing mutations)
    
    mutation_config = {
        # Growing mutations - high weight when small, low when large
        'AddAtom':        (2.0, -1.5),   # Small addition
        'AddFragment':    (4.0, -3.5),   # Large addition - strongly favor when small
        'InsertAromatic': (3.0, -2.5),   # Large addition
        'Napthalenate':   (2.0, -1.8),   # Large addition
        'Glycolate':      (3.0, -2.5),   # Large addition - good for heat transfer
        'Esterify':       (3.0, -2.5),   # Large addition
        
        # Shrinking mutations - low weight when small, high when large
        'RemoveAtom':     (1.0, +2.0),   # Small removal
        'RemoveFragment': (1.0, +3.0),   # Large removal - strongly favor when large
        
        # Neutral mutations - relatively constant, slight increase when mid-sized
        'ReplaceAtom':    (2.0, +0.5),   # Slight increase with size (more atoms to replace)
        'ReplaceBond':    (2.0, +0.5),   # Slight increase with size (more bonds to replace)
    }
    
    weights = {}
    for mutation, (base, scaling) in mutation_config.items():
        # Linear interpolation: weight = base + scaling * size_ratio
        weight = base + scaling * size_ratio
        weights[mutation] = max(0.1, weight)  # Ensure minimum weight of 0.1
    
    return weights


def select_weighted_mutation(mutations_list, weights_dict):
    """
    Select a mutation from the list using the provided weights.
    """
    available_mutations = [m for m in mutations_list if m in weights_dict]
    weights = [weights_dict[m] for m in available_mutations]
    
    return random.choices(available_mutations, weights=weights, k=1)[0]


def apply_mutations_to_population(
    df,
    MUTATIONS,
    MUTATION_RATE,
    NewAtoms,
    BondTypes,
    fragments,
    AromaticMolecule,
    Napthalenes,
    MIN_HEAVY_ATOMS,
    MAX_HEAVY_ATOMS,
    MAX_CARBONS,
    MAX_OXYGENS,
    seen_smiles,
    max_attempts=50,
):
    """
    Mutate a population of SMILES strings with size-dependent mutation selection.
    
    Small molecules preferentially get "growing" mutations.
    Large molecules preferentially get "shrinking" mutations.
    """
    mutated_smiles = []
    attempted = 0
    successful = 0
    failed = 0
    
    # Track mutation usage for diagnostics
    mutation_counts = {m: 0 for m in MUTATIONS}
    mutation_success = {m: 0 for m in MUTATIONS}

    for smi in df["SMILES"]:

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        new_smi = smi
        mutated = False

        if random.random() < MUTATION_RATE:
            attempted += 1
            
            # Get current molecule size
            n_heavy = mol.GetNumHeavyAtoms()
            
            # Calculate size-dependent weights
            weights = get_size_dependent_weights(n_heavy, MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS)

            for _ in range(max_attempts):
                
                # Select mutation based on size-dependent weights
                mutation = select_weighted_mutation(MUTATIONS, weights)
                mutation_counts[mutation] += 1

                mutated_mol, mutated_smi = mutate(
                    mol,
                    mutation,
                    NewAtoms,
                    BondTypes,
                    fragments,
                    AromaticMolecule,
                    Napthalenes,
                    False,
                    False
                )

                if mutated_mol is None or mutated_smi is None:
                    continue

                # Heavy atom constraints
                ha = mutated_mol.GetNumHeavyAtoms()
                if not (MIN_HEAVY_ATOMS <= ha <= MAX_HEAVY_ATOMS):
                    continue

                # Atom count constraints
                num_c = sum(1 for a in mutated_mol.GetAtoms() if a.GetAtomicNum() == 6)
                num_o = sum(1 for a in mutated_mol.GetAtoms() if a.GetAtomicNum() == 8)

                if num_c > MAX_CARBONS or num_o > MAX_OXYGENS:
                    continue

                canonical = strict_canonicalize_smiles(mutated_smi)
                if canonical in seen_smiles:
                    continue

                # SUCCESS
                new_smi = canonical
                seen_smiles.add(canonical)
                successful += 1
                mutation_success[mutation] += 1
                mutated = True
                break

            if not mutated:
                failed += 1

        mutated_smiles.append(new_smi)

    # Print summary with mutation breakdown
    print("\n=== Mutation Summary ===")
    print(f"Population size: {len(df)}")
    print(f"Attempted mutations: {attempted}")
    print(f"Successful mutations: {successful}")
    print(f"Failed mutation attempts: {failed}")
    print("\n--- Mutation Breakdown ---")
    print(f"{'Mutation':<18} {'Attempted':>10} {'Succeeded':>10} {'Rate':>8}")
    print("-" * 48)
    for m in MUTATIONS:
        att = mutation_counts[m]
        suc = mutation_success[m]
        rate = f"{suc/att*100:.1f}%" if att > 0 else "N/A"
        print(f"{m:<18} {att:>10} {suc:>10} {rate:>8}")
    print("=" * 48 + "\n")

    out_df = pd.DataFrame({"SMILES": mutated_smiles})
    out_df = out_df.drop_duplicates().reset_index(drop=True)

    return out_df, seen_smiles


def evaluate_molecules(df, thermo_models, sc_model, tox21_models, biodeg_model, drop_descriptors=True, fom1_direct_models=None, molprice_model=None):
    """
    Evaluate molecules by predicting all thermophysical properties.
    
    Handles model dependencies - e.g., Density_100C may depend on Density_40C prediction.
    Models are separated into independent (no auxiliary features) and dependent (uses 
    predictions from other models as input features).
    """
    # Remove rows with missing SMILES
    df = df[df['SMILES'].notna()].copy()

    # Convert to RDKit Mol objects safely
    df['Mol'] = df['SMILES'].apply(lambda smi: Chem.MolFromSmiles(smi) if smi else None)
    df = df[df['Mol'].notna()].reset_index(drop=True)

    # ----------------------------
    # Compute descriptors
    # ----------------------------
    desc_data = [calc_descriptors(mol, descriptor_funcs) for mol in df['Mol']]
    desc_df = pd.DataFrame(desc_data, columns=descriptor_names)
    
    # CRITICAL: Convert all descriptor columns to numeric, coercing errors to NaN
    for col in desc_df.columns:
        desc_df[col] = pd.to_numeric(desc_df[col], errors='coerce')
    
    # Fill NaN with 0 (or could use column medians)
    desc_df = desc_df.fillna(0)
    
    df = pd.concat([df[['SMILES', 'Mol']], desc_df], axis=1)

    # ----------------------------
    # Predict SCScore
    # ----------------------------
    df["SCScore"] = df["SMILES"].apply(lambda smi: get_scscore_cached(sc_model, smi))

    # ----------------------------
    # Predict MolPrice (log USD/mmol)
    # ----------------------------
    if molprice_model is not None:
        df["MolPrice"] = df["SMILES"].apply(molprice_model.predict)

    # ----------------------------
    # Predict Tox21 (using descriptors for new-style models)
    # ----------------------------
    df["Tox21_Score"] = predict_tox21_batch(df, tox21_models)

    # ----------------------------
    # Predict Thermo properties (with dependency handling)
    # ----------------------------
    # Separate models into those with and without auxiliary features
    independent_models = {}
    dependent_models = {}
    
    for target, data in thermo_models.items():
        aux_feature = data.get('auxiliary_feature')
        aux_feature_name = data.get('auxiliary_feature_name')
        if aux_feature and aux_feature_name:
            dependent_models[target] = data
        else:
            independent_models[target] = data
    
    # First pass: predict all independent models
    for target, data in independent_models.items():
        model = data['model']
        features = data['features']
        log_target = data.get('log_target', False)
        try:
            X = df[features].copy()
            y_pred = model.predict(X)
            if log_target:
                y_pred = np.expm1(y_pred)
            df[target] = y_pred
        except Exception as e:
            print(f"Skipping {target}, error: {e}")
            df[target] = np.nan
    
    # Second pass: predict dependent models (these use predictions from first pass)
    for target, data in dependent_models.items():
        model = data['model']
        features = data['features']
        log_target = data.get('log_target', False)
        aux_feature = data['auxiliary_feature']  # e.g., "Density_40C_g_cm^3"
        aux_feature_name = data['auxiliary_feature_name']  # e.g., "AUX_Density_40C_g_cm^3"
        
        try:
            # Build feature matrix
            # Some features come from descriptors, one comes from a previous prediction
            X = pd.DataFrame(index=df.index)
            
            for feat in features:
                if feat == aux_feature_name:
                    # This feature comes from a previous prediction
                    if aux_feature not in df.columns:
                        raise ValueError(f"Auxiliary feature {aux_feature} not yet predicted. "
                                         f"Ensure {aux_feature} is predicted before {target}.")
                    X[feat] = df[aux_feature].values
                else:
                    # Regular descriptor feature
                    X[feat] = df[feat].values
            
            y_pred = model.predict(X)
            if log_target:
                y_pred = np.expm1(y_pred)
            df[target] = y_pred
            
        except Exception as e:
            print(f"Skipping {target}, error: {e}")
            df[target] = np.nan

    # ----------------------------
    # Predict biodegradability
    # ----------------------------
    df["Biodegradable"] = df["SMILES"].apply(lambda smi: is_biodegradable(smi, biodeg_model))

    # ----------------------------
    # Compute thermal properties (alpha, beta, Ra, h, FOM1)
    # ----------------------------
    g = 9.81
    dT = 60  # 40→100°C

    df["MW"] = df["Mol"].apply(lambda mol: Descriptors.MolWt(mol))

    df["Cp_40"] = df["Heat_Capacity_Constant_Pressure_40C_J_K_Mol"] / df["MW"] * 1000
    df["Cp_100"] = df["Heat_Capacity_Constant_Pressure_100C_J_K_Mol"] / df["MW"] * 1000

    df["alpha_40"] = df["Thermal_Conductivity_40C"] / (df["Density_40C_g_cm^3"] * 1000 * df["Cp_40"])
    df["alpha_100"] = df["Thermal_Conductivity_100C"] / (df["Density_100C_g_cm^3"] * 1000 * df["Cp_100"])

    df["nu_40"] = df["Kinematic_Viscosity_40C"] * 1e-6
    df["nu_100"] = df["Kinematic_Viscosity_100C"] * 1e-6

    df["beta_40"] = -(1 / df["Density_40C_g_cm^3"]) * ((df["Density_100C_g_cm^3"] - df["Density_40C_g_cm^3"]) / dT)
    df["beta_100"] = -(1 / df["Density_100C_g_cm^3"]) * ((df["Density_100C_g_cm^3"] - df["Density_40C_g_cm^3"]) / dT)

    df["Ra_40"] = g * df["beta_40"] / (df["nu_40"] * df["alpha_40"])
    df["Ra_100"] = g * df["beta_100"] / (df["nu_100"] * df["alpha_100"])

    n_exp = 1/3
    df["h_40"] = df["Thermal_Conductivity_40C"] * np.power(df["Ra_40"].clip(lower=0), n_exp)
    df["h_100"] = df["Thermal_Conductivity_100C"] * np.power(df["Ra_100"].clip(lower=0), n_exp)

    df["FOM1_40"] = df["Thermal_Conductivity_40C"] * ((df["beta_40"] * df["Cp_40"] * df["Density_40C_g_cm^3"] * 1000) / (df["nu_40"] * df["Thermal_Conductivity_40C"]))**0.2813
    df["FOM1_100"] = df["Thermal_Conductivity_100C"] * ((df["beta_100"] * df["Cp_100"] * df["Density_100C_g_cm^3"] * 1000) / (df["nu_100"] * df["Thermal_Conductivity_100C"]))**0.2813

    # Prandtl number: Pr = nu / alpha = (mu * Cp) / k
    df["Pr_40"] = df["nu_40"] / df["alpha_40"]
    df["Pr_100"] = df["nu_100"] / df["alpha_100"]

    # Grashof number: Gr = g * beta * dT * L^3 / nu^2
    # Using L = 1 (characteristic length = 1m for dimensionless comparison)
    L = 1.0
    df["Gr_40"] = g * df["beta_40"] * dT * (L**3) / (df["nu_40"]**2)
    df["Gr_100"] = g * df["beta_100"] * dT * (L**3) / (df["nu_100"]**2)

    # Nusselt number: Nu = h * L / k (using h from natural convection correlation)
    # For natural convection: Nu ~ Ra^(1/3), so Nu = (Ra)^(1/3)
    df["Nu_40"] = np.power(df["Ra_40"].clip(lower=0), n_exp)
    df["Nu_100"] = np.power(df["Ra_100"].clip(lower=0), n_exp)

    # Compute averages
    df["alpha_avg"] = (df["alpha_40"] + df["alpha_100"]) / 2
    df["beta_avg"] = (df["beta_40"] + df["beta_100"]) / 2
    df["Ra_avg"] = (df["Ra_40"] + df["Ra_100"]) / 2
    df["h_avg"] = (df["h_40"] + df["h_100"]) / 2
    df["FOM1_avg"] = (df["FOM1_40"] + df["FOM1_100"]) / 2
    df["Pr_avg"] = (df["Pr_40"] + df["Pr_100"]) / 2
    df["Gr_avg"] = (df["Gr_40"] + df["Gr_100"]) / 2
    df["Nu_avg"] = (df["Nu_40"] + df["Nu_100"]) / 2

    # ----------------------------
    # Predict FOM1 directly (XGBoost+Descriptor ensemble)
    # ----------------------------
    if fom1_direct_models is not None:
        for temp_key, col_name in [('fom1_40', 'FOM1_40C_direct'), ('fom1_100', 'FOM1_100C_direct')]:
            if temp_key not in fom1_direct_models:
                continue
            mdata = fom1_direct_models[temp_key]
            desc_cols = mdata['descriptor_columns']

            # Build feature matrix from the descriptors already in df
            X_raw = df[desc_cols].copy()
            X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_raw = X_raw.replace([np.inf, -np.inf], 0)

            # Ensemble prediction: average across 5 folds (each with its own scaler)
            preds_all = []
            for model, scaler in zip(mdata['models'], mdata['scalers']):
                X_scaled = scaler.transform(X_raw.values)
                preds_all.append(model.predict(X_scaled))
            df[col_name] = np.mean(preds_all, axis=0)

        df["FOM1_direct_avg"] = (df["FOM1_40C_direct"] + df["FOM1_100C_direct"]) / 2

    # ----------------------------
    # Drop descriptors/Mol if requested
    # ----------------------------
    if drop_descriptors:
        df = df.drop(columns=descriptor_names + ['Mol'], errors='ignore')

    return df


def compute_fitness(df, TARGET, TARGET_CONFIG):
    target_col = TARGET_CONFIG[TARGET]["column"]
    maximize = TARGET_CONFIG[TARGET]["maximize"]

    if maximize:
        fitness = df[target_col]
    else:
        fitness = -df[target_col]

    df["FitnessScore"] = fitness * df["is_valid"]
    return df


def apply_mp_penalty(df, soft_threshold=-30, hard_threshold=-10):
    """
    Apply soft penalty for melting point between soft and hard thresholds.
    
    Penalty factor P:
    - MP <= soft_threshold (-30°C): P = 1.0 (no penalty)
    - MP >= hard_threshold (-10°C): P = 0.0 (full penalty, molecule eliminated)
    - soft_threshold < MP < hard_threshold: P decreases smoothly from 1 to 0
    
    Uses a smooth S-curve (cosine-based) for gradual transition.
    
    Args:
        df: DataFrame with 'FitnessScore' and 'MP-Measured' columns
        soft_threshold: Below this, no penalty (default -30°C)
        hard_threshold: At or above this, fitness = 0 (default -10°C)
    
    Returns:
        DataFrame with updated FitnessScore and new MP_Penalty column
    """
    mp = df["MP-Measured"]
    
    # Compute penalty factor
    penalty = np.ones(len(df))
    
    # Region: soft < MP < hard - smooth transition
    in_transition = (mp > soft_threshold) & (mp < hard_threshold)
    
    # Normalized position in transition zone (0 at soft, 1 at hard)
    t = (mp[in_transition] - soft_threshold) / (hard_threshold - soft_threshold)
    
    # Smooth S-curve using cosine: P = (1 + cos(π*t)) / 2
    # At t=0 (soft threshold): P = 1
    # At t=1 (hard threshold): P = 0
    penalty[in_transition] = (1 + np.cos(np.pi * t)) / 2
    
    # Region: MP >= hard - zero fitness
    penalty[mp >= hard_threshold] = 0.0
    
    # Apply penalty to fitness
    df["MP_Penalty"] = penalty
    df["FitnessScore"] = df["FitnessScore"] * penalty

    return df


def apply_molprice_penalty(df, soft_threshold=3.0, hard_threshold=6.0):
    """
    Apply soft penalty for MolPrice (log USD/mmol) between thresholds.

    Uses the same cosine S-curve as apply_mp_penalty.

    Penalty factor P:
    - MolPrice <= soft_threshold: P = 1.0 (cheap, no penalty)
    - MolPrice >= hard_threshold: P = 0.0 (too expensive, eliminated)
    - Between: smooth cosine transition from 1 to 0

    Args:
        df: DataFrame with 'FitnessScore' and 'MolPrice' columns
        soft_threshold: Below this, no penalty (default 0.0 log USD/mmol)
        hard_threshold: At or above this, fitness = 0 (default 3.0 log USD/mmol)

    Returns:
        DataFrame with updated FitnessScore and new MolPrice_Penalty column
    """
    if "MolPrice" not in df.columns:
        df["MolPrice_Penalty"] = 1.0
        return df

    mp = df["MolPrice"]
    penalty = np.ones(len(df))

    in_transition = (mp > soft_threshold) & (mp < hard_threshold)
    t = (mp[in_transition] - soft_threshold) / (hard_threshold - soft_threshold)
    penalty[in_transition] = (1 + np.cos(np.pi * t)) / 2

    penalty[mp >= hard_threshold] = 0.0

    df["MolPrice_Penalty"] = penalty
    df["FitnessScore"] = df["FitnessScore"] * penalty

    return df


# ============================================================
# Invalid Fragment SMARTS (banned substructures)
# ============================================================

INVALID_SMARTS = {
    # ----- Cumulenes & Allenes -----
    "cumulene": Chem.MolFromSmarts("C=C=C"),
    "long_cumulene": Chem.MolFromSmarts("C=C=C=C"),
    
     # ----- Alkynes (reactive triple bonds) -----
    "alkyne": Chem.MolFromSmarts("C#C"),
    
    # ----- Peroxides & O-O bonds -----
    "dioxygen": Chem.MolFromSmarts("[O]=[O]"),
    "peroxide": Chem.MolFromSmarts("[O]-[O]"),
    "hydroperoxide": Chem.MolFromSmarts("[O][OH]"),
    "trioxide": Chem.MolFromSmarts("[O]-[O]-[O]"),
    
    # ----- Hemiacetals & Acetals -----
    "hemiacetal": Chem.MolFromSmarts("[CX4]([OH])([O])"),
    "acetal": Chem.MolFromSmarts("[CX4]([O])([O])"),
    "orthoester": Chem.MolFromSmarts("[CX4]([O])([O])([O])"),
    "orthocarbonate": Chem.MolFromSmarts("[CX4]([O])([O])([O])([O])"),
    
    # ----- Reactive Carbonyls -----
    "alpha_dicarbonyl": Chem.MolFromSmarts("[#6](=O)[#6](=O)"),
    
    # ----- Enols & Vinyl ethers -----
    "enol": Chem.MolFromSmarts("[OH][C]=[C]"),
    "vinyl_ether": Chem.MolFromSmarts("[C]-[O]-[C]=[C]"),
    "ketene": Chem.MolFromSmarts("C=C=O"),
    "ketene_acetal": Chem.MolFromSmarts("[C]-[O]-[C]=[C]-[O]-[C]"),
    "vinyl_alcohol": Chem.MolFromSmarts("[CH2]=[CH][OH]"),
    
    # ----- Anhydrides -----
    "acid_anhydride": Chem.MolFromSmarts("[C](=O)[O][C](=O)"),
    
    # ----- Strained rings -----
    "epoxide": Chem.MolFromSmarts("C1OC1"),
    "cyclopropene": Chem.MolFromSmarts("C1=CC1"),
    
    # ----- Other unstable -----
    "carbon_dioxide": Chem.MolFromSmarts("O=C=O"),
    "gem_diol": Chem.MolFromSmarts("[CX4]([OH])([OH])"),
    "enediol": Chem.MolFromSmarts("[OH]C=C[OH]"),
    "peroxyacid": Chem.MolFromSmarts("[C](=O)[O][O]"),
    "acyl_peroxide": Chem.MolFromSmarts("[C](=O)[O][O][C](=O)"),
    
    # ----- Polyunsaturated -----
    "conjugated_triene": Chem.MolFromSmarts("C=CC=CC=C"),
    
    # ----- Formates (hydrolyze to corrosive formic acid) -----
    "formate_ester": Chem.MolFromSmarts("[CH](=O)O[C]"),
    
    # ----- Lactols (cyclic hemiacetals) -----
    "lactol_4": Chem.MolFromSmarts("[OH]C1OCC1"),
    "lactol_5": Chem.MolFromSmarts("[OH]C1OCCC1"),
    "lactol_6": Chem.MolFromSmarts("[OH]C1OCCCC1"),
    
    # ----- Carbene (highly reactive) -----
    "carbene": Chem.MolFromSmarts("[C;X2;v2]"),
    
    # ----- Strained small rings -----
    "3_membered_ring": Chem.MolFromSmarts("[R3]"),  
    "4_membered_ring": Chem.MolFromSmarts("[R4]"),  
}


def has_invalid_fragments(smiles):
    """
    Returns True if molecule contains forbidden substructures
    that would make it unsuitable as a thermal/cooling fluid.
    Invalid SMILES are treated as invalid.
    """
    if not isinstance(smiles, str):
        return True

    if '.' in smiles:
        return True

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True

    for name, pattern in INVALID_SMARTS.items():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return True

    if has_small_rings(smiles):
        return True

    return False

def has_small_rings(smiles):
    """Returns True if the molecule contains 3- or 4-membered rings."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True
        ring_info = mol.GetRingInfo()
        for size in ring_info.AtomRings():
            if len(size) in (3, 4):
                return True
        return False
    except:
        return True

def has_rdkit_valence_errors(smiles):
    """Check if SMILES has valence errors when parsed by RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return mol is None
    except:
        return True


def assign_validity(
    df,
    sc_threshold=3,
    mp_max=-30,  # Kept for reference but NOT used in hard filter
    bp_min=100,
    dc_max=7,
    min_fp=423,
    use_biodeg=True,
    max_tox21=3,
):
    """
    Assign validity flag to molecules based on property thresholds
    and structural filters.
    
    NOTE: MP is NOT included here - it's handled by soft penalty separately.
    """
    invalid_fragment = df["SMILES"].apply(has_invalid_fragments)
    rdkit_invalid = df["SMILES"].apply(has_rdkit_valence_errors)

    # Check for radicals (unpaired electrons - not physically stable coolants)
    def _has_radicals(smi):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            return True
        return Descriptors.NumRadicalElectrons(mol) > 0
    has_radical = df["SMILES"].apply(_has_radicals)

    # Check for negative beta (physically impossible - would mean density increases with temp)
    negative_beta = (df["beta_40"] < 0) | (df["beta_100"] < 0)

    # MP removed from hard constraints - handled by soft penalty
    conditions = (
        (df["SCScore"] <= sc_threshold) &
        # (df["MP-Measured"] <= mp_max) &  # REMOVED - now soft penalty
        (df["BP-Measured"] >= bp_min) &
        (df["DC_exp"] <= dc_max) &
        (df["flashpoint"] >= min_fp) &
        ((not use_biodeg) | (df["Biodegradable"] == True)) &
        (df["Tox21_Score"] <= max_tox21) &
        (~invalid_fragment) &
        (~rdkit_invalid) &
        (~has_radical) &
        (~negative_beta)
    )

    df["is_valid"] = conditions.astype(int)
    return df


def apply_niching(df, tau=0.05, alpha=1000, p=2):
    """
    Apply niching penalty based on Tanimoto similarity to promote diversity.
    Uses precomputed fingerprints for efficiency.
    """
    # Precompute Morgan fingerprints
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=8, fpSize=2048)

    def mol_fp_gen(smiles_list):
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            fp = fpgen.GetFingerprint(mol) if mol else None
            yield fp

    fingerprints = list(mol_fp_gen(df["SMILES"]))

    # Compute average similarity for each molecule
    avg_similarities = []
    for i, fp_i in enumerate(fingerprints):
        if fp_i is None:
            avg_similarities.append(0.0)
            continue
        sims = [DataStructs.TanimotoSimilarity(fp_i, fp_j)
                for j, fp_j in enumerate(fingerprints) if i != j and fp_j is not None]
        avg_similarities.append(np.mean(sims) if sims else 0.0)

    df["AvgTanimotoSimilarity"] = avg_similarities

    # Apply niching penalty
    penalty = np.exp(-alpha * np.maximum(0, df["AvgTanimotoSimilarity"] - tau) ** p)
    df["NichedFitnessScore"] = df["FitnessScore"] * penalty * (1 - df["AvgTanimotoSimilarity"])
    
    return df


def k_way_tournament(elite_df, k=3):
    """
    Select one parent via k-way tournament from elite_df.
    elite_df must have columns: 'SMILES' and 'NichedFitnessScore'.
    """
    competitors = elite_df.sample(n=k, replace=False)
    winner = competitors.loc[competitors['NichedFitnessScore'].idxmax()]
    return winner['SMILES']


def TanimotoSimilarity(SMILES, SMILESList):
    """
    Calculate average Tanimoto similarity between a molecule and a list.
    Note: This is O(n) per call - for bulk operations use apply_niching instead.
    """
    SMILESList = [x for x in SMILESList if x != SMILES]

    ms = [Chem.MolFromSmiles(x) for x in SMILESList]
    SMILESms = Chem.MolFromSmiles(SMILES)

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=8, fpSize=2048)
    SMILESfps = fpgen.GetFingerprint(SMILESms)
    fps = [fpgen.GetFingerprint(x) for x in ms if x is not None]

    SimScores = [DataStructs.TanimotoSimilarity(SMILESfps, fp) for fp in fps]

    return sum(SimScores) / len(SimScores) if SimScores else 0.0


# ============================================================
# Tox21 Model Functions
# ============================================================

TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
    'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]


def load_tox21_models(model_dir):
    """
    Load all Tox21 classification models from directory.
    
    Supports two directory structures:
    1. Old style: model_dir/*.joblib (e.g., NR-AR.joblib)
    2. New style: model_dir/{target}/model/xgb_model.joblib
    """
    models = {}
    
    # Check for old style first (flat directory with .joblib files)
    old_style_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    
    if old_style_files:
        # Old style loading - raw models trained on MACCS fingerprints
        for filename in sorted(old_style_files):
            task_name = filename.replace(".joblib", "")
            model_path = os.path.join(model_dir, filename)
            models[task_name] = joblib.load(model_path)
        print(f"Loaded {len(models)} Tox21 models (old style - MACCS fingerprints)")
    else:
        # New style loading (subdirectories) - models trained on RDKit descriptors
        for task_name in TOX21_TASKS:
            model_path = os.path.join(model_dir, task_name, "model", "xgb_model.joblib")
            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                    # New style stores dict with 'model' and 'features' keys
                    # IMPORTANT: Keep the full dict so we can access features for prediction
                    if isinstance(model_data, dict) and 'model' in model_data:
                        models[task_name] = {
                            'model': model_data['model'],
                            'features': model_data.get('features', [])
                        }
                    else:
                        # Fallback if it's just a raw model
                        models[task_name] = model_data
                    print(f"  Loaded Tox21 model: {task_name}")
                except Exception as e:
                    print(f"  Warning: Could not load Tox21 model {task_name}: {e}")
            else:
                print(f"  Warning: Tox21 model not found: {model_path}")
        print(f"Loaded {len(models)} Tox21 models (new style - RDKit descriptors)")
    
    return models


def load_biodeg_model(model_dir):
    """
    Load biodegradability model from directory.
    
    Supports two structures:
    1. Old style: model_dir/biodegradability_model.pkl
    2. New style: model_dir/Activity/model/xgb_model.joblib
    """
    # Try old style first
    old_path = os.path.join(model_dir, "biodegradability_model.pkl")
    if os.path.exists(old_path):
        with open(old_path, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded biodegradability model (old style): {old_path}")
        return model
    
    # Try new style
    new_path = os.path.join(model_dir, "Activity", "model", "xgb_model.joblib")
    if os.path.exists(new_path):
        model_data = joblib.load(new_path)
        print(f"Loaded biodegradability model (new style): {new_path}")
        return model_data  # Return full dict so is_biodegradable can use features
    
    raise FileNotFoundError(f"Biodegradability model not found in {model_dir}. "
                            f"Expected either {old_path} or {new_path}")


def smiles_to_maccs(smiles):
    """Convert SMILES string to MACCS fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)


def predict_tox21(smiles_str, models, threshold=0.5):
    """
    Predict Tox21 toxicity score as sum of positive class probabilities.
    Lower is better (less toxic).
    
    NOTE: This function uses MACCS fingerprints for OLD-STYLE models only.
    For new-style models trained on descriptors, use predict_tox21_batch instead.
    """
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return 12.0  # Max score for invalid molecules

    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    x = arr.reshape(1, -1)

    total_prob = 0.0
    for model in models.values():
        try:
            prob = model.predict_proba(x)[0, 1]
            total_prob += prob
        except Exception:
            # If model fails, add 0.5 (neutral contribution)
            total_prob += 0.5

    return total_prob


def predict_tox21_batch(df, tox21_models):
    """
    Predict Tox21 scores for all molecules in a dataframe.
    
    Handles both old-style (MACCS fingerprints) and new-style (RDKit descriptors) models.
    
    Args:
        df: DataFrame with SMILES and descriptor columns
        tox21_models: Dict of model name -> model data
        
    Returns:
        pd.Series of Tox21 scores (sum of probabilities across all targets)
    """
    # Check if models are new-style (dict with 'features') or old-style (raw model)
    first_model_data = list(tox21_models.values())[0]
    is_new_style = isinstance(first_model_data, dict) and 'model' in first_model_data
    
    if not is_new_style:
        # Old style: use MACCS fingerprints
        return df["SMILES"].apply(lambda smi: predict_tox21(smi, tox21_models))
    
    # New style: use RDKit descriptors
    scores = []
    n_targets = len(tox21_models)
    
    for idx in range(len(df)):
        total_prob = 0.0
        
        for target_name, model_data in tox21_models.items():
            model = model_data['model']
            features = model_data.get('features', [])
            
            try:
                # Build feature vector from dataframe row
                X_values = []
                for feat in features:
                    if feat in df.columns:
                        val = df.iloc[idx][feat]
                        X_values.append(float(val) if pd.notna(val) else 0.0)
                    else:
                        X_values.append(0.0)
                
                X_row = np.array(X_values, dtype=np.float64).reshape(1, -1)
                
                # Predict probability of positive class
                prob = model.predict_proba(X_row)[0, 1]
                total_prob += prob
                
            except Exception as e:
                # If prediction fails, add neutral probability
                total_prob += 0.5
        
        scores.append(total_prob)
    
    return pd.Series(scores, index=df.index)


# ============================================================
# Regression Model Loading with Auxiliary Feature Support
# ============================================================

def load_fom1_direct_models(fom1_model_dir):
    """
    Load XGBoost+Descriptor FOM1 direct prediction models (ensemble of 5 folds)
    for both 40C and 100C.

    Each fold's joblib contains: {'model': XGBRegressor, 'scaler': StandardScaler, 'params': dict}
    The models were trained on a specific set of RDKit descriptors listed in descriptor_columns.json.

    Args:
        fom1_model_dir: Path to the FOM1 architecture comparison results directory,
                        e.g. training/FOM1_architecture_comparison/results

    Returns:
        dict with keys 'fom1_40' and 'fom1_100', each containing:
            - 'models': list of 5 XGBRegressor models (one per fold)
            - 'scalers': list of 5 StandardScaler objects (one per fold)
            - 'descriptor_columns': list of descriptor column names
    """
    import json

    fom1_models = {}

    for temp_key in ['fom1_40', 'fom1_100']:
        temp_dir = os.path.join(fom1_model_dir, temp_key)

        # Load descriptor columns
        desc_cols_path = os.path.join(temp_dir, 'descriptor_columns.json')
        if not os.path.exists(desc_cols_path):
            raise FileNotFoundError(f"FOM1 descriptor columns not found: {desc_cols_path}")
        with open(desc_cols_path) as f:
            descriptor_columns = json.load(f)

        # Load all 5 fold models
        models = []
        scalers = []
        for fold_id in range(5):
            model_path = os.path.join(temp_dir, f'xgboost_descriptors_fold{fold_id}_model.joblib')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"FOM1 model not found: {model_path}")
            model_data = joblib.load(model_path)
            models.append(model_data['model'])
            scalers.append(model_data['scaler'])

            # Prefer descriptor_columns from joblib (saved alongside model during training)
            # over descriptor_columns.json which may have been regenerated with a different RDKit
            if fold_id == 0 and 'descriptor_columns' in model_data:
                if model_data['descriptor_columns'] != descriptor_columns:
                    print(f"  NOTE: Using descriptor_columns from model joblib "
                          f"({len(model_data['descriptor_columns'])} cols) instead of JSON "
                          f"({len(descriptor_columns)} cols)")
                descriptor_columns = model_data['descriptor_columns']

        # Final check: ensure descriptor count matches scaler expectation
        expected_n = scalers[0].n_features_in_
        if expected_n != len(descriptor_columns):
            raise ValueError(
                f"FOM1 {temp_key}: descriptor_columns has {len(descriptor_columns)} entries "
                f"but scaler expects {expected_n} features. The models need to be retrained "
                f"with train_xgboost_descriptors.py to embed the correct feature names."
            )

        fom1_models[temp_key] = {
            'models': models,
            'scalers': scalers,
            'descriptor_columns': descriptor_columns,
        }
        print(f"  Loaded FOM1 direct model: {temp_key} (5-fold ensemble, {len(descriptor_columns)} descriptors)")

    return fom1_models


def load_regression_models_with_aux(targets, model_dir):
    """
    Load regression models with support for auxiliary features.
    
    This function loads models that may depend on predictions from other models.
    For example, Density_100C may use Density_40C as an input feature.
    
    The model metadata should contain:
    - 'model': the trained model
    - 'features': list of feature names
    - 'log_target': whether the target was log-transformed
    - 'auxiliary_feature': the original target name used as auxiliary (e.g., 'Density_40C_g_cm^3')
    - 'auxiliary_feature_name': the feature name in the model (e.g., 'AUX_Density_40C_g_cm^3')
    
    Returns:
        dict: Dictionary mapping target names to model data dictionaries
    """
    models = {}
    
    for target in targets:
        model_path = os.path.join(model_dir, target, "model", "xgb_model.joblib")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found for {target} at {model_path}")
            continue
        
        try:
            model_data = joblib.load(model_path)
            
            # Extract model components
            models[target] = {
                'model': model_data['model'],
                'features': model_data['features'],
                'log_target': model_data.get('log_target', False),
                'auxiliary_feature': model_data.get('auxiliary_feature'),
                'auxiliary_feature_name': model_data.get('auxiliary_feature_name')
            }
            
            # Log if auxiliary feature is present
            if model_data.get('auxiliary_feature'):
                print(f"  Loaded {target} (uses auxiliary: {model_data['auxiliary_feature']})")
            else:
                print(f"  Loaded {target}")
                
        except Exception as e:
            print(f"Error loading model for {target}: {e}")
            continue
    
    return models
