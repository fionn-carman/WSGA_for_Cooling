import os
import pandas as pd
import pickle
import random
from rdkit import Chem
from rdkit.Chem import Draw, rdchem, Descriptors
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import joblib
import time

from mutations import mutate
from descriptors import (
    descriptor_funcs, descriptor_names, calc_descriptors,
    mordred_descriptor_names, rdkit_prefixed_names, calc_mordred_descriptors,
)
from evaluation import get_scscore_cached, strict_canonicalize_smiles

from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler


def is_biodegradable(smiles, biodeg_model, desc_row=None):
    """
    Predict biodegradability using the trained classification model.

    Supports Mordred-pipeline models (dict with 'selected_features') and
    legacy models. When desc_row (a Series with all descriptors) is provided,
    avoids recomputing descriptors.
    """
    if not isinstance(biodeg_model, dict) or 'selected_features' not in biodeg_model:
        # Legacy path (old-style models)
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            return False
        try:
            desc_values = [func(mol) for func in descriptor_funcs]
        except Exception:
            return False
        base_features = pd.DataFrame([desc_values], columns=descriptor_names)
        model = biodeg_model['model'] if isinstance(biodeg_model, dict) else biodeg_model
        features = biodeg_model.get('features', descriptor_names) if isinstance(biodeg_model, dict) else descriptor_names
        try:
            X = base_features[features]
        except KeyError:
            X = base_features
        try:
            return model.predict(X)[0] == 1
        except Exception:
            return False

    # Mordred-pipeline path
    model = biodeg_model['model']
    features = biodeg_model['selected_features']

    if desc_row is not None:
        try:
            X = pd.DataFrame([desc_row[features].values], columns=features)
        except KeyError:
            return False
    else:
        # Compute descriptors from scratch
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            return False
        try:
            rdkit_vals = calc_descriptors(mol, descriptor_funcs)
            mordred_vals = calc_mordred_descriptors(mol)
        except Exception:
            return False
        all_names = (
            [f"rdkit_{n}" for n in descriptor_names] +
            mordred_descriptor_names
        )
        all_vals = rdkit_vals + mordred_vals
        row = pd.Series(all_vals, index=all_names)
        try:
            X = pd.DataFrame([row[features].values], columns=features)
        except KeyError:
            return False

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    try:
        return model.predict(X)[0] == 1
    except Exception:
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


def evaluate_molecules(df, thermo_models, sc_model, tox21_models, biodeg_model,
                       drop_descriptors=True, molprice_model=None,
                       mahal_params=None, fom1_mlp_data=None,
                       fold_ensembles=None, conformal_quantiles=None):
    """
    Evaluate molecules: predict all properties using Mordred-pipeline models.

    All models use mordred_*/rdkit_* prefixed features.  40C only — no 100C,
    no Pr/Gr/Ra/Nu, no averaging.  Includes on-the-fly Mahalanobis OOD detection
    when mahal_params is provided.
    """
    df = df[df['SMILES'].notna()].copy()

    df['Mol'] = df['SMILES'].apply(lambda smi: Chem.MolFromSmiles(smi) if smi else None)
    df = df[df['Mol'].notna()].reset_index(drop=True)

    # ----------------------------
    # Compute descriptors (RDKit unprefixed + Mordred + rdkit-prefixed)
    # ----------------------------
    rdkit_data = [calc_descriptors(mol, descriptor_funcs) for mol in df['Mol']]
    rdkit_df = pd.DataFrame(rdkit_data, columns=descriptor_names)

    mordred_data = [calc_mordred_descriptors(mol) for mol in df['Mol']]
    mordred_df = pd.DataFrame(mordred_data, columns=mordred_descriptor_names)

    rdkit_prefixed_df = rdkit_df.copy()
    rdkit_prefixed_df.columns = rdkit_prefixed_names

    df = pd.concat([df[['SMILES', 'Mol']], rdkit_df, mordred_df, rdkit_prefixed_df], axis=1).copy()

    # Numeric coercion + NaN fill (vectorised)
    all_desc_cols = list(rdkit_df.columns) + list(mordred_df.columns) + list(rdkit_prefixed_df.columns)
    df[all_desc_cols] = df[all_desc_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

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
    # Predict Tox21 (GT4SD PaccMann MCA)
    # ----------------------------
    df["Tox21_Score"] = predict_tox21_batch(df, tox21_models)

    # ----------------------------
    # Predict thermophysical properties (all independent, no aux features)
    # ----------------------------
    for target, data in thermo_models.items():
        model = data['model']
        features = data['features']
        log_transform = data.get('log_transform', False)
        try:
            X = df[features].copy()
            y_pred = model.predict(X)
            if log_transform:
                y_pred = np.expm1(y_pred)
            df[target] = y_pred
        except Exception as e:
            print(f"Skipping {target}, error: {e}")
            df[target] = np.nan

    # ----------------------------
    # Override fom1_40C with MLP predictions (if enabled)
    # ----------------------------
    if fom1_mlp_data is not None:
        mlp_model = fom1_mlp_data['model']
        mlp_scaler_X = fom1_mlp_data['scaler_X']
        mlp_scaler_y = fom1_mlp_data['scaler_y']
        mlp_features = fom1_mlp_data['selected_features']
        try:
            X_mlp = df[mlp_features].copy()
            X_mlp_scaled = mlp_scaler_X.transform(X_mlp)
            y_mlp_scaled = mlp_model.predict(X_mlp_scaled)
            y_mlp = mlp_scaler_y.inverse_transform(
                y_mlp_scaled.reshape(-1, 1)
            ).ravel()
            if fom1_mlp_data.get('log_transform', False):
                y_mlp = np.expm1(y_mlp)
            df['fom1_40C'] = y_mlp
        except Exception as e:
            print(f"WARNING: MLP FOM1 prediction failed, keeping XGBoost: {e}")

    # ----------------------------
    # Predict biodegradability (using precomputed descriptor row)
    # ----------------------------
    biodeg_preds = []
    for i, row in df.iterrows():
        biodeg_preds.append(is_biodegradable(row['SMILES'], biodeg_model, desc_row=row))
    df["Biodegradable"] = biodeg_preds

    # ----------------------------
    # Compute derived thermal properties (40C only)
    # ----------------------------
    df["MW"] = df["Mol"].apply(lambda mol: Descriptors.MolWt(mol))

    df["Cp_40"] = df["cpsat_40C"] / df["MW"] * 1000
    df["alpha_40"] = df["tc_40C"] / (df["density_40C"] * 1000 * df["Cp_40"])
    df["nu_40"] = df["viscosity_40C"] * 1e-6
    df["FOM1_40"] = df["tc_40C"] * (
        (df["beta_40C"] * df["Cp_40"] * df["density_40C"] * 1000)
        / (df["nu_40"] * df["tc_40C"])
    ) ** 0.2813

    # ----------------------------
    # Mahalanobis OOD detection (per model)
    # ----------------------------
    if mahal_params is not None:
        for model_name, params in mahal_params.items():
            compute_mahalanobis_batch(df, params, model_name)
        # Summary columns
        ood_cols = [c for c in df.columns if c.startswith('OOD_')]
        if ood_cols:
            df['OOD_any'] = df[ood_cols].max(axis=1)
            df['OOD_count'] = (df[ood_cols] > 0).sum(axis=1)

    # ----------------------------
    # Fold ensemble uncertainty (per model)
    # ----------------------------
    if fold_ensembles is not None:
        df = predict_fold_ensemble(df, fold_ensembles, conformal_quantiles)

    # ----------------------------
    # Drop descriptors/Mol if requested
    # ----------------------------
    if drop_descriptors:
        drop_cols = list(descriptor_names) + list(mordred_descriptor_names) + list(rdkit_prefixed_names) + ['Mol']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

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


def apply_molprice_penalty(df, soft_threshold=3.0, hard_threshold=6.0):
    """
    Apply soft penalty for MolPrice (log USD/mmol) between thresholds.

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
    if "MolPrice" not in df.columns or df["MolPrice"].isna().all():
        df["MolPrice_Penalty"] = 1.0
        return df

    # No penalty when thresholds are zero/equal (nocost mode)
    if soft_threshold >= hard_threshold:
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
    
    # ----- Reactive alkenes / dienes -----
    "any_alkene": Chem.MolFromSmarts("[C]=[C]"),
    "conjugated_diene": Chem.MolFromSmarts("C=CC=C"),

    # ----- Reactive Carbonyls -----
    "aldehyde": Chem.MolFromSmarts("[CH]=O"),
    "carboxylic_acid": Chem.MolFromSmarts("[C](=O)[OH]"),
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
    mp_max=-30,
    bp_min=100,
    dc_max=7,
    min_fp=423,
    use_biodeg=True,
    max_tox21=3,
):
    """
    Assign validity flag to molecules based on property thresholds
    and structural filters.
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
    negative_beta = (df["beta_40C"] < 0) if "beta_40C" in df.columns else pd.Series(False, index=df.index)

    conditions = (
        (df["SCScore"] <= sc_threshold) &
        (df["mp"] <= mp_max) &
        (df["bp"] >= bp_min) &
        (df["dc"] <= dc_max) &
        (df["fp"] >= min_fp) &
        ((not use_biodeg) | (df["Biodegradable"] == True)) &
        (df["Tox21_Score"] <= max_tox21) &
        (~invalid_fragment) &
        (~rdkit_invalid) &
        (~has_radical) &
        (~negative_beta)
    )

    df["is_valid"] = conditions.astype(int)
    return df


def compute_tanimoto_similarities(df):
    """
    Compute average Tanimoto similarity for each molecule in the population.
    Adds 'AvgTanimotoSimilarity' column to df.
    """
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=8, fpSize=2048)

    def mol_fp_gen(smiles_list):
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            fp = fpgen.GetFingerprint(mol) if mol else None
            yield fp

    fingerprints = list(mol_fp_gen(df["SMILES"]))

    avg_similarities = []
    for i, fp_i in enumerate(fingerprints):
        if fp_i is None:
            avg_similarities.append(0.0)
            continue
        sims = [DataStructs.TanimotoSimilarity(fp_i, fp_j)
                for j, fp_j in enumerate(fingerprints) if i != j and fp_j is not None]
        avg_similarities.append(np.mean(sims) if sims else 0.0)

    df["AvgTanimotoSimilarity"] = avg_similarities
    return df


def apply_niching(df, tau=0.05, alpha=1000, p=2):
    """
    Apply niching penalty based on Tanimoto similarity to promote diversity.
    Uses precomputed fingerprints for efficiency.
    """
    df = compute_tanimoto_similarities(df)

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
# Tox21 Model Functions (GT4SD)
# ============================================================

TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
    'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]


def load_tox21_predictor(model_path):
    """
    Initialise the PaccMann MCA Tox21 toxicity predictor.

    Uses the pretrained MCA (Multiscale Convolutional Attentive) model from
    the GT4SD model hub, trained on the Tox21 challenge dataset (12 endpoints).
    Weights are stored locally in models/tox21_gt4sd/.

    Args:
        model_path: Path to the tox21_gt4sd model directory containing
                    model_params.json, smiles_language.pkl, and weights/.

    Returns:
        Tox21Predictor callable (SMILES -> list of 12 floats)
    """
    from tox21_gt4sd import Tox21Predictor
    predictor = Tox21Predictor(model_path)
    print("Loaded PaccMann MCA Tox21 predictor (12 endpoints)")
    return predictor


def load_biodeg_model(model_dir):
    """
    Load biodegradability model from directory.

    Supports three structures (tried in order):
    1. Mordred pipeline: model_dir/model/xgb_model.joblib (has 'selected_features')
    2. Old joblib: model_dir/Activity/model/xgb_model.joblib
    3. Old pickle: model_dir/biodegradability_model.pkl
    """
    # Mordred pipeline (new standard)
    mordred_path = os.path.join(model_dir, "model", "xgb_model.joblib")
    if os.path.exists(mordred_path):
        model_data = joblib.load(mordred_path)
        n_feat = len(model_data.get('selected_features', []))
        print(f"Loaded biodegradability model (Mordred pipeline): {mordred_path} ({n_feat} features)")
        return model_data

    # Old joblib
    old_joblib = os.path.join(model_dir, "Activity", "model", "xgb_model.joblib")
    if os.path.exists(old_joblib):
        model_data = joblib.load(old_joblib)
        print(f"Loaded biodegradability model (old joblib): {old_joblib}")
        return model_data

    # Old pickle
    old_pkl = os.path.join(model_dir, "biodegradability_model.pkl")
    if os.path.exists(old_pkl):
        with open(old_pkl, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded biodegradability model (old pickle): {old_pkl}")
        return model

    raise FileNotFoundError(f"Biodegradability model not found in {model_dir}")


def predict_tox21_batch(df, tox21_predictor):
    """
    Predict Tox21 scores for all molecules using the GT4SD predictor.

    For each molecule the predictor returns 12 probabilities (one per Tox21
    endpoint).  The score is their sum (range 0-12, lower = less toxic).

    Args:
        df: DataFrame with a SMILES column
        tox21_predictor: GT4SD Tox21 property predictor (from load_tox21_predictor)

    Returns:
        pd.Series of Tox21 scores
    """
    scores = []
    for smi in df["SMILES"]:
        try:
            probs = tox21_predictor(smi)  # list of 12 floats
            scores.append(sum(probs))
        except Exception:
            scores.append(12.0)  # worst-case for unparseable molecules
    return pd.Series(scores, index=df.index)


# ============================================================
# Regression Model Loading with Auxiliary Feature Support
# ============================================================

# ============================================================
# On-the-Fly Mahalanobis OOD Detection
# ============================================================

def _drop_correlated_features(X, corr_threshold=0.95):
    """Remove highly correlated features. Returns indices to keep."""
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    drop = set()
    while True:
        counts = np.zeros(n, dtype=int)
        for i in range(n):
            if i in drop:
                continue
            for j in range(i + 1, n):
                if j in drop:
                    continue
                if abs(corr[i, j]) > corr_threshold:
                    counts[i] += 1
                    counts[j] += 1
        active = {i: counts[i] for i in range(n) if i not in drop and counts[i] > 0}
        if not active:
            break
        drop.add(max(active, key=active.get))
    return sorted(set(range(n)) - drop)


def compute_mahalanobis_params(model_name, model_features, data_dir,
                                threshold_pct=0.975, corr_threshold=0.95):
    """
    Precompute Mahalanobis parameters for one property model.

    Uses Ledoit-Wolf shrinkage covariance in decorrelated descriptor space.
    OOD threshold is the 97.5th percentile of training D_M^2.

    Supports both Mordred-pipeline models (prefixed features, separate descriptor CSV)
    and legacy models (unprefixed features, descriptors inline in training CSV).

    Args:
        model_name: Property name (e.g., 'density_40C', 'bp')
        model_features: List of selected feature names
        data_dir: Path to training/data directory
        threshold_pct: Percentile for OOD threshold
        corr_threshold: Correlation threshold for feature decorrelation

    Returns:
        dict with keys: features, scaler, keep_idx, location, precision,
                        threshold_sq   (or None if data unavailable)
    """
    # Determine training data source
    thermo_props = {'density_40C': 'density', 'viscosity_40C': 'viscosity',
                    'tc_40C': 'tc', 'cpsat_40C': 'cpsat',
                    'beta_40C': 'beta', 'fom1_40C': 'fom1'}
    constraint_props = {'bp': 'BP-Measured', 'mp': 'MP-Measured',
                        'fp': 'flashpoint', 'dc': 'DC_exp'}

    # Detect if model uses prefixed features (Mordred pipeline) or unprefixed (legacy)
    is_prefixed = any(f.startswith('mordred_') or f.startswith('rdkit_') for f in model_features[:5])

    if model_name in thermo_props:
        prop_key = thermo_props[model_name]
        if is_prefixed:
            # Mordred pipeline: descriptors in separate CSV, SMILES in property CSV
            desc_csv = os.path.join(data_dir, 'nist_8100', 'descriptors_rdkit_mordred.csv')
            smiles_csv = os.path.join(data_dir, 'nist_8100', f'{prop_key}_cho_cleaned.csv')
        else:
            # Legacy: descriptors inline in property CSV
            desc_csv = os.path.join(data_dir, 'nist_8100', f'{prop_key}_cho_cleaned.csv')
            smiles_csv = desc_csv  # same file
    elif model_name in constraint_props:
        prop_key = constraint_props[model_name]
        if is_prefixed:
            desc_csv = os.path.join(data_dir, 'constraints', 'descriptors_rdkit_mordred.csv')
            smiles_csv = os.path.join(data_dir, 'constraints', f'{prop_key}_cleaned.csv')
        else:
            desc_csv = os.path.join(data_dir, 'constraints', f'{prop_key}_cleaned.csv')
            smiles_csv = desc_csv
    else:
        print(f"  Mahalanobis: unknown model {model_name}, skipping")
        return None

    if not os.path.exists(desc_csv):
        print(f"  Mahalanobis: missing data for {model_name} ({desc_csv})")
        return None

    if is_prefixed and desc_csv != smiles_csv:
        # Mordred pipeline: filter descriptor CSV to training SMILES
        if not os.path.exists(smiles_csv):
            print(f"  Mahalanobis: missing SMILES data for {model_name}")
            return None
        smiles_df = pd.read_csv(smiles_csv, usecols=['SMILES'])
        train_smiles = set(smiles_df['SMILES'].dropna().unique())
        desc_df = pd.read_csv(desc_csv)
        desc_df = desc_df[desc_df['SMILES'].isin(train_smiles)].copy()
    else:
        # Legacy: descriptors are in the training CSV itself
        desc_df = pd.read_csv(desc_csv)

    # Check feature availability
    available = [f for f in model_features if f in desc_df.columns]
    if len(available) < 2:
        print(f"  Mahalanobis: too few features for {model_name} "
              f"({len(available)}/{len(model_features)} available)")
        return None

    X_train = desc_df[available].values.astype(float)

    # Drop rows with NaN
    valid = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[valid]
    n_train = X_train.shape[0]

    if n_train < 10:
        print(f"  Mahalanobis: too few training samples for {model_name} ({n_train})")
        return None

    # Standardise
    scaler = StandardScaler().fit(X_train)
    X_sc = scaler.transform(X_train)

    # Decorrelate
    keep_idx = _drop_correlated_features(X_sc, corr_threshold)
    n_kept = len(keep_idx)
    X_sc = X_sc[:, keep_idx]

    # Ledoit-Wolf covariance
    lw = LedoitWolf().fit(X_sc)

    # Training threshold
    diff = X_sc - lw.location_
    train_mahal_sq = np.maximum(np.sum(diff @ lw.precision_ * diff, axis=1), 0)
    threshold_sq = np.percentile(train_mahal_sq, 100 * threshold_pct)

    print(f"  Mahalanobis {model_name}: {n_train} train, {len(available)} feat -> {n_kept} kept, "
          f"p97.5={np.sqrt(threshold_sq):.1f}")

    return {
        'features': available,
        'scaler': scaler,
        'keep_idx': keep_idx,
        'location': lw.location_,
        'precision': lw.precision_,
        'threshold_sq': threshold_sq,
    }


def compute_mahalanobis_batch(df, params, model_name):
    """Compute Mahalanobis distance for a batch of molecules (in-place)."""
    p = params
    try:
        X = df[p['features']].values.astype(float)
    except KeyError:
        df[f'Mahal_{model_name}'] = np.nan
        df[f'OOD_{model_name}'] = 1
        return

    X = np.nan_to_num(X)
    X_sc = p['scaler'].transform(X)[:, p['keep_idx']]
    diff = X_sc - p['location']
    mahal_sq = np.maximum(np.sum(diff @ p['precision'] * diff, axis=1), 0)
    df[f'Mahal_{model_name}'] = np.sqrt(mahal_sq)
    df[f'OOD_{model_name}'] = (mahal_sq > p['threshold_sq']).astype(int)


def load_fom1_direct_models(fom1_model_dir):
    """
    Load XGBoost+Descriptor FOM1 direct prediction models (ensemble of 5 folds)
    for both 40C and 100C.

    Each fold's joblib contains:
        {'model': XGBRegressor, 'scaler': StandardScaler,
         'params': dict, 'descriptor_columns': list[str]}

    Supports two directory layouts:
        1. Flat: fom1_model_dir contains fom1_40/ and fom1_100/ subdirectories
        2. Split: separate directories specified via a dict mapping
           (used when 40C and 100C models live in models/FOM1_direct_5fold_40C etc.)

    Args:
        fom1_model_dir: Either a single path containing fom1_40/ and fom1_100/ subdirs,
                        or the base models/ directory containing FOM1_direct_5fold_40C
                        and FOM1_direct_5fold_100C.

    Returns:
        dict with keys 'fom1_40' and 'fom1_100', each containing:
            - 'models': list of 5 XGBRegressor models (one per fold)
            - 'scalers': list of 5 StandardScaler objects (one per fold)
            - 'descriptor_columns': list of descriptor column names
    """
    import json

    fom1_models = {}

    # Resolve directory for each temperature
    dir_candidates = {
        'fom1_40': [
            os.path.join(fom1_model_dir, 'fom1_40'),
            os.path.join(fom1_model_dir, 'FOM1_direct_5fold_40C'),
        ],
        'fom1_100': [
            os.path.join(fom1_model_dir, 'fom1_100'),
            os.path.join(fom1_model_dir, 'FOM1_direct_5fold_100C'),
        ],
    }

    for temp_key, candidates in dir_candidates.items():
        temp_dir = None
        for cand in candidates:
            if os.path.isdir(cand):
                temp_dir = cand
                break
        if temp_dir is None:
            raise FileNotFoundError(
                f"FOM1 {temp_key} model directory not found. Tried: {candidates}"
            )

        # Load descriptor columns from JSON if available
        descriptor_columns = None
        desc_cols_path = os.path.join(temp_dir, 'descriptor_columns.json')
        if os.path.exists(desc_cols_path):
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

            # Use descriptor_columns from joblib if not loaded from JSON
            if fold_id == 0 and 'descriptor_columns' in model_data:
                if descriptor_columns is None:
                    descriptor_columns = model_data['descriptor_columns']
                elif model_data['descriptor_columns'] != descriptor_columns:
                    print(f"  NOTE: Using descriptor_columns from model joblib "
                          f"({len(model_data['descriptor_columns'])} cols) instead of JSON "
                          f"({len(descriptor_columns)} cols)")
                    descriptor_columns = model_data['descriptor_columns']

        if descriptor_columns is None:
            raise FileNotFoundError(
                f"FOM1 {temp_key}: no descriptor_columns found in JSON or joblib"
            )

        # Final check: ensure descriptor count matches scaler expectation
        expected_n = scalers[0].n_features_in_
        if expected_n != len(descriptor_columns):
            raise ValueError(
                f"FOM1 {temp_key}: descriptor_columns has {len(descriptor_columns)} entries "
                f"but scaler expects {expected_n} features."
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
    Load regression models (Mordred-pipeline format).

    Each model joblib contains:
    - 'model': trained XGBRegressor/Classifier
    - 'selected_features': list of mordred_*/rdkit_* feature names
    - 'log_transform': bool (target was log1p-transformed)
    - 'property': str
    - 'best_params': dict

    Also supports legacy format ('features', 'log_target') for backward compat.
    """
    models = {}

    for target in targets:
        model_path = os.path.join(model_dir, target, "model", "xgb_model.joblib")

        if not os.path.exists(model_path):
            print(f"Warning: Model not found for {target} at {model_path}")
            continue

        try:
            model_data = joblib.load(model_path)

            # Auto-detect format
            if 'selected_features' in model_data:
                features = model_data['selected_features']
                log_transform = model_data.get('log_transform', False)
            elif 'features' in model_data:
                features = model_data['features']
                log_transform = model_data.get('log_target', False)
            else:
                print(f"Warning: No feature list found for {target}")
                continue

            models[target] = {
                'model': model_data['model'],
                'features': features,
                'log_transform': log_transform,
            }
            print(f"  Loaded {target} ({len(features)} features, log={log_transform})")

        except Exception as e:
            print(f"Error loading model for {target}: {e}")
            continue

    return models


def load_fold_ensemble_models(targets, model_dir):
    """
    Load 5-fold XGBoost ensembles for uncertainty estimation.

    Each fold model joblib contains:
    - 'model': XGBRegressor
    - 'scaler': StandardScaler (or None)
    - 'selected_features': list of feature names
    - 'log_transform': bool

    Returns:
        dict mapping target → list of 5 fold dicts
    """
    ensembles = {}

    for target in targets:
        fold_models = []
        all_found = True
        for fold_id in range(5):
            path = os.path.join(model_dir, target, "model", f"fold{fold_id}_model.joblib")
            if not os.path.exists(path):
                print(f"  Warning: fold model not found: {path}")
                all_found = False
                break

            data = joblib.load(path)
            fold_models.append({
                'model': data['model'],
                'scaler': data.get('scaler'),
                'features': data.get('selected_features', data.get('features')),
                'log_transform': data.get('log_transform', data.get('log_target', False)),
            })

        if all_found and len(fold_models) == 5:
            ensembles[target] = fold_models
            n_feat = len(fold_models[0]['features'])
            print(f"  Loaded fold ensemble: {target} (5 folds, ~{n_feat} features)")
        else:
            print(f"  Skipping fold ensemble for {target} (missing fold models)")

    return ensembles


def compute_conformal_quantiles(model_dir, targets, alpha=0.1):
    """
    Compute conformal prediction quantiles from out-of-fold residuals.

    Uses constant-width conformal: q_hat = quantile(|OOF_residual|, 1-alpha).
    At inference, the 90% prediction interval is [y_pred - q_hat, y_pred + q_hat].

    Args:
        model_dir: path to models/ directory
        targets: list of target names
        alpha: miscoverage rate (0.1 = 90% CI)

    Returns:
        dict mapping target → q_hat (float)
    """
    quantiles = {}

    for target in targets:
        preds_path = os.path.join(model_dir, target, "model", "predictions.csv")
        if not os.path.exists(preds_path):
            print(f"  Warning: no predictions.csv for {target}, skipping conformal")
            continue

        df = pd.read_csv(preds_path)
        if 'residual' not in df.columns:
            print(f"  Warning: no 'residual' column in {preds_path}")
            continue

        abs_resid = df['residual'].abs().values
        n = len(abs_resid)
        # Finite-sample correction: (1-alpha)(n+1)/n
        level = min((1 - alpha) * (n + 1) / n, 1.0)
        q = np.quantile(abs_resid, level)
        quantiles[target] = q
        print(f"  Conformal {target}: q_{1-alpha:.0%} = {q:.2f} (n={n})")

    return quantiles


def predict_fold_ensemble(df, fold_ensembles, conformal_quantiles=None):
    """
    Predict with 5-fold ensembles and compute uncertainty columns.

    For each target, predicts with all 5 fold models and outputs:
    - {target}_fold_mean: mean of 5 fold predictions
    - {target}_fold_std: std of 5 fold predictions
    - {target}_ci_lower: fold_mean - q_hat (if conformal available)
    - {target}_ci_upper: fold_mean + q_hat (if conformal available)

    Args:
        df: DataFrame with descriptor columns
        fold_ensembles: dict from load_fold_ensemble_models()
        conformal_quantiles: dict from compute_conformal_quantiles() (optional)

    Returns:
        df with uncertainty columns added
    """
    if fold_ensembles is None:
        return df

    for target, folds in fold_ensembles.items():
        fold_preds = []
        for fold_data in folds:
            try:
                X_fold = df[fold_data['features']].copy()
                if fold_data['scaler'] is not None:
                    X_fold = pd.DataFrame(
                        fold_data['scaler'].transform(X_fold),
                        columns=fold_data['features'],
                        index=df.index,
                    )
                pred = fold_data['model'].predict(X_fold)
                if fold_data['log_transform']:
                    pred = np.expm1(pred)
                fold_preds.append(pred)
            except Exception as e:
                print(f"  Warning: fold prediction failed for {target}: {e}")
                break
        else:
            # All 5 folds succeeded
            fold_preds = np.array(fold_preds)  # (5, n_molecules)
            df[f'{target}_fold_mean'] = fold_preds.mean(axis=0)
            df[f'{target}_fold_std'] = fold_preds.std(axis=0, ddof=1)

            if conformal_quantiles and target in conformal_quantiles:
                q = conformal_quantiles[target]
                df[f'{target}_ci_lower'] = df[f'{target}_fold_mean'] - q
                df[f'{target}_ci_upper'] = df[f'{target}_fold_mean'] + q

    return df


# ============================================================
# NSGA-II Core Functions
# ============================================================

def fast_non_dominated_sort(objectives):
    """
    Standard NSGA-II non-dominated sort.

    Args:
        objectives: (N, M) array where each row is maximised.

    Returns:
        List of fronts, each front a list of row indices.
        fronts[0] = Pareto-optimal set, fronts[1] = next front, etc.
    """
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)   # how many dominate me
    dominated_set = [[] for _ in range(n)]       # whom do I dominate
    ranks = np.full(n, -1, dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            diff = objectives[i] - objectives[j]
            if np.all(diff >= 0) and np.any(diff > 0):
                # i dominates j
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif np.all(diff <= 0) and np.any(diff < 0):
                # j dominates i
                dominated_set[j].append(i)
                domination_count[i] += 1

    fronts = []
    current_front = np.where(domination_count == 0)[0].tolist()

    rank = 0
    while current_front:
        for idx in current_front:
            ranks[idx] = rank
        fronts.append(current_front)
        next_front = []
        for idx in current_front:
            for dominated_idx in dominated_set[idx]:
                domination_count[dominated_idx] -= 1
                if domination_count[dominated_idx] == 0:
                    next_front.append(dominated_idx)
        current_front = next_front
        rank += 1

    return fronts


def crowding_distance_3d(objectives_3d, front_indices):
    """
    Crowding distance in 3D: FOM1, -MolPrice, -AvgTanimoto.

    Boundary points (best/worst in any dimension) get inf distance
    to ensure they are always preserved.

    Args:
        objectives_3d: (N, 3) array of objective values for the full population.
        front_indices: list of indices belonging to this front.

    Returns:
        dict mapping index -> crowding distance.
    """
    front_indices = list(front_indices)
    n = len(front_indices)
    if n <= 2:
        return {idx: float('inf') for idx in front_indices}

    distances = {idx: 0.0 for idx in front_indices}
    obj_vals = objectives_3d[front_indices]  # (n, 3)

    for m in range(3):
        sorted_order = np.argsort(obj_vals[:, m])
        sorted_indices = [front_indices[i] for i in sorted_order]

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        obj_range = obj_vals[sorted_order[-1], m] - obj_vals[sorted_order[0], m]
        if obj_range == 0:
            continue

        for k in range(1, n - 1):
            distances[sorted_indices[k]] += (
                (obj_vals[sorted_order[k + 1], m] - obj_vals[sorted_order[k - 1], m])
                / obj_range
            )

    return distances


def nsga2_tournament(pareto_ranks, crowding_distances, k=2):
    """
    Binary tournament: prefer lower pareto_rank, then higher crowding_distance.

    Args:
        pareto_ranks: array of Pareto ranks for the population (length N).
        crowding_distances: array of crowding distances (length N).
        k: tournament size (default 2).

    Returns:
        Index of the tournament winner.
    """
    n = len(pareto_ranks)
    competitors = random.sample(range(n), min(k, n))

    best = competitors[0]
    for c in competitors[1:]:
        if pareto_ranks[c] < pareto_ranks[best]:
            best = c
        elif pareto_ranks[c] == pareto_ranks[best]:
            if crowding_distances[c] > crowding_distances[best]:
                best = c
    return best


def compute_hypervolume_2d(front_points, ref_point):
    """
    2D hypervolume by sorting + rectangle summation.

    Both objectives are maximised. Points dominated by the reference
    are excluded. The reference point should be *below* the front
    (e.g. [0, -10] for FOM1 and -MolPrice).

    Args:
        front_points: (K, 2) array of non-dominated objective vectors.
        ref_point: (2,) reference point (lower bound).

    Returns:
        Hypervolume (float).
    """
    pts = np.asarray(front_points)
    ref = np.asarray(ref_point)

    # Filter points that are worse than reference in any objective
    mask = np.all(pts > ref, axis=1)
    pts = pts[mask]

    if len(pts) == 0:
        return 0.0

    # Sort by first objective descending
    order = np.argsort(-pts[:, 0])
    pts = pts[order]

    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y

    return hv


def compute_nsga2_objectives(df, target, target_config):
    """
    Compute the two NSGA-II objectives (both maximised).

    Obj 1: FOM1 (or chosen target), zeroed for invalid molecules.
    Obj 2: Affordability = -MolPrice, set to -1e6 for invalid molecules.

    Args:
        df: evaluated DataFrame with 'is_valid', target column, and 'MolPrice'.
        target: target key (e.g. 'FOM1_direct').
        target_config: dict mapping target key to {column, maximize}.

    Returns:
        (N, 2) numpy array of objectives.
    """
    target_col = target_config[target]["column"]
    maximize = target_config[target]["maximize"]

    fom = df[target_col].values.copy().astype(float)
    if not maximize:
        fom = -fom

    # Zero out invalid molecules on objective 1
    invalid = df["is_valid"].values == 0
    fom[invalid] = 0.0

    # Objective 2: affordability = -MolPrice
    if "MolPrice" in df.columns and not df["MolPrice"].isna().all():
        afford = -df["MolPrice"].values.copy().astype(float)
    else:
        afford = np.zeros(len(df))

    afford[invalid] = -1e6

    return np.column_stack([fom, afford])
