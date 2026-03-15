import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import Draw
from SCScorer import SCScorer
from pathlib import Path
from joblib import load

# === Utility functions ===

def canonicalize_smiles(smi):
    """Basic canonicalization."""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def strict_canonicalize_smiles(smi):
    """Strict canonical SMILES (non-isomeric) to enforce uniqueness."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

# Global cache for SCScore
_scscore_cache = {}

def get_scscore_cached(sc_model, smi):
    """Efficient SCScore with caching."""
    if smi in _scscore_cache:
        return _scscore_cache[smi]
    _, score = sc_model.get_score_from_smi(smi)
    _scscore_cache[smi] = score
    return score

def Toxicity(smiles, model_path="../models/tox21_gt4sd"):
    """Standalone Tox21 score (sum of 12 endpoint probabilities, range 0-12)."""
    from tox21_gt4sd import Tox21Predictor
    predictor = Tox21Predictor(model_path)
    return sum(predictor(smiles))

# === Model loading ===

def load_models(models_dir):
    models_dict = {}
    for fname in os.listdir(models_dir):
        if fname.endswith(".pkl"):
            key = fname.replace(".pkl", "")
            model_data = joblib.load(os.path.join(models_dir, fname))
            models_dict[key] = model_data
    return models_dict


def load_regression_models(targets, base_dir="../models"):
    """
    Load XGBoost regression models trained on RDKit descriptors.

    Returns:
        dict[target] = {
            'model': xgb_model,
            'features': list[str],
            'log_target': bool
        }
    """
    models = {}

    for target in targets:
        model_path = Path(base_dir) / target / "model" / "xgb_model.joblib"
        if not model_path.exists():
            print(f"Warning: Missing model for {target}, skipping: {model_path}")
            continue  # Skip this target instead of raising an error

        try:
            model_data = load(model_path)
        except Exception as e:
            print(f"Warning: Failed to load model for {target}: {e}")
            continue

        models[target] = {
            "model": model_data["model"],
            "features": model_data["features"],
            "log_target": model_data.get("log_target", False)
        }

    return models

# === Property prediction and fitness ===

def compute_missing_properties(df, models_dict, sc_model):
    """
    Ensures all model-predicted properties and SCScore are present in the DataFrame.
    Updates the DataFrame in-place and returns it.
    """
    for prop, model_data in models_dict.items():
        if prop in df.columns:
            continue

        model = model_data['model']
        descriptors = model_data['descriptor_names']
        use_log = model_data.get('use_log_target', False)

        X = df[descriptors]
        y_pred = model.predict(X)
        if use_log:
            y_pred = np.expm1(y_pred)
        df[prop] = y_pred

    if "SCScore" not in df.columns:
        df["SCScore"] = df["SMILES"].apply(lambda smi: get_scscore_cached(sc_model, smi))

    return df

def direct_ranking(models, combined_df, top_n=100, sc_model=None):
    if sc_model is None:
        raise ValueError("SCScore model instance required")

    combined_df = combined_df.copy()

    # Predict model properties
    for prop, model_data in models.items():
        model = model_data['model']
        descriptors = model_data['descriptor_names']
        use_log = model_data.get('use_log_target', False)
        X = combined_df[descriptors]
        y_pred = model.predict(X)
        if use_log:
            y_pred = np.expm1(y_pred)
        combined_df[prop] = y_pred

    # Add SCScore
    combined_df["SCScore"] = combined_df["SMILES"].apply(lambda smi: get_scscore_cached(sc_model, smi))

    # Define weights and objectives
    PROPERTY_WEIGHTS = {
        "Thermal_Conductivity_40C": {"weight": 0.125, "maximize": True},
        "Thermal_Conductivity_100C": {"weight": 0.125, "maximize": True},
        "Heat_Capacity_Constant_Pressure_40C_J_K_Mol": {"weight": 0.125, "maximize": True},
        "Heat_Capacity_Constant_Pressure_100C_J_K_Mol": {"weight": 0.125, "maximize": True},
        "Kinematic_Viscosity_40C": {"weight": 0.125, "maximize": False},
        "Kinematic_Viscosity_100C": {"weight": 0.125, "maximize": False},
        "SCScore": {"weight": 0.25, "maximize": False},
    }

    fitness = np.zeros(len(combined_df))
    for prop, settings in PROPERTY_WEIGHTS.items():
        vals = combined_df[prop].values.reshape(-1, 1)
        scaled = MinMaxScaler().fit_transform(vals).flatten()
        if not settings["maximize"]:
            scaled = 1.0 - scaled
        weighted = scaled * settings["weight"]
        fitness += weighted

    combined_df["FitnessScore"] = fitness
    combined_df.sort_values(by="FitnessScore", ascending=False, inplace=True)

    top_df = combined_df.head(top_n).reset_index(drop=True)
    return top_df

# === NSGA-II ===

def dominates(row1, row2):
    return all(r1 <= r2 for r1, r2 in zip(row1, row2)) and any(r1 < r2 for r1, r2 in zip(row1, row2))

def fast_non_dominated_sort(F):
    S = [[] for _ in range(len(F))]
    n = [0] * len(F)
    rank = [0] * len(F)
    fronts = [[]]

    for p in range(len(F)):
        for q in range(len(F)):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1], rank

def crowding_distance(F, front):
    distance = np.zeros(len(front))
    num_objectives = F.shape[1]

    for m in range(num_objectives):
        values = F[front, m]
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        distance[sorted_idx[0]] = distance[sorted_idx[-1]] = np.inf
        if sorted_values[-1] == sorted_values[0]:
            continue
        for i in range(1, len(front) - 1):
            distance[sorted_idx[i]] += (
                (sorted_values[i + 1] - sorted_values[i - 1]) /
                (sorted_values[-1] - sorted_values[0])
            )
    return distance

def nsga_parent_selection(models_dict, df, sc_model, num_selections=25, tournament_k=3):
    objectives = {
        'Conductivity': {
            'props': ['Thermal_Conductivity_40C', 'Thermal_Conductivity_100C'],
            'weight': 0.25,
            'invert': False
        },
        'Heat_Capacity': {
            'props': ['Heat_Capacity_Constant_Pressure_40C_J_K_Mol', 'Heat_Capacity_Constant_Pressure_100C_J_K_Mol'],
            'weight': 0.25,
            'invert': False
        },
        'Viscosity': {
            'props': ['Kinematic_Viscosity_40C', 'Kinematic_Viscosity_100C'],
            'weight': 0.25,
            'invert': True
        },
        'SCScore': {
            'props': ['SCScore'],
            'weight': 0.25,
            'invert': True
        }
    }

    df = compute_missing_properties(df, models_dict, sc_model)

    weighted_obj_matrix = []
    for group in objectives.values():
        prop_values = []
        for prop in group['props']:
            preds = np.array(df[prop]).reshape(-1, 1)
            scaled = MinMaxScaler().fit_transform(preds).flatten()
            if group['invert']:
                scaled = 1.0 - scaled
            prop_values.append(scaled)

        group_score = np.mean(np.vstack(prop_values), axis=0) * group['weight']
        weighted_obj_matrix.append(group_score)

    F = np.vstack(weighted_obj_matrix).T

    fronts, rank = fast_non_dominated_sort(F)
    df["Rank"] = rank
    crowding = np.zeros(len(df))
    for front in fronts:
        dists = crowding_distance(F, front)
        for i, idx in enumerate(front):
            crowding[idx] = dists[i]
    df["Crowding"] = crowding

    selected_indices = []
    while len(selected_indices) < num_selections:
        candidates = df.sample(tournament_k)
        best = candidates.sort_values(by=["Rank", "Crowding"], ascending=[True, False]).iloc[0]
        selected_indices.append(best.name)

    return df.loc[selected_indices].reset_index(drop=True)