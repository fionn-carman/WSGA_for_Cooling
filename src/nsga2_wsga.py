"""
NSGA-II WSGA - Multi-Objective Genetic Algorithm for Cooling Fluid Discovery

Optimises two objectives simultaneously:
  1. FOM1 (thermal figure of merit) — maximised
  2. Affordability (-MolPrice) — maximised

Uses NSGA-II non-dominated sorting + crowding distance in 3D
(FOM1, -MolPrice, -AvgTanimoto) for selection.  All other properties
remain hard validity constraints via assign_validity().

Fork of wsga.py — shares all mutation, crossover, evaluation, and
validity infrastructure.
"""

import os
import sys
import pandas as pd
import random
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdchem
from tqdm import tqdm
import numpy as np
import argparse

from wsga_helper import (
    apply_mutations_to_population,
    evaluate_molecules,
    load_tox21_predictor,
    load_biodeg_model,
    compute_tanimoto_similarities,
    assign_validity,
    load_regression_models_with_aux,
    load_fom1_direct_models,
    fast_non_dominated_sort,
    crowding_distance_3d,
    nsga2_tournament,
    compute_hypervolume_2d,
    compute_nsga2_objectives,
)
from evaluation import get_scscore_cached, strict_canonicalize_smiles
from fragment_utils import prepare_fragments, crossover_fragments, crossover_mol_fragments
from SCScorer import SCScorer
from generate_molecules import generate_initial_population, load_combined_training_data


# =============================
# Logging Setup
# =============================

jobid_raw = os.environ.get("PBS_JOBID", "local")
jobid = jobid_raw.split(".")[0].split("[")[0]
array_idx = os.environ.get("PBS_ARRAY_INDEX", "0")

logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", jobid)
os.makedirs(logs_dir, exist_ok=True)

log_path = os.path.join(logs_dir, f"nsga2_{array_idx}.log")
log = open(log_path, "a", buffering=1)
sys.stdout = log
sys.stderr = log

print(f"Logging to: {log_path}")


# =============================
# Command-line Arguments
# =============================

parser = argparse.ArgumentParser(description="Run NSGA-II Multi-Objective WSGA")
parser.add_argument("--population_size", type=int, default=1000, help="Population size")
parser.add_argument("--mutation_rate", type=float, default=0.8, help="Mutation rate")
parser.add_argument("--output_dir", type=str, default="../outputs", help="Output directory")
parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
parser.add_argument("--model_dir", type=str, default="../models", help="Model directory")
parser.add_argument("--Tau", type=float, default=0.05, help="Niching threshold for Tanimoto similarity")
parser.add_argument("--target", type=str, default="FOM1_direct",
    choices=["FOM1", "FOM1_40", "FOM1_100",
             "FOM1_direct", "FOM1_direct_40", "FOM1_direct_100"],
    help="Target property (first objective)")
parser.add_argument("--fom1_model_dir", type=str, default="../models",
    help="Directory containing FOM1 direct models")
parser.add_argument("--top_n", type=int, default=40, help="Number of top molecules to track")
parser.add_argument("--num_generations", type=int, default=200, help="Number of generations")
parser.add_argument("--tournament_k", type=int, default=2, help="Tournament size")

# Validity thresholds
parser.add_argument("--mp_threshold", type=float, default=-30, help="Max melting point (C)")
parser.add_argument("--bp_threshold", type=float, default=70, help="Min boiling point (C)")
parser.add_argument("--dc_threshold", type=float, default=8, help="Max dielectric constant")
parser.add_argument("--fp_threshold", type=float, default=373, help="Min flash point (K)")
parser.add_argument("--sc_threshold", type=float, default=3, help="Max SCScore")
parser.add_argument("--tox_threshold", type=float, default=3, help="Max Tox21 score")

parser.add_argument("--no_biodeg", action="store_true",
    help="Disable biodegradability filter")

args = parser.parse_args()

USE_BIODEG_FILTER = not args.no_biodeg


# =============================
# Configuration
# =============================

GENERATION_SIZE = args.population_size
INITIAL_POPULATION_SIZE = 4 * GENERATION_SIZE
BASE_MUTATION_RATE = args.mutation_rate
NUM_GENERATIONS = args.num_generations
FRAGMENT_LIMIT = 2000
TOURNAMENT_K = args.tournament_k
TOP_N = args.top_n

# Stagnation — hypervolume-based
STAGNATION_WINDOW = 5
STAGNATION_THRESHOLD = 0.001   # relative hypervolume improvement
STAGNATION_RESTART_RATIO = 0.3
MAX_STAGNATION_RESTARTS = 5

# Adaptive mutation
MIN_MUTATION_RATE = 0.3
MAX_MUTATION_RATE = 0.95
MUTATION_BOOST_FACTOR = 0.15

# Target
TARGET = args.target
TARGET_CONFIG = {
    "FOM1":  {"column": "FOM1_avg",  "maximize": True},
    "FOM1_40":  {"column": "FOM1_40",  "maximize": True},
    "FOM1_100": {"column": "FOM1_100", "maximize": True},
    "FOM1_direct":     {"column": "FOM1_direct_avg",  "maximize": True},
    "FOM1_direct_40":  {"column": "FOM1_40C_direct",  "maximize": True},
    "FOM1_direct_100": {"column": "FOM1_100C_direct", "maximize": True},
}

MP_THRESHOLD = args.mp_threshold
BP_THRESHOLD = args.bp_threshold
DC_THRESHOLD = args.dc_threshold
MIN_FLASHPOINT = args.fp_threshold
MAX_SCSCORE = args.sc_threshold
MAX_TOX21 = args.tox_threshold

# Structural constraints
MAX_HEAVY_ATOMS = 30
MIN_HEAVY_ATOMS = 5
MAX_CARBONS = 30
MAX_OXYGENS = 6

# Paths
DATA_DIR = args.data_dir
MODEL_DIR = args.model_dir
OUTPUT_DIR = args.output_dir
ALL_EVALUATED_PATH = os.path.join(OUTPUT_DIR, "all_evaluated_molecules.csv")
PARETO_TRACKING_PATH = os.path.join(OUTPUT_DIR, "pareto_front_tracking.csv")
GENERATION_STATS_PATH = os.path.join(OUTPUT_DIR, "generation_stats.csv")
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "ga_top_molecules.png")

TAU = args.Tau

# Hypervolume reference point (below worst expected values)
HV_REF_POINT = np.array([0.0, -10.0])

print(f"=== NSGA-II WSGA Configuration ===")
print(f"Target: {TARGET}")
print(f"Biodeg filter: {USE_BIODEG_FILTER}")
print(f"Population size: {GENERATION_SIZE}")
print(f"Mutation rate: {BASE_MUTATION_RATE}")
print(f"Tournament k: {TOURNAMENT_K}")
print(f"Tau (niching): {TAU}")
print(f"Generations: {NUM_GENERATIONS}")
print(f"HV reference point: {HV_REF_POINT}")
print(f"--- Validity Thresholds ---")
print(f"MP threshold: {MP_THRESHOLD}C")
print(f"BP threshold: {BP_THRESHOLD}C")
print(f"DC threshold: {DC_THRESHOLD}")
print(f"Flash point: {MIN_FLASHPOINT}K")
print(f"SCScore max: {MAX_SCSCORE}")
print(f"Tox21 max: {MAX_TOX21}")
print(f"==============================================")


# =============================
# Mutation Configuration
# =============================

MUTATIONS = [
    'AddAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveAtom',
    'AddFragment', 'RemoveFragment', 'InsertAromatic',
    'Napthalenate', 'Glycolate', 'Esterify'
]

NewAtoms = [6, 8]
BondTypes = [rdchem.BondType.SINGLE]

fragments = [Chem.MolFromSmiles(smi) for smi in [
    'CCCC', 'CCCCC', 'CCCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC', 'CCCCCCCCCC',
    'CC(C)C', 'CC(C)CC', 'CCC(C)C', 'CC(C)(C)C', 'CC(C)CCC', 'CCC(C)CC',
    'CC(C)C(C)C', 'CCCC(C)C', 'CC(C)CCCC', 'CCC(CC)CC', 'CC(C)(C)CC',
    'CCCCC(C)C', 'CC(C)CC(C)C',
    'COC', 'CCOC', 'CCOCC', 'CCCOCC', 'CCCCOCC', 'CCOCCCC', 'CCOCCC',
    'CCCOCCC', 'CCCOCCCC', 'CCCCOCCCC',
    'COCCO', 'COCCOC', 'CCOCCOC', 'CCOCCOCC', 'COCCOCCO', 'COCCOCCOC',
    'COCCOCCOCCO', 'COCCOCCOCCOC', 'CCOCC(C)OCC', 'COCC(C)OC', 'CCOCCOCCOCC',
    'CC(C)OC', 'CC(C)OCC', 'CC(C)OC(C)C', 'CC(C)(C)OC', 'CCOC(C)C',
    'CC(C)COCC', 'CCOC(C)(C)C', 'CC(C)COC(C)C', 'CC(C)OCCC', 'CCC(C)OCC',
    'CC(=O)OC', 'CC(=O)OCC', 'CCC(=O)OC', 'CC(=O)OCCC', 'CCC(=O)OCC',
    'CCCC(=O)OC', 'CC(=O)OCCCC', 'CCC(=O)OCCC', 'CCCC(=O)OCC', 'CCCCC(=O)OC',
    'CC(=O)OC(C)C', 'CC(C)C(=O)OC', 'CC(=O)OCC(C)C',
    'CC(=O)OCCOC(=O)C', 'COC(=O)CCCC(=O)OC', 'CCOC(=O)CCC(=O)OCC',
    'COC(=O)CCC(=O)OC', 'CCOC(=O)CCCC(=O)OCC', 'COC(=O)CC(=O)OC',
    'CCOC(=O)CC(=O)OCC', 'COC(=O)CCCCC(=O)OC',
    'C1CCCCC1', 'C1CCCC1', 'CC1CCCCC1', 'C1CCCCC1C', 'CC1CCCC1',
    'C1CCC(CC1)C', 'C1CCCCC1CC', 'CCC1CCCCC1', 'C1CCC(C)CC1', 'C1CCC(CC)CC1',
    'C1CCOCC1', 'C1CCOC1', 'C1COCCO1', 'CC1CCOCC1', 'C1COCC1',
    'C1CCOCC1C', 'CC1CCOC1', 'C1CCOC1C', 'C1CCOCCO1',
    'COC(=O)OC', 'CCOC(=O)OCC', 'COC(=O)OCC', 'CCCOC(=O)OCCC',
]]

Napthalenes = [Chem.MolFromSmiles(smi) for smi in [
    'C12=CC=CC=C2C=CC=C1', 'C1CCCC2=CC=CC=C12',
    'C1CCCC2CCCCC12', 'C1CCC2=CC=CC=C12'
]]

AromaticMolecule = Chem.MolFromSmiles('c1ccccc1')



# =============================
# Properties to Track
# =============================

TRACKED_PROPERTIES = [
    'pareto_rank', 'crowding_distance', 'AvgTanimotoSimilarity', 'is_valid',
    'Density_40C_g_cm^3', 'Kinematic_Viscosity_40C', 'Thermal_Conductivity_40C',
    'Heat_Capacity_Constant_Pressure_40C_J_K_Mol',
    'Density_100C_g_cm^3', 'Kinematic_Viscosity_100C', 'Thermal_Conductivity_100C',
    'Heat_Capacity_Constant_Pressure_100C_J_K_Mol',
    'alpha_40', 'alpha_100', 'alpha_avg',
    'beta_40', 'beta_100', 'beta_avg',
    'Ra_40', 'Ra_100', 'Ra_avg',
    'FOM1_40', 'FOM1_100', 'FOM1_avg',
    'FOM1_40C_direct', 'FOM1_100C_direct', 'FOM1_direct_avg',
    'Pr_40', 'Pr_100', 'Pr_avg',
    'Gr_40', 'Gr_100', 'Gr_avg',
    'Nu_40', 'Nu_100', 'Nu_avg',
    'SCScore', 'Tox21_Score', 'Biodegradable',
    'MolPrice',
    'MP-Measured', 'BP-Measured', 'DC_exp', 'flashpoint'
]


# =============================
# Helper Functions
# =============================

def nsga2_select(population_df, n_select, target, target_config, tau):
    """
    NSGA-II selection: non-dominated sort + crowding distance.

    1. Compute 2 objectives (FOM1, -MolPrice)
    2. Compute Tanimoto similarities via apply_niching (for 3rd crowding dim)
    3. Non-dominated sort on 2 objectives
    4. Crowding distance on 3 dimensions per front
    5. Fill n_select slots iterating through fronts

    Returns:
        Selected DataFrame with pareto_rank and crowding_distance columns.
    """
    # Compute Tanimoto similarities (similarity only, no fitness penalty)
    population_df = compute_tanimoto_similarities(population_df)

    # Compute 2 objectives
    objectives_2d = compute_nsga2_objectives(population_df, target, target_config)

    # Non-dominated sort
    fronts = fast_non_dominated_sort(objectives_2d)

    # Build 3D objectives for crowding distance
    avg_sim = population_df["AvgTanimotoSimilarity"].values
    objectives_3d = np.column_stack([objectives_2d, -avg_sim])

    # Assign ranks and crowding distances
    pareto_ranks = np.full(len(population_df), -1, dtype=int)
    crowding_dists = np.zeros(len(population_df))

    selected_indices = []
    for rank, front in enumerate(fronts):
        for idx in front:
            pareto_ranks[idx] = rank

        cd = crowding_distance_3d(objectives_3d, front)
        for idx, dist in cd.items():
            crowding_dists[idx] = dist

        if len(selected_indices) + len(front) <= n_select:
            # Entire front fits
            selected_indices.extend(front)
        else:
            # Partial front — sort by crowding distance descending
            remaining_slots = n_select - len(selected_indices)
            sorted_front = sorted(front, key=lambda i: crowding_dists[i], reverse=True)
            selected_indices.extend(sorted_front[:remaining_slots])
            break

    population_df["pareto_rank"] = pareto_ranks
    population_df["crowding_distance"] = crowding_dists

    selected_df = population_df.iloc[selected_indices].reset_index(drop=True)
    return selected_df


def get_pareto_front(df, target, target_config):
    """Extract Pareto-optimal molecules (rank 0) from a population."""
    objectives = compute_nsga2_objectives(df, target, target_config)
    fronts = fast_non_dominated_sort(objectives)
    if fronts:
        return df.iloc[fronts[0]].copy()
    return df.head(0)


def track_pareto_front(pareto_df, generation, tracking_data, target, target_config):
    """Record the Pareto front for this generation."""
    target_col = target_config[target]["column"]
    for rank_on_front, (_, row) in enumerate(pareto_df.iterrows()):
        record = {
            'generation': generation,
            'front_rank': rank_on_front,
            'SMILES': row['SMILES'],
        }
        for prop in TRACKED_PROPERTIES:
            record[prop] = row.get(prop, np.nan)
        # Also store the objective values explicitly
        record['obj_fom1'] = row.get(target_col, np.nan)
        record['obj_molprice'] = row.get('MolPrice', np.nan)
        tracking_data.append(record)


def compute_generation_stats(pop_df, generation, hypervolume, target, target_config):
    """Compute per-generation summary statistics."""
    target_col = target_config[target]["column"]
    stats = {
        'generation': generation,
        'hypervolume': hypervolume,
        'n_pareto_front': int((pop_df.get('pareto_rank', pd.Series()) == 0).sum()),
        'n_valid': int(pop_df['is_valid'].sum()),
        'pct_valid': 100 * pop_df['is_valid'].mean(),
    }

    for prop in [target_col, 'MolPrice', 'AvgTanimotoSimilarity']:
        if prop in pop_df.columns:
            vals = pop_df[prop].dropna()
            if len(vals) > 0:
                stats[f'{prop}_mean'] = vals.mean()
                stats[f'{prop}_std'] = vals.std()
                stats[f'{prop}_min'] = vals.min()
                stats[f'{prop}_max'] = vals.max()

    return stats


def save_molecule_grid_image(df, filepath, gen_num=0):
    """Save a grid image of Pareto front molecules."""
    if "SMILES" not in df.columns or df.empty:
        return

    target_col = TARGET_CONFIG[TARGET]["column"]
    df = df.sort_values(by=target_col, ascending=False)

    mols, legends = [], []
    for _, row in df.head(40).iterrows():
        smi = row["SMILES"]
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol:
            mols.append(mol)
            short_smi = smi if len(smi) <= 35 else smi[:32] + "..."
            fom1_val = row.get(target_col, float('nan'))
            mp_val = row.get('MolPrice', float('nan'))
            legend = f"{short_smi}\nFOM1: {fom1_val:.2f} | MP: {mp_val:.2f}"
            legends.append(legend)

    if not mols:
        return

    img = Draw.MolsToGridImage(
        mols, molsPerRow=5, subImgSize=(300, 300),
        legends=legends, useSVG=False, returnPNG=False,
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    base, ext = os.path.splitext(filepath)
    img.save(f"{base}_gen{gen_num}.png")
    print(f"Saved molecule grid: {base}_gen{gen_num}.png")


# =============================
# Main Function
# =============================

def main():
    RDLogger.DisableLog('rdApp.error')

    tracking_data = []
    generation_stats = []

    # ----- Load Models -----
    print("\nLoading models...")

    thermo_targets = [
        "BP-Measured", "DC_exp", "Density_100C_g_cm^3", "Density_40C_g_cm^3",
        "flashpoint", "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",
        "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",
        "Kinematic_Viscosity_40C", "Kinematic_Viscosity_100C",
        "MP-Measured", "Thermal_Conductivity_100C", "Thermal_Conductivity_40C"
    ]
    thermo_models = load_regression_models_with_aux(thermo_targets, MODEL_DIR)

    sc_model = SCScorer()
    sc_model.restore(os.path.join(MODEL_DIR, "SCScorer/scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"))

    tox21_dir = os.path.join(MODEL_DIR, "tox21_gt4sd")
    tox21_models = load_tox21_predictor(tox21_dir)

    biodeg_dir = os.path.join(MODEL_DIR, "biodegradability")
    biodeg_model = load_biodeg_model(biodeg_dir)

    # MolPrice — always loaded (it's an objective, not a penalty)
    from molprice import MolPriceModel
    _mp_path = os.path.join(MODEL_DIR, "MolPrice", "MP_Morgan_hybrid.pkl")
    molprice_model = None
    if os.path.exists(_mp_path):
        molprice_model = MolPriceModel(_mp_path)
        print(f"Loaded MolPrice model from {_mp_path}")
    else:
        print(f"WARNING: MolPrice model not found at {_mp_path}")

    # FOM1 direct models
    fom1_direct_models = None
    try:
        fom1_direct_models = load_fom1_direct_models(args.fom1_model_dir)
    except FileNotFoundError:
        if TARGET.startswith("FOM1_direct"):
            raise
        print("FOM1 direct models not found — columns will not be computed")

    print("Models loaded successfully.\n")

    # ----- Initialize Population -----
    print("Generating initial population...")
    training_smiles = load_combined_training_data(DATA_DIR)
    print(f"Combined training set: {len(training_smiles)} unique C/O molecules")

    df = generate_initial_population(
        n=INITIAL_POPULATION_SIZE,
        max_heavy_atoms=MAX_HEAVY_ATOMS,
        min_heavy_atoms=MIN_HEAVY_ATOMS,
        max_carbons=MAX_CARBONS,
        max_oxygens=MAX_OXYGENS,
        training_smiles=training_smiles,
    )

    df = df[["SMILES"]].copy()
    df["SMILES"] = df["SMILES"].apply(strict_canonicalize_smiles)
    df = df.drop_duplicates().reset_index(drop=True)
    seen_smiles = set(df["SMILES"])

    # ----- Mutate Initial Population -----
    df, seen_smiles = apply_mutations_to_population(
        df=df, MUTATIONS=MUTATIONS, MUTATION_RATE=BASE_MUTATION_RATE,
        NewAtoms=NewAtoms, BondTypes=BondTypes, fragments=fragments,
        AromaticMolecule=AromaticMolecule, Napthalenes=Napthalenes,
        MIN_HEAVY_ATOMS=MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS=MAX_HEAVY_ATOMS,
        MAX_CARBONS=MAX_CARBONS, MAX_OXYGENS=MAX_OXYGENS,
        seen_smiles=seen_smiles
    )

    # ----- Evaluate Initial Population -----
    evaluated_df = evaluate_molecules(
        df, thermo_models=thermo_models, sc_model=sc_model,
        tox21_models=tox21_models, biodeg_model=biodeg_model,
        fom1_direct_models=fom1_direct_models, molprice_model=molprice_model
    )

    evaluated_df = assign_validity(
        evaluated_df,
        sc_threshold=MAX_SCSCORE, mp_max=MP_THRESHOLD, bp_min=BP_THRESHOLD,
        dc_max=DC_THRESHOLD, min_fp=MIN_FLASHPOINT, use_biodeg=USE_BIODEG_FILTER,
        max_tox21=MAX_TOX21,
    )

    # ----- NSGA-II Selection on Initial Population -----
    population_df = nsga2_select(evaluated_df, GENERATION_SIZE, TARGET, TARGET_CONFIG, TAU)

    print(f"Initial population: {len(population_df)} molecules "
          f"({population_df['is_valid'].sum()} valid)")

    # ----- Setup Output Files -----
    evaluated_df["CanonicalSMILES"] = evaluated_df["SMILES"].apply(strict_canonicalize_smiles)
    evaluated_df["generation"] = 0
    seen_smiles.update(evaluated_df["CanonicalSMILES"])

    if os.path.exists(ALL_EVALUATED_PATH):
        os.remove(ALL_EVALUATED_PATH)
    os.makedirs(os.path.dirname(ALL_EVALUATED_PATH), exist_ok=True)
    CSV_COLUMNS = list(evaluated_df.columns)
    evaluated_df[CSV_COLUMNS].to_csv(ALL_EVALUATED_PATH, mode='w', header=True, index=False)

    # ----- Track Generation 0 -----
    pareto_front_df = get_pareto_front(population_df, TARGET, TARGET_CONFIG)
    objectives_2d = compute_nsga2_objectives(pareto_front_df, TARGET, TARGET_CONFIG)
    hv = compute_hypervolume_2d(objectives_2d, HV_REF_POINT)

    track_pareto_front(pareto_front_df, 0, tracking_data, TARGET, TARGET_CONFIG)
    generation_stats.append(
        compute_generation_stats(population_df, 0, hv, TARGET, TARGET_CONFIG)
    )

    print(f"\nGeneration 0:")
    print(f"  Pareto front size: {len(pareto_front_df)}")
    print(f"  Hypervolume: {hv:.4f}")
    print(f"  Valid: {population_df['is_valid'].sum()}/{len(population_df)}")

    # =============================
    # Stagnation Tracking
    # =============================
    hv_history = [hv]
    generations_since_improvement = 0
    last_best_hv = hv
    total_restarts = 0
    current_mutation_rate = BASE_MUTATION_RATE

    # =============================
    # Genetic Algorithm Loop
    # =============================

    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n{'='*50}")
        print(f"Generation {gen}/{NUM_GENERATIONS}")
        print(f"Mutation rate: {current_mutation_rate:.3f} | Restarts: {total_restarts}")
        print(f"{'='*50}")

        # ----- Generate Offspring via Tournament + Crossover -----
        pareto_ranks = population_df["pareto_rank"].values
        crowding_dists = population_df["crowding_distance"].values
        required_offspring = GENERATION_SIZE
        new_population_smiles = []

        with tqdm(total=required_offspring, desc="Crossover") as pbar:
            while len(new_population_smiles) < required_offspring:
                idx1 = nsga2_tournament(pareto_ranks, crowding_dists, k=TOURNAMENT_K)
                idx2 = nsga2_tournament(pareto_ranks, crowding_dists, k=TOURNAMENT_K)
                p1 = population_df.iloc[idx1]["SMILES"]
                p2 = population_df.iloc[idx2]["SMILES"]

                child = crossover_mol_fragments(p1, p2, max_heavy_atoms=MAX_HEAVY_ATOMS)
                if child is None:
                    child = crossover_fragments(
                        p1, p2,
                        lambda smi, side: prepare_fragments(smi, side, limit_=FRAGMENT_LIMIT),
                        max_heavy_atoms=MAX_HEAVY_ATOMS
                    )
                if not child:
                    continue

                mol = Chem.MolFromSmiles(child)
                if not mol or not (MIN_HEAVY_ATOMS <= mol.GetNumHeavyAtoms() <= MAX_HEAVY_ATOMS):
                    continue

                canonical_child = strict_canonicalize_smiles(child)
                if canonical_child not in seen_smiles:
                    new_population_smiles.append(canonical_child)
                    pbar.update(1)

        offspring_df = pd.DataFrame({'SMILES': new_population_smiles})

        # ----- Mutate Offspring -----
        mutated_offspring_df, _ = apply_mutations_to_population(
            df=offspring_df, MUTATIONS=MUTATIONS, MUTATION_RATE=current_mutation_rate,
            NewAtoms=NewAtoms, BondTypes=BondTypes, fragments=fragments,
            AromaticMolecule=AromaticMolecule, Napthalenes=Napthalenes,
            MIN_HEAVY_ATOMS=MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS=MAX_HEAVY_ATOMS,
            MAX_CARBONS=MAX_CARBONS, MAX_OXYGENS=MAX_OXYGENS,
            seen_smiles=seen_smiles
        )

        # ----- Evaluate Offspring -----
        evaluated_offspring_df = evaluate_molecules(
            mutated_offspring_df, thermo_models=thermo_models, sc_model=sc_model,
            tox21_models=tox21_models, biodeg_model=biodeg_model,
            fom1_direct_models=fom1_direct_models, molprice_model=molprice_model
        )

        evaluated_offspring_df = assign_validity(
            evaluated_offspring_df,
            sc_threshold=MAX_SCSCORE, mp_max=MP_THRESHOLD, bp_min=BP_THRESHOLD,
            dc_max=DC_THRESHOLD, min_fp=MIN_FLASHPOINT, use_biodeg=USE_BIODEG_FILTER,
            max_tox21=MAX_TOX21,
        )

        # Update seen SMILES
        evaluated_offspring_df["CanonicalSMILES"] = evaluated_offspring_df["SMILES"].apply(strict_canonicalize_smiles)
        seen_smiles.update(evaluated_offspring_df["CanonicalSMILES"])

        # ----- Combine Parents + Offspring (2N pool) -----
        # Drop NSGA-II columns from parents before combining
        parent_cols_to_drop = ['pareto_rank', 'crowding_distance', 'NichedFitnessScore']
        pop_clean = population_df.drop(columns=[c for c in parent_cols_to_drop if c in population_df.columns], errors='ignore')
        combined_df = pd.concat([pop_clean, evaluated_offspring_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset='SMILES', keep='first')

        # ----- NSGA-II Selection (N from 2N) -----
        population_df = nsga2_select(combined_df, GENERATION_SIZE, TARGET, TARGET_CONFIG, TAU)

        # ----- Append offspring to CSV -----
        evaluated_offspring_df["generation"] = gen
        for col in CSV_COLUMNS:
            if col not in evaluated_offspring_df.columns:
                evaluated_offspring_df[col] = np.nan
        evaluated_offspring_df[CSV_COLUMNS].to_csv(ALL_EVALUATED_PATH, mode='a', header=False, index=False)

        # ----- Track Pareto Front -----
        pareto_front_df = get_pareto_front(population_df, TARGET, TARGET_CONFIG)
        objectives_2d = compute_nsga2_objectives(pareto_front_df, TARGET, TARGET_CONFIG)
        hv = compute_hypervolume_2d(objectives_2d, HV_REF_POINT)

        track_pareto_front(pareto_front_df, gen, tracking_data, TARGET, TARGET_CONFIG)
        generation_stats.append(
            compute_generation_stats(population_df, gen, hv, TARGET, TARGET_CONFIG)
        )

        # ----- Progress Report -----
        target_col = TARGET_CONFIG[TARGET]["column"]
        print(f"\n  Pareto front size: {len(pareto_front_df)}")
        print(f"  Hypervolume: {hv:.4f}")
        print(f"  Valid: {population_df['is_valid'].sum()}/{len(population_df)}")
        if target_col in population_df.columns:
            valid_pop = population_df[population_df['is_valid'] == 1]
            if len(valid_pop) > 0:
                print(f"  Best {target_col}: {valid_pop[target_col].max():.4f}")
        if 'MolPrice' in population_df.columns:
            valid_pop = population_df[population_df['is_valid'] == 1]
            if len(valid_pop) > 0 and not valid_pop['MolPrice'].isna().all():
                print(f"  MolPrice range: [{valid_pop['MolPrice'].min():.2f}, {valid_pop['MolPrice'].max():.2f}]")
        if 'AvgTanimotoSimilarity' in population_df.columns:
            print(f"  Mean Tanimoto: {population_df['AvgTanimotoSimilarity'].mean():.4f}")

        # =============================
        # Stagnation Detection (Hypervolume-based)
        # =============================
        hv_history.append(hv)

        # Relative improvement
        rel_improvement = (hv - last_best_hv) / max(abs(last_best_hv), 1e-10)
        if rel_improvement > STAGNATION_THRESHOLD:
            generations_since_improvement = 0
            last_best_hv = hv
            current_mutation_rate = BASE_MUTATION_RATE
            print(f"  Improvement! HV: {hv:.4f} (+{rel_improvement*100:.2f}%)")
        else:
            generations_since_improvement += 1
            print(f"  No improvement for {generations_since_improvement} generations")

            current_mutation_rate = min(
                MAX_MUTATION_RATE,
                BASE_MUTATION_RATE + (generations_since_improvement * MUTATION_BOOST_FACTOR)
            )

            if generations_since_improvement >= STAGNATION_WINDOW:
                if total_restarts < MAX_STAGNATION_RESTARTS:
                    total_restarts += 1
                    print(f"\n{'!'*50}")
                    print(f"STAGNATION DETECTED - Restart #{total_restarts}")
                    print(f"{'!'*50}")

                    n_fresh = int(GENERATION_SIZE * STAGNATION_RESTART_RATIO)
                    n_keep = GENERATION_SIZE - n_fresh

                    # Keep top by pareto rank then crowding distance
                    kept_df = population_df.sort_values(
                        ['pareto_rank', 'crowding_distance'],
                        ascending=[True, False]
                    ).head(n_keep)

                    # Generate fresh molecules
                    print(f"Generating {n_fresh} fresh molecules...")
                    fresh_df = generate_initial_population(
                        n=n_fresh * 3, training_smiles=training_smiles,
                        min_heavy_atoms=MIN_HEAVY_ATOMS, max_heavy_atoms=MAX_HEAVY_ATOMS,
                        max_carbons=MAX_CARBONS, max_oxygens=MAX_OXYGENS
                    )

                    fresh_smiles = [s for s in fresh_df['SMILES']
                                    if strict_canonicalize_smiles(s) not in seen_smiles][:n_fresh]

                    if fresh_smiles:
                        fresh_df = pd.DataFrame({'SMILES': fresh_smiles})
                        fresh_evaluated = evaluate_molecules(
                            fresh_df, thermo_models=thermo_models, sc_model=sc_model,
                            tox21_models=tox21_models, biodeg_model=biodeg_model,
                            fom1_direct_models=fom1_direct_models, molprice_model=molprice_model
                        )
                        fresh_evaluated = assign_validity(
                            fresh_evaluated,
                            sc_threshold=MAX_SCSCORE, mp_max=MP_THRESHOLD, bp_min=BP_THRESHOLD,
                            dc_max=DC_THRESHOLD, min_fp=MIN_FLASHPOINT, use_biodeg=USE_BIODEG_FILTER,
                            max_tox21=MAX_TOX21,
                        )

                        fresh_evaluated["CanonicalSMILES"] = fresh_evaluated["SMILES"].apply(strict_canonicalize_smiles)
                        seen_smiles.update(fresh_evaluated["CanonicalSMILES"])

                        fresh_evaluated["generation"] = gen
                        for col in CSV_COLUMNS:
                            if col not in fresh_evaluated.columns:
                                fresh_evaluated[col] = np.nan
                        fresh_evaluated[CSV_COLUMNS].to_csv(ALL_EVALUATED_PATH, mode='a', header=False, index=False)

                        # Drop NSGA-II columns from kept before combining
                        kept_clean = kept_df.drop(
                            columns=[c for c in parent_cols_to_drop if c in kept_df.columns],
                            errors='ignore'
                        )
                        combined_restart = pd.concat([kept_clean, fresh_evaluated], ignore_index=True)
                        population_df = nsga2_select(combined_restart, GENERATION_SIZE, TARGET, TARGET_CONFIG, TAU)

                        print(f"Restart complete: kept {n_keep}, added {len(fresh_evaluated)} fresh")
                    else:
                        print("Warning: Could not generate fresh molecules")

                    generations_since_improvement = 0
                    current_mutation_rate = MAX_MUTATION_RATE
                else:
                    print(f"Max restarts ({MAX_STAGNATION_RESTARTS}) reached")

        # ----- Periodic Save -----
        if gen % 1 == 0:
            pd.DataFrame(tracking_data).to_csv(PARETO_TRACKING_PATH, index=False)
            pd.DataFrame(generation_stats).to_csv(GENERATION_STATS_PATH, index=False)

    # =============================
    # Save Final Results
    # =============================
    pd.DataFrame(tracking_data).to_csv(PARETO_TRACKING_PATH, index=False)
    pd.DataFrame(generation_stats).to_csv(GENERATION_STATS_PATH, index=False)

    # Save final Pareto front
    pareto_front_df = get_pareto_front(population_df, TARGET, TARGET_CONFIG)
    pareto_path = os.path.join(OUTPUT_DIR, "final_pareto_front.csv")
    pareto_front_df.to_csv(pareto_path, index=False)

    # Save final top molecules (by FOM1)
    target_col = TARGET_CONFIG[TARGET]["column"]
    top_25 = population_df.sort_values(target_col, ascending=False).head(25)
    top_path = os.path.join(OUTPUT_DIR, f"top_25_molecules_{TARGET}.csv")
    top_25.to_csv(top_path, index=False)

    print(f"\n{'='*50}")
    print(f"NSGA-II WSGA COMPLETE")
    print(f"{'='*50}")
    print(f"Final Pareto front: {pareto_path} ({len(pareto_front_df)} molecules)")
    print(f"Final hypervolume: {hv:.4f}")
    print(f"All evaluated: {ALL_EVALUATED_PATH}")
    print(f"Pareto tracking: {PARETO_TRACKING_PATH}")
    print(f"Generation stats: {GENERATION_STATS_PATH}")


if __name__ == "__main__":
    main()
