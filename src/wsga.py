"""
WSGA - Weighted Sum Genetic Algorithm for Cooling Fluid Discovery

Uses combined n-gram model trained on all available datasets for diverse
initial population generation. Features soft MP penalty, hybrid elite
selection, stagnation detection with adaptive restart, and size-dependent
mutation weighting.

Optimizes molecular structures for thermophysical properties relevant to
cooling fluids (thermal conductivity, heat capacity, viscosity, etc.)
"""

import os
import sys
import pandas as pd
import pickle
import random
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdchem
from tqdm import tqdm
import numpy as np
import argparse

from wsga_helper import (
    apply_mutations_to_population,
    evaluate_molecules,
    k_way_tournament,
    load_tox21_models,
    load_biodeg_model,
    apply_niching,
    assign_validity,
    compute_fitness,
    apply_mp_penalty,
    load_regression_models_with_aux
)
from evaluation import get_scscore_cached, strict_canonicalize_smiles
from fragment_utils import prepare_fragments, crossover_fragments
from SCScorer import SCScorer
from generate_molecules import generate_initial_population, load_combined_training_data


# =============================
# Logging Setup
# =============================

jobid_raw = os.environ.get("PBS_JOBID", "local")
# Extract base job ID: "1632441[0].pbs" -> "1632441"
jobid = jobid_raw.split(".")[0].split("[")[0]
array_idx = os.environ.get("PBS_ARRAY_INDEX", "0")

# Create logs directory
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", jobid)
os.makedirs(logs_dir, exist_ok=True)

# Open log file with array index
log_path = os.path.join(logs_dir, f"wsga_{array_idx}.log")
log = open(log_path, "a", buffering=1)
sys.stdout = log
sys.stderr = log

print(f"Logging to: {log_path}")


# =============================
# Command-line Arguments
# =============================

parser = argparse.ArgumentParser(description="Run WSGA Genetic Algorithm")
parser.add_argument("--elitism_rate", type=float, default=0.4, help="Elitism rate")
parser.add_argument("--population_size", type=int, default=1000, help="Population size")
parser.add_argument("--mutation_rate", type=float, default=0.8, help="Mutation rate")
parser.add_argument("--output_dir", type=str, default="../outputs", help="Output directory")
parser.add_argument("--data_dir", type=str, default="../data", help="Data directory containing all CSV files")
parser.add_argument("--model_dir", type=str, default="../models", help="Model directory")
parser.add_argument("--Tau", type=float, default=0.05, help="Threshold for niching")
parser.add_argument("--target", type=str, default="FOM1",
    choices=["alpha", "beta", "Pr", "Gr", "Ra", "Nu", "FOM1", "FOM1_40", "FOM1_100"],
    help="Target property to optimize"
)
parser.add_argument("--top_n", type=int, default=40, help="Number of top molecules to track and visualize")
parser.add_argument("--num_generations", type=int, default=200, help="Number of generations to run")

# Validity threshold arguments (for case studies)
parser.add_argument("--mp_soft", type=float, default=-30, help="Soft MP threshold - no penalty below this (°C)")
parser.add_argument("--mp_hard", type=float, default=-10, help="Hard MP threshold - zero fitness at/above this (°C)")
parser.add_argument("--bp_threshold", type=float, default=70, help="Min boiling point (°C)")
parser.add_argument("--dc_threshold", type=float, default=8, help="Max dielectric constant")
parser.add_argument("--fp_threshold", type=float, default=373, help="Min flash point (K)")
parser.add_argument("--sc_threshold", type=float, default=3, help="Max SCScore")
parser.add_argument("--tox_threshold", type=float, default=3, help="Max Tox21 score")
parser.add_argument("--no_biodeg", action="store_true", help="Disable biodegradability filter")

args = parser.parse_args()


# =============================
# Configuration
# =============================

# GA Parameters
GENERATION_SIZE = args.population_size
INITIAL_POPULATION_SIZE = 4 * GENERATION_SIZE
ELITISM_RATE = args.elitism_rate
ELITE_COUNT = int(GENERATION_SIZE * ELITISM_RATE)
BASE_MUTATION_RATE = args.mutation_rate  # Base mutation rate
NUM_GENERATIONS = args.num_generations
FRAGMENT_LIMIT = 2000

# Elite Selection Split
# - BEST_ELITE_RATIO of elites are selected by raw FitnessScore (guaranteed best performers)
# - Remaining (1 - BEST_ELITE_RATIO) are selected by NichedFitnessScore (diversity)
BEST_ELITE_RATIO = 0.3  # 30% by raw fitness, 70% by niched fitness
BEST_ELITE_COUNT = int(ELITE_COUNT * BEST_ELITE_RATIO)
DIVERSE_ELITE_COUNT = ELITE_COUNT - BEST_ELITE_COUNT

# Stagnation Detection & Restart Configuration
STAGNATION_WINDOW = 5           # Number of generations to look back for improvement
STAGNATION_THRESHOLD = 0.01    # Minimum improvement required (in FOM1 units)
STAGNATION_RESTART_RATIO = 0.3  # Replace 30% of elite with fresh molecules on restart
MAX_STAGNATION_RESTARTS = 5     # Maximum number of restarts before giving up

# Adaptive Mutation Rate Configuration
MIN_MUTATION_RATE = 0.3         # Minimum mutation rate
MAX_MUTATION_RATE = 0.95        # Maximum mutation rate during stagnation boost
MUTATION_BOOST_FACTOR = 0.15    # How much to increase mutation rate per stagnation generation

# Tracking
TOP_N = args.top_n  # Number of top molecules to track per generation

# Fitness Target
TARGET = args.target
TARGET_CONFIG = {
    "alpha": {"column": "alpha_avg", "maximize": True},
    "beta":  {"column": "beta_avg",  "maximize": True},
    "Pr":    {"column": "Pr_avg",    "maximize": False},
    "Gr":    {"column": "Gr_avg",    "maximize": True},
    "Ra":    {"column": "Ra_avg",    "maximize": True},
    "Nu":    {"column": "Nu_avg",    "maximize": True},
    "FOM1":  {"column": "FOM1_avg",  "maximize": True},
    "FOM1_40":  {"column": "FOM1_40",  "maximize": True},
    "FOM1_100": {"column": "FOM1_100", "maximize": True},
}

# Selection Criteria (validity thresholds) - from args for case studies
# MP now uses soft penalty instead of hard threshold
MP_SOFT = args.mp_soft            # Soft threshold - no penalty below this (°C)
MP_HARD = args.mp_hard            # Hard threshold - zero fitness at/above this (°C)
BP_THRESHOLD = args.bp_threshold      # Min boiling point (°C)
DC_THRESHOLD = args.dc_threshold      # Max dielectric constant
MIN_FLASHPOINT = args.fp_threshold    # Min flash point (K)
MAX_SCSCORE = args.sc_threshold       # Max synthetic complexity
MAX_TOX21 = args.tox_threshold        # Max toxicity score
USE_BIODEG_FILTER = not args.no_biodeg

# Structural Constraints
MAX_HEAVY_ATOMS = 30
MIN_HEAVY_ATOMS = 5
MAX_CARBONS = 30
MAX_OXYGENS = 6

# Paths
DATA_DIR = args.data_dir
DATA_PATH = "../data/processed_full_hydrocarbon_dataset.csv"  # Kept for backwards compatibility
MODEL_DIR = args.model_dir
OUTPUT_IMAGE_PATH = os.path.join(args.output_dir, "ga_top_molecules.png")
ALL_EVALUATED_PATH = os.path.join(args.output_dir, "all_evaluated_molecules.csv")
TOP_N_TRACKING_PATH = os.path.join(args.output_dir, "top_n_tracking.csv")
GENERATION_STATS_PATH = os.path.join(args.output_dir, "generation_stats.csv")

# Display
VISUALIZE = True
TAU = args.Tau

print(f"=== WSGA Configuration (Combined Dataset + Soft MP Penalty) ===")
print(f"Target: {TARGET}")
print(f"Population size: {GENERATION_SIZE}")
print(f"Elite count: {ELITE_COUNT}")
print(f"  - Best by FitnessScore: {BEST_ELITE_COUNT} ({BEST_ELITE_RATIO*100:.0f}%)")
print(f"  - Best by NichedFitness: {DIVERSE_ELITE_COUNT} ({(1-BEST_ELITE_RATIO)*100:.0f}%)")
print(f"Elitism rate: {ELITISM_RATE}")
print(f"Base mutation rate: {BASE_MUTATION_RATE}")
print(f"Tau (niching): {TAU}")
print(f"Top N tracking: {TOP_N}")
print(f"Model directory: {MODEL_DIR}")
print(f"--- MP Soft Penalty ---")
print(f"Soft threshold: {MP_SOFT}°C (P=1.0 below)")
print(f"Hard threshold: {MP_HARD}°C (P=0.0 at/above)")
print(f"--- Stagnation Settings ---")
print(f"Stagnation window: {STAGNATION_WINDOW} generations")
print(f"Stagnation threshold: {STAGNATION_THRESHOLD}")
print(f"Restart ratio: {STAGNATION_RESTART_RATIO*100:.0f}%")
print(f"Max restarts: {MAX_STAGNATION_RESTARTS}")
print(f"--- Adaptive Mutation ---")
print(f"Min mutation rate: {MIN_MUTATION_RATE}")
print(f"Max mutation rate: {MAX_MUTATION_RATE}")
print(f"Boost factor: {MUTATION_BOOST_FACTOR}")
print(f"--- Other Validity Thresholds (Hard) ---")
print(f"BP threshold: {BP_THRESHOLD}°C")
print(f"DC threshold: {DC_THRESHOLD}")
print(f"Flash point: {MIN_FLASHPOINT}K")
print(f"SCScore max: {MAX_SCSCORE}")
print(f"Tox21 max: {MAX_TOX21}")
print(f"Biodeg filter: {USE_BIODEG_FILTER}")
print(f"==============================================")


# =============================
# Mutation Configuration
# =============================

MUTATIONS = [
    'AddAtom', 
    'ReplaceAtom',
    'ReplaceBond',
    'RemoveAtom',
    'AddFragment',
    'RemoveFragment',
    'InsertAromatic',
    'Napthalenate',
    'Glycolate',
    'Esterify'
]

# Allowed atoms (C=6, O=8)
NewAtoms = [6, 8]
BondTypes = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE]

# Fragment libraries for mutations
fragments = [Chem.MolFromSmiles(smi) for smi in [
    # Linear alkyl
    'CCCC', 'CCCCC', 'CCCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC', 'CCCCCCCCCC',
    # Branched alkyl  
    'CC(C)C', 'CC(C)CC', 'CCC(C)C', 'CC(C)(C)C', 'CC(C)CCC', 'CCC(C)CC',
    'CC(C)C(C)C', 'CCCC(C)C', 'CC(C)CCCC', 'CCC(CC)CC', 'CC(C)(C)CC',
    'CCCCC(C)C', 'CC(C)CC(C)C',
    # Simple ethers
    'COC', 'CCOC', 'CCOCC', 'CCCOCC', 'CCCCOCC', 'CCOCCCC', 'CCOCCC',
    'CCCOCCC', 'CCCOCCCC', 'CCCCOCCCC',
    # Glycol ethers
    'COCCO', 'COCCOC', 'CCOCCOC', 'CCOCCOCC', 'COCCOCCO', 'COCCOCCOC',
    'COCCOCCOCCO', 'COCCOCCOCCOC', 'CCOCC(C)OCC', 'COCC(C)OC', 'CCOCCOCCOCC',
    # Branched ethers
    'CC(C)OC', 'CC(C)OCC', 'CC(C)OC(C)C', 'CC(C)(C)OC', 'CCOC(C)C',
    'CC(C)COCC', 'CCOC(C)(C)C', 'CC(C)COC(C)C', 'CC(C)OCCC', 'CCC(C)OCC',
    # Esters
    'CC(=O)OC', 'CC(=O)OCC', 'CCC(=O)OC', 'CC(=O)OCCC', 'CCC(=O)OCC',
    'CCCC(=O)OC', 'CC(=O)OCCCC', 'CCC(=O)OCCC', 'CCCC(=O)OCC', 'CCCCC(=O)OC',
    'CC(=O)OC(C)C', 'CC(C)C(=O)OC', 'CC(=O)OCC(C)C',
    # Diesters
    'CC(=O)OCCOC(=O)C', 'COC(=O)CCCC(=O)OC', 'CCOC(=O)CCC(=O)OCC',
    'COC(=O)CCC(=O)OC', 'CCOC(=O)CCCC(=O)OCC', 'COC(=O)CC(=O)OC',
    'CCOC(=O)CC(=O)OCC', 'COC(=O)CCCCC(=O)OC',
    # Cyclic alkanes
    'C1CCCCC1', 'C1CCCC1', 'CC1CCCCC1', 'C1CCCCC1C', 'CC1CCCC1',
    'C1CCC(CC1)C', 'C1CCCCC1CC', 'CCC1CCCCC1', 'C1CCC(C)CC1', 'C1CCC(CC)CC1',
    # Cyclic ethers
    'C1CCOCC1', 'C1CCOC1', 'C1COCCO1', 'CC1CCOCC1', 'C1COCC1',
    'C1CCOCC1C', 'CC1CCOC1', 'C1CCOC1C', 'C1CCOCCO1',
    # Carbonates
    'COC(=O)OC', 'CCOC(=O)OCC', 'COC(=O)OCC', 'CCCOC(=O)OCCC',
    # Unsaturated
    'C=CC', 'CC=C', 'C=CCC', 'CC=CC', 'C=CCCC', 'CC=CCC', 'CCC=CC', 'C=CC=C'
]]

Napthalenes = [Chem.MolFromSmiles(smi) for smi in [
    'C12=CC=CC=C2C=CC=C1', 'C1CCCC2=CC=CC=C12', 
    'C1CCCC2CCCCC12', 'C1CCC2=CC=CC=C12'
]]

AromaticMolecule = Chem.MolFromSmiles('c1ccccc1')


# =============================
# Properties to Track
# =============================

# These are the columns we'll track for top N molecules each generation
TRACKED_PROPERTIES = [
    # Fitness
    'FitnessScore', 'NichedFitnessScore', 'AvgTanimotoSimilarity', 'is_valid',
    'MP_Penalty',  # NEW: MP soft penalty factor
    # Thermophysical - 40C
    'Density_40C_g_cm^3', 'Kinematic_Viscosity_40C', 'Thermal_Conductivity_40C',
    'Heat_Capacity_Constant_Pressure_40C_J_K_Mol',
    # Thermophysical - 100C
    'Density_100C_g_cm^3', 'Kinematic_Viscosity_100C', 'Thermal_Conductivity_100C',
    'Heat_Capacity_Constant_Pressure_100C_J_K_Mol',
    # Derived properties
    'alpha_40', 'alpha_100', 'alpha_avg',
    'beta_40', 'beta_100', 'beta_avg',
    'Ra_40', 'Ra_100', 'Ra_avg',
    'FOM1_40', 'FOM1_100', 'FOM1_avg',
    'Pr_40', 'Pr_100', 'Pr_avg',
    'Gr_40', 'Gr_100', 'Gr_avg',
    'Nu_40', 'Nu_100', 'Nu_avg',
    # Safety/Synthesis
    'SCScore', 'Tox21_Score', 'Biodegradable',
    'MP-Measured', 'BP-Measured', 'DC_exp', 'flashpoint'
]


# =============================
# Helper Functions
# =============================

def save_molecule_grid_image(df, filepath, gen_num=0):
    """Save a grid image of top molecules with fitness scores."""
    if "SMILES" not in df.columns or df.empty:
        print("No SMILES data to visualize.")
        return

    df = df.sort_values(by="NichedFitnessScore", ascending=False)

    mols = []
    legends = []

    for _, row in df.iterrows():
        smi = row["SMILES"]
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol:
            mols.append(mol)
            short_smi = smi if len(smi) <= 35 else smi[:32] + "..."
            fitness = row.get("FitnessScore")
            niched = row.get("NichedFitnessScore")
            tanimoto = row.get("AvgTanimotoSimilarity")
            fitness_str = f"Fit: {fitness:.2f}" if fitness is not None else "Fit: -"
            niched_str = f"Nch: {niched:.2f}" if niched is not None else ""
            ts_str = f"Sim: {tanimoto:.3f}" if tanimoto is not None else ""
            legend = f"{short_smi}\n{fitness_str} | {niched_str}\n{ts_str}"
            legends.append(legend)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(300, 300),
        legends=legends,
        useSVG=False,
        returnPNG=False,
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    base, ext = os.path.splitext(filepath)
    output_path = f"{base}_gen{gen_num}.png"
    img.save(output_path)
    print(f"Saved molecule grid: {output_path}")


def get_top_n_molecules(elite_df, n):
    """
    Get the top N molecules from elite population.
    Returns a copy sorted by raw FitnessScore (true best performers).
    """
    return (
        elite_df
        .sort_values("FitnessScore", ascending=False)
        .head(n)
        .reset_index(drop=True)
        .copy()
    )


def track_generation(top_n_df, generation, tracking_data):
    """
    Record properties of top N molecules for this generation.
    
    Args:
        top_n_df: DataFrame of top N molecules
        generation: Current generation number
        tracking_data: List to append records to
    """
    for rank, (_, row) in enumerate(top_n_df.iterrows(), start=1):
        record = {
            'generation': generation,
            'rank': rank,
            'SMILES': row['SMILES']
        }
        # Add all tracked properties
        for prop in TRACKED_PROPERTIES:
            if prop in row:
                record[prop] = row[prop]
            else:
                record[prop] = np.nan
        
        tracking_data.append(record)


def compute_generation_stats(top_n_df, generation):
    """
    Compute summary statistics for the top N molecules.
    
    Returns a dict with mean, std, min, max for key properties.
    """
    stats = {'generation': generation}
    
    # Properties to compute stats for
    stat_properties = [
        'FitnessScore', 'NichedFitnessScore', 'AvgTanimotoSimilarity',
        'alpha_avg', 'beta_avg', 'FOM1_avg', 'Ra_avg',
        'Thermal_Conductivity_40C', 'Kinematic_Viscosity_40C',
        'SCScore', 'Tox21_Score', 'MP_Penalty', 'MP-Measured'
    ]
    
    for prop in stat_properties:
        if prop in top_n_df.columns:
            values = top_n_df[prop].dropna()
            if len(values) > 0:
                stats[f'{prop}_mean'] = values.mean()
                stats[f'{prop}_std'] = values.std()
                stats[f'{prop}_min'] = values.min()
                stats[f'{prop}_max'] = values.max()
    
    # Count valid molecules
    if 'is_valid' in top_n_df.columns:
        stats['n_valid'] = top_n_df['is_valid'].sum()
        stats['pct_valid'] = 100 * top_n_df['is_valid'].mean()
    
    # Count molecules with full MP penalty (P=1)
    if 'MP_Penalty' in top_n_df.columns:
        stats['n_full_mp_pass'] = (top_n_df['MP_Penalty'] == 1.0).sum()
        stats['n_mp_penalized'] = ((top_n_df['MP_Penalty'] > 0) & (top_n_df['MP_Penalty'] < 1)).sum()
    
    return stats


def save_tracking_data(tracking_data, generation_stats, output_dir):
    """Save tracking data and generation stats to CSV files."""
    # Save individual molecule tracking
    tracking_df = pd.DataFrame(tracking_data)
    tracking_df.to_csv(TOP_N_TRACKING_PATH, index=False)
    
    # Save generation-level stats
    stats_df = pd.DataFrame(generation_stats)
    stats_df.to_csv(GENERATION_STATS_PATH, index=False)
    
    print(f"Saved tracking data: {TOP_N_TRACKING_PATH}")
    print(f"Saved generation stats: {GENERATION_STATS_PATH}")


# =============================
# Main Function
# =============================

def main():
    RDLogger.DisableLog('rdApp.error')

    # Initialize tracking lists
    tracking_data = []      # Individual molecule tracking
    generation_stats = []   # Per-generation summary stats

    # ----- Load Models -----
    print("\nLoading models...")
    
    # Regression models (thermophysical properties)
    thermo_targets = [
        "BP-Measured", "DC_exp", "Density_100C_g_cm^3", "Density_40C_g_cm^3",
        "flashpoint", "Heat_Capacity_Constant_Pressure_100C_J_K_Mol",
        "Heat_Capacity_Constant_Pressure_40C_J_K_Mol",
        "Kinematic_Viscosity_40C", "Kinematic_Viscosity_100C",
        "MP-Measured", "Thermal_Conductivity_100C", "Thermal_Conductivity_40C"
    ]
    thermo_models = load_regression_models_with_aux(thermo_targets, MODEL_DIR)

    # SCScorer (separate pre-trained model)
    sc_model = SCScorer()
    sc_model.restore(os.path.join(MODEL_DIR, "SCScorer/scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"))

    # Tox21 models (classification)
    tox21_dir = os.path.join(MODEL_DIR, "tox21")
    tox21_models = load_tox21_models(tox21_dir)

    # Biodegradability model (classification)
    biodeg_dir = os.path.join(MODEL_DIR, "biodegradability")
    biodeg_model = load_biodeg_model(biodeg_dir)

    print("Models loaded successfully.\n")

    # ----- Initialize Population -----
    print("Generating initial population...")
    print("Loading combined training data for n-gram model...")
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
        df=df,
        MUTATIONS=MUTATIONS,
        MUTATION_RATE=BASE_MUTATION_RATE,
        NewAtoms=NewAtoms,
        BondTypes=BondTypes,
        fragments=fragments,
        AromaticMolecule=AromaticMolecule,
        Napthalenes=Napthalenes,
        MIN_HEAVY_ATOMS=MIN_HEAVY_ATOMS,
        MAX_HEAVY_ATOMS=MAX_HEAVY_ATOMS,
        MAX_CARBONS=MAX_CARBONS,
        MAX_OXYGENS=MAX_OXYGENS,
        seen_smiles=seen_smiles
    )

    # ----- Evaluate Initial Population -----
    evaluated_df = evaluate_molecules(
        df,
        thermo_models=thermo_models,
        sc_model=sc_model,
        tox21_models=tox21_models,
        biodeg_model=biodeg_model
    )

    evaluated_df = assign_validity(
        evaluated_df,
        sc_threshold=MAX_SCSCORE,
        # mp_max removed - now using soft penalty
        bp_min=BP_THRESHOLD,
        dc_max=DC_THRESHOLD,
        min_fp=MIN_FLASHPOINT,
        use_biodeg=USE_BIODEG_FILTER,
        max_tox21=MAX_TOX21
    )

    evaluated_df = compute_fitness(evaluated_df, TARGET, TARGET_CONFIG)
    evaluated_df = apply_mp_penalty(evaluated_df, soft_threshold=MP_SOFT, hard_threshold=MP_HARD)
    evaluated_df = apply_niching(evaluated_df)

    # Select initial GA population
    initial_population_df = (
        evaluated_df
        .sort_values("NichedFitnessScore", ascending=False)
        .head(GENERATION_SIZE)
        .reset_index(drop=True)
    )

    print(f"Initial population: {len(initial_population_df)} molecules "
          f"({initial_population_df['is_valid'].sum()} valid)")

    # ----- Setup Output Files -----
    evaluated_df["CanonicalSMILES"] = evaluated_df["SMILES"].apply(strict_canonicalize_smiles)
    evaluated_df["generation"] = 0
    seen_smiles.update(evaluated_df["CanonicalSMILES"])

    if os.path.exists(ALL_EVALUATED_PATH):
        os.remove(ALL_EVALUATED_PATH)
    os.makedirs(os.path.dirname(ALL_EVALUATED_PATH), exist_ok=True)
    evaluated_df.to_csv(ALL_EVALUATED_PATH, mode='w', header=True, index=False)

    # ----- Initialize Elite Population (Hybrid Selection) -----
    # Remove duplicates first
    evaluated_df = evaluated_df.sort_values('FitnessScore', ascending=False).drop_duplicates(subset='SMILES', keep='first')
    
    # Select top molecules by raw fitness (guaranteed survival)
    best_by_fitness = evaluated_df.nlargest(BEST_ELITE_COUNT, 'FitnessScore')
    
    # Select remaining by niched fitness (excluding already selected)
    remaining = evaluated_df[~evaluated_df.index.isin(best_by_fitness.index)]
    diverse_elites = remaining.nlargest(DIVERSE_ELITE_COUNT, 'NichedFitnessScore')
    
    # Combine
    elite_df = pd.concat([best_by_fitness, diverse_elites]).reset_index(drop=True)
    print(f"Elite population: {len(elite_df)} molecules ({BEST_ELITE_COUNT} by fitness, {DIVERSE_ELITE_COUNT} by diversity)")

    # ----- Track Generation 0 -----
    top_n_df = get_top_n_molecules(elite_df, TOP_N)
    track_generation(top_n_df, generation=0, tracking_data=tracking_data)
    generation_stats.append(compute_generation_stats(top_n_df, generation=0))

    if VISUALIZE:
        save_molecule_grid_image(top_n_df, OUTPUT_IMAGE_PATH, gen_num=0)

    print(f"\nGeneration 0 - Top {TOP_N} stats:")
    print(f"  Best FitnessScore: {top_n_df['FitnessScore'].max():.4f}")
    print(f"  Mean FitnessScore: {top_n_df['FitnessScore'].mean():.4f}")
    print(f"  Best NichedFitness: {top_n_df['NichedFitnessScore'].max():.4f}")
    print(f"  Valid: {top_n_df['is_valid'].sum()}/{TOP_N}")

    # =============================
    # Stagnation Tracking Variables
    # =============================
    best_fitness_history = [top_n_df['FitnessScore'].max()]  # Track best fitness per generation
    generations_since_improvement = 0
    last_best_fitness = best_fitness_history[0]
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

        # ----- Generate Offspring via Crossover -----
        required_offspring = GENERATION_SIZE - len(elite_df)
        new_population_smiles = []

        with tqdm(total=required_offspring, desc="Crossover") as pbar:
            while len(new_population_smiles) < required_offspring:
                p1 = k_way_tournament(elite_df, k=3)
                p2 = k_way_tournament(elite_df, k=3)

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

        # ----- Mutate Offspring (with adaptive mutation rate) -----
        mutated_offspring_df, _ = apply_mutations_to_population(
            df=offspring_df,
            MUTATIONS=MUTATIONS,
            MUTATION_RATE=current_mutation_rate,  # Use adaptive rate
            NewAtoms=NewAtoms,
            BondTypes=BondTypes,
            fragments=fragments,
            AromaticMolecule=AromaticMolecule,
            Napthalenes=Napthalenes,
            MIN_HEAVY_ATOMS=MIN_HEAVY_ATOMS,
            MAX_HEAVY_ATOMS=MAX_HEAVY_ATOMS,
            MAX_CARBONS=MAX_CARBONS,
            MAX_OXYGENS=MAX_OXYGENS,
            seen_smiles=seen_smiles
        )

        # ----- Evaluate Offspring -----
        evaluated_offspring_df = evaluate_molecules(
            mutated_offspring_df,
            thermo_models=thermo_models,
            sc_model=sc_model,
            tox21_models=tox21_models,
            biodeg_model=biodeg_model
        )

        evaluated_offspring_df = assign_validity(
            evaluated_offspring_df,
            sc_threshold=MAX_SCSCORE,
            # mp_max removed - now using soft penalty
            bp_min=BP_THRESHOLD,
            dc_max=DC_THRESHOLD,
            min_fp=MIN_FLASHPOINT,
            use_biodeg=USE_BIODEG_FILTER,
            max_tox21=MAX_TOX21
        )

        evaluated_offspring_df = compute_fitness(evaluated_offspring_df, TARGET, TARGET_CONFIG)
        evaluated_offspring_df = apply_mp_penalty(evaluated_offspring_df, soft_threshold=MP_SOFT, hard_threshold=MP_HARD)
        # Note: Don't apply niching here - we'll do it on the combined population

        # ----- Update Seen SMILES -----
        evaluated_offspring_df["CanonicalSMILES"] = evaluated_offspring_df["SMILES"].apply(strict_canonicalize_smiles)
        seen_smiles.update(evaluated_offspring_df["CanonicalSMILES"])

        # ----- Select New Elites -----
        # Combine elite and offspring populations
        combined_df = pd.concat([elite_df, evaluated_offspring_df], ignore_index=True)
        
        # CRITICAL: Apply niching to the COMBINED population
        # This ensures elite molecules get re-penalized if population becomes similar to them
        # and new molecules are penalized based on similarity to entire pool
        combined_df = apply_niching(combined_df, tau=TAU)
        
        # HYBRID ELITE SELECTION:
        # 1. Reserve slots for BEST molecules by raw FitnessScore (guaranteed survival of top performers)
        # 2. Fill remaining slots by NichedFitnessScore (diversity-promoting)
        
        # Remove duplicates first (keep best by FitnessScore)
        combined_df = combined_df.sort_values('FitnessScore', ascending=False).drop_duplicates(subset='SMILES', keep='first')
        
        # Select top molecules by raw fitness (guaranteed survival)
        best_by_fitness = combined_df.nlargest(BEST_ELITE_COUNT, 'FitnessScore')
        
        # Select remaining by niched fitness (excluding already selected)
        remaining = combined_df[~combined_df.index.isin(best_by_fitness.index)]
        diverse_elites = remaining.nlargest(DIVERSE_ELITE_COUNT, 'NichedFitnessScore')
        
        # Combine and reset index
        elite_df = pd.concat([best_by_fitness, diverse_elites]).reset_index(drop=True)
        
        # ----- Append to Output CSV (AFTER niching so all columns are populated) -----
        # Only save the offspring portion (they now have proper niched scores)
        offspring_with_niching = combined_df[combined_df['SMILES'].isin(evaluated_offspring_df['SMILES'])]
        offspring_with_niching["generation"] = gen
        offspring_with_niching.to_csv(ALL_EVALUATED_PATH, mode='a', header=False, index=False)

        # ----- Track Top N for This Generation -----
        top_n_df = get_top_n_molecules(elite_df, TOP_N)
        track_generation(top_n_df, generation=gen, tracking_data=tracking_data)
        generation_stats.append(compute_generation_stats(top_n_df, generation=gen))

        # ----- Progress Report -----
        valid_count = top_n_df['is_valid'].sum()
        best_fitness = top_n_df['NichedFitnessScore'].max()
        mean_fitness = top_n_df['NichedFitnessScore'].mean()
        best_raw_fitness = top_n_df['FitnessScore'].max()
        mean_raw_fitness = top_n_df['FitnessScore'].mean()
        
        # Diversity metrics
        mean_similarity = elite_df['AvgTanimotoSimilarity'].mean()
        unique_in_elite = elite_df['SMILES'].nunique()
        
        print(f"\nTop {TOP_N} stats:")
        print(f"  Best raw fitness:    {best_raw_fitness:.4f}")
        print(f"  Best niched fitness: {best_fitness:.4f}")
        print(f"  Mean raw fitness:    {mean_raw_fitness:.4f}")
        print(f"  Mean niched fitness: {mean_fitness:.4f}")
        print(f"  Valid:               {valid_count}/{TOP_N}")
        print(f"  Elite diversity:     {unique_in_elite} unique, avg_sim={mean_similarity:.4f}")
        
        if 'FOM1_avg' in top_n_df.columns:
            print(f"  Best FOM1:           {top_n_df['FOM1_avg'].max():.4f}")
            print(f"  Mean FOM1:           {top_n_df['FOM1_avg'].mean():.4f}")
        if 'alpha_avg' in top_n_df.columns:
            print(f"  Mean alpha:          {top_n_df['alpha_avg'].mean():.6f}")

        # =============================
        # Stagnation Detection & Response
        # =============================
        best_fitness_history.append(best_raw_fitness)
        
        # Check for improvement
        if best_raw_fitness > last_best_fitness + STAGNATION_THRESHOLD:
            # Improvement found - reset counters
            generations_since_improvement = 0
            last_best_fitness = best_raw_fitness
            current_mutation_rate = BASE_MUTATION_RATE  # Reset to base rate
            print(f"  ✓ Improvement! New best: {best_raw_fitness:.4f}")
        else:
            # No improvement
            generations_since_improvement += 1
            print(f"  ✗ No improvement for {generations_since_improvement} generations")
            
            # Adaptive mutation rate - increase during stagnation
            current_mutation_rate = min(
                MAX_MUTATION_RATE,
                BASE_MUTATION_RATE + (generations_since_improvement * MUTATION_BOOST_FACTOR)
            )
            
            # Check if we need a restart
            if generations_since_improvement >= STAGNATION_WINDOW:
                if total_restarts < MAX_STAGNATION_RESTARTS:
                    total_restarts += 1
                    print(f"\n{'!'*50}")
                    print(f"STAGNATION DETECTED - Initiating Restart #{total_restarts}")
                    print(f"{'!'*50}")
                    
                    # Calculate how many fresh molecules to inject
                    n_fresh = int(ELITE_COUNT * STAGNATION_RESTART_RATIO)
                    n_keep = ELITE_COUNT - n_fresh
                    
                    # Keep top performers by raw fitness
                    elite_df_kept = elite_df.nlargest(n_keep, 'FitnessScore')
                    
                    # Generate fresh molecules
                    print(f"Generating {n_fresh} fresh molecules...")
                    fresh_df = generate_initial_population(
                        n=n_fresh * 3,  # Generate extra to account for filtering
                        training_smiles=training_smiles,
                        min_heavy_atoms=MIN_HEAVY_ATOMS,
                        max_heavy_atoms=MAX_HEAVY_ATOMS,
                        max_carbons=MAX_CARBONS,
                        max_oxygens=MAX_OXYGENS
                    )
                    
                    # Filter for unseen molecules
                    fresh_smiles = [s for s in fresh_df['SMILES'] if strict_canonicalize_smiles(s) not in seen_smiles][:n_fresh]
                    
                    if len(fresh_smiles) > 0:
                        fresh_df = pd.DataFrame({'SMILES': fresh_smiles})
                        
                        # Evaluate fresh molecules
                        fresh_evaluated = evaluate_molecules(
                            fresh_df,
                            thermo_models=thermo_models,
                            sc_model=sc_model,
                            tox21_models=tox21_models,
                            biodeg_model=biodeg_model
                        )
                        fresh_evaluated = assign_validity(
                            fresh_evaluated,
                            sc_threshold=MAX_SCSCORE,
                            # mp_max removed - now using soft penalty
                            bp_min=BP_THRESHOLD,
                            dc_max=DC_THRESHOLD,
                            min_fp=MIN_FLASHPOINT,
                            use_biodeg=USE_BIODEG_FILTER,
                            max_tox21=MAX_TOX21
                        )
                        fresh_evaluated = compute_fitness(fresh_evaluated, TARGET, TARGET_CONFIG)
                        fresh_evaluated = apply_mp_penalty(fresh_evaluated, soft_threshold=MP_SOFT, hard_threshold=MP_HARD)
                        
                        # Add to seen smiles
                        fresh_evaluated["CanonicalSMILES"] = fresh_evaluated["SMILES"].apply(strict_canonicalize_smiles)
                        seen_smiles.update(fresh_evaluated["CanonicalSMILES"])
                        
                        # Save fresh molecules to CSV
                        fresh_evaluated["generation"] = gen
                        fresh_evaluated.to_csv(ALL_EVALUATED_PATH, mode='a', header=False, index=False)
                        
                        # Combine kept elites with fresh molecules
                        combined_restart = pd.concat([elite_df_kept, fresh_evaluated], ignore_index=True)
                        combined_restart = apply_niching(combined_restart, tau=TAU)
                        
                        # Re-select elites using hybrid approach
                        combined_restart = combined_restart.sort_values('FitnessScore', ascending=False).drop_duplicates(subset='SMILES', keep='first')
                        best_by_fitness = combined_restart.nlargest(BEST_ELITE_COUNT, 'FitnessScore')
                        remaining = combined_restart[~combined_restart.index.isin(best_by_fitness.index)]
                        diverse_elites = remaining.nlargest(DIVERSE_ELITE_COUNT, 'NichedFitnessScore')
                        elite_df = pd.concat([best_by_fitness, diverse_elites]).reset_index(drop=True)
                        
                        print(f"Restart complete: kept {n_keep}, added {len(fresh_evaluated)} fresh molecules")
                        print(f"New elite population: {len(elite_df)} molecules")
                    else:
                        print("Warning: Could not generate fresh molecules for restart")
                    
                    # Reset stagnation counter but keep elevated mutation rate for a bit
                    generations_since_improvement = 0
                    current_mutation_rate = MAX_MUTATION_RATE  # Keep high after restart
                else:
                    print(f"\n⚠ Max restarts ({MAX_STAGNATION_RESTARTS}) reached - continuing with current population")

        # ----- Save Visualisation -----
        if VISUALIZE:
            save_molecule_grid_image(top_n_df, OUTPUT_IMAGE_PATH, gen_num=gen)

        # ----- Periodic Save of Tracking Data -----
        if gen % 1 == 0:
            save_tracking_data(tracking_data, generation_stats, args.output_dir)

    # =============================
    # Save Final Results
    # =============================
    
    # Save final tracking data
    save_tracking_data(tracking_data, generation_stats, args.output_dir)

    # Save final top molecules
    TOP_FINAL = 25
    final_top_df = (
        elite_df
        .sort_values("FitnessScore", ascending=False)
        .head(TOP_FINAL)
        .reset_index(drop=True)
    )

    output_top_path = os.path.join(args.output_dir, f"top_{TOP_FINAL}_molecules_{TARGET}.csv")
    final_top_df.to_csv(output_top_path, index=False)

    print(f"\n{'='*50}")
    print(f"WSGA COMPLETE")
    print(f"{'='*50}")
    print(f"Saved top {TOP_FINAL} molecules to: {output_top_path}")
    print(f"All evaluated molecules: {ALL_EVALUATED_PATH}")
    print(f"Top N tracking: {TOP_N_TRACKING_PATH}")
    print(f"Generation stats: {GENERATION_STATS_PATH}")


if __name__ == "__main__":
    main()