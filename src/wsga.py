"""
WSGA - Weighted Sum Genetic Algorithm for Cooling Fluid Discovery

Uses a GRU prior trained on the unified training corpus for initial
population generation. Features hybrid elite selection, stagnation
detection with population restart, and size-dependent mutation weighting.

Optimizes molecular structures for thermophysical properties relevant to
cooling fluids (thermal conductivity, heat capacity, viscosity, etc.)
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
import joblib

from wsga_helper import (
    apply_mutations_to_population,
    evaluate_molecules,
    k_way_tournament,
    load_tox21_predictor,
    load_biodeg_model,
    apply_niching,
    assign_validity,
    compute_fitness,
    apply_molprice_penalty,
    load_regression_models_with_aux,
    load_fold_ensemble_models,
    compute_conformal_quantiles,
    compute_mahalanobis_params,
)
from evaluation import get_scscore_cached, strict_canonicalize_smiles
from fragment_utils import prepare_fragments, crossover_fragments, crossover_mol_fragments
from SCScorer import SCScorer
from generate_molecules import generate_initial_population, load_combined_training_data, sample_pubchem_population, generate_gru_population


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
parser.add_argument("--training_data_dir", type=str, default="../training/data",
    help="Training data directory (for Mahalanobis OOD detection)")
parser.add_argument("--Tau", "--tau", type=float, default=0.05, help="Threshold for niching")
parser.add_argument("--target", type=str, default="FOM1",
    choices=["FOM1", "FOM1_direct"],
    help="Target property to optimize"
)
parser.add_argument("--top_n", type=int, default=40, help="Number of top molecules to track and visualize")
parser.add_argument("--num_generations", type=int, default=200, help="Number of generations to run")

# Validity threshold arguments (for case studies)
parser.add_argument("--mp_threshold", type=float, default=-30, help="Max melting point (°C) - hard cutoff")
parser.add_argument("--bp_threshold", type=float, default=70, help="Min boiling point (°C)")
parser.add_argument("--dc_threshold", type=float, default=8, help="Max dielectric constant")
parser.add_argument("--fp_threshold", type=float, default=373, help="Min flash point (K)")
parser.add_argument("--sc_threshold", type=float, default=3, help="Max SCScore")
parser.add_argument("--tox_threshold", type=float, default=3, help="Max Tox21 score")
parser.add_argument("--no_biodeg", action="store_true", help="Disable biodegradability filter")
parser.add_argument("--molprice_model", type=str, default=None,
    help="Path to MolPrice model weights (.pkl). Enables MolPrice cost penalty.")
parser.add_argument("--molprice_soft", type=float, default=3.0,
    help="MolPrice soft threshold (log USD/mmol) - no penalty below this")
parser.add_argument("--molprice_hard", type=float, default=6.0,
    help="MolPrice hard threshold (log USD/mmol) - zero fitness at/above this")
parser.add_argument("--tournament_k", type=int, default=3, help="Tournament selection size (k)")
parser.add_argument("--best_elite_ratio", type=float, default=0.3,
    help="Fraction of elites selected by raw FitnessScore (rest by NichedFitnessScore)")
parser.add_argument("--init_method", type=str, default="gru",
    choices=["gru", "ngram", "pubchem"],
    help="Initial population method: 'gru' (default), 'ngram', or 'pubchem'")
parser.add_argument("--gru_prior", type=str, default=None,
    help="Path to pre-trained GRU prior.pt (default: models/init_corpus/gru_prior.pt)")
parser.add_argument("--gru_vocab", type=str, default=None,
    help="Path to GRU vocabulary.json (default: models/init_corpus/vocabulary.json)")
parser.add_argument("--pubchem_path", type=str, default=None,
    help="Path to PubChem CHO CSV file (required when --init_method pubchem)")
parser.add_argument("--seed", type=int, default=None,
    help="Random seed for reproducibility")

args = parser.parse_args()

if args.init_method == "pubchem" and args.pubchem_path is None:
    parser.error("--pubchem_path is required when --init_method is 'pubchem'")

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)


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
TOURNAMENT_K = args.tournament_k

# Elite Selection Split
# - BEST_ELITE_RATIO of elites are selected by raw FitnessScore (guaranteed best performers)
# - Remaining (1 - BEST_ELITE_RATIO) are selected by NichedFitnessScore (diversity)
BEST_ELITE_RATIO = args.best_elite_ratio
BEST_ELITE_COUNT = int(ELITE_COUNT * BEST_ELITE_RATIO)
DIVERSE_ELITE_COUNT = ELITE_COUNT - BEST_ELITE_COUNT

# Stagnation Detection & Restart Configuration
STAGNATION_WINDOW = 5           # Number of generations to look back for improvement
STAGNATION_THRESHOLD = 0.01    # Minimum improvement required (in FOM1 units)
STAGNATION_RESTART_RATIO = 0.3  # Replace 30% of elite with fresh molecules on restart
MAX_STAGNATION_RESTARTS = 5     # Maximum number of restarts before giving up

# Tracking
TOP_N = args.top_n  # Number of top molecules to track per generation

# Fitness Target
TARGET = args.target
TARGET_CONFIG = {
    "FOM1":        {"column": "fom1_40C",  "maximize": True},
    "FOM1_direct": {"column": "fom1_40C",  "maximize": True},
}

# Selection Criteria (validity thresholds) - from args for case studies
MP_THRESHOLD = args.mp_threshold      # Max melting point (°C) - hard cutoff
BP_THRESHOLD = args.bp_threshold      # Min boiling point (°C)
DC_THRESHOLD = args.dc_threshold      # Max dielectric constant
MIN_FLASHPOINT = args.fp_threshold    # Min flash point (K)
MAX_SCSCORE = args.sc_threshold       # Max synthetic complexity
MAX_TOX21 = args.tox_threshold        # Max toxicity score
USE_BIODEG_FILTER = not args.no_biodeg
MOLPRICE_MODEL_PATH = args.molprice_model
MOLPRICE_SOFT = args.molprice_soft
MOLPRICE_HARD = args.molprice_hard

# Structural Constraints
MAX_HEAVY_ATOMS = 30
MIN_HEAVY_ATOMS = 5
MAX_CARBONS = 30
MAX_OXYGENS = 6

# Paths
DATA_DIR = args.data_dir
MODEL_DIR = args.model_dir
OUTPUT_IMAGE_PATH = os.path.join(args.output_dir, "ga_top_molecules.png")
ALL_EVALUATED_PATH = os.path.join(args.output_dir, "all_evaluated_molecules.csv")
TOP_N_TRACKING_PATH = os.path.join(args.output_dir, "top_n_tracking.csv")
GENERATION_STATS_PATH = os.path.join(args.output_dir, "generation_stats.csv")

# Display
VISUALIZE = False
TAU = args.Tau

print(f"=== WSGA Configuration ===")
print(f"Target: {TARGET}")
print(f"Population size: {GENERATION_SIZE}")
print(f"Elite count: {ELITE_COUNT}")
print(f"  - Best by FitnessScore: {BEST_ELITE_COUNT} ({BEST_ELITE_RATIO*100:.0f}%)")
print(f"  - Best by NichedFitness: {DIVERSE_ELITE_COUNT} ({(1-BEST_ELITE_RATIO)*100:.0f}%)")
print(f"Elitism rate: {ELITISM_RATE}")
print(f"Base mutation rate: {BASE_MUTATION_RATE}")
print(f"Tournament k: {TOURNAMENT_K}")
print(f"Tau (niching): {TAU}")
print(f"Top N tracking: {TOP_N}")
print(f"Model directory: {MODEL_DIR}")
print(f"--- Stagnation Settings ---")
print(f"Stagnation window: {STAGNATION_WINDOW} generations")
print(f"Stagnation threshold: {STAGNATION_THRESHOLD}")
print(f"Restart ratio: {STAGNATION_RESTART_RATIO*100:.0f}%")
print(f"Max restarts: {MAX_STAGNATION_RESTARTS}")
print(f"--- Validity Thresholds ---")
print(f"MP threshold: {MP_THRESHOLD}°C")
print(f"BP threshold: {BP_THRESHOLD}°C")
print(f"DC threshold: {DC_THRESHOLD}")
print(f"Flash point: {MIN_FLASHPOINT}K")
print(f"SCScore max: {MAX_SCSCORE}")
print(f"Tox21 max: {MAX_TOX21}")
print(f"Biodeg filter: {USE_BIODEG_FILTER}")
print(f"Crossover: bond-level (with string fallback)")
print(f"--- MolPrice ---")
print(f"MolPrice prediction: always on (for logging)")
if MOLPRICE_MODEL_PATH:
    print(f"MolPrice penalty: enabled ({MOLPRICE_MODEL_PATH})")
    print(f"MolPrice soft threshold: {MOLPRICE_SOFT} log(USD/mmol)")
    print(f"MolPrice hard threshold: {MOLPRICE_HARD} log(USD/mmol)")
else:
    print(f"MolPrice penalty: disabled (no --molprice_model)")
print(f"--- Init Method ---")
print(f"Init method: {args.init_method}")
if args.init_method == "pubchem":
    print(f"PubChem path: {args.pubchem_path}")
if args.seed is not None:
    print(f"Random seed: {args.seed}")
print(f"Training data dir: {args.training_data_dir}")
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
BondTypes = [rdchem.BondType.SINGLE]

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
    # Thermophysical predictions (40C)
    'density_40C', 'viscosity_40C', 'tc_40C', 'cpsat_40C', 'beta_40C', 'fom1_40C',
    # Derived
    'MW', 'Cp_40', 'alpha_40', 'nu_40', 'FOM1_40',
    # Constraints
    'bp', 'mp', 'fp', 'dc',
    # Safety/Synthesis/Cost
    'SCScore', 'Tox21_Score', 'Biodegradable',
    'MolPrice', 'MolPrice_Penalty',
    # OOD
    'OOD_any', 'OOD_count',
    # Fold ensemble uncertainty (key properties)
    'fom1_40C_fold_mean', 'fom1_40C_fold_std',
    'fom1_40C_ci_lower', 'fom1_40C_ci_upper',
    'fp_fold_std', 'mp_fold_std', 'bp_fold_std', 'dc_fold_std',
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
        'alpha_40', 'beta_40C', 'fom1_40C', 'FOM1_40',
        'tc_40C', 'viscosity_40C',
        'SCScore', 'Tox21_Score', 'mp', 'OOD_count'
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
    
    # Regression models (all use Mordred-pipeline format)
    thermo_targets = [
        "bp", "mp", "fp", "dc",
        "density_40C", "viscosity_40C", "tc_40C", "cpsat_40C", "beta_40C",
        "fom1_40C",
    ]
    thermo_models = load_regression_models_with_aux(thermo_targets, MODEL_DIR)

    # SCScorer (separate pre-trained model)
    sc_model = SCScorer()
    sc_model.restore(os.path.join(MODEL_DIR, "SCScorer/scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"))

    # Tox21 predictor (PaccMann MCA from GT4SD model hub)
    tox21_dir = os.path.join(MODEL_DIR, "tox21_gt4sd")
    tox21_models = load_tox21_predictor(tox21_dir)

    # Biodegradability model (classification)
    biodeg_dir = os.path.join(MODEL_DIR, "biodeg")
    biodeg_model = load_biodeg_model(biodeg_dir)

    # MolPrice model (cost prediction) — always loaded for logging,
    # but the cost *penalty* is only applied when --molprice_model is set.
    from molprice import MolPriceModel
    _default_mp_path = os.path.join(MODEL_DIR, "MolPrice", "MP_Morgan_hybrid.pkl")
    _mp_path = MOLPRICE_MODEL_PATH or _default_mp_path
    molprice_model = None
    if os.path.exists(_mp_path):
        molprice_model = MolPriceModel(_mp_path)
        print(f"Loaded MolPrice model from {_mp_path}")
    else:
        print(f"WARNING: MolPrice model not found at {_mp_path} — MolPrice column will be empty")

    # Mahalanobis OOD params (precomputed per model at startup)
    print("\nPrecomputing Mahalanobis OOD parameters...")
    mahal_params = {}
    training_data_dir = args.training_data_dir
    for target, data in thermo_models.items():
        params = compute_mahalanobis_params(
            target, data['features'], training_data_dir)
        if params is not None:
            mahal_params[target] = params
    if mahal_params:
        print(f"Mahalanobis ready for {len(mahal_params)} models")
    else:
        print("Mahalanobis: no params computed (training data not found)")
        mahal_params = None

    # Fold ensemble models for uncertainty estimation
    print("\nLoading fold ensemble models...")
    fold_ensembles = load_fold_ensemble_models(thermo_targets, MODEL_DIR)
    if fold_ensembles:
        conformal_quantiles = compute_conformal_quantiles(
            MODEL_DIR, list(fold_ensembles.keys()), alpha=0.1)
        print(f"Fold ensembles ready for {len(fold_ensembles)} models")
    else:
        fold_ensembles = None
        conformal_quantiles = None
        print("No fold ensembles found — uncertainty columns will be absent")

    print("Models loaded successfully.\n")

    # ----- Initialize Population -----
    print("Generating initial population...")

    if args.init_method == "gru":
        prior_path = args.gru_prior or os.path.join(MODEL_DIR, "init_corpus", "gru_prior.pt")
        vocab_path = args.gru_vocab or os.path.join(MODEL_DIR, "init_corpus", "vocabulary.json")
        print(f"Generating initial population via GRU prior: {prior_path}")
        df = generate_gru_population(
            n=INITIAL_POPULATION_SIZE,
            prior_path=prior_path,
            vocab_path=vocab_path,
            max_heavy_atoms=MAX_HEAVY_ATOMS,
            min_heavy_atoms=MIN_HEAVY_ATOMS,
            max_carbons=MAX_CARBONS,
            max_oxygens=MAX_OXYGENS,
        )
        training_smiles = load_combined_training_data(DATA_DIR)
    elif args.init_method == "pubchem":
        print(f"Sampling from PubChem corpus: {args.pubchem_path}")
        df = sample_pubchem_population(
            n=INITIAL_POPULATION_SIZE,
            pubchem_path=args.pubchem_path,
            max_heavy_atoms=MAX_HEAVY_ATOMS,
            min_heavy_atoms=MIN_HEAVY_ATOMS,
            max_carbons=MAX_CARBONS,
            max_oxygens=MAX_OXYGENS,
            seed=args.seed,
        )
        training_smiles = load_combined_training_data(DATA_DIR)
    else:
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
        biodeg_model=biodeg_model,
        molprice_model=molprice_model,
        mahal_params=mahal_params,
        fom1_mlp_data=None,
        fold_ensembles=fold_ensembles,
        conformal_quantiles=conformal_quantiles,
    )

    evaluated_df = assign_validity(
        evaluated_df,
        sc_threshold=MAX_SCSCORE,
        mp_max=MP_THRESHOLD,
        bp_min=BP_THRESHOLD,
        dc_max=DC_THRESHOLD,
        min_fp=MIN_FLASHPOINT,
        use_biodeg=USE_BIODEG_FILTER,
        max_tox21=MAX_TOX21,
    )

    evaluated_df = compute_fitness(evaluated_df, TARGET, TARGET_CONFIG)
    if MOLPRICE_MODEL_PATH:
        evaluated_df = apply_molprice_penalty(evaluated_df, soft_threshold=MOLPRICE_SOFT, hard_threshold=MOLPRICE_HARD)
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
    # Fix column order for CSV — all subsequent appends must use this same order
    CSV_COLUMNS = list(evaluated_df.columns)
    evaluated_df[CSV_COLUMNS].to_csv(ALL_EVALUATED_PATH, mode='w', header=True, index=False)

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

    # =============================
    # Genetic Algorithm Loop
    # =============================
    
    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n{'='*50}")
        print(f"Generation {gen}/{NUM_GENERATIONS}")
        print(f"Mutation rate: {BASE_MUTATION_RATE:.3f} | Restarts: {total_restarts}")
        print(f"{'='*50}")

        # ----- Generate Offspring via Crossover -----
        required_offspring = GENERATION_SIZE - len(elite_df)
        new_population_smiles = []

        with tqdm(total=required_offspring, desc="Crossover") as pbar:
            while len(new_population_smiles) < required_offspring:
                p1 = k_way_tournament(elite_df, k=TOURNAMENT_K)
                p2 = k_way_tournament(elite_df, k=TOURNAMENT_K)

                # Try bond-level crossover first
                child = crossover_mol_fragments(p1, p2, max_heavy_atoms=MAX_HEAVY_ATOMS)

                if child is None:
                    # Fallback to string-level crossover
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
            df=offspring_df,
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

        # ----- Evaluate Offspring -----
        evaluated_offspring_df = evaluate_molecules(
            mutated_offspring_df,
            thermo_models=thermo_models,
            sc_model=sc_model,
            tox21_models=tox21_models,
            biodeg_model=biodeg_model,
            molprice_model=molprice_model,
            mahal_params=mahal_params,
            fom1_mlp_data=None,
            fold_ensembles=fold_ensembles,
            conformal_quantiles=conformal_quantiles,
        )

        evaluated_offspring_df = assign_validity(
            evaluated_offspring_df,
            sc_threshold=MAX_SCSCORE,
            mp_max=MP_THRESHOLD,
            bp_min=BP_THRESHOLD,
            dc_max=DC_THRESHOLD,
            min_fp=MIN_FLASHPOINT,
            use_biodeg=USE_BIODEG_FILTER,
            max_tox21=MAX_TOX21,
        )

        evaluated_offspring_df = compute_fitness(evaluated_offspring_df, TARGET, TARGET_CONFIG)
        if MOLPRICE_MODEL_PATH:
            evaluated_offspring_df = apply_molprice_penalty(evaluated_offspring_df, soft_threshold=MOLPRICE_SOFT, hard_threshold=MOLPRICE_HARD)
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
        # Reorder columns to match the header written at generation 0
        offspring_with_niching[CSV_COLUMNS].to_csv(ALL_EVALUATED_PATH, mode='a', header=False, index=False)

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
        
        if 'fom1_40C' in top_n_df.columns:
            print(f"  Best FOM1 (direct):  {top_n_df['fom1_40C'].max():.4f}")
            print(f"  Mean FOM1 (direct):  {top_n_df['fom1_40C'].mean():.4f}")
        if 'FOM1_40' in top_n_df.columns:
            print(f"  Best FOM1 (comp):    {top_n_df['FOM1_40'].max():.4f}")
        if 'OOD_count' in top_n_df.columns:
            print(f"  Mean OOD count:      {top_n_df['OOD_count'].mean():.1f}")

        # =============================
        # Stagnation Detection & Response
        # =============================
        best_fitness_history.append(best_raw_fitness)
        
        # Check for improvement
        if best_raw_fitness > last_best_fitness + STAGNATION_THRESHOLD:
            # Improvement found - reset counters
            generations_since_improvement = 0
            last_best_fitness = best_raw_fitness
            print(f"  ✓ Improvement! New best: {best_raw_fitness:.4f}")
        else:
            # No improvement
            generations_since_improvement += 1
            print(f"  ✗ No improvement for {generations_since_improvement} generations")

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
                    
                    # Generate fresh molecules using same method as init
                    print(f"Generating {n_fresh} fresh molecules...")
                    if args.init_method == "gru":
                        fresh_df = generate_gru_population(
                            n=n_fresh * 3,
                            prior_path=args.gru_prior or os.path.join(MODEL_DIR, "init_corpus", "gru_prior.pt"),
                            vocab_path=args.gru_vocab or os.path.join(MODEL_DIR, "init_corpus", "vocabulary.json"),
                            max_heavy_atoms=MAX_HEAVY_ATOMS,
                            min_heavy_atoms=MIN_HEAVY_ATOMS,
                            max_carbons=MAX_CARBONS,
                            max_oxygens=MAX_OXYGENS,
                        )
                    elif args.init_method == "pubchem":
                        fresh_df = sample_pubchem_population(
                            n=n_fresh * 3,
                            pubchem_path=args.pubchem_path,
                            max_heavy_atoms=MAX_HEAVY_ATOMS,
                            min_heavy_atoms=MIN_HEAVY_ATOMS,
                            max_carbons=MAX_CARBONS,
                            max_oxygens=MAX_OXYGENS,
                            seed=None,
                        )
                    else:
                        fresh_df = generate_initial_population(
                            n=n_fresh * 3,
                            training_smiles=training_smiles,
                            min_heavy_atoms=MIN_HEAVY_ATOMS,
                            max_heavy_atoms=MAX_HEAVY_ATOMS,
                            max_carbons=MAX_CARBONS,
                            max_oxygens=MAX_OXYGENS,
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
                            biodeg_model=biodeg_model,
                            molprice_model=molprice_model,
                            mahal_params=mahal_params,
                            fom1_mlp_data=None,
                            fold_ensembles=fold_ensembles,
                            conformal_quantiles=conformal_quantiles,
                        )
                        fresh_evaluated = assign_validity(
                            fresh_evaluated,
                            sc_threshold=MAX_SCSCORE,
                            mp_max=MP_THRESHOLD,
                            bp_min=BP_THRESHOLD,
                            dc_max=DC_THRESHOLD,
                            min_fp=MIN_FLASHPOINT,
                            use_biodeg=USE_BIODEG_FILTER,
                            max_tox21=MAX_TOX21,
                        )
                        fresh_evaluated = compute_fitness(fresh_evaluated, TARGET, TARGET_CONFIG)
                        if MOLPRICE_MODEL_PATH:
                            fresh_evaluated = apply_molprice_penalty(fresh_evaluated, soft_threshold=MOLPRICE_SOFT, hard_threshold=MOLPRICE_HARD)
                        
                        # Add to seen smiles
                        fresh_evaluated["CanonicalSMILES"] = fresh_evaluated["SMILES"].apply(strict_canonicalize_smiles)
                        seen_smiles.update(fresh_evaluated["CanonicalSMILES"])
                        
                        # Save fresh molecules to CSV
                        fresh_evaluated["generation"] = gen
                        # Reorder columns to match the header written at generation 0
                        for col in CSV_COLUMNS:
                            if col not in fresh_evaluated.columns:
                                fresh_evaluated[col] = np.nan
                        fresh_evaluated[CSV_COLUMNS].to_csv(ALL_EVALUATED_PATH, mode='a', header=False, index=False)
                        
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
                    
                    # Reset stagnation counter
                    generations_since_improvement = 0
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

    # Save top molecules by raw FOM1 (ignoring MolPrice penalty)
    # Ensures high-FOM1 discoveries are never hidden by cost penalties
    fom1_col = "fom1_40C" if "fom1_40C" in elite_df.columns else "FOM1_40"
    valid_elite = elite_df[elite_df["is_valid"] == 1] if "is_valid" in elite_df.columns else elite_df
    if fom1_col in valid_elite.columns and not valid_elite.empty:
        final_top_fom1_df = (
            valid_elite
            .sort_values(fom1_col, ascending=False)
            .head(TOP_FINAL)
            .reset_index(drop=True)
        )
        output_top_fom1_path = os.path.join(args.output_dir, f"top_{TOP_FINAL}_molecules_{TARGET}_raw.csv")
        final_top_fom1_df.to_csv(output_top_fom1_path, index=False)

    print(f"\n{'='*50}")
    print(f"WSGA COMPLETE")
    print(f"{'='*50}")
    print(f"Saved top {TOP_FINAL} molecules to: {output_top_path}")
    print(f"Saved top {TOP_FINAL} by raw FOM1 to: {output_top_fom1_path}")
    print(f"All evaluated molecules: {ALL_EVALUATED_PATH}")
    print(f"Top N tracking: {TOP_N_TRACKING_PATH}")
    print(f"Generation stats: {GENERATION_STATS_PATH}")


if __name__ == "__main__":
    main()