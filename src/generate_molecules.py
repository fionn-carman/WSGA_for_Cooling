import random
import pandas as pd
import os
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

# Suppress noisy SMILES parse warnings from n-gram generated strings
RDLogger.DisableLog("rdApp.*")

# ---------------------------
# N-gram SMILES model
# ---------------------------

def build_ngram_model(smiles_list, n=8):
    """
    Build character-level n-gram model from SMILES strings.
    
    Default n=8 provides optimal trade-off between:
    - Distribution match (~97%): generated molecules have similar size 
      and functional group distributions to training data
    - Novelty (~50%): good balance of novel vs known molecules
    """
    model = {}
    for smi in smiles_list:
        if not isinstance(smi, str):
            continue
        s = "^" * n + smi + "$"
        for i in range(len(s) - n):
            prefix = s[i:i+n]
            next_char = s[i+n]
            model.setdefault(prefix, []).append(next_char)
    return model


def generate_smiles_from_ngram(model, n=8, max_len=80):
    """
    Generate a SMILES string using the n-gram model.
    
    Args:
        model: n-gram model dictionary
        n: n-gram order (must match model)
        max_len: maximum SMILES length
    """
    prefix = "^" * n
    result = ""
    for _ in range(max_len):
        if prefix not in model:
            break
        next_char = random.choice(model[prefix])
        if next_char == "$":
            break
        result += next_char
        prefix = prefix[1:] + next_char
    return result


# ---------------------------
# Chemical filters
# ---------------------------

def is_CO_only(mol):
    """Check if molecule contains only C and O atoms."""
    if mol is None:
        return False
    atoms = set(a.GetSymbol() for a in mol.GetAtoms())
    return atoms.issubset({'C', 'O'})


def has_valid_rings(mol, allowed_sizes=(5, 6)):
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        if len(ring) not in allowed_sizes:
            return False
    return True


def atom_counts(mol):
    counts = {"C": 0, "O": 0}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
    return counts


def is_diverse(mol, mol_pool, threshold=0.6):
    if not mol_pool:
        return True
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    for m in mol_pool:
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        if DataStructs.TanimotoSimilarity(fp, fp2) > threshold:
            return False
    return True


# ---------------------------
# Combined dataset loader
# ---------------------------

def load_combined_training_data(data_dir="."):
    """
    Load and combine all available datasets for n-gram training.
    
    This provides a more diverse chemical space than using only the
    thermo dataset, which should help the GA explore new scaffolds.
    
    Returns:
        list: Combined list of unique canonical SMILES (C/O only)
    """
    datasets = [
        ('processed_full_hydrocarbon_dataset.csv', 'Thermo'),
        ('biodegradability_cleaned.csv', 'Biodeg'),
        ('BP-Measured_cleaned.csv', 'BP'),
        ('DC_exp_cleaned.csv', 'DC'),
        ('flashpoint_cleaned.csv', 'FP'),
        ('MP-Measured_cleaned.csv', 'MP'),
    ]
    
    all_smiles = []
    
    for filename, name in datasets:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        
        try:
            df = pd.read_csv(filepath)
            
            # Find SMILES column (case-insensitive)
            smiles_col = None
            for col in df.columns:
                if col.lower() == 'smiles':
                    smiles_col = col
                    break
            
            if smiles_col is None:
                print(f"Warning: Could not find SMILES column in {filename}")
                continue
            
            smiles_list = df[smiles_col].dropna().tolist()
            
            # Filter for C/O only and canonicalize
            count = 0
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol and is_CO_only(mol):
                    all_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                    count += 1
            print(f"  {name}: {count} C/O molecules loaded")
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
    
    # Remove duplicates
    unique_smiles = list(set(all_smiles))
    print(f"Combined: {len(unique_smiles)} unique C/O molecules from all datasets")
    
    return unique_smiles


# ---------------------------
# Main population generator
# ---------------------------

def generate_initial_population(
    n,
    max_heavy_atoms,
    min_heavy_atoms,
    max_carbons,
    max_oxygens,
    training_smiles,
    ngram_order=8,
    similarity_threshold=0.9,
    max_attempts_factor=100,
    diversity_start_fraction=0.3,
):
    """
    Generate an initial GA population as a DataFrame with a SMILES column.
    
    Args:
        n: number of molecules to generate
        max_heavy_atoms: maximum heavy atom count
        min_heavy_atoms: minimum heavy atom count
        max_carbons: maximum carbon count
        max_oxygens: maximum oxygen count
        training_smiles: list of SMILES to train n-gram model
        ngram_order: n-gram order (default 8, optimal for distribution match vs novelty)
        similarity_threshold: Tanimoto threshold for diversity filter
        max_attempts_factor: max_attempts = n * max_attempts_factor
        diversity_start_fraction: fraction of population before diversity check starts
    """

    ngram_model = build_ngram_model(training_smiles, n=ngram_order)
    print(f"Built {ngram_order}-gram model with {len(ngram_model)} unique prefixes")

    population_mols = []
    population_smiles = []
    seen_smiles = set()

    max_attempts = n * max_attempts_factor
    attempts = 0

    while len(population_mols) < n and attempts < max_attempts:
        attempts += 1

        # --- Generate SMILES ---
        smi = generate_smiles_from_ngram(ngram_model, n=ngram_order)

        if not smi:
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # --- C/O only filter ---
        if not is_CO_only(mol):
            continue

        # --- Canonicalise early ---
        smi = Chem.MolToSmiles(mol, canonical=True)
        if smi in seen_smiles:
            continue

        # --- Atom constraints ---
        heavy_atoms = mol.GetNumHeavyAtoms()
        if not (min_heavy_atoms <= heavy_atoms <= max_heavy_atoms):
            continue

        counts = atom_counts(mol)
        if counts["C"] > max_carbons or counts["O"] > max_oxygens:
            continue

        # --- Ring constraint ---
        if not has_valid_rings(mol):
            continue

        # --- Diversity (delayed!) ---
        if len(population_mols) > int(diversity_start_fraction * n):
            if not is_diverse(mol, population_mols, similarity_threshold):
                continue

        # --- Accept molecule ---
        population_mols.append(mol)
        population_smiles.append(smi)
        seen_smiles.add(smi)

    yield_pct = (len(population_smiles) / attempts) * 100 if attempts > 0 else 0
    
    if len(population_smiles) < n:
        print(
            f"Warning: Only generated {len(population_smiles)}/{n} molecules "
            f"after {attempts} attempts (yield={yield_pct:.2f}%)."
        )
    else:
        print(
            f"Generated {len(population_smiles)}/{n} molecules "
            f"after {attempts} attempts (yield={yield_pct:.2f}%)."
        )

    return pd.DataFrame({"SMILES": population_smiles})


# ---------------------------
# PubChem direct sampling
# ---------------------------

def sample_pubchem_population(
    n,
    pubchem_path,
    max_heavy_atoms,
    min_heavy_atoms,
    max_carbons,
    max_oxygens,
    seed=None,
):
    """
    Sample initial population directly from a PubChem CHO CSV file.

    Applies the same chemical filters as generate_initial_population
    (C/O only, atom counts, valid rings) but draws from a pre-existing
    corpus instead of generating via n-gram.

    Args:
        n: number of molecules to collect
        pubchem_path: path to CSV with a SMILES column
        max_heavy_atoms: maximum heavy atom count
        min_heavy_atoms: minimum heavy atom count
        max_carbons: maximum carbon count
        max_oxygens: maximum oxygen count
        seed: optional random seed for reproducible shuffling
    """
    df_pubchem = pd.read_csv(pubchem_path)
    all_smiles = df_pubchem["SMILES"].dropna().tolist()

    rng = random.Random(seed)
    rng.shuffle(all_smiles)

    population_smiles = []
    seen_smiles = set()
    scanned = 0

    for smi in all_smiles:
        if len(population_smiles) >= n:
            break
        scanned += 1

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        if not is_CO_only(mol):
            continue

        smi = Chem.MolToSmiles(mol, canonical=True)
        if smi in seen_smiles:
            continue

        heavy_atoms = mol.GetNumHeavyAtoms()
        if not (min_heavy_atoms <= heavy_atoms <= max_heavy_atoms):
            continue

        counts = atom_counts(mol)
        if counts["C"] > max_carbons or counts["O"] > max_oxygens:
            continue

        if not has_valid_rings(mol):
            continue

        population_smiles.append(smi)
        seen_smiles.add(smi)

    yield_pct = (len(population_smiles) / scanned) * 100 if scanned > 0 else 0

    if len(population_smiles) < n:
        print(
            f"Warning: Only sampled {len(population_smiles)}/{n} molecules "
            f"from {scanned} scanned (yield={yield_pct:.1f}%)."
        )
    else:
        print(
            f"Sampled {len(population_smiles)}/{n} molecules "
            f"from {scanned} scanned (yield={yield_pct:.1f}%)."
        )

    return pd.DataFrame({"SMILES": population_smiles})


# ---------------------------
# GRU prior sampling
# ---------------------------

def generate_gru_population(
    n,
    prior_path,
    vocab_path,
    max_heavy_atoms,
    min_heavy_atoms,
    max_carbons,
    max_oxygens,
    temperature=1.0,
    batch_size=512,
):
    """
    Generate initial population using a pre-trained GRU prior.

    Loads a trained GRU model and vocabulary, samples SMILES in batches,
    and filters to CHO molecules matching the atom count constraints.

    Args:
        n: number of unique valid molecules to collect
        prior_path: path to gru_prior.pt state dict
        vocab_path: path to vocabulary.json
        max_heavy_atoms: maximum heavy atom count
        min_heavy_atoms: minimum heavy atom count
        max_carbons: maximum carbon count
        max_oxygens: maximum oxygen count
        temperature: sampling temperature (1.0 = standard)
        batch_size: number of SMILES to sample per batch
    """
    import sys
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'reinvent'))
    from reinvent.vocabulary import SMILESVocabulary
    from reinvent.model import GRUModel

    # Load model
    vocab = SMILESVocabulary.load(vocab_path)
    model = GRUModel(len(vocab))
    model.load_state_dict(torch.load(prior_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"Loaded GRU prior from {prior_path} (vocab={len(vocab)})")

    population_smiles = []
    seen = set()
    attempts = 0

    while len(population_smiles) < n:
        with torch.no_grad():
            batch_smiles, _ = model.sample(vocab, batch_size=batch_size,
                                           temperature=temperature)
        for smi in batch_smiles:
            attempts += 1
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            if not is_CO_only(mol):
                continue

            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical in seen:
                continue

            heavy_atoms = mol.GetNumHeavyAtoms()
            if not (min_heavy_atoms <= heavy_atoms <= max_heavy_atoms):
                continue

            counts = atom_counts(mol)
            if counts["C"] > max_carbons or counts["O"] > max_oxygens:
                continue

            if not has_valid_rings(mol):
                continue

            population_smiles.append(canonical)
            seen.add(canonical)

            if len(population_smiles) >= n:
                break

    yield_pct = (len(population_smiles) / attempts) * 100 if attempts > 0 else 0
    print(f"GRU generated {len(population_smiles)}/{n} molecules "
          f"from {attempts} samples (yield={yield_pct:.1f}%)")

    return pd.DataFrame({"SMILES": population_smiles})