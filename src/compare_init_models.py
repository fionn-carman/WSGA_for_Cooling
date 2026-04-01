#!/usr/bin/env python3
"""
Compare 8-gram vs GRU initial population models.

1. Curate unified corpus from all training datasets (CHO, no banned fragments)
2. Train both models on the corpus
3. Generate 5k molecules from each
4. Compare: validity rates, MW distribution, functional groups vs corpus
"""

import os
import sys
import json
import glob
import time
import random
import numpy as np
import pandas as pd
from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog('rdApp.error')

# Add reinvent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'reinvent'))

from generate_molecules import build_ngram_model, generate_smiles_from_ngram
from wsga_helper import has_invalid_fragments

# ============================================================
# Step 1: Curate unified corpus
# ============================================================

def curate_corpus(training_data_dir):
    """Collect union of all training SMILES, filtered to CHO + no banned fragments."""
    all_smiles = set()

    # NIST 8100 thermo datasets
    nist_dir = os.path.join(training_data_dir, "nist_8100")
    for csv_path in sorted(glob.glob(os.path.join(nist_dir, "*_cho_cleaned.csv"))):
        df = pd.read_csv(csv_path, usecols=["SMILES"])
        all_smiles.update(df["SMILES"].dropna().tolist())
        print(f"  {os.path.basename(csv_path)}: {len(df)} rows")

    # Constraint datasets
    constraints_dir = os.path.join(training_data_dir, "constraints")
    for name in ["BP-Measured_cleaned.csv", "MP-Measured_cleaned.csv",
                 "flashpoint_cleaned.csv", "DC_exp_cleaned.csv",
                 "biodegradability_cleaned.csv"]:
        path = os.path.join(constraints_dir, name)
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=["SMILES"])
            all_smiles.update(df["SMILES"].dropna().tolist())
            print(f"  {name}: {len(df)} rows")

    print(f"\nTotal raw SMILES collected: {len(all_smiles)}")

    # Filter
    valid = []
    rejected = Counter()
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rejected["parse_fail"] += 1
            continue

        # CHO only
        atoms = set(a.GetAtomicNum() for a in mol.GetAtoms())
        if not atoms <= {6, 1, 8}:
            rejected["non_CHO"] += 1
            continue

        # Heavy atom count
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 5 or n_heavy > 30:
            rejected["heavy_atoms"] += 1
            continue

        # No charges/radicals
        if Chem.GetFormalCharge(mol) != 0:
            rejected["charge"] += 1
            continue
        if Descriptors.NumRadicalElectrons(mol) > 0:
            rejected["radicals"] += 1
            continue

        # No banned fragments
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        if has_invalid_fragments(canonical):
            rejected["banned_fragments"] += 1
            continue

        valid.append(canonical)

    valid = sorted(set(valid))
    print(f"\nFiltered corpus: {len(valid)} unique SMILES")
    print(f"Rejected: {dict(rejected)}")
    return valid


# ============================================================
# Step 2: Analyse a set of molecules
# ============================================================

def analyse_molecules(smiles_list, label):
    """Compute MW, heavy atoms, and functional group distributions."""
    mws, has_, n_ester, n_ether, n_ketone, n_alkene, n_alcohol, n_ring = [], [], [], [], [], [], [], []
    n_aromatic = []

    ester_pat = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
    ether_pat = Chem.MolFromSmarts("[OD2]([CX4])[CX4]")
    ketone_pat = Chem.MolFromSmarts("[CX3](=O)([CX4])[CX4]")
    alkene_pat = Chem.MolFromSmarts("[CX3]=[CX3]")
    alcohol_pat = Chem.MolFromSmarts("[OX2H]")

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mws.append(Descriptors.MolWt(mol))
        has_.append(mol.GetNumHeavyAtoms())
        n_ester.append(len(mol.GetSubstructMatches(ester_pat)))
        n_ether.append(len(mol.GetSubstructMatches(ether_pat)))
        n_ketone.append(len(mol.GetSubstructMatches(ketone_pat)))
        n_alkene.append(len(mol.GetSubstructMatches(alkene_pat)))
        n_alcohol.append(len(mol.GetSubstructMatches(alcohol_pat)))
        n_ring.append(rdMolDescriptors.CalcNumRings(mol))
        n_aromatic.append(rdMolDescriptors.CalcNumAromaticRings(mol))

    stats = {
        "label": label,
        "count": len(mws),
        "MW_mean": np.mean(mws), "MW_std": np.std(mws),
        "MW_median": np.median(mws),
        "HA_mean": np.mean(has_),
        "pct_ester": 100 * np.mean([x > 0 for x in n_ester]),
        "pct_ether": 100 * np.mean([x > 0 for x in n_ether]),
        "pct_ketone": 100 * np.mean([x > 0 for x in n_ketone]),
        "pct_alkene": 100 * np.mean([x > 0 for x in n_alkene]),
        "pct_alcohol": 100 * np.mean([x > 0 for x in n_alcohol]),
        "pct_ring": 100 * np.mean([x > 0 for x in n_ring]),
        "pct_aromatic": 100 * np.mean([x > 0 for x in n_aromatic]),
        "mean_n_ester": np.mean(n_ester),
        "mean_n_ether": np.mean(n_ether),
    }
    return stats, mws


# ============================================================
# Step 3: Generate from 8-gram
# ============================================================

def generate_ngram(corpus, n_generate=5000, ngram_order=8):
    """Build 8-gram model and generate molecules."""
    print("\nBuilding 8-gram model...")
    model = build_ngram_model(corpus, n=ngram_order)
    print(f"  Model size: {len(model)} prefixes")

    print(f"Generating {n_generate} molecules from 8-gram...")
    generated = set()
    attempts = 0
    valid_attempts = 0
    max_attempts = n_generate * 100

    t0 = time.time()
    while len(generated) < n_generate and attempts < max_attempts:
        smi = generate_smiles_from_ngram(model, n=ngram_order, max_len=80)
        attempts += 1
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        valid_attempts += 1
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        generated.add(canonical)

    elapsed = time.time() - t0
    print(f"  Attempts: {attempts}, Valid SMILES: {valid_attempts} "
          f"({100*valid_attempts/attempts:.1f}%), Unique valid: {len(generated)}")
    print(f"  Time: {elapsed:.1f}s ({attempts/elapsed:.0f} attempts/s)")

    return list(generated), model, {
        "attempts": attempts,
        "valid": valid_attempts,
        "unique": len(generated),
        "validity_rate": valid_attempts / attempts,
        "time_s": elapsed,
    }


# ============================================================
# Step 4: Generate from GRU
# ============================================================

def generate_gru(corpus, n_generate=5000, epochs=30, batch_size=512, temperature=1.0,
                 n_augmentations=3):
    """Train GRU prior and generate molecules."""
    import torch
    from reinvent.vocabulary import SMILESVocabulary
    from reinvent.model import GRUModel
    from reinvent.trainer import pretrain
    from reinvent.augment import augment_corpus

    if n_augmentations and n_augmentations > 0:
        print(f"\nAugmenting corpus for GRU training ({n_augmentations}x)...")
        augmented = augment_corpus(corpus, n_augmentations=n_augmentations)
        print(f"  Augmented: {len(corpus)} → {len(augmented)} SMILES")
    else:
        print("\nNo augmentation — training on canonical SMILES only")
        augmented = list(corpus)

    print("Building vocabulary...")
    vocab = SMILESVocabulary().build(augmented)
    print(f"  Vocab size: {len(vocab)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training GRU on {device}...")
    model = GRUModel(len(vocab)).to(device)
    model = pretrain(model, vocab, augmented, epochs=epochs, batch_size=batch_size,
                     lr=1e-3, device=device)

    print(f"\nGenerating {n_generate} molecules from GRU (T={temperature})...")
    model.eval()
    generated = set()
    attempts = 0
    valid_attempts = 0

    t0 = time.time()
    while len(generated) < n_generate:
        batch_smiles, _ = model.sample(vocab, batch_size=512, temperature=temperature)
        for smi in batch_smiles:
            attempts += 1
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            valid_attempts += 1
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            generated.add(canonical)

    elapsed = time.time() - t0
    print(f"  Attempts: {attempts}, Valid SMILES: {valid_attempts} "
          f"({100*valid_attempts/attempts:.1f}%), Unique valid: {len(generated)}")
    print(f"  Time: {elapsed:.1f}s")

    return list(generated), model, vocab, {
        "attempts": attempts,
        "valid": valid_attempts,
        "unique": len(generated),
        "validity_rate": valid_attempts / attempts,
        "time_s": elapsed,
    }


# ============================================================
# Main
# ============================================================

def main():
    random.seed(42)
    np.random.seed(42)

    training_data_dir = os.path.join(os.path.dirname(__file__), "..", "training", "data")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "models", "init_corpus")
    os.makedirs(output_dir, exist_ok=True)

    N_GENERATE = 5000

    # --- Curate corpus ---
    print("=" * 60)
    print("STEP 1: Curate unified corpus")
    print("=" * 60)
    corpus = curate_corpus(training_data_dir)

    corpus_path = os.path.join(output_dir, "corpus.txt")
    with open(corpus_path, "w") as f:
        f.write("\n".join(corpus))
    print(f"Saved corpus to {corpus_path}")

    # --- Analyse corpus ---
    print("\n" + "=" * 60)
    print("STEP 2: Analyse corpus")
    print("=" * 60)
    corpus_stats, corpus_mws = analyse_molecules(corpus, "Corpus")

    # --- Generate from 8-gram ---
    print("\n" + "=" * 60)
    print("STEP 3: 8-gram generation")
    print("=" * 60)
    ngram_smiles, ngram_model, ngram_gen_stats = generate_ngram(corpus, N_GENERATE)

    # Save 8-gram model
    ngram_path = os.path.join(output_dir, "ngram_8.json")
    with open(ngram_path, "w") as f:
        json.dump(ngram_model, f)
    print(f"Saved 8-gram model to {ngram_path}")

    ngram_stats, ngram_mws = analyse_molecules(ngram_smiles, "8-gram")

    # --- Generate from GRU (no augmentation) ---
    print("\n" + "=" * 60)
    print("STEP 4a: GRU generation (no augmentation)")
    print("=" * 60)
    gru0_smiles, gru0_model, gru0_vocab, gru0_gen_stats = generate_gru(
        corpus, N_GENERATE, n_augmentations=0)

    # Save GRU model (no aug)
    import torch
    torch.save(gru0_model.state_dict(), os.path.join(output_dir, "gru_prior_noaug.pt"))
    gru0_vocab.save(os.path.join(output_dir, "vocabulary_noaug.json"))

    gru0_stats, gru0_mws = analyse_molecules(gru0_smiles, "GRU-0x")

    # --- Generate from GRU (3x augmentation) ---
    print("\n" + "=" * 60)
    print("STEP 4b: GRU generation (3x augmentation)")
    print("=" * 60)
    gru3_smiles, gru3_model, gru3_vocab, gru3_gen_stats = generate_gru(
        corpus, N_GENERATE, n_augmentations=3)

    # Save GRU model (3x)
    torch.save(gru3_model.state_dict(), os.path.join(output_dir, "gru_prior_3x.pt"))
    gru3_vocab.save(os.path.join(output_dir, "vocabulary_3x.json"))

    gru3_stats, gru3_mws = analyse_molecules(gru3_smiles, "GRU-3x")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON: Corpus vs 8-gram vs GRU(0x) vs GRU(3x)")
    print("=" * 60)

    all_stats = [corpus_stats, ngram_stats, gru0_stats, gru3_stats]
    headers = ['Corpus', '8-gram', 'GRU-0x', 'GRU-3x']

    print(f"\n{'Metric':<25} " + " ".join(f"{h:>10}" for h in headers))
    print("-" * 70)
    for key in ['count', 'MW_mean', 'MW_std', 'MW_median', 'HA_mean',
                'pct_ester', 'pct_ether', 'pct_ketone', 'pct_alkene',
                'pct_alcohol', 'pct_ring', 'pct_aromatic',
                'mean_n_ester', 'mean_n_ether']:
        fmt = ".1f" if 'pct' in key or 'MW' in key or 'HA' in key else ".2f"
        if key == 'count':
            print(f"{key:<25} " + " ".join(f"{s[key]:>10}" for s in all_stats))
        else:
            print(f"{key:<25} " + " ".join(f"{s[key]:>10{fmt}}" for s in all_stats))

    gen_stats_list = [ngram_gen_stats, gru0_gen_stats, gru3_gen_stats]
    gen_headers = ['8-gram', 'GRU-0x', 'GRU-3x']

    print(f"\n{'Generation stats':<25} " + " ".join(f"{h:>10}" for h in gen_headers))
    print("-" * 58)
    print(f"{'Validity rate':<25} " + " ".join(f"{s['validity_rate']:>10.1%}" for s in gen_stats_list))
    print(f"{'Attempts for 5k unique':<25} " + " ".join(f"{s['attempts']:>10}" for s in gen_stats_list))
    print(f"{'Generation time (s)':<25} " + " ".join(f"{s['time_s']:>10.1f}" for s in gen_stats_list))

    # Overlap with corpus
    corpus_set = set(corpus)
    gen_smiles_list = [ngram_smiles, gru0_smiles, gru3_smiles]
    overlaps = [len(set(s) & corpus_set) for s in gen_smiles_list]
    print(f"\n{'Overlap with corpus':<25} " + " ".join(f"{o:>10}" for o in overlaps))
    print(f"{'% novel (not in corpus)':<25} " + " ".join(
        f"{100*(1-o/len(s)):>10.1f}" for o, s in zip(overlaps, gen_smiles_list)))

    # CHO-only rate
    def pct_cho(smiles_list):
        cho = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol and set(a.GetAtomicNum() for a in mol.GetAtoms()) <= {6, 1, 8}:
                cho += 1
        return 100 * cho / len(smiles_list)

    print(f"{'% CHO-only':<25} {'100.0':>10} " + " ".join(
        f"{pct_cho(s):>10.1f}" for s in gen_smiles_list))


if __name__ == "__main__":
    main()
