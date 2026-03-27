#!/usr/bin/env python3
"""
Train GRU prior for WSGA initial population generation.

Curates a unified CHO corpus from all training datasets, augments with
randomised SMILES, and trains a character-level GRU model.

Usage:
    python train_gru_prior.py                          # defaults
    python train_gru_prior.py --output_dir ../models/init_corpus
    python train_gru_prior.py --n_augmentations 5 --epochs 30
"""

import argparse
import os
import sys
import glob
import json
import numpy as np
from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.error')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'reinvent'))

from wsga_helper import has_invalid_fragments
from reinvent.vocabulary import SMILESVocabulary
from reinvent.model import GRUModel
from reinvent.trainer import pretrain
from reinvent.augment import augment_corpus


def curate_corpus(training_data_dir):
    """Collect union of all training SMILES, filtered to CHO + no banned fragments."""
    import pandas as pd

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

    valid = []
    rejected = Counter()
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rejected["parse_fail"] += 1
            continue
        atoms = set(a.GetAtomicNum() for a in mol.GetAtoms())
        if not atoms <= {6, 1, 8}:
            rejected["non_CHO"] += 1
            continue
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 5 or n_heavy > 30:
            rejected["heavy_atoms"] += 1
            continue
        if Chem.GetFormalCharge(mol) != 0:
            rejected["charge"] += 1
            continue
        if Descriptors.NumRadicalElectrons(mol) > 0:
            rejected["radicals"] += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        if has_invalid_fragments(canonical, stability_mode=None):
            rejected["banned_fragments"] += 1
            continue
        valid.append(canonical)

    valid = sorted(set(valid))
    print(f"Filtered corpus: {len(valid)} unique SMILES")
    print(f"Rejected: {dict(rejected)}")
    return valid


def main():
    parser = argparse.ArgumentParser(description="Train GRU prior for WSGA init population")
    parser.add_argument("--training_data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "training", "data"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "models", "init_corpus"))
    parser.add_argument("--n_augmentations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None,
                        help="torch device (auto-detect if omitted)")
    args = parser.parse_args()

    import torch

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # 1. Curate corpus
    print("=" * 60)
    print("Curating unified CHO corpus")
    print("=" * 60)
    corpus = curate_corpus(args.training_data_dir)

    corpus_path = os.path.join(args.output_dir, "corpus.txt")
    with open(corpus_path, "w") as f:
        f.write("\n".join(corpus))
    print(f"Saved corpus ({len(corpus)} SMILES) to {corpus_path}")

    # 2. Augment
    print(f"\n{'=' * 60}")
    print(f"Augmenting corpus ({args.n_augmentations}x)")
    print("=" * 60)
    augmented = augment_corpus(corpus, n_augmentations=args.n_augmentations)

    # 3. Build vocabulary
    vocab = SMILESVocabulary().build(augmented)
    vocab_path = os.path.join(args.output_dir, "vocabulary.json")
    vocab.save(vocab_path)
    print(f"Vocabulary: {len(vocab)} tokens, saved to {vocab_path}")

    # 4. Train GRU
    print(f"\n{'=' * 60}")
    print(f"Training GRU on {device}")
    print("=" * 60)
    model = GRUModel(len(vocab)).to(device)
    model = pretrain(model, vocab, augmented, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr, device=device)

    # 5. Save model
    prior_path = os.path.join(args.output_dir, "gru_prior.pt")
    torch.save(model.state_dict(), prior_path)
    print(f"\nSaved GRU prior to {prior_path}")

    # 6. Quick validation
    print("\nValidation: sampling 1000 molecules...")
    model.eval()
    smiles, _ = model.sample(vocab, batch_size=1000, temperature=1.0)
    n_valid = sum(1 for s in smiles if Chem.MolFromSmiles(s) is not None)
    print(f"Validity: {n_valid}/1000 ({100*n_valid/1000:.1f}%)")

    print(f"\nDone. Files saved to {args.output_dir}/")
    print(f"  corpus.txt       ({len(corpus)} SMILES)")
    print(f"  vocabulary.json  ({len(vocab)} tokens)")
    print(f"  gru_prior.pt     (GRU, {args.n_augmentations}x aug, {args.epochs} epochs)")


if __name__ == "__main__":
    main()
