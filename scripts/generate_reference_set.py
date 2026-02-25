"""
Generate Reference Set for Chemical Space Analysis.

Two modes:
  - ngram (default): Generate molecules from the 8-gram character-level
    SMILES model trained on the combined C/O dataset.
  - training: Extract the combined training data directly (all unique C/O
    SMILES from the 6 experimental datasets used in this work).

Usage:
    python generate_reference_set.py --mode ngram                  # 10k n-gram molecules
    python generate_reference_set.py --mode training               # combined training data
    python generate_reference_set.py --mode training --force       # overwrite existing
"""

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def generate_ngram_reference(args, repo_root):
    """Generate molecules from 8-gram SMILES model."""
    # Set seeds BEFORE any generation for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    src_dir = os.path.join(repo_root, "src")
    sys.path.insert(0, src_dir)

    from generate_molecules import generate_initial_population, load_combined_training_data

    print(f"Loading training data from {args.data_dir}...")
    training_smiles = load_combined_training_data(args.data_dir)
    training_smiles = sorted(training_smiles)
    print(f"  {len(training_smiles)} unique C/O training SMILES")

    print(f"\nGenerating {args.n_mol} molecules via {args.ngram_order}-gram model (seed={args.seed})...")
    ref_df = generate_initial_population(
        n=args.n_mol,
        max_heavy_atoms=30,
        min_heavy_atoms=5,
        max_carbons=30,
        max_oxygens=6,
        training_smiles=training_smiles,
        ngram_order=args.ngram_order,
        similarity_threshold=1.0,
        max_attempts_factor=50,
    )

    smiles_list = ref_df["SMILES"].tolist()
    print(f"  Generated {len(smiles_list)} molecules")
    return smiles_list


def generate_training_reference(args, repo_root):
    """Extract combined training data as reference set."""
    src_dir = os.path.join(repo_root, "src")
    sys.path.insert(0, src_dir)

    from generate_molecules import load_combined_training_data

    print(f"Loading combined training data from {args.data_dir}...")
    training_smiles = load_combined_training_data(args.data_dir)
    training_smiles = sorted(training_smiles)
    print(f"  {len(training_smiles)} unique C/O training SMILES")
    return training_smiles


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference set for chemical space analysis"
    )
    parser.add_argument("--mode", type=str, default="ngram",
                        choices=["ngram", "training"],
                        help="Reference set type: 'ngram' (generated molecules) "
                             "or 'training' (combined training data) (default: ngram)")
    parser.add_argument("--n_mol", type=int, default=10000,
                        help="Number of molecules to generate in ngram mode (default: 10000)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory for training SMILES (auto-detected)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (auto-set per mode if not given)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for ngram mode (default: 42)")
    parser.add_argument("--ngram_order", type=int, default=8,
                        help="N-gram order for ngram mode (default: 8)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    # Auto-detect directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    if args.data_dir is None:
        args.data_dir = os.path.join(repo_root, "data")
        if not os.path.isdir(args.data_dir):
            print(f"ERROR: Could not find data directory at {args.data_dir}")
            print("       Use --data_dir to specify the path")
            sys.exit(1)

    if args.output is None:
        if args.mode == "ngram":
            args.output = os.path.join(repo_root, "data", "ngram_reference_10k.csv")
        else:
            args.output = os.path.join(repo_root, "data", "training_reference.csv")

    # Check if output already exists
    if os.path.exists(args.output) and not args.force:
        print(f"Reference set already exists: {args.output}")
        print(f"  Use --force to regenerate")
        df = pd.read_csv(args.output)
        print(f"  Contains {len(df)} molecules")
        return

    # Generate reference set
    if args.mode == "ngram":
        smiles_list = generate_ngram_reference(args, repo_root)
    else:
        smiles_list = generate_training_reference(args, repo_root)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pd.DataFrame({"SMILES": smiles_list}).to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    print(f"  {len(smiles_list)} molecules ({args.mode} mode)")


if __name__ == "__main__":
    main()
