"""
Generate N-gram Reference Set for Chemical Space Analysis.

Generates molecules from the 8-gram character-level SMILES model trained
on the combined C/O dataset and saves them to a CSV for reuse across
analyses.  Uses a fixed random seed for reproducibility.

Usage:
    python generate_reference_set.py
    python generate_reference_set.py --n_mol 10000 --seed 42
    python generate_reference_set.py --force          # overwrite existing
"""

import os
import sys
import random
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate n-gram reference molecules and save to CSV"
    )
    parser.add_argument("--n_mol", type=int, default=10000,
                        help="Number of molecules to generate (default: 10000)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory for training SMILES (auto-detected)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: data/ngram_reference_10k.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--ngram_order", type=int, default=8,
                        help="N-gram order (default: 8)")
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
        args.output = os.path.join(repo_root, "data", "ngram_reference_10k.csv")

    # Check if output already exists
    if os.path.exists(args.output) and not args.force:
        print(f"Reference set already exists: {args.output}")
        print(f"  Use --force to regenerate")
        # Print summary of existing file
        import pandas as pd
        df = pd.read_csv(args.output)
        print(f"  Contains {len(df)} molecules")
        return

    # Set seeds BEFORE any generation for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Import generator
    src_dir = os.path.join(repo_root, "src")
    sys.path.insert(0, src_dir)

    from generate_molecules import generate_initial_population, load_combined_training_data

    # Load training data
    print(f"Loading training data from {args.data_dir}...")
    training_smiles = load_combined_training_data(args.data_dir)
    # Sort for deterministic n-gram model construction
    training_smiles = sorted(training_smiles)
    print(f"  {len(training_smiles)} unique C/O training SMILES")

    # Generate molecules
    print(f"\nGenerating {args.n_mol} molecules via {args.ngram_order}-gram model (seed={args.seed})...")
    ref_df = generate_initial_population(
        n=args.n_mol,
        max_heavy_atoms=30,
        min_heavy_atoms=5,
        max_carbons=30,
        max_oxygens=6,
        training_smiles=training_smiles,
        ngram_order=args.ngram_order,
        similarity_threshold=1.0,       # disable diversity filter
        max_attempts_factor=50,          # faster without diversity filter
    )

    smiles_list = ref_df["SMILES"].tolist()
    print(f"  Generated {len(smiles_list)} molecules")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ref_df[["SMILES"]].to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    print(f"  {len(smiles_list)} molecules, seed={args.seed}")


if __name__ == "__main__":
    main()
