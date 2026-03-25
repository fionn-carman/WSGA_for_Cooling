#!/usr/bin/env python3
"""Curate a C/H/O prior corpus from PubChem CID-SMILES bulk download.

Downloads the PubChem CID-SMILES.gz file (~6-8 GB) and filters to
molecules containing only C, H, O with 5-30 heavy atoms, no charges,
no radicals, no fragments.  Outputs a deduplicated CSV of canonical
SMILES suitable for REINVENT prior pretraining.

Usage:
    python curate_pubchem_prior.py --output data/pubchem_cho_5_30ha.csv

    # Skip download if file already exists:
    python curate_pubchem_prior.py --gz_path CID-SMILES.gz --output data/pubchem_cho_5_30ha.csv

    # Subsample to max N molecules:
    python curate_pubchem_prior.py --max_molecules 500000 --output data/pubchem_cho_5_30ha.csv
"""

import argparse
import gzip
import os
import random
import sys
import time
import urllib.request

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

PUBCHEM_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
ALLOWED_ATOMS = {6, 1, 8}  # C, H, O


def download_pubchem(output_gz, url=PUBCHEM_URL):
    """Download PubChem CID-SMILES.gz with progress reporting."""
    print(f"Downloading {url}")
    print(f"  -> {output_gz}")

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)
        else:
            mb = downloaded / 1e6
            print(f"\r  {mb:.0f} MB downloaded", end="", flush=True)

    urllib.request.urlretrieve(url, output_gz, reporthook=_report)
    print(f"\n  Download complete: {os.path.getsize(output_gz) / 1e9:.2f} GB")


def passes_filter(smi):
    """Check if a SMILES passes the CHO filter.

    Returns:
        (canonical_smiles, None) on success, or (None, reason) on failure.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "parse_fail"

    # Single connected component (no fragments)
    frags = Chem.GetMolFrags(mol)
    if len(frags) > 1:
        return None, "fragments"

    # Only C, H, O
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ALLOWED_ATOMS:
            return None, "elements"

    # Heavy atom count 5-30
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < 5 or n_heavy > 30:
        return None, "heavy_atoms"

    # No charges
    if Chem.GetFormalCharge(mol) != 0:
        return None, "charge"

    # No radicals
    if Descriptors.NumRadicalElectrons(mol) > 0:
        return None, "radicals"

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return canonical, None


def curate(gz_path, max_molecules=None):
    """Stream through CID-SMILES.gz, filter, and deduplicate.

    Returns:
        (smiles_set, stats_dict)
    """
    unique_smiles = set()
    stats = {
        "total_lines": 0,
        "parse_fail": 0,
        "fragments": 0,
        "elements": 0,
        "heavy_atoms": 0,
        "charge": 0,
        "radicals": 0,
        "duplicates": 0,
        "accepted": 0,
    }

    t0 = time.time()
    print(f"Streaming {gz_path}...")

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            stats["total_lines"] = line_num

            # Progress every 5M lines
            if line_num % 5_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  {line_num / 1e6:.0f}M lines, "
                      f"{stats['accepted']} accepted, "
                      f"{len(unique_smiles)} unique "
                      f"({elapsed:.0f}s)")

            parts = line.strip().split("\t")
            if len(parts) < 2:
                stats["parse_fail"] += 1
                continue

            smi = parts[1].strip()
            if not smi:
                stats["parse_fail"] += 1
                continue

            canonical, reason = passes_filter(smi)
            if reason:
                stats[reason] += 1
                continue

            if canonical in unique_smiles:
                stats["duplicates"] += 1
                continue

            unique_smiles.add(canonical)
            stats["accepted"] += 1

            # Early exit if max reached (before subsampling)
            if max_molecules and len(unique_smiles) >= max_molecules * 2:
                print(f"  Reached {len(unique_smiles)} unique molecules, "
                      f"stopping early (2x max_molecules buffer)")
                break

    elapsed = time.time() - t0
    print(f"Done: {stats['total_lines'] / 1e6:.1f}M lines in {elapsed:.0f}s")
    return unique_smiles, stats


def main():
    parser = argparse.ArgumentParser(
        description="Curate PubChem C/H/O corpus for REINVENT prior")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path (single SMILES column)")
    parser.add_argument("--gz_path", type=str, default=None,
                        help="Path to CID-SMILES.gz (downloads if absent)")
    parser.add_argument("--max_molecules", type=int, default=500000,
                        help="Max molecules to keep (random subsample if "
                             "exceeded). Set 0 for no limit.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Download if needed
    gz_path = args.gz_path or "CID-SMILES.gz"
    if not os.path.exists(gz_path):
        download_pubchem(gz_path)

    # Curate
    unique_smiles, stats = curate(gz_path, args.max_molecules or None)

    # Subsample if needed
    smiles_list = sorted(unique_smiles)
    if args.max_molecules and len(smiles_list) > args.max_molecules:
        random.seed(args.seed)
        smiles_list = sorted(random.sample(smiles_list, args.max_molecules))
        print(f"Subsampled to {len(smiles_list)} molecules")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("SMILES\n")
        for smi in smiles_list:
            f.write(smi + "\n")
    print(f"Saved {len(smiles_list)} SMILES to {args.output}")

    # Report
    print(f"\n{'='*50}")
    print("Curation Statistics")
    print(f"{'='*50}")
    print(f"  Total lines parsed:  {stats['total_lines']:>12,}")
    print(f"  Parse failures:      {stats['parse_fail']:>12,}")
    print(f"  Rejected (elements): {stats['elements']:>12,}")
    print(f"  Rejected (heavy atoms): {stats['heavy_atoms']:>12,}")
    print(f"  Rejected (fragments):{stats['fragments']:>12,}")
    print(f"  Rejected (charge):   {stats['charge']:>12,}")
    print(f"  Rejected (radicals): {stats['radicals']:>12,}")
    print(f"  Duplicates removed:  {stats['duplicates']:>12,}")
    print(f"  Accepted (unique):   {stats['accepted']:>12,}")
    print(f"  Final output:        {len(smiles_list):>12,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
