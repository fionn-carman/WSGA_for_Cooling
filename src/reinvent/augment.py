"""SMILES augmentation via randomized enumeration.

Generates non-canonical (randomized) SMILES for the same molecule,
expanding the effective training set for the prior.  Each epoch gets
fresh random SMILES, maximising diversity without pre-computing a
large static dataset.
"""

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def randomize_smiles(smi, n_augmentations=5):
    """Generate n randomized SMILES for a molecule.

    Args:
        smi: canonical SMILES string.
        n_augmentations: number of random SMILES to generate.

    Returns:
        List of randomized SMILES (may include the canonical form).
        Returns [smi] if molecule cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]

    results = set()
    # Always include canonical
    results.add(smi)

    attempts = 0
    max_attempts = n_augmentations * 5  # Allow extra attempts for duplicates
    while len(results) < n_augmentations + 1 and attempts < max_attempts:
        random_smi = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        results.add(random_smi)
        attempts += 1

    return list(results)


def augment_corpus(smiles_list, n_augmentations=None):
    """Augment a SMILES corpus with randomized enumerations.

    Automatically selects augmentation level based on corpus size:
        <50k molecules:   10x augmentation
        50k-200k:          5x augmentation
        >200k:             2x augmentation

    Args:
        smiles_list: list of canonical SMILES.
        n_augmentations: override automatic selection (number of random
                         SMILES per molecule, excluding the canonical).

    Returns:
        List of augmented SMILES (canonical + randomized).
    """
    n = len(smiles_list)

    if n_augmentations is None:
        if n < 50_000:
            n_augmentations = 10
        elif n < 200_000:
            n_augmentations = 5
        else:
            n_augmentations = 2

    print(f"Augmenting {n} SMILES with {n_augmentations}x randomization...")

    augmented = []
    for i, smi in enumerate(smiles_list):
        variants = randomize_smiles(smi, n_augmentations=n_augmentations)
        augmented.extend(variants)

        if (i + 1) % 50_000 == 0:
            print(f"  {i + 1}/{n} molecules augmented "
                  f"({len(augmented)} total SMILES)")

    print(f"Augmentation complete: {n} -> {len(augmented)} SMILES "
          f"({len(augmented) / n:.1f}x)")
    return augmented
