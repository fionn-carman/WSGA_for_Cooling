"""Hash-based deterministic fold assignment.

Same SMILES always maps to the same fold regardless of which property dataset
it appears in, ensuring consistent cross-property analysis.
"""

import hashlib

import numpy as np


def assign_fold(smiles: str, n_folds: int = 5) -> int:
    """Map a single SMILES to a deterministic fold ID."""
    return int(hashlib.md5(smiles.encode()).hexdigest(), 16) % n_folds


def assign_folds(smiles_array, n_folds: int = 5) -> np.ndarray:
    """Vectorised fold assignment for an array of SMILES."""
    return np.array([assign_fold(s, n_folds) for s in smiles_array])


def get_fold_indices(fold_array: np.ndarray, fold_id: int):
    """Return (train_idx, test_idx) for a given fold."""
    test_idx = np.where(fold_array == fold_id)[0]
    train_idx = np.where(fold_array != fold_id)[0]
    return train_idx, test_idx
