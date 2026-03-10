"""
MolPrice wrapper — predicts log(USD/mmol) for molecules using the
MolPrice NumPy neural network model from OptiMaL-PSE-Lab/MolPrice.

Self-contained: embeds the feature extractor logic so we don't need
ray, gin, torch, or the full MolPrice src package.
"""

import os
import pickle
import numpy as np
import joblib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors


# ---------------------------------------------------------------------------
# 2D molecular feature extraction (adapted from MolPrice model_utils.py)
# ---------------------------------------------------------------------------

def _calculate_2d_features(smi):
    """
    Calculate the 10 2D molecular descriptors used by MolPrice.
    Returns np.ndarray of shape (10,).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(10, dtype=np.float32)

    sp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    # SpacialScore may not be available in older RDKit builds
    try:
        from rdkit.Chem import SpacialScore
        sps = SpacialScore.SPS(mol)
    except (ImportError, AttributeError):
        sps = 0.0

    stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    heterocyc = rdMolDescriptors.CalcNumHeterocycles(mol)
    no_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    no_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    # Macrocycles and multi-ring atoms
    ri = mol.GetRingInfo()
    n_atoms = mol.GetNumAtoms()
    n_macrocycles = 0
    multi_ring_atoms = {i: 0 for i in range(n_atoms)}
    for ring_atoms in ri.AtomRings():
        if len(ring_atoms) > 6:
            n_macrocycles += 1
        for atom in ring_atoms:
            multi_ring_atoms[atom] += 1
    n_multi = sum(v - 1 for v in multi_ring_atoms.values() if v > 1)

    return np.array([
        sp3, sps, stereo, rot_bonds, tpsa,
        heterocyc, no_spiro, no_bridgehead, n_macrocycles, n_multi
    ], dtype=np.float32)


class MolPriceModel:
    """
    Wrapper around the MolPrice NumPy neural-network model.

    Predicts log(USD/mmol) from a SMILES string. Uses Morgan fingerprints
    (radius=3, 4096 bits) concatenated with 10 standardised 2D descriptors.

    Args:
        model_path: Path to the .pkl weights file (MP_Morgan_hybrid.pkl)
        scaler_path: Path to directory containing std.bin (StandardScaler
                     for 2D features).  Defaults to same directory as model_path.
    """

    # Default high cost returned for invalid SMILES
    DEFAULT_COST = 10.0

    def __init__(self, model_path, scaler_path=None):
        self.model_path = Path(model_path)
        if scaler_path is None:
            scaler_path = self.model_path.parent
        self.scaler_path = Path(scaler_path)

        # Fingerprint parameters (must match training)
        self.fp_rad = 3
        self.fp_len = 4096
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fp_rad, fpSize=self.fp_len, countSimulation=False
        )

        # Load scaler
        scaler_file = self.scaler_path / "std.bin"
        if not scaler_file.exists():
            raise FileNotFoundError(f"MolPrice scaler not found: {scaler_file}")
        self.scaler = joblib.load(scaler_file)

        # Load neural-network weights
        self._load_weights(self.model_path)

        # Per-SMILES cache
        self._cache = {}

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(self, weights_path):
        with open(weights_path, "rb") as f:
            w = pickle.load(f)

        self.nn_weights = []
        self.nn_biases = []
        for idx in [0, 3, 6, 9]:
            wk = f"neural_network.{idx}.weight"
            bk = f"neural_network.{idx}.bias"
            if wk in w and bk in w:
                self.nn_weights.append(w[wk].T)
                self.nn_biases.append(w[bk])

        self.final_weight = w["linear.weight"].T
        self.final_bias = w["linear.bias"]

    # ------------------------------------------------------------------
    # Featurisation
    # ------------------------------------------------------------------

    def _smi_to_fp(self, smi):
        """SMILES → Morgan FP (4096) + standardised 2D features (10)."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        fp = self.fp_gen.GetFingerprintAsNumPy(mol).astype(np.float32)

        features = _calculate_2d_features(smi)
        features = features.reshape(1, -1)
        features = self.scaler.transform(features).squeeze().astype(np.float32)

        return np.concatenate([fp, features])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, x):
        """Numpy forward pass through the MolPrice NN."""
        z = x
        for i, (weight, bias) in enumerate(zip(self.nn_weights, self.nn_biases)):
            z = z @ weight + bias
            if i < len(self.nn_weights) - 1:
                z = np.maximum(0, z)  # ReLU
        return (z @ self.final_weight + self.final_bias)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, smiles):
        """
        Predict log(USD/mmol) for a single SMILES.

        Returns DEFAULT_COST (10.0) for invalid molecules.
        """
        if smiles in self._cache:
            return self._cache[smiles]

        fp = self._smi_to_fp(smiles)
        if fp is None:
            self._cache[smiles] = self.DEFAULT_COST
            return self.DEFAULT_COST

        score = float(self._forward(fp.reshape(1, -1)).squeeze())
        self._cache[smiles] = score
        return score

    def predict_batch(self, smiles_list):
        """Predict log(USD/mmol) for a list of SMILES."""
        return [self.predict(smi) for smi in smiles_list]
