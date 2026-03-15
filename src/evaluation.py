from rdkit import Chem
from SCScorer import SCScorer

# === Utility functions ===

def canonicalize_smiles(smi):
    """Basic canonicalization."""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def strict_canonicalize_smiles(smi):
    """Strict canonical SMILES (non-isomeric) to enforce uniqueness."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

# Global cache for SCScore
_scscore_cache = {}

def get_scscore_cached(sc_model, smi):
    """Efficient SCScore with caching."""
    if smi in _scscore_cache:
        return _scscore_cache[smi]
    _, score = sc_model.get_score_from_smi(smi)
    _scscore_cache[smi] = score
    return score
