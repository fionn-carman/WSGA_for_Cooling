import numpy as np
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors as mordred_descriptors

# ── RDKit descriptors (unprefixed — backward compat + MW calc) ──
descriptor_names = [desc[0] for desc in Descriptors._descList]
descriptor_funcs = [desc[1] for desc in Descriptors._descList]


def calc_descriptors(mol, descriptor_funcs):
    return [func(mol) if mol is not None else None for func in descriptor_funcs]


# ── Mordred 2D descriptors ──
mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)
mordred_descriptor_names = [f"mordred_{str(d)}" for d in mordred_calc.descriptors]

# ── RDKit descriptors with rdkit_ prefix (for Mordred-pipeline models) ──
rdkit_prefixed_names = [f"rdkit_{n}" for n in descriptor_names]


def calc_mordred_descriptors(mol):
    """Compute Mordred 2D descriptors. Returns list matching mordred_descriptor_names."""
    if mol is None:
        return [np.nan] * len(mordred_descriptor_names)
    result = mordred_calc(mol)
    values = []
    for v in result:
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(np.nan)
    return values
