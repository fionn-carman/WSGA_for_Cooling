import numpy as np
import random
import re
from collections import Counter
from rdkit import Chem

def set_avoid_ring(_smiles):
    """
    Identifies ring positions and creates a set of indices to avoid slicing.
    Properly ignores isotope or atom label digits inside square brackets.
    """
    avoid_ring = set()

    # Mask content inside square brackets to avoid counting isotopes/atom indices
    masked_smiles = re.sub(r"\[.*?\]", lambda m: ' ' * len(m.group(0)), _smiles)

    # Find ring digit positions outside of brackets
    ring_digits = re.findall(r'\d', masked_smiles)
    ring_nums = set(ring_digits)

    for num in ring_nums:
        positions = [i for i, val in enumerate(masked_smiles) if val == num]

        if len(positions) % 2 != 0:
            print(f"Warning: Unmatched ring reference detected for ring {num} in SMILES string")
            continue

        for i in range(0, len(positions), 2):
            start = positions[i]
            end = positions[i + 1]
            avoid_ring.update(range(start, end + 1))  # inclusive

    return avoid_ring


def set_avoid_special(_smiles):
    """
    Identifies position of special characters and creates a set of indices to avoid slicing.
    """
    avoid_special = set()
    for i in range(len(_smiles)):
        if _smiles[i] in "/\\=#+-.():[]%":
            avoid_special.add(i)
        # Also skip one before if relevant (e.g., avoid_special cuts just *before* a slash)
        if i < len(_smiles) - 1 and _smiles[i + 1] in "/\\":  
            avoid_special.add(i + 1)
    return avoid_special


def prepare_fragments(_smiles, side, _minimum_len=4, limit_=2000):
    """Creates fragments from parent smiles for crossover.
    :param _smiles: SMILES (str)
    :param side: Left SMILES or Right SMILES ['L'|'R'] (str)
    :param _minimum_len: minimum cut size (int)
    :return:
    """

    # Check correct inout given for side
    if side not in ["L", "R"]:
        raise Exception("You must choice in L(Left) or R(Right)")

    # Check molecule large enough to create two fragments with length > min length
    _smiles_len = len(_smiles)
    if _smiles_len <= 2 * _minimum_len:
        raise ValueError(f"SMILES too short to cut safely: {_smiles}")

    _smi = None

    # Identify ring positions to avoid cutting in rings
    avoid_ring_list = set_avoid_ring(_smiles)
    avoid_special_list = set_avoid_special(_smiles)
    avoid_list = avoid_ring_list.union(avoid_special_list)

    p = 0
    _start = None
    _end = None
    _gate = False

    
    while not _gate:
        if p == limit_:
            raise ValueError(f"main_gate fail ({side}): {_smiles}")

        ring_gate = False
        j = 0
        while not ring_gate:
            if j == 30:
                raise ValueError(f"ring_gate fail ({side}): {_smiles}")

            if side == "L":
                _end = np.random.randint(_minimum_len, _smiles_len - _minimum_len + 1)
                if _end not in avoid_list:
                    _start = 0
                    ring_gate = True
            else:  # side == "R"
                _start = np.random.randint(_minimum_len, _smiles_len - _minimum_len)
                if _start not in avoid_list:
                    _end = _smiles_len
                    ring_gate = True

            j += 1

        _smi = _smiles[_start:_end]

        # Skip fragments with disconnections
        if "." in _smi:
            p += 1
            continue

        # Check bracket balance
        n_chk_sq = 0  # [ ]
        n_chk_br = 0  # ( )
        for char in _smi:
            if char == "[":
                n_chk_sq += 1
            elif char == "]":
                n_chk_sq -= 1
            elif char == "(":
                n_chk_br += 1
            elif char == ")":
                n_chk_br -= 1

        # Check ring closure digit balance
        digits = re.findall(r'\d', _smi)
        digit_counts = Counter(digits)

        rings_balanced = all(count % 2 == 0 for count in digit_counts.values())

        if n_chk_sq != 0 or n_chk_br != 0 or not rings_balanced:
            p += 1
            continue

        # Valid fragment
        _gate = True

    return _smi

def crossover_fragments(smi1, smi2, cut_func, ring_safe=True, max_attempts=30, max_heavy_atoms=30):
    """
    Performs a crossover between two SMILES strings using fragment-based recombination.
    Filters out invalid molecules and those with too many heavy atoms.
    """
    for attempt in range(max_attempts):
        try:
            # Try first direction
            l_smi = cut_func(smi1, "L")
            r_smi = cut_func(smi2, "R")
            new_smi = l_smi + r_smi
            mol = Chem.MolFromSmiles(new_smi)
            if mol and mol.GetNumHeavyAtoms() <= max_heavy_atoms:
                return new_smi

            # Try reversed direction
            l_smi = cut_func(smi2, "L")
            r_smi = cut_func(smi1, "R")
            new_smi = l_smi + r_smi
            mol = Chem.MolFromSmiles(new_smi)
            if mol and mol.GetNumHeavyAtoms() <= max_heavy_atoms:
                return new_smi

        except ValueError:
            continue

    return None

