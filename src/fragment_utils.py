import numpy as np
import random
import re
from collections import Counter
from rdkit import Chem
from rdkit.Chem import RWMol, rdmolops

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


# ============================================================
# Bond-Level Crossover (RDKit-based)
# ============================================================

def _get_breakable_bonds(mol):
    """
    Find single, non-ring bonds that can be broken for crossover.
    Returns list of bond indices.
    """
    ring_info = mol.GetRingInfo()
    breakable = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if ring_info.NumBondRings(bond.GetIdx()) > 0:
            continue
        # Skip bonds to hydrogen (shouldn't exist in SMILES but be safe)
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        # Require both sides to have at least 2 heavy atoms
        breakable.append(bond.GetIdx())
    return breakable


def _fragment_on_bond(mol, bond_idx):
    """
    Fragment a molecule on a single bond, returning two fragments as mol objects.
    Uses dummy atoms (*) at the break points.
    Returns (frag1, frag2) or None on failure.
    """
    try:
        frags = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True)
        frag_smiles = Chem.MolToSmiles(frags)
        parts = frag_smiles.split(".")
        if len(parts) != 2:
            return None
        mol1 = Chem.MolFromSmiles(parts[0])
        mol2 = Chem.MolFromSmiles(parts[1])
        if mol1 is None or mol2 is None:
            return None
        return mol1, mol2
    except Exception:
        return None


def _join_fragments(frag1, frag2):
    """
    Join two fragments at their dummy atoms ([*] / atom number 0).
    Returns a sanitised SMILES string or None on failure.
    """
    try:
        combo = Chem.CombineMols(frag1, frag2)
        rw = RWMol(combo)

        # Find dummy atoms (atomic number 0)
        dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) < 2:
            return None

        # Pick one dummy from each original fragment
        # After CombineMols, frag1 atoms come first, frag2 atoms second
        n1 = frag1.GetNumAtoms()
        d1 = [d for d in dummies if d < n1]
        d2 = [d for d in dummies if d >= n1]
        if not d1 or not d2:
            return None

        # Get the neighbor of each dummy (the real atom it's bonded to)
        atom1_idx = d1[0]
        atom2_idx = d2[0]

        neighbors1 = [n.GetIdx() for n in rw.GetAtomWithIdx(atom1_idx).GetNeighbors()]
        neighbors2 = [n.GetIdx() for n in rw.GetAtomWithIdx(atom2_idx).GetNeighbors()]
        if not neighbors1 or not neighbors2:
            return None

        real1 = neighbors1[0]
        real2 = neighbors2[0]

        # Add bond between the real atoms
        rw.AddBond(real1, real2, Chem.BondType.SINGLE)

        # Remove dummy atoms (remove higher index first to preserve indices)
        to_remove = sorted([atom1_idx, atom2_idx], reverse=True)
        for idx in to_remove:
            rw.RemoveAtom(idx)

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)

        # Verify the result is a single connected molecule
        if "." in smi:
            return None

        return smi
    except Exception:
        return None


def crossover_mol_fragments(smi1, smi2, max_heavy_atoms=30, max_attempts=20):
    """
    Bond-level crossover: break one bond in each parent, recombine fragments.

    Steps:
    1. Convert parents to RDKit mol objects
    2. Find breakable bonds (single, non-ring)
    3. Randomly break one bond per parent to get 2 fragments each
    4. Recombine: left of parent1 + right of parent2
    5. Validate and return canonical SMILES

    Returns canonical SMILES of child, or None if crossover fails.
    """
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return None

    bonds1 = _get_breakable_bonds(mol1)
    bonds2 = _get_breakable_bonds(mol2)

    if not bonds1 or not bonds2:
        return None

    for _ in range(max_attempts):
        b1 = random.choice(bonds1)
        b2 = random.choice(bonds2)

        result1 = _fragment_on_bond(mol1, b1)
        result2 = _fragment_on_bond(mol2, b2)
        if result1 is None or result2 is None:
            continue

        frag1_a, frag1_b = result1
        frag2_a, frag2_b = result2

        # Try 4 recombination options
        for fa, fb in [(frag1_a, frag2_b), (frag1_b, frag2_a),
                       (frag2_a, frag1_b), (frag2_b, frag1_a)]:
            child_smi = _join_fragments(fa, fb)
            if child_smi is None:
                continue

            child_mol = Chem.MolFromSmiles(child_smi)
            if child_mol is None:
                continue

            n_heavy = child_mol.GetNumHeavyAtoms()
            if n_heavy > max_heavy_atoms or n_heavy < 2:
                continue

            return Chem.MolToSmiles(child_mol)

    return None

