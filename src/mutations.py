import random
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
from rdkit.Chem import rdchem
from rdkit.Chem import rdFMCS
import io

rnd = random.choice

# ---- Utility to show difference ----
def view_difference(mol1, mol2, outpath="../outputs/mutation_diff.png"):
    mcs = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

    match1 = mol1.GetSubstructMatch(mcs_mol)
    match2 = mol2.GetSubstructMatch(mcs_mol)

    highlight1 = [atom.GetIdx() for atom in mol1.GetAtoms() if atom.GetIdx() not in match1]
    highlight2 = [atom.GetIdx() for atom in mol2.GetAtoms() if atom.GetIdx() not in match2]

    # This returns a PIL image if useSVG=False
    img = Draw.MolsToGridImage(
        [mol1, mol2],
        highlightAtomLists=[highlight1, highlight2],
        subImgSize=(300, 300),
        useSVG=False,
        returnPNG=False
    )

    try:
        img.show()
        #print(f"Saved mutation difference image to {outpath}")
    except Exception as e:
        print(f"Could not show image: {e}")

# ---- Validity Check Function ----
def MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff, Verbose=False):
    Mut_Mol = StartingMolecule.GetMol()
    MutMolSMILES = Chem.MolToSmiles(Mut_Mol)
    RI = Mut_Mol.GetRingInfo()

    try:
        NumRings = RI.GetNumRings()
    except:
        RI = None

    if RI is not None:
        NumAromaticRings = Chem.rdMolDescriptors.CalcNumAromaticRings(StartingMolecule)
        if NumAromaticRings == 0:
            for op in [
                rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS,
                rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_CLEANUP,
                rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS,
                rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_PROPERTIES,
                rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_SETCONJUGATION,
                rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
                rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_SYMMRINGS,
            ]:
                Chem.SanitizeMol(StartingMolecule, sanitizeOps=op, catchErrors=True)
            print('Non-Aromatic Cyclic Hydrocarbon')
            print(MutMolSMILES)
        else:
            Chem.SanitizeMol(Mut_Mol, catchErrors=True)
    else:
        Chem.SanitizeMol(Mut_Mol, catchErrors=True)

    if len(Chem.GetMolFrags(Mut_Mol)) != 1:
        if Verbose:
            print('Fragmented result, trying new mutation')
        Mut_Mol, MutMolSMILES = None, None
        if showdiff:
            view_difference(StartingMoleculeUnedited, Mut_Mol)
    elif Chem.SanitizeMol(Mut_Mol, catchErrors=True) != rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        if Verbose:
            print('Validity Check Failed')
        Mut_Mol, MutMolSMILES = None, None
    else:
        if Verbose:
            print('Validity Check Passed')
        if showdiff:
            view_difference(StartingMoleculeUnedited, Mut_Mol)

    return Mut_Mol, MutMolSMILES

# ---- Mutation Functions ----

def AddAtom(StartingMolecule, NewAtoms, BondTypes, showdiff=False, Verbose=False):
    """
    Adds a randomly selected new atom to an existing RDKit molecule, forming a new bond
    between the added atom and a randomly chosen existing atom.

    Parameters:
    ----------
    StartingMolecule : rdkit.Chem.Mol
        The original molecule to which a new atom will be added.
        
    NewAtoms : list of int
        A list of atomic numbers (e.g., [6, 7, 8] for C, N, O) to choose from for the new atom.
        
    BondTypes : list of rdkit.Chem.rdchem.BondType
        A list of RDKit bond types (e.g., [Chem.rdchem.BondType.SINGLE]) to choose from
        when forming a bond between the new atom and the existing molecule.
        
    showdiff : bool, optional (default=False)
        If True, will generate a visual comparison between the original and mutated molecules.

    Verbose : bool, optional (default=False)
        If True, prints debug information, especially in case of errors.

    Returns:
    -------
    Mut_Mol : rdkit.Chem.Mol or None
        The mutated molecule with the new atom and bond, or None if the mutation failed.

    MutMolSMILES : str or None
        The SMILES string of the mutated molecule, or None if the mutation failed.

    Procedure (Pseudo-code):
    ------------------------
    1. Make a deep copy of the original molecule to preserve it for comparison.
    2. Convert the original molecule to an editable RWMol (Read-Write Mol).
    3. Randomly select a new atom type from the input list and add it to the molecule.
    4. Randomly select an existing atom in the molecule to attach the new atom to.
       - Skip if there's no atom to attach to (e.g., molecule is empty).
    5. Randomly choose a bond type and create a bond between the new atom and the selected existing atom.
    6. Call MolCheckandPlot to:
        - Sanitize the molecule
        - Optionally visualize the changes
        - Return the mutated molecule and its SMILES string.
    7. If any exception occurs during this process, handle it gracefully and return None values.

    Example usage:
    --------------
    mol = Chem.MolFromSmiles("CC")
    new_mol, new_smiles = AddAtom(mol, NewAtoms=[6, 8], BondTypes=[Chem.rdchem.BondType.SINGLE])
    """

    StartingMoleculeUnedited = deepcopy(StartingMolecule)

    try:
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
        new_atom = Chem.Atom(int(random.choice(NewAtoms)))
        new_idx = StartingMolecule.AddAtom(new_atom)

        all_atoms = list(range(StartingMolecule.GetNumAtoms()))
        all_atoms.remove(new_idx)
        if not all_atoms:
            raise ValueError("No atoms to bond to.")

        attach_idx = random.choice(all_atoms)
        bond_type = random.choice(BondTypes)

        StartingMolecule.AddBond(attach_idx, new_idx, bond_type)

        Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff, Verbose)
    except Exception as e:
        if Verbose:
            print(f'Add atom mutation failed: {e}')
        Mut_Mol, MutMolSMILES = None, None

    return Mut_Mol, MutMolSMILES

def ReplaceAtom(StartingMolecule, NewAtoms, fromAromatic=False, showdiff=False, Verbose=False):
    """
    Randomly replaces a non-aromatic (or optionally aromatic) atom in a molecule
    with another atom from the provided list, excluding the current atom type.

    Parameters:
    ----------
    StartingMolecule : rdkit.Chem.Mol
        The molecule in which an atom will be replaced.

    NewAtoms : list of str or int
        A list of new atomic types to use (as atomic numbers or element symbols).

    fromAromatic : bool, optional (default=False)
        If True, allows replacing atoms from aromatic systems.

    showdiff : bool, optional (default=False)
        If True, displays a visual difference between original and mutated molecules.

    Verbose : bool, optional (default=False)
        If True, prints debug information.

    Returns:
    -------
    Mut_Mol : rdkit.Chem.Mol or None
        The mutated molecule or None if replacement failed.

    MutMolSMILES : str or None
        The SMILES string of the mutated molecule or None if replacement failed.

    Procedure:
    ----------
    1. Identify atoms eligible for replacement based on aromaticity/ring status.
    2. Randomly select one eligible atom.
    3. Choose a new atom from the list that is different from the original.
    4. Replace the atom, preserving its properties.
    5. Sanitize and validate the result.
    6. Return the new molecule and SMILES, or None if mutation failed.
    """

    StartingMoleculeUnedited = deepcopy(StartingMolecule)
    AtomIdxs = []

    # Find replaceable atom indices
    for atom in StartingMoleculeUnedited.GetAtoms():
        if not fromAromatic:
            if atom.GetIsAromatic() or atom.IsInRing():
                continue
        AtomIdxs.append(atom.GetIdx())

    if len(AtomIdxs) == 0:
        if Verbose:
            print('No valid atoms found for replacement.')
        return None, None

    # Choose atom to replace
    ReplaceAtomIdx = random.choice(AtomIdxs)
    ReplaceAtomSymbol = StartingMoleculeUnedited.GetAtomWithIdx(ReplaceAtomIdx).GetSymbol()

    # Ensure new atom is different
    AtomReplacements = [x for x in NewAtoms if (str(x).capitalize() != ReplaceAtomSymbol)]
    if not AtomReplacements:
        if Verbose:
            print('No valid replacement atoms.')
        return None, None

    # Perform replacement
    Replacement = random.choice(AtomReplacements)
    if isinstance(Replacement, str):
        ReplacementAtom = Chem.Atom(Chem.MolFromSmiles(Replacement).GetAtomWithIdx(0).GetAtomicNum())
    else:
        ReplacementAtom = Chem.Atom(int(Replacement))

    StartingMoleculeEditable = Chem.RWMol(StartingMoleculeUnedited)
    StartingMoleculeEditable.ReplaceAtom(ReplaceAtomIdx, ReplacementAtom, preserveProps=True)

    if Verbose:
        print(f"Replaced {ReplaceAtomSymbol} with {ReplacementAtom.GetSymbol()} at index {ReplaceAtomIdx}")

    # Validate and visualize
    try:
        Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMoleculeEditable, showdiff, Verbose)
    except Exception as e:
        if Verbose:
            print(f"Replacement failed during sanitization or visualization: {e}")
        Mut_Mol, MutMolSMILES = None, None

    return Mut_Mol, MutMolSMILES

def RemoveAtom(StartingMolecule, BondTypes, fromAromatic=False, showdiff=True, Verbose=False):
    """
    Function to remove an atom from the molecule, optionally from aromatic rings.

    Args:
    - StartingMolecule: rdkit.Chem.Mol, input molecule
    - BondTypes: list of rdkit.Chem.rdchem.BondType, possible bond types to add when reconnecting neighbors
    - fromAromatic: bool, if True, allow removing atoms from aromatic rings
    - showdiff: bool, whether to show molecule differences
    - Verbose: bool, print debug info

    Returns:
    - Mut_Mol: mutated molecule or None if mutation failed
    - MutMolSMILES: SMILES string of mutated molecule or None
    """
    StartingMoleculeUnedited = deepcopy(StartingMolecule)

    if len(StartingMoleculeUnedited.GetAromaticAtoms()) == len(StartingMoleculeUnedited.GetAtoms()):
        if Verbose:
            print("Starting molecule is completely aromatic")
        return None, None

    StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

    AtomIdxs = [atom.GetIdx() for atom in StartingMolecule.GetAtoms()]
    OneBondAtoms = []
    TwoBondAtoms = []
    AromaticAtoms = []

    for idx in AtomIdxs:
        atom = StartingMolecule.GetAtomWithIdx(idx)
        n_bonds = len(atom.GetBonds())
        if atom.GetIsAromatic() and n_bonds == 2:
            AromaticAtoms.append(idx)
        elif n_bonds == 2:
            TwoBondAtoms.append(idx)
        elif n_bonds == 1:
            OneBondAtoms.append(idx)

    # Choose candidate atoms to remove
    if fromAromatic:
        Candidates = OneBondAtoms + TwoBondAtoms + AromaticAtoms
    else:
        Candidates = OneBondAtoms + TwoBondAtoms

    if not Candidates:
        if Verbose:
            print("No suitable atoms found to remove.")
        return None, None

    # Select atom to remove
    RemoveAtomIdx = random.choice(Candidates)
    RemoveAtomNeighbors = StartingMolecule.GetAtomWithIdx(RemoveAtomIdx).GetNeighbors()
    removed_atom_symbol = StartingMolecule.GetAtomWithIdx(RemoveAtomIdx).GetSymbol()

    if len(RemoveAtomNeighbors) == 1:
        StartingMolecule.RemoveAtom(RemoveAtomIdx)
        if Verbose:
            print(f"Removed atom {removed_atom_symbol} at index {RemoveAtomIdx} with 1 neighbor.")
    elif len(RemoveAtomNeighbors) == 2:
        n1 = RemoveAtomNeighbors[0].GetIdx()
        n2 = RemoveAtomNeighbors[1].GetIdx()
        new_bond_type = random.choice(BondTypes)

        StartingMolecule.RemoveAtom(RemoveAtomIdx)

        try:
            if n1 < StartingMolecule.GetNumAtoms() and n2 < StartingMolecule.GetNumAtoms():
                if StartingMolecule.GetBondBetweenAtoms(n1, n2) is None:
                    StartingMolecule.AddBond(n1, n2, new_bond_type)
                    if Verbose:
                        print(f"Added bond between atoms {n1} and {n2} of type {new_bond_type}")
                else:
                    if Verbose:
                        print(f"Bond already exists between atoms {n1} and {n2}, skipping addition.")
            else:
                if Verbose:
                    print("Invalid neighbor indices after atom removal; skipping bond addition.")
                return None, None
        except Exception as e:
            if Verbose:
                print(f"Exception during bond addition: {e}")
            return None, None
    else:
        if Verbose:
            print(f"Atom {RemoveAtomIdx} has {len(RemoveAtomNeighbors)} neighbors; cannot safely remove.")
        return None, None

    if StartingMoleculeUnedited.GetNumHeavyAtoms() == StartingMolecule.GetNumHeavyAtoms():
        if Verbose:
            print("Atom removal failed: heavy atom count did not decrease.")
        return None, None

    Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff, Verbose=Verbose)
    return Mut_Mol, MutMolSMILES

def ReplaceBond(StartingMolecule, Bonds, showdiff=True, Verbose=False):
    """
    Function to replace a bond in a molecule with a different bond type.

    Args:
    - StartingMolecule: rdkit.Chem.Mol, the input molecule
    - Bonds: list of rdkit.Chem.rdchem.BondType, potential replacement bond types
    - showdiff: bool, whether to display the difference image
    - Verbose: bool, whether to print debug information

    Returns:
    - Mut_Mol: the mutated molecule (or None if invalid)
    - MutMolSMILES: SMILES of mutated molecule (or None if invalid)
    """
    StartingMoleculeUnedited = deepcopy(StartingMolecule)

    BondIdxs = [bond.GetIdx() for bond in StartingMoleculeUnedited.GetBonds() if not bond.GetIsAromatic()]

    if len(BondIdxs) > 0:
        ReplaceBondIdx = random.choice(BondIdxs)
        ReplaceBondType = StartingMoleculeUnedited.GetBondWithIdx(ReplaceBondIdx).GetBondType()
        BondReplacements = [x for x in Bonds if x != ReplaceBondType]

        if len(BondReplacements) == 0:
            if Verbose:
                print('No alternative bond types available.')
            return None, None

        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

        ReplacedBond = StartingMolecule.GetBondWithIdx(ReplaceBondIdx)
        Atom1 = ReplacedBond.GetBeginAtomIdx()
        Atom2 = ReplacedBond.GetEndAtomIdx()

        NewBondType = random.choice(BondReplacements)

        StartingMolecule.RemoveBond(Atom1, Atom2)
        StartingMolecule.AddBond(Atom1, Atom2, NewBondType)

        if Verbose:
            print(f'Replaced bond between atoms {Atom1} and {Atom2} from {ReplaceBondType} to {NewBondType}')

        Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff, Verbose=Verbose)
    else:
        if Verbose:
            print('No non-aromatic bonds to replace.')
        Mut_Mol, MutMolSMILES = None, None

    return Mut_Mol, MutMolSMILES


def AddFragment(StartingMolecule, Fragment, InsertStyle, showdiff=True, Verbose=False):
    """
    Function to add a fragment to a starting molecule.
    
    Args:
    - StartingMolecule: Mol object
    - Fragment: Mol object to be added
    - InsertStyle: 'Within' or 'Edge'
    - showdiff: Whether to visualize differences (passed to MolCheckandPlot)
    - Verbose: Verbosity flag
    """
    StartingMoleculeUnedited = StartingMolecule

    try:
        # Skip if fragment or starting molecule is fully aromatic
        if len(Fragment.GetAromaticAtoms()) == len(Fragment.GetAtoms()):
            if Verbose:
                print('Fragment aromatic, inappropriate function')
            return None, None

        elif len(StartingMoleculeUnedited.GetAromaticAtoms()) == len(StartingMoleculeUnedited.GetAtoms()):
            if Verbose:
                print('Starting molecule is completely aromatic')
            return None, None

        # Merge molecule and fragment
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
        StartingMolecule.InsertMol(Fragment)

        frags = Chem.GetMolFrags(StartingMolecule)
        StartMolIdxs = frags[0]
        FragIdxs = frags[1]

        OneBondAtomsMolecule = []
        TwoBondAtomsMolecule = []
        AromaticAtomsMolecule = []
        OneBondAtomsFragment = []
        TwoBondAtomsFragment = []

        for index in StartMolIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(index)
            if Atom.GetIsAromatic() and len(Atom.GetBonds()) == 2:
                AromaticAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 1:
                OneBondAtomsMolecule.append(index)

        for index in FragIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(index)
            if len(Atom.GetBonds()) == 1:
                OneBondAtomsFragment.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsFragment.append(index)

        if InsertStyle == 'Within':
            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                return None, None

            AtomRmv = rnd(TwoBondAtomsMolecule)
            neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]
            SeverIdx = rnd([0, 1])
            StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

            if SeverIdx == 0:
                StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv - 1, Chem.BondType.SINGLE)
                StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)
            else:
                StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv + 1, Chem.BondType.SINGLE)
                StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

            Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

        elif InsertStyle == 'Edge':
            FragAdd = rnd(['Aromatic', 'Non-Aromatic'])

            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                return None, None

            elif FragAdd == 'Non-Aromatic' or len(AromaticAtomsMolecule) == 0:
                FragAtom = rnd(FragIdxs)
                AtomRmv = rnd(OneBondAtomsMolecule)
                StartingMolecule.AddBond(FragAtom, AtomRmv, Chem.BondType.SINGLE)
                Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            elif len(AromaticAtomsMolecule) > 0:
                ArmtcAtom = rnd(AromaticAtomsMolecule)
                FragAtom = rnd(OneBondAtomsFragment)
                StartingMolecule.AddBond(ArmtcAtom, FragAtom, Chem.BondType.SINGLE)
                Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

        else:
            if Verbose:
                print('Unknown InsertStyle. Returning empty.')
            return None, None

    except Exception as E:
        if Verbose:
            print(E)
            print('Index error, starting molecule probably too short, trying different mutation')
        return None, None

    return Mut_Mol, MutMolSMILES


def RemoveFragment(InputMolecule, BondTypes, showdiff=False, Verbose=False, Visualize=False):

    def try_add_bond(mol, idx1, idx2, bondtype):
        try:
            mol.AddBond(idx1, idx2, bondtype)
            Chem.SanitizeMol(mol)
            return True
        except Exception:
            return False

    try:
        StartingMoleculeUnedited = deepcopy(InputMolecule)
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

        RI = StartingMolecule.GetRingInfo()
        AromaticAtoms = [a.GetIdx() for a in StartingMolecule.GetAromaticAtoms()]
        RemoveRing = rnd([True, False]) if AromaticAtoms else False

        if Verbose:
            print(f"Attempting to remove aromatic ring: {RemoveRing}")

        if RemoveRing:
            try:
                # Tag atoms for tracking
                for atom in StartingMolecule.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIdx())

                ChosenRing = rnd(RI.AtomRings())
                BondIdxs, AtomIdxs = [], []

                for AtomIndex in ChosenRing:
                    Atom = StartingMolecule.GetAtomWithIdx(AtomIndex)
                    for Bond in Atom.GetBonds():
                        if not Bond.IsInRing():
                            BondAtoms = [Bond.GetBeginAtom(), Bond.GetEndAtom()]
                            BondIdxs.append(BondAtoms)
                            AtomIdxs.extend([a.GetIdx() for a in BondAtoms if not a.GetIsAromatic()])

                for B in BondIdxs:
                    StartingMolecule.RemoveBond(B[0].GetIdx(), B[1].GetIdx())

                if len(AtomIdxs) > 1:
                    # Try to add bond with random bond type, fallback to SINGLE if fails
                    bond_type = rnd(BondTypes)
                    success = try_add_bond(StartingMolecule, AtomIdxs[0], AtomIdxs[1], bond_type)
                    if not success:
                        if Verbose:
                            print(f"Failed to add bond {AtomIdxs[0]}-{AtomIdxs[1]} as {bond_type}, trying SINGLE bond")
                        success = try_add_bond(StartingMolecule, AtomIdxs[0], AtomIdxs[1], Chem.BondType.SINGLE)
                    if not success:
                        if Verbose:
                            print("Failed to add bond after fallback; aborting mutation.")
                        return None, None

                elif len(AtomIdxs) == 1:
                    if Verbose:
                        print("Aromatic ring was terminal — removing without reconnection.")
                else:
                    if Verbose:
                        print("Not enough atoms after aromatic ring removal.")
                    return None, None

                mol_frags = Chem.GetMolFrags(StartingMolecule, asMols=True, sanitizeFrags=True)
                if not mol_frags:
                    return None, None

                ring_atom_set = set(ChosenRing)
                for frag in mol_frags:
                    frag_atom_map_nums = [a.GetAtomMapNum() for a in frag.GetAtoms()]
                    if not any(a in ring_atom_set for a in frag_atom_map_nums):
                        StartingMolecule = Chem.RWMol(frag)
                        for atom in StartingMolecule.GetAtoms():
                            atom.SetAtomMapNum(0)
                        return MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

                if Verbose:
                    print("All fragments contained ring atoms; nothing to return.")
                return None, None

            except Exception as E:
                if Verbose:
                    print(f"Remove Aromatic ring failed: {E}")
                return None, None

        else:
            StartingMoleculeAtoms = StartingMolecule.GetAtoms()
            AtomIdxs = [x.GetIdx() for x in StartingMoleculeAtoms if x.GetIdx() not in AromaticAtoms]

            FinalAtomIdxs = AtomIdxs

            SeveringBonds = []
            used_bond_idxs = set()
            attempts, max_attempts = 0, 10
            while len(SeveringBonds) < 2 and attempts < max_attempts:
                atom_idx = rnd(FinalAtomIdxs)
                atom = StartingMolecule.GetAtomWithIdx(atom_idx)
                valid_bonds = [
                    b for b in atom.GetBonds()
                    if not b.IsInRing()
                    and b.GetIdx() not in used_bond_idxs
                    and not b.GetBeginAtom().GetIsAromatic()
                    and not b.GetEndAtom().GetIsAromatic()
                ]
                random.shuffle(valid_bonds)
                if valid_bonds:
                    bond = valid_bonds[0]
                    SeveringBonds.append(bond)
                    used_bond_idxs.add(bond.GetIdx())
                    if Verbose:
                        print(f"Selected bond: {bond.GetBeginAtomIdx()}–{bond.GetEndAtomIdx()} "
                              f"({bond.GetBeginAtom().GetSymbol()}–{bond.GetEndAtom().GetSymbol()})")
                attempts += 1

            if len(SeveringBonds) < 2:
                if Verbose:
                    print("Not enough bonds found to sever (non-aromatic only).")
                return None, None

            for bond in SeveringBonds:
                StartingMolecule.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

            frags = Chem.GetMolFrags(StartingMolecule, asMols=False, sanitizeFrags=True)

            if Visualize:
                frag_mols = Chem.GetMolFrags(StartingMolecule, asMols=True, sanitizeFrags=True)
                for i, mol in enumerate(frag_mols):
                    print(f"Fragment {i + 1}: {Chem.MolToSmiles(mol)}")

            if len(frags) == 3:
                sizes = [len(x) for x in frags]
                sorted_frags = sorted(zip(sizes, frags), key=lambda x: x[0])
                mid_frag = sorted_frags[0][1]
                if 1 <= len(mid_frag) <= 0.4 * InputMolecule.GetNumAtoms():
                    f1 = sorted_frags[1][1]
                    f2 = sorted_frags[2][1]
                    f1_atoms = [x for x in f1 if x in FinalAtomIdxs]
                    f2_atoms = [x for x in f2 if x in FinalAtomIdxs]

                    if not f1_atoms:
                        f1_atoms = list(f1)
                    if not f2_atoms:
                        f2_atoms = list(f2)

                    if f1_atoms and f2_atoms:
                        bond_type = rnd(BondTypes)
                        success = try_add_bond(StartingMolecule, f1_atoms[-1], f2_atoms[0], bond_type)
                        if not success:
                            if Verbose:
                                print(f"Failed to add bond {f1_atoms[-1]}-{f2_atoms[0]} as {bond_type}, trying SINGLE bond")
                            success = try_add_bond(StartingMolecule, f1_atoms[-1], f2_atoms[0], Chem.BondType.SINGLE)
                        if not success:
                            if Verbose:
                                print("Failed to add bond after fallback; aborting mutation.")
                            return None, None

                        mol_frags = Chem.GetMolFrags(StartingMolecule, asMols=True)
                        if mol_frags and any(mol_frags):
                            largest_frag = max(mol_frags, key=lambda m: m.GetNumAtoms())
                            if largest_frag is not None:
                                StartingMolecule = Chem.RWMol(largest_frag)
                            else:
                                raise ValueError("Largest fragment is None.")
                        else:
                            raise ValueError("No valid fragments found after mutation.")
                        return MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)
                    else:
                        raise ValueError("No valid atoms to reconnect (even after fallback).")
            elif Verbose:
                print("Remove fragment failed, returning empty objects - not 3 frags")
            return None, None

    except Exception as E:
        if Verbose:
            print(f"Remove fragment failed: {E}")
        return None, None

def InsertAromatic(StartingMolecule, AromaticMolecule, InsertStyle='Within', showdiff=False, Verbose=False):
    """
    Function to insert an aromatic atom into a starting molecule
    4. If inserting aromatic ring:
    - Check if fragment is aromatic
    - Combine starting molecule and fragment into single disconnected Mol object 
    - Split atom indexes of starting molecule and fragment and save in different objects
    - Check number of bonds each atom has in starting molecule and fragment, exclude any atoms that don't have 
    exactly two bonds
    - Randomly select one of 2-bonded atoms and store the index of the Atom, and store bond objects of atom's bonds
    - Get the atom neighbors of selected atom 
    - Remove selected atom 
    - Select two unique atoms from cyclic atom
    - Create a new bond between each of the neighbors of the removed atom and each of the terminal atoms on the 
    fragment 
    """

    

    StartingMoleculeUnedited = deepcopy(StartingMolecule)
    Fragment = AromaticMolecule

    #Always check if fragment or starting molecule is a cyclic (benzene) molecule

    try:
        if len(Fragment.GetAromaticAtoms()) != len(Fragment.GetAtoms()):
            if Verbose:
                print('Fragment is not cyclic')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
        
        elif len((StartingMoleculeUnedited.GetAromaticAtoms())) == len(StartingMoleculeUnedited.GetAtoms()):
            if Verbose:
                print('Starting molecule is completely aromatic')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

        else:
            Chem.SanitizeMol(Fragment)

            # Add fragment to Mol object of starting molecule
            StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
            StartingMolecule.InsertMol(Fragment)

            # Get indexes of starting molecule and fragment, store them in separate objects 
            frags = Chem.GetMolFrags(StartingMolecule)
            StartMolIdxs = frags[0]
            FragIdxs = frags[1]

            OneBondAtomsMolecule = []
            TwoBondAtomsMolecule = []
            AromaticAtomsMolecule = []

            # Getting atoms in starting molecule with different amount of bonds, storing indexes in lists
            for index in StartMolIdxs:
                Atom = StartingMolecule.GetAtomWithIdx(int(index))
                if Atom.GetIsAromatic():
                    AromaticAtomsMolecule.append(index) 
                if len(Atom.GetBonds()) == 2:
                    TwoBondAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 1:
                    OneBondAtomsMolecule.append(index)
                else:
                    continue

            if InsertStyle == 'Within':
                # Randomly choose two unique atoms from aromatic molecule
                ArmtcAtoms = random.sample(FragIdxs, 2)

                # Select random two bonded atom
                AtomRmv = rnd(TwoBondAtomsMolecule)

                # Get atom neighbor indexes, remove bonds between selected atom and neighbors 
                neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]

                # Randomly choose which bond of target atom to sever
                SeverIdx = rnd([0,1])

                # Sever the selected bond
                StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

                #For situation where bond before target atom is severed
                if SeverIdx == 0:
                    StartingMolecule.AddBond(ArmtcAtoms[0], AtomRmv - 1, Chem.BondType.SINGLE)
                    StartingMolecule.AddBond(ArmtcAtoms[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, 
                                                                                StartingMolecule, 
                                                                                showdiff)

                #For situation where bond after target atom is severed
                else:
                    StartingMolecule.AddBond(ArmtcAtoms[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                    StartingMolecule.AddBond(ArmtcAtoms[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, 
                                                                                StartingMolecule, 
                                                                                showdiff)

            elif InsertStyle == 'Edge':

                if len(OneBondAtomsMolecule) == 0:
                    if Verbose:
                        print('No one-bonded terminal atoms in starting molecule, returning empty object')
                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
                
                else:
                    # Randomly choose two unique atoms from aromatic molecule
                    ArmtcAtoms = rnd(FragIdxs)

                    # Select random one bonded atom
                    AtomRmv = rnd(OneBondAtomsMolecule)

                    StartingMolecule.AddBond(ArmtcAtoms, AtomRmv, Chem.BondType.SINGLE) 

                    Mut_Mol,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, 
                                                                                StartingMolecule, 
                                                                                showdiff)
                
            else:
                if Verbose:
                    print('Edge case, returning empty objects')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    except Exception as e:
        if Verbose:
            print(f'[InsertAromatic ERROR] {e}')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    return Mut_Mol, MutMolSMILES


def Napthalenate(StartingMolecule, Napthalene, InsertStyle, showdiff=True, Verbose=False):
    """
    Function to add a fragment from a selected list of fragments to starting molecule.
    
    Args:
    - StartingMolecule: SMILES String of Starting Molecule
    - Fragments: List of fragments 
    - showdiff
    - InsertStyle: Whether to add fragment to edge or within molecule
    """
    """
    Steps:
    2. Determine whether fragment is going to be inserted within or appended to end of molecule
    
    3. If inserting fragment within molecule:
        - Combine starting molecule and fragment into single disconnected Mol object 
        - Split atom indexes of starting molecule and fragment and save in different objects
        - Check number of bonds each atom has in starting molecule and fragment, exclude any atoms that don't have 
        exactly two bonds
        - Get the atom neighbors of selected atom 
        - Remove one of selected atom's bonds with its neighbors, select which randomly but store which bond is severed 
        - Calculate terminal atoms of fragment (atoms with only one bond, will be at least two, return total number of
        fragments, store terminal atom indexes in a list)
        - Randomly select two terminal atoms from terminal atoms list of fragment
        - Create a new bond between each of the neighbors of the target atom and each of the terminal atoms on the 
        fragment, specify which type of bond, with highest likelihood that it is a single bond
        """
    
    StartingMoleculeUnedited = StartingMolecule

    try:
        if isinstance(StartingMoleculeUnedited, str):
            StartingMoleculeUnedited = Chem.MolFromSmiles(StartingMolecule)
        if isinstance(Napthalene, str):
            Napthalene = Chem.MolFromSmiles(Napthalene)

        SMRings = StartingMoleculeUnedited.GetRingInfo() #Starting molecule rings

        if SMRings.NumRings() > 0:
            if Verbose:
                print('Starting molecule has rings, abandoning mutations')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

        else:
            # Add fragment to Mol object of starting molecule
            StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
            StartingMolecule.InsertMol(Napthalene)
            Chem.SanitizeMol(StartingMolecule, catchErrors=True)

            # Get indexes of starting molecule and fragment, store them in separate objects 
            frags = Chem.GetMolFrags(StartingMolecule)
            StartMolIdxs = frags[0]
            FragIdxs = frags[1]

            OneBondAtomsMolecule = []
            TwoBondAtomsMolecule = []
            AromaticAtomsMolecule = []
            OneBondAtomsFragment = []
            TwoBondAtomsFragment = []

            # Getting atoms in starting molecule with different amount of bonds, storing indexes in list
            for index in StartMolIdxs:
                Atom = StartingMolecule.GetAtomWithIdx(int(index))
                if Atom.GetIsAromatic() and len(Atom.GetBonds()) == 2:
                    AromaticAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 2:
                    TwoBondAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 1:
                    OneBondAtomsMolecule.append(index)
                else:
                    continue

            # Getting atoms in fragment with varying bonds, storing indexes in list
            for index in FragIdxs:
                Atom = StartingMolecule.GetAtomWithIdx(int(index))
                if len(Atom.GetBonds()) == 1:
                    OneBondAtomsFragment.append(index)
                elif len(Atom.GetBonds()) == 2:
                    TwoBondAtomsFragment.append(index)

            ### INSERTING FRAGMENT AT RANDOM POINT WITHIN STARTING MOLECULE
            if InsertStyle == 'Within':
                # Select random two bonded atom, not including aromatic
                AtomRmv = rnd(TwoBondAtomsMolecule)

                # Get atom neighbor indexes, remove bonds between selected atom and neighbors 
                neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]

                # Randomly choose which bond of target atom to sever
                SeverIdx = rnd([0,1])

                StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

                #Create a bond between the fragment and the now segmented fragments of the starting molecule
                #For situation where bond before target atom is severed
                if SeverIdx == 0:

                    StartingMolecule.AddBond(TwoBondAtomsFragment[0], AtomRmv - 1, Chem.BondType.SINGLE)
                    StartingMolecule.AddBond(TwoBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

                #For situation where bond after target atom is severed
                elif SeverIdx != 0:
                    StartingMolecule.AddBond(TwoBondAtomsFragment[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                    StartingMolecule.AddBond(TwoBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            ### INSERTING FRAGMENT AT END OF STARTING MOLECULE

            elif InsertStyle == 'Edge':
                # Choose whether fragment will be added to an aromatic or non-aromatic bond
                FragAdd = rnd(['Aromatic', 'Non-Aromatic'])

                if len(OneBondAtomsMolecule) == 0:
                    if Verbose:
                        print('No one-bonded terminal atoms in starting molecule, returning empty object')
                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

                elif FragAdd == 'Non-Aromatic' or len(AromaticAtomsMolecule) == 0: 
                    #Randomly choose atom from fragment (including aromatic)
                    FragAtom = rnd(FragIdxs)

                    #Select random terminal from starting molecule
                    AtomRmv = rnd(OneBondAtomsMolecule)

                    #Attach fragment to end of molecule
                    StartingMolecule.AddBond(FragAtom, AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            else:
                if Verbose:
                    print('Edge case, returning empty objects')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
    except Exception as E:
        if Verbose:
            print(E)
            print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
      
    return Mut_Mol, MutMolSMILES

def Esterify(StartingMolecule, InsertStyle, showdiff=True, Verbose=False):
    StartingMoleculeUnedited = StartingMolecule
    EsterSMILES = 'CC(=O)OC'
    EsterMol = Chem.MolFromSmiles(EsterSMILES)
    
    if isinstance(StartingMoleculeUnedited, str):
        StartingMoleculeUnedited = Chem.MolFromSmiles(StartingMolecule)

    try:
        # Add fragment to Mol object of starting molecule
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
        StartingMolecule.InsertMol(EsterMol)

        # Get indexes of starting molecule and fragment, store them in separate objects 
        frags = Chem.GetMolFrags(StartingMolecule)
        StartMolIdxs = frags[0]
        FragIdxs = frags[1]

        OneBondAtomsMolecule = []
        TwoBondAtomsMolecule = []
        AromaticAtomsMolecule = []
        OneBondAtomsFragment = []
        TwoBondAtomsFragment = []

        # Getting atoms in starting molecule with different amount of bonds, storing indexes in list
        for index in StartMolIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(int(index))
            if Atom.GetIsAromatic() and len(Atom.GetBonds()) == 2:
                AromaticAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 1:
                OneBondAtomsMolecule.append(index)
            else:
                continue

        # Getting atoms in fragment with varying bonds, storing indexes in listv
        for index in FragIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(int(index))
            if len(Atom.GetBonds()) == 1:
                if Atom.GetAtomicNum() == 6:
                    OneBondAtomsFragment.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsFragment.append(index)
        
        ### INSERTING FRAGMENT AT RANDOM POINT WITHIN STARTING MOLECULE
        if InsertStyle == 'Within':

            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                Mut_Mol = None
                Mut_Mol_Sanitized = None
                MutMolSMILES = None
            
            else:
                # Select random two bonded atom, not including aromatic
                AtomRmv = rnd(TwoBondAtomsMolecule)

                # Get atom neighbor indexes, remove bonds between selected atom and neighbors 
                neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]

                # Randomly choose which bond of target atom to sever
                SeverIdx = rnd([0,1])

                StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

                #Create a bond between the fragment and the now segmented fragments of the starting molecule
                #For situation where bond before target atom is severed
                if SeverIdx == 0:

                    StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv - 1, Chem.BondType.SINGLE)
                    StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

                #For situation where bond after target atom is severed
                elif SeverIdx != 0:
                    StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                    StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

        ### INSERTING FRAGMENT AT END OF STARTING MOLECULE

        elif InsertStyle == 'Edge':
            # Choose whether fragment will be added to an aromatic or non-aromatic bond
            FragAdd = rnd(['Aromatic', 'Non-Aromatic'])

            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

            elif FragAdd == 'Non-Aromatic' or len(AromaticAtomsMolecule) == 0: 
                #Randomly choose atom from fragment (including aromatic)
                FragAtom = rnd(FragIdxs)

                #Select random terminal from starting molecule
                AtomRmv = rnd(OneBondAtomsMolecule)

                #Attach fragment to end of molecule
                StartingMolecule.AddBond(FragAtom, AtomRmv, Chem.BondType.SINGLE)

                Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            elif len(AromaticAtomsMolecule) > 0:
                #Randomly select 2 bonded (non-branch base) aromatic atom 
                ArmtcAtom = rnd(AromaticAtomsMolecule)

                #Randomly select terminal atom from fragment
                FragAtom = rnd(OneBondAtomsFragment)

                #Add Bond
                StartingMolecule.AddBond(ArmtcAtom, FragAtom, Chem.BondType.SINGLE)

                Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)
                
        else:
            if Verbose:
                print('Edge case, returning empty objects')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
    except Exception as E:
        if Verbose:
            print(E)
            print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
      
    return Mut_Mol, MutMolSMILES

def Generate_Glycol_Chain(ChainLength):
    """Generate ethylene glycol (PEG) chain: -O-CH2CH2-O-CH2CH2-O- pattern.

    ChainLength=1 -> COCC        (one -O-CH2CH2- unit)
    ChainLength=2 -> COCCOCC     (two units)
    ChainLength=3 -> COCCOCCOC   (three units, capped)
    """
    if ChainLength < 1:
        raise ValueError("Repeat count must be at least 1")
    return "COCC" + "OCC" * (ChainLength - 1)

def Glycolate(StartingMolecule, InsertStyle, showdiff=True, Verbose=False):
    StartingMoleculeUnedited = StartingMolecule
    GlycolSMILES = Generate_Glycol_Chain(rnd([1,2,3]))
    GlycolMol = Chem.MolFromSmiles(GlycolSMILES)

    if isinstance(StartingMoleculeUnedited, str):
        StartingMoleculeUnedited = Chem.MolFromSmiles(StartingMolecule)

    try:
        # Add fragment to Mol object of starting molecule
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
        StartingMolecule.InsertMol(GlycolMol)

        # Get indexes of starting molecule and fragment, store them in separate objects 
        frags = Chem.GetMolFrags(StartingMolecule)
        StartMolIdxs = frags[0]
        FragIdxs = frags[1]

        OneBondAtomsMolecule = []
        TwoBondAtomsMolecule = []
        AromaticAtomsMolecule = []
        OneBondAtomsFragment = []
        TwoBondAtomsFragment = []

        # Getting atoms in starting molecule with different amount of bonds, storing indexes in list
        for index in StartMolIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(int(index))
            if Atom.GetIsAromatic() and len(Atom.GetBonds()) == 2:
                AromaticAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsMolecule.append(index)
            elif len(Atom.GetBonds()) == 1:
                OneBondAtomsMolecule.append(index)
            else:
                continue

        # Getting atoms in fragment with varying bonds, storing indexes in listv
        for index in FragIdxs:
            Atom = StartingMolecule.GetAtomWithIdx(int(index))
            if len(Atom.GetBonds()) == 1:
                if Atom.GetAtomicNum() == 6:
                    OneBondAtomsFragment.append(index)
            elif len(Atom.GetBonds()) == 2:
                TwoBondAtomsFragment.append(index)
        
        ### INSERTING FRAGMENT AT RANDOM POINT WITHIN STARTING MOLECULE
        if InsertStyle == 'Within':

            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                Mut_Mol = None
                Mut_Mol_Sanitized = None
                MutMolSMILES = None
            
            else:
                # Select random two bonded atom, not including aromatic
                AtomRmv = rnd(TwoBondAtomsMolecule)

                # Get atom neighbor indexes, remove bonds between selected atom and neighbors 
                neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]

                # Randomly choose which bond of target atom to sever
                SeverIdx = rnd([0,1])

                StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

                #Create a bond between the fragment and the now segmented fragments of the starting molecule
                #For situation where bond before target atom is severed
                if SeverIdx == 0:

                    StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv - 1, Chem.BondType.SINGLE)
                    StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

                #For situation where bond after target atom is severed
                elif SeverIdx != 0:
                    StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                    StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

        ### INSERTING FRAGMENT AT END OF STARTING MOLECULE

        elif InsertStyle == 'Edge':
            # Choose whether fragment will be added to an aromatic or non-aromatic bond
            FragAdd = rnd(['Aromatic', 'Non-Aromatic'])

            if len(OneBondAtomsMolecule) == 0:
                if Verbose:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

            elif FragAdd == 'Non-Aromatic' or len(AromaticAtomsMolecule) == 0: 
                #Randomly choose atom from fragment (including aromatic)
                FragAtom = rnd(FragIdxs)

                #Select random terminal from starting molecule
                AtomRmv = rnd(OneBondAtomsMolecule)

                #Attach fragment to end of molecule
                StartingMolecule.AddBond(FragAtom, AtomRmv, Chem.BondType.SINGLE)

                Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            elif len(AromaticAtomsMolecule) > 0:
                #Randomly select 2 bonded (non-branch base) aromatic atom 
                ArmtcAtom = rnd(AromaticAtomsMolecule)

                #Randomly select terminal atom from fragment
                FragAtom = rnd(OneBondAtomsFragment)

                #Add Bond
                StartingMolecule.AddBond(ArmtcAtom, FragAtom, Chem.BondType.SINGLE)

                Mut_Mol, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)
                
        else:
            if Verbose:
                print('Edge case, returning empty objects')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
    except Exception as E:
        if Verbose:
            print(E)
            print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
      
    return Mut_Mol, MutMolSMILES



def mutate(mol, selected_mutation_name, NewAtoms, BondTypes, Fragment, AromaticMolecule, Napthalenes, Showdiff, Verbose_check):
    
    if selected_mutation_name == "AddAtom":
        mutated_mol, mutated_smiles = AddAtom(mol, NewAtoms, BondTypes, showdiff=Showdiff, Verbose=Verbose_check)

    elif selected_mutation_name == "ReplaceAtom":
        mutated_mol, mutated_smiles = ReplaceAtom(mol, NewAtoms, fromAromatic=False, showdiff=Showdiff, Verbose=Verbose_check)

    elif selected_mutation_name == "RemoveAtom":
        mutated_mol, mutated_smiles = RemoveAtom(mol, BondTypes, fromAromatic=False, showdiff=Showdiff, Verbose=Verbose_check)

    elif selected_mutation_name == "ReplaceBond":
        mutated_mol, mutated_smiles = ReplaceBond(mol, BondTypes, showdiff=Showdiff)

    elif selected_mutation_name == "AddFragment":
        mutated_mol, mutated_smiles = AddFragment(mol, rnd(Fragment), InsertStyle=rnd(['Within', 'Edge']), showdiff=Showdiff, Verbose=Verbose_check)

    elif selected_mutation_name == "RemoveFragment":
        mutated_mol, mutated_smiles = RemoveFragment(mol, BondTypes, showdiff=Showdiff, Verbose=Verbose_check)

    elif selected_mutation_name == "InsertAromatic":
        mutated_mol, mutated_smiles = InsertAromatic(mol, AromaticMolecule, InsertStyle=rnd(['Within', 'Edge']), showdiff=Showdiff, Verbose=Verbose_check)

    elif selected_mutation_name == 'Napthalenate':
        mutated_mol, mutated_smiles = Napthalenate(mol, rnd(Napthalenes), InsertStyle=rnd(['Within', 'Edge']), showdiff=Showdiff, Verbose=Verbose_check)

    elif selected_mutation_name == 'Glycolate':
        mutated_mol, mutated_smiles = Glycolate(mol, InsertStyle=rnd(['Within', 'Edge']), showdiff=Showdiff, Verbose=Verbose_check)

    elif selected_mutation_name == 'Esterify':
        mutated_mol, mutated_smiles = Esterify(mol, InsertStyle=rnd(['Within', 'Edge']), showdiff=Showdiff, Verbose=Verbose_check)

    else:
        print("Mutation function not yet fully integrated.")
        return None, None

    return mutated_mol, mutated_smiles

# ---- Available Mutation Names ----
mutation_names = [
    "AddAtom",
    "ReplaceAtom",
    "RemoveAtom",
    "ReplaceBond",
    "AddFragment",
    "RemoveFragment",
    "InsertAromatic",
    "Napthalenate",
    "Glycolate",
    "Esterify"
]

# ---- Main Mutation Execution ----
if __name__ == "__main__":
    print("Initializing Mutation System...\n")

    # --- Input and Setup ---
    smiles = input("Enter starting SMILES (default: CC(O)Cc1ccccc1): ").strip() or "c1ccccc1CC(O)CCCCCCCc1ccccc1CCCCCCCC"
    mol = Chem.MolFromSmiles(smiles)

    Atoms = ['C', 'O']
    AtomMolObjects = [Chem.MolFromSmiles(x) for x in Atoms]
    AtomicNumbers = [atom.GetAtomicNum() for mol in AtomMolObjects for atom in mol.GetAtoms()]
    NewAtoms = AtomicNumbers

    BondTypes = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE]

    fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 
                 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C', 'CC(=O)OCC(CC=O)OCOCO']
    fragments = [Chem.MolFromSmiles(x) for x in fragments]

    Napthalenes = ['C12=CC=CC=C2C=CC=C1', 'C1CCCC2=CC=CC=C12', 'C1CCCC2CCCCC12', 'C1CCC2=CC=CC=C12']
    Napthalenes = [Chem.MolFromSmiles(x) for x in Napthalenes]

    AromaticMolecule = Chem.MolFromSmiles('c1ccccc1')


    print("\nOriginal SMILES:", smiles)

    # --- Mutation Menu ---
    print("\nAvailable Mutations:")
    for idx, name in enumerate(mutation_names, start=1):
        print(f"[{idx}] {name}")

    try:
        selection = int(input("\nChoose a mutation to perform (number): ").strip())
        selected_mutation_name = mutation_names[selection - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        exit()

    print(f"\n--- Running {selected_mutation_name} Mutation ---")

    # --- Execute via mutate() ---
    mutated_mol, mutated_smiles = mutate(
        mol=mol,
        selected_mutation_name=selected_mutation_name,
        NewAtoms=NewAtoms,
        BondTypes=BondTypes,
        Fragment=fragments,
        AromaticMolecule=AromaticMolecule,
        Napthalenes=Napthalenes,
        Showdiff=True,
        Verbose_check=True
    )

    # --- Output ---
    if mutated_mol is not None:
        print("Mutated SMILES:", mutated_smiles)
    else:
        print("Mutation failed.")


