from rdkit.Chem import Descriptors

def calc_descriptors(mol, descriptor_funcs):
    return [func(mol) if mol is not None else None for func in descriptor_funcs]

descriptor_names = [desc[0] for desc in Descriptors._descList]
descriptor_funcs = [desc[1] for desc in Descriptors._descList]
