import os
import gzip
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import six

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SCScorer:
    score_scale = 5.0
    FP_len = 1024
    FP_rad = 2

    def __init__(self, score_scale=score_scale):
        self.vars = []
        self.score_scale = score_scale
        self._restored = False
        self.fingerprint_generator = GetMorganGenerator(
            radius=self.FP_rad,
            includeChirality=True,
            fpSize=self.FP_len
        )

    def restore(self, weight_path):
        self.FP_len = self.FP_len
        self.FP_rad = self.FP_rad
        self._load_vars(weight_path)

        def mol_to_fp(self, mol):
            if mol is None:
                return np.zeros((self.FP_len,), dtype=np.float32)
            fp = self.fingerprint_generator.GetFingerprint(mol)
            return np.array(fp, dtype=bool)

        self.mol_to_fp = mol_to_fp
        self._restored = True
        return self

    def smi_to_fp(self, smi):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(self, Chem.MolFromSmiles(smi))

    def apply(self, x):
        if not self._restored:
            raise ValueError("Must restore model weights first.")
        for i in range(0, len(self.vars), 2):
            W = self.vars[i]
            b = self.vars[i + 1]
            x = np.matmul(x, W) + b
            if i < len(self.vars) - 2:
                x = x * (x > 0)  # ReLU
        x = 1 + (self.score_scale - 1) * sigmoid(x)
        return x

    def get_score_from_smi(self, smi='', v=False):
        if not smi:
            return ('', 0.0)
        fp = np.array(self.smi_to_fp(smi), dtype=np.float32)
        if sum(fp) == 0:
            if v: print("Could not get fingerprint")
            cur_score = 0.0
        else:
            cur_score = float(self.apply(fp)[0])  # Extract scalar safely
            if v: print(f"Score: {cur_score}")
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ''
        return (smi, cur_score)

    def _load_vars(self, weight_path):
        if weight_path.endswith('json.gz'):
            with gzip.open(weight_path, 'r') as f:
                json_str = f.read().decode('utf-8')
                self.vars = json.loads(json_str)
                self.vars = [np.array(x) for x in self.vars]
        else:
            raise ValueError("Unsupported model format.")