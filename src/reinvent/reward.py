"""Reward function wrapping the exact WSGA evaluation pipeline.

Critical for fair comparison: uses the same XGBoost models, the same
descriptor computation, the same validity filtering, the same fitness
function, and the same MolPrice penalty as WSGA.
"""

import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

# Ensure src/ is on the path so we can import wsga_helper, etc.
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wsga_helper import (
    evaluate_molecules,
    assign_validity,
    compute_fitness,
    apply_molprice_penalty,
    load_regression_models_with_aux,
    load_tox21_predictor,
    load_biodeg_model,
    compute_mahalanobis_params,
)
from SCScorer import SCScorer
from evaluation import strict_canonicalize_smiles


# Target configuration (must match wsga.py exactly)
TARGET_CONFIG = {
    "alpha":       {"column": "alpha_40",  "maximize": True},
    "FOM1":        {"column": "fom1_40C",  "maximize": True},
    "FOM1_40":     {"column": "FOM1_40",   "maximize": True},
    "FOM1_direct": {"column": "fom1_40C",  "maximize": True},
}


def load_nist8100_corpus(training_data_dir):
    """Collect unique canonical CHO SMILES from NIST 8100 training data.

    Reads all *_cho_cleaned.csv files in {training_data_dir}/nist_8100/
    and returns deduplicated canonical SMILES.
    """
    import glob

    nist_dir = os.path.join(training_data_dir, "nist_8100")
    csv_files = glob.glob(os.path.join(nist_dir, "*_cho_cleaned.csv"))

    all_smiles = set()
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, usecols=["SMILES"])
            for smi in df["SMILES"].dropna():
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                    all_smiles.add(canonical)
        except Exception as e:
            print(f"Warning: could not load {csv_path}: {e}")

    smiles_list = sorted(all_smiles)
    print(f"Loaded {len(smiles_list)} unique CHO SMILES from NIST 8100")
    return smiles_list


class CoolingFluidReward:
    """Reward function using the identical WSGA evaluation pipeline.

    Loads the same XGBoost models, SCScorer, Tox21 predictor, biodeg model,
    and MolPrice model.  Applies the same validity filtering, fitness
    computation, and MolPrice penalty.

    The reward for each molecule is its FitnessScore — the same value that
    WSGA uses for selection.  Invalid or unparseable molecules get reward 0.
    """

    def __init__(
        self,
        model_dir,
        training_data_dir,
        target="FOM1",
        sc_threshold=3,
        mp_threshold=-30,
        bp_threshold=100,
        dc_threshold=7,
        fp_threshold=373,
        tox_threshold=3,
        use_biodeg=True,
        stability_mode=None,
        molprice_soft=0.0,
        molprice_hard=0.0,
    ):
        self.target = target
        self.sc_threshold = sc_threshold
        self.mp_threshold = mp_threshold
        self.bp_threshold = bp_threshold
        self.dc_threshold = dc_threshold
        self.fp_threshold = fp_threshold
        self.tox_threshold = tox_threshold
        self.use_biodeg = use_biodeg
        self.stability_mode = stability_mode
        self.molprice_soft = molprice_soft
        self.molprice_hard = molprice_hard

        # --- Load all models (same as wsga.py) ---
        print("\nLoading models for REINVENT reward...")
        thermo_targets = [
            "bp", "mp", "fp", "dc",
            "density_40C", "viscosity_40C", "tc_40C", "cpsat_40C",
            "beta_40C", "fom1_40C",
        ]
        self.thermo_models = load_regression_models_with_aux(
            thermo_targets, model_dir)

        self.sc_model = SCScorer()
        self.sc_model.restore(os.path.join(
            model_dir,
            "SCScorer/scscore/models/full_reaxys_model_1024bool/"
            "model.ckpt-10654.as_numpy.json.gz",
        ))

        self.tox21_models = load_tox21_predictor(
            os.path.join(model_dir, "tox21_gt4sd"))
        self.biodeg_model = load_biodeg_model(
            os.path.join(model_dir, "biodeg"))

        # MolPrice — always load for logging (same as wsga.py)
        from molprice import MolPriceModel
        mp_path = os.path.join(model_dir, "MolPrice", "MP_Morgan_hybrid.pkl")
        self.molprice_model = None
        if os.path.exists(mp_path):
            self.molprice_model = MolPriceModel(mp_path)
            print(f"Loaded MolPrice model from {mp_path}")
        else:
            print(f"WARNING: MolPrice model not found at {mp_path}")

        # Mahalanobis OOD parameters
        print("Precomputing Mahalanobis OOD parameters...")
        mahal = {}
        for t, data in self.thermo_models.items():
            params = compute_mahalanobis_params(
                t, data["features"], training_data_dir)
            if params is not None:
                mahal[t] = params
        self.mahal_params = mahal if mahal else None
        if self.mahal_params:
            print(f"Mahalanobis ready for {len(self.mahal_params)} models")

        print("REINVENT reward models loaded.\n")

    def evaluate_batch(self, smiles_list):
        """Evaluate a batch of SMILES through the full WSGA pipeline.

        Args:
            smiles_list: list of SMILES strings (may contain invalid entries).

        Returns:
            rewards: np.ndarray of FitnessScore values (0 for invalid/unparseable).
            df: DataFrame with all properties for parseable molecules, or None.
        """
        # Canonicalize and filter unparseable SMILES
        canonical = []
        valid_mask = []
        for smi in smiles_list:
            if not isinstance(smi, str) or not smi:
                valid_mask.append(False)
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                valid_mask.append(False)
                continue
            csmi = strict_canonicalize_smiles(smi)
            if csmi is None:
                valid_mask.append(False)
                continue
            valid_mask.append(True)
            canonical.append(csmi)

        rewards = np.zeros(len(smiles_list))

        if not canonical:
            return rewards, None

        df = pd.DataFrame({"SMILES": canonical})

        # --- Exact WSGA pipeline ---
        df = evaluate_molecules(
            df,
            thermo_models=self.thermo_models,
            sc_model=self.sc_model,
            tox21_models=self.tox21_models,
            biodeg_model=self.biodeg_model,
            molprice_model=self.molprice_model,
            mahal_params=self.mahal_params,
        )

        df = assign_validity(
            df,
            sc_threshold=self.sc_threshold,
            mp_max=self.mp_threshold,
            bp_min=self.bp_threshold,
            dc_max=self.dc_threshold,
            min_fp=self.fp_threshold,
            use_biodeg=self.use_biodeg,
            max_tox21=self.tox_threshold,
            stability_mode=self.stability_mode,
        )

        df = compute_fitness(df, self.target, TARGET_CONFIG)

        df = apply_molprice_penalty(
            df,
            soft_threshold=self.molprice_soft,
            hard_threshold=self.molprice_hard,
        )

        # Map rewards back to original positions
        j = 0
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                rewards[i] = df.iloc[j]["FitnessScore"]
                j += 1

        return rewards, df
