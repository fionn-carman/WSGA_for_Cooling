"""REINVENT reward wrapping the lubricant WSGA evaluation pipeline.

Mirrors src/reinvent/reward.py but uses lube_wsga_helper + lube_fitness so
the REINVENT agent is optimising the same FOM_LUBE fitness that the GA uses.
"""

import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Add LubeOil/src on the path so we can import lube modules.
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from lube_wsga_helper import (
    evaluate_molecules,
    assign_validity,
    compute_fitness,
    apply_molprice_penalty,
    load_regression_models_with_aux,
    load_tox21_predictor,
    load_biodeg_model,
    compute_mahalanobis_params,
    has_invalid_fragments,
)
from SCScorer import SCScorer
from evaluation import strict_canonicalize_smiles
from lube_fitness import add_dvi_and_lube_fitness


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def soft_constraint_score(row, bp_threshold, mp_threshold, fp_threshold,
                          sc_threshold, tox_threshold, use_biodeg):
    """Continuous constraint penalty in [0, 1] for the lubricant task.

    Dropped dc_threshold — dielectric constant is not a lubricant constraint.
    """
    score = 1.0

    bp = row.get("bp", 0)
    if bp < bp_threshold:
        score *= float(_sigmoid((bp - bp_threshold) / 10))

    mp = row.get("mp", 0)
    if mp > mp_threshold:
        score *= float(_sigmoid((mp_threshold - mp) / 10))

    fp = row.get("fp", 999)
    if fp < fp_threshold:
        score *= float(_sigmoid((fp - fp_threshold) / 20))

    tox = row.get("Tox21_Score", 0)
    if tox > tox_threshold:
        score *= float(_sigmoid((tox_threshold - tox) / 1))

    sc = row.get("SCScore", 0)
    if sc > sc_threshold:
        score *= float(_sigmoid((sc_threshold - sc) / 0.5))

    if use_biodeg and not row.get("Biodegradable", False):
        score *= 0.1

    return score


def _has_structural_issues(smiles):
    if has_invalid_fragments(smiles):
        return True
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return True
    if Descriptors.NumRadicalElectrons(mol) > 0:
        return True
    return False


TARGET_CONFIG = {
    "FOM_LUBE": {"column": "FOM_LUBE", "maximize": True},
}


class LubricantReward:
    """REINVENT reward for lubricant base-oil design, matching the GA fitness."""

    def __init__(
        self,
        model_dir,
        training_data_dir,
        target="FOM_LUBE",
        weight_profile="even",
        sc_threshold=3,
        mp_threshold=-30,
        bp_threshold=100,
        fp_threshold=373,
        tox_threshold=3,
        use_biodeg=False,
        molprice_soft=0.0,
        molprice_hard=0.0,
        soft_constraints=False,
    ):
        self.target = target
        self.weight_profile = weight_profile
        self.sc_threshold = sc_threshold
        self.mp_threshold = mp_threshold
        self.bp_threshold = bp_threshold
        self.fp_threshold = fp_threshold
        self.tox_threshold = tox_threshold
        self.use_biodeg = use_biodeg
        self.molprice_soft = molprice_soft
        self.molprice_hard = molprice_hard
        self.soft_constraints = soft_constraints

        print("\nLoading models for lubricant REINVENT reward...")
        thermo_targets = [
            "bp", "mp", "fp",
            "density_40C", "viscosity_40C", "viscosity_100C",
            "tc_40C", "cpsat_40C", "beta_40C",
        ]
        self.thermo_models = load_regression_models_with_aux(thermo_targets, model_dir)

        self.sc_model = SCScorer()
        self.sc_model.restore(os.path.join(
            model_dir,
            "SCScorer/scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz",
        ))

        self.tox21_models = load_tox21_predictor(os.path.join(model_dir, "tox21_gt4sd"))
        self.biodeg_model = load_biodeg_model(os.path.join(model_dir, "biodeg"))

        from molprice import MolPriceModel
        mp_path = os.path.join(model_dir, "MolPrice", "MP_Morgan_hybrid.pkl")
        self.molprice_model = None
        if os.path.exists(mp_path):
            self.molprice_model = MolPriceModel(mp_path)
            print(f"Loaded MolPrice model from {mp_path}")

        print("Precomputing Mahalanobis OOD parameters...")
        mahal = {}
        for t, data in self.thermo_models.items():
            params = compute_mahalanobis_params(t, data["features"], training_data_dir)
            if params is not None:
                mahal[t] = params
        self.mahal_params = mahal if mahal else None

        if self.soft_constraints:
            print("Soft constraints ENABLED (sigmoid penalties)")
        print("LubricantReward ready.\n")

    def evaluate_batch(self, smiles_list):
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
            dc_max=999,                # disabled for lubricants
            min_fp=self.fp_threshold,
            use_biodeg=self.use_biodeg,
            max_tox21=self.tox_threshold,
        )

        df = add_dvi_and_lube_fitness(df, weight_profile=self.weight_profile)

        if self.soft_constraints:
            soft_scores = np.zeros(len(df))
            for j in range(len(df)):
                row = df.iloc[j]
                if _has_structural_issues(row["SMILES"]):
                    soft_scores[j] = 0.0
                    continue
                fom = row.get("FOM_LUBE", 0)
                if not np.isfinite(fom) or fom <= 0:
                    soft_scores[j] = 0.0
                    continue
                c = soft_constraint_score(
                    row,
                    bp_threshold=self.bp_threshold,
                    mp_threshold=self.mp_threshold,
                    fp_threshold=self.fp_threshold,
                    sc_threshold=self.sc_threshold,
                    tox_threshold=self.tox_threshold,
                    use_biodeg=self.use_biodeg,
                )
                soft_scores[j] = fom * c
            df["FitnessScore"] = soft_scores
            df = apply_molprice_penalty(df, soft_threshold=self.molprice_soft,
                                        hard_threshold=self.molprice_hard)
        else:
            df = compute_fitness(df, self.target, TARGET_CONFIG)
            df = apply_molprice_penalty(df, soft_threshold=self.molprice_soft,
                                        hard_threshold=self.molprice_hard)

        j = 0
        for i, ok in enumerate(valid_mask):
            if ok:
                rewards[i] = df.iloc[j]["FitnessScore"]
                j += 1
        return rewards, df
