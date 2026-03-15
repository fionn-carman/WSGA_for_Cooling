"""
GT4SD / PaccMann Tox21 toxicity predictor.

Thin wrapper around the PaccMann MCA (Multiscale Convolutional Attentive)
model trained on the Tox21 challenge dataset (12 binary toxicity endpoints).

Pretrained weights are from the GT4SD model hub (IBM COS) and stored locally
in models/tox21_gt4sd/.

This module vendors the essential logic from paccmann_generator.drug_evaluators
to avoid importing the full paccmann_generator package (which pulls in
paccmann_chemistry and the REINFORCE framework).
"""

import json
import os

import torch
from paccmann_predictor.utils.utils import get_device
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.smiles.transforms import (
    Canonicalization,
    Kekulize,
    NotKekulize,
    RemoveIsomery,
    Selfies,
)
from pytoda.transforms import Compose
from toxsmi.models import MODEL_FACTORY

TOX21_ENDPOINTS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
    'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
]


class _TokenizeAndTensorize:
    """Convert SMILES string to token-index tensor via the loaded vocabulary."""

    def __init__(self, smiles_language: SMILESLanguage):
        self.lang = smiles_language

    def __call__(self, smiles: str) -> torch.Tensor:
        tokens = self.lang.smiles_tokenizer(smiles)
        indexes = (
            [self.lang.start_index]
            + [self.lang.token_to_index.get(t, self.lang.unknown_index) for t in tokens]
            + [self.lang.stop_index]
        )
        return torch.tensor(indexes, dtype=torch.short)


class Tox21Predictor:
    """
    Predicts Tox21 toxicity probabilities from a SMILES string.

    Usage::

        predictor = Tox21Predictor("../models/tox21_gt4sd")
        probs = predictor("CCO")   # list of 12 floats in [0, 1]
        score = sum(probs)          # aggregate score (0-12, lower = safer)
    """

    def __init__(self, model_path: str):
        self.device = get_device()
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        with open(os.path.join(model_path, 'model_params.json')) as f:
            params = json.load(f)

        self.smiles_language = SMILESLanguage.load(
            os.path.join(model_path, 'smiles_language.pkl')
        )
        self.transforms = self._build_transforms(params)

        self.model = MODEL_FACTORY['mca'](params)
        self.model.load(
            os.path.join(model_path, 'weights', 'best_ROC-AUC_mca.pt'),
            map_location=self.device,
        )
        self.model.eval()

    def _build_transforms(self, params: dict) -> Compose:
        """Replicate the SMILES transforms used during training."""
        transforms = []

        if params.get('canonical', False):
            transforms.append(Canonicalization())
        else:
            if params.get('remove_bonddir', False) or params.get('remove_chirality', False):
                transforms.append(RemoveIsomery(
                    bonddir=params.get('remove_bonddir', False),
                    chirality=params.get('remove_chirality', False),
                ))
            if params.get('kekulize', False):
                transforms.append(Kekulize(
                    all_bonds_explicit=params.get('all_bonds_explicit', False),
                    all_hs_explicit=params.get('all_hs_explicit', False),
                ))
            elif params.get('all_bonds_explicit', False) or params.get('all_hs_explicit', False):
                transforms.append(NotKekulize(
                    all_bonds_explicit=params.get('all_bonds_explicit', False),
                    all_hs_explicit=params.get('all_hs_explicit', False),
                ))
            if params.get('selfies', False):
                transforms.append(Selfies())

        transforms.append(_TokenizeAndTensorize(self.smiles_language))
        return Compose(transforms)

    @torch.no_grad()
    def __call__(self, smiles: str) -> list:
        """
        Predict Tox21 toxicity probabilities for a single molecule.

        Args:
            smiles: SMILES string.

        Returns:
            List of 12 floats (one per Tox21 endpoint), each in [0, 1].
            Higher values indicate higher predicted toxicity.
        """
        token_tensor = torch.unsqueeze(self.transforms(smiles), 0).repeat(2, 1)
        predictions, _ = self.model(token_tensor)
        return predictions[0, :].cpu().tolist()
