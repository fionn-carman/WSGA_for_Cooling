#!/usr/bin/env python3
"""
ChemBERTa-2 fine-tuning for FOM1 prediction.

Backbone: DeepChem/ChemBERTa-77M-MTR (384-dim, 3 layers)
Strategy: Freeze backbone for 5 epochs, then unfreeze with differential LR.
Head: [CLS] → 128 → ReLU → Dropout(0.1) → 1
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = str(Path(__file__).resolve().parent / "chemberta_model")


class SMILESDataset(Dataset):
    def __init__(self, smiles, targets, tokenizer, max_len=128):
        self.smiles = smiles
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class ChemBERTaRegressor(nn.Module):
    def __init__(self, backbone, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] pooling
        return self.head(cls_token).squeeze(-1)


def train_one_fold(smiles_train, y_train, smiles_test, y_test, tokenizer, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train/val split (85/15)
    n = len(smiles_train)
    perm = np.random.permutation(n)
    n_val = int(0.15 * n)
    val_idx, trn_idx = perm[:n_val], perm[n_val:]

    trn_ds = SMILESDataset(smiles_train[trn_idx], y_train[trn_idx], tokenizer)
    val_ds = SMILESDataset(smiles_train[val_idx], y_train[val_idx], tokenizer)
    tst_ds = SMILESDataset(smiles_test, y_test, tokenizer)

    trn_dl = DataLoader(trn_ds, batch_size=32, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    tst_dl = DataLoader(tst_ds, batch_size=32, shuffle=False, num_workers=0)

    backbone = AutoModel.from_pretrained(MODEL_NAME)
    model = ChemBERTaRegressor(backbone).to(device)
    criterion = nn.MSELoss()

    # Phase 1: Freeze backbone, train head only (5 epochs)
    for param in model.backbone.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.head.parameters(), lr=2e-3)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(50):
        # Unfreeze backbone after 5 epochs with differential LR
        if epoch == 5:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": model.backbone.parameters(), "lr": 1e-4},
                {"params": model.head.parameters(), "lr": 2e-3},
            ])
            logger.info("  Epoch %d: Backbone unfrozen", epoch)

        model.train()
        losses = []
        for batch in trn_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target"].to(device)
                preds = model(input_ids, attention_mask)
                val_losses.append(criterion(preds, targets).item())
        val_loss = np.mean(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info("  Early stopping at epoch %d", epoch)
                break

        if epoch % 5 == 0:
            logger.info("  Epoch %d: train_loss=%.4f, val_loss=%.4f", epoch, train_loss, val_loss)

    # Predict on test
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tst_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds), best_state, history


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ChemBERTa-2 for FOM1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--target", type=str, default="FOM1_avg",
                        help="Target column to predict (FOM1_40, FOM1_100, or FOM1_avg)")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "fom1_dataset.csv")
    with open(data_dir / "fold_indices.json") as f:
        fold_indices = json.load(f)

    smiles = df["SMILES"].values
    y = df[args.target].values.astype(np.float32)
    logger.info("Target: %s", args.target)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    folds = range(5) if args.fold is None else [args.fold]

    for fold_id in folds:
        logger.info("=" * 50)
        logger.info("FOLD %d — ChemBERTa-2", fold_id)
        logger.info("=" * 50)
        t0 = time.time()

        train_idx = fold_indices[str(fold_id)]["train"]
        test_idx = fold_indices[str(fold_id)]["test"]

        y_pred, best_state, history = train_one_fold(
            smiles[train_idx], y[train_idx],
            smiles[test_idx], y[test_idx],
            tokenizer, seed=args.seed + fold_id, device=device,
        )

        y_test = y[test_idx]
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - t0
        logger.info("Test R²: %.4f, RMSE: %.4f, MAE: %.4f (%.1fs)", r2, rmse, mae, elapsed)

        pd.DataFrame({"SMILES": df.iloc[test_idx]["SMILES"].values, "y_true": y_test, "y_pred": y_pred})\
            .to_csv(data_dir / f"chemberta_fold{fold_id}_predictions.csv", index=False)

        with open(data_dir / f"chemberta_fold{fold_id}_metrics.json", "w") as f:
            json.dump({"model": "chemberta", "target": args.target, "fold": fold_id, "r2": float(r2),
                        "rmse": float(rmse), "mae": float(mae), "training_time_s": float(elapsed)}, f, indent=2)

        # Save only head weights (backbone is large)
        head_state = {k: v for k, v in best_state.items() if k.startswith("head.")}
        torch.save({"head_state_dict": head_state, "backbone_name": MODEL_NAME},
                    data_dir / f"chemberta_fold{fold_id}_model.pt")

        with open(data_dir / f"chemberta_fold{fold_id}_history.json", "w") as f:
            json.dump(history, f)

    logger.info("ChemBERTa training complete.")


if __name__ == "__main__":
    main()
