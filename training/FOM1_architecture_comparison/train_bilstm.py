#!/usr/bin/env python3
"""
Bidirectional LSTM on character-level SMILES for FOM1 prediction.

Embedding(vocab_size, 64) → BiLSTM(128, 2 layers, dropout=0.1) →
last hidden → FC(256) → ReLU → Dropout(0.1) → FC(1)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SMILESVocab:
    """Character-level vocabulary for SMILES."""
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.char2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2char = {0: self.PAD, 1: self.UNK}

    def build(self, smiles_list):
        chars = set()
        for smi in smiles_list:
            chars.update(smi)
        for i, ch in enumerate(sorted(chars), start=2):
            self.char2idx[ch] = i
            self.idx2char[i] = ch
        return self

    def encode(self, smi):
        return [self.char2idx.get(ch, 1) for ch in smi]

    def __len__(self):
        return len(self.char2idx)


class SMILESDatasetBiLSTM(Dataset):
    def __init__(self, smiles, targets, vocab):
        self.smiles = smiles
        self.targets = targets
        self.vocab = vocab

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.vocab.encode(self.smiles[idx])
        return (
            torch.tensor(enc, dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            len(enc),
        )


def collate_fn(batch):
    seqs, targets, lengths = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.stack(targets), torch.tensor(lengths, dtype=torch.long)


class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x, lengths):
        emb = self.embedding(x)
        # Use plain LSTM (no packing — avoids OMP library conflicts)
        output, (h_n, _) = self.lstm(emb)
        # Masked mean pooling over valid timesteps
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (B, T, 1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def train_one_fold(smiles_train, y_train, smiles_test, y_test, vocab, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(smiles_train)
    perm = np.random.permutation(n)
    n_val = int(0.15 * n)
    val_idx, trn_idx = perm[:n_val], perm[n_val:]

    trn_ds = SMILESDatasetBiLSTM(smiles_train[trn_idx], y_train[trn_idx], vocab)
    val_ds = SMILESDatasetBiLSTM(smiles_train[val_idx], y_train[val_idx], vocab)
    tst_ds = SMILESDatasetBiLSTM(smiles_test, y_test, vocab)

    trn_dl = DataLoader(trn_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)
    tst_dl = DataLoader(tst_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = BiLSTMRegressor(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    logger.info("  Starting training loop (100 epochs)...")
    for epoch in range(100):
        model.train()
        losses = []
        for x, y_batch, lengths in trn_dl:
            x, y_batch, lengths = x.to(device), y_batch.to(device), lengths.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y_batch, lengths in val_dl:
                x, y_batch = x.to(device), y_batch.to(device)
                val_losses.append(criterion(model(x, lengths), y_batch).item())
        val_loss = np.mean(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _, lengths in tst_dl:
            x = x.to(device)
            all_preds.append(model(x, lengths).cpu().numpy())
    return np.concatenate(all_preds), best_state, history


def main():
    parser = argparse.ArgumentParser(
        description="Train BiLSTM on character-level SMILES for FOM1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--target", type=str, default="FOM1_avg",
                        help="Target column to predict (FOM1_40, FOM1_100, or FOM1_avg)")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Force CPU — pack_padded_sequence doesn't work well on MPS
    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "fom1_dataset.csv")
    with open(data_dir / "fold_indices.json") as f:
        fold_indices = json.load(f)

    smiles = df["SMILES"].values
    y = df[args.target].values.astype(np.float32)
    logger.info("Target: %s", args.target)

    # Build vocabulary from all SMILES
    vocab = SMILESVocab()
    vocab.build(smiles)
    logger.info("Vocabulary size: %d", len(vocab))

    # Save vocabulary
    with open(data_dir / "bilstm_vocab.json", "w") as f:
        json.dump(vocab.char2idx, f)

    folds = range(5) if args.fold is None else [args.fold]

    for fold_id in folds:
        logger.info("=" * 50)
        logger.info("FOLD %d — BiLSTM", fold_id)
        logger.info("=" * 50)
        t0 = time.time()

        train_idx = fold_indices[str(fold_id)]["train"]
        test_idx = fold_indices[str(fold_id)]["test"]

        y_pred, best_state, history = train_one_fold(
            smiles[train_idx], y[train_idx],
            smiles[test_idx], y[test_idx],
            vocab, seed=args.seed + fold_id, device=device,
        )

        y_test = y[test_idx]
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - t0
        logger.info("Test R²: %.4f, RMSE: %.4f, MAE: %.4f (%.1fs)", r2, rmse, mae, elapsed)

        pd.DataFrame({"SMILES": df.iloc[test_idx]["SMILES"].values, "y_true": y_test, "y_pred": y_pred})\
            .to_csv(data_dir / f"bilstm_fold{fold_id}_predictions.csv", index=False)

        with open(data_dir / f"bilstm_fold{fold_id}_metrics.json", "w") as f:
            json.dump({"model": "bilstm", "target": args.target, "fold": fold_id, "r2": float(r2),
                        "rmse": float(rmse), "mae": float(mae), "training_time_s": float(elapsed)}, f, indent=2)

        torch.save({"state_dict": best_state, "vocab_size": len(vocab)},
                    data_dir / f"bilstm_fold{fold_id}_model.pt")

        with open(data_dir / f"bilstm_fold{fold_id}_history.json", "w") as f:
            json.dump(history, f)

    logger.info("BiLSTM training complete.")


if __name__ == "__main__":
    main()
