#!/usr/bin/env python3
"""
MLP on Morgan fingerprints (2048-dim) for FOM1 prediction.
Same architecture as MLP descriptors but input dim 2048.
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(X_train, y_train, X_test, y_test, input_dim, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X_train)
    perm = np.random.permutation(n)
    n_val = int(0.15 * n)
    val_idx, trn_idx = perm[:n_val], perm[n_val:]

    X_trn = torch.FloatTensor(X_train[trn_idx]).to(device)
    y_trn = torch.FloatTensor(y_train[trn_idx]).to(device)
    X_val = torch.FloatTensor(X_train[val_idx]).to(device)
    y_val = torch.FloatTensor(y_train[val_idx]).to(device)
    X_tst = torch.FloatTensor(X_test).to(device)

    trn_ds = TensorDataset(X_trn, y_trn)
    trn_dl = DataLoader(trn_ds, batch_size=64, shuffle=True)

    model = MLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(200):
        model.train()
        losses = []
        for xb, yb in trn_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tst).cpu().numpy()

    return y_pred, best_state, history


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP on Morgan fingerprints for FOM1",
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

    X = np.load(data_dir / "fingerprints.npy")
    y = df[args.target].values.astype(np.float32)
    logger.info("Target: %s", args.target)

    folds = range(5) if args.fold is None else [args.fold]

    for fold_id in folds:
        logger.info("=" * 50)
        logger.info("FOLD %d — MLP Fingerprints", fold_id)
        logger.info("=" * 50)
        t0 = time.time()

        train_idx = fold_indices[str(fold_id)]["train"]
        test_idx = fold_indices[str(fold_id)]["test"]

        # Fingerprints are binary, but scaling still helps MLP
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        y_pred, best_state, history = train_one_fold(
            X_train, y_train, X_test, y_test,
            input_dim=X_train.shape[1], seed=args.seed + fold_id, device=device,
        )

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - t0
        logger.info("Test R²: %.4f, RMSE: %.4f, MAE: %.4f (%.1fs)", r2, rmse, mae, elapsed)

        pd.DataFrame({"SMILES": df.iloc[test_idx]["SMILES"].values, "y_true": y_test, "y_pred": y_pred})\
            .to_csv(data_dir / f"mlp_fingerprints_fold{fold_id}_predictions.csv", index=False)

        with open(data_dir / f"mlp_fingerprints_fold{fold_id}_metrics.json", "w") as f:
            json.dump({"model": "mlp_fingerprints", "target": args.target, "fold": fold_id, "r2": float(r2),
                        "rmse": float(rmse), "mae": float(mae), "training_time_s": float(elapsed)}, f, indent=2)

        torch.save({"state_dict": best_state, "input_dim": X_train.shape[1]},
                    data_dir / f"mlp_fingerprints_fold{fold_id}_model.pt")

        with open(data_dir / f"mlp_fingerprints_fold{fold_id}_history.json", "w") as f:
            json.dump(history, f)

    logger.info("MLP Fingerprints training complete.")


if __name__ == "__main__":
    main()
