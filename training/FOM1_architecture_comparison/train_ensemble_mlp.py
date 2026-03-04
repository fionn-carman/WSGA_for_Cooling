#!/usr/bin/env python3
"""
Deep Ensemble MLP for FOM1 prediction with uncertainty estimation.

5 independent MLPs (same architecture) with seeds 0-4.
Each outputs (mean, log_var) and is trained with Gaussian NLL loss.
Provides calibrated epistemic + aleatoric uncertainty — critical for BO.
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


class GaussianMLP(nn.Module):
    """MLP that outputs (mean, log_var) for Gaussian NLL training."""

    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, 1)
        self.logvar_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mean, logvar


def gaussian_nll_loss(y_true, mean, logvar):
    """Gaussian negative log-likelihood loss."""
    var = torch.exp(logvar).clamp(min=1e-6)
    return 0.5 * torch.mean(logvar + (y_true - mean) ** 2 / var)


def train_single_member(X_train, y_train, X_val, y_val, input_dim, seed, device):
    """Train one ensemble member."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_trn = torch.FloatTensor(X_train).to(device)
    y_trn = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)

    trn_ds = TensorDataset(X_trn, y_trn)
    trn_dl = DataLoader(trn_ds, batch_size=64, shuffle=True)

    model = GaussianMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(200):
        model.train()
        losses = []
        for xb, yb in trn_dl:
            optimizer.zero_grad()
            mean, logvar = model(xb)
            loss = gaussian_nll_loss(yb, mean, logvar)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        model.eval()
        with torch.no_grad():
            v_mean, v_logvar = model(X_v)
            val_loss = gaussian_nll_loss(y_v, v_mean, v_logvar).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    return best_state, history


def ensemble_predict(models, X, device):
    """Compute ensemble predictions with uncertainty."""
    X_t = torch.FloatTensor(X).to(device)
    all_means = []
    all_vars = []

    for model in models:
        model.eval()
        with torch.no_grad():
            mean, logvar = model(X_t)
            all_means.append(mean.cpu().numpy())
            all_vars.append(np.exp(logvar.cpu().numpy()))

    all_means = np.array(all_means)  # (M, N)
    all_vars = np.array(all_vars)    # (M, N)

    # Inverse-variance weighted mean
    inv_vars = 1.0 / np.clip(all_vars, 1e-6, None)
    weights = inv_vars / inv_vars.sum(axis=0, keepdims=True)
    ensemble_mean = (weights * all_means).sum(axis=0)

    # Uncertainties
    epistemic_var = np.var(all_means, axis=0)
    aleatoric_var = np.mean(all_vars, axis=0)
    total_std = np.sqrt(epistemic_var + aleatoric_var)

    return ensemble_mean, total_std, epistemic_var, aleatoric_var


def compute_calibration(y_true, y_pred, total_std):
    """Compute empirical coverage at 1sigma and 2sigma."""
    residuals = np.abs(y_true - y_pred)
    coverage_1sigma = np.mean(residuals <= total_std)
    coverage_2sigma = np.mean(residuals <= 2 * total_std)
    return coverage_1sigma, coverage_2sigma


def main():
    parser = argparse.ArgumentParser(
        description="Train deep ensemble MLP for FOM1 with uncertainty",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--target", type=str, default="FOM1_avg",
                        help="Target column to predict (FOM1_40, FOM1_100, or FOM1_avg)")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n_members", type=int, default=5, help="Ensemble members")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "fom1_dataset.csv")
    with open(data_dir / "fold_indices.json") as f:
        fold_indices = json.load(f)
    with open(data_dir / "descriptor_columns.json") as f:
        descriptor_columns = json.load(f)

    X = df[descriptor_columns].values.astype(np.float32)
    y = df[args.target].values.astype(np.float32)
    logger.info("Target: %s", args.target)

    folds = range(5) if args.fold is None else [args.fold]

    for fold_id in folds:
        logger.info("=" * 50)
        logger.info("FOLD %d — Deep Ensemble (%d members)", fold_id, args.n_members)
        logger.info("=" * 50)
        t0 = time.time()

        train_idx = fold_indices[str(fold_id)]["train"]
        test_idx = fold_indices[str(fold_id)]["test"]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        # Split train into train/val
        n = len(X_train)
        perm = np.random.RandomState(args.seed).permutation(n)
        n_val = int(0.15 * n)
        val_idx_inner, trn_idx_inner = perm[:n_val], perm[n_val:]

        X_trn = X_train[trn_idx_inner]
        y_trn = y_train[trn_idx_inner]
        X_val = X_train[val_idx_inner]
        y_val = y_train[val_idx_inner]

        # Train ensemble members
        member_states = []
        for m in range(args.n_members):
            logger.info("  Training member %d/%d...", m + 1, args.n_members)
            state, hist = train_single_member(
                X_trn, y_trn, X_val, y_val,
                input_dim=X_trn.shape[1], seed=m, device=device,
            )
            member_states.append(state)

        # Create models for prediction
        models = []
        for state in member_states:
            model = GaussianMLP(X_train.shape[1]).to(device)
            model.load_state_dict(state)
            models.append(model)

        # Ensemble prediction
        y_pred, total_std, epistemic_var, aleatoric_var = ensemble_predict(models, X_test, device)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - t0

        # Calibration
        cov_1s, cov_2s = compute_calibration(y_test, y_pred, total_std)
        logger.info("Test R²: %.4f, RMSE: %.4f, MAE: %.4f (%.1fs)", r2, rmse, mae, elapsed)
        logger.info("Calibration: 1sigma coverage=%.3f (ideal 0.683), 2sigma coverage=%.3f (ideal 0.954)",
                     cov_1s, cov_2s)

        # Save predictions with uncertainty
        pred_df = pd.DataFrame({
            "SMILES": df.iloc[test_idx]["SMILES"].values,
            "y_true": y_test,
            "y_pred": y_pred,
            "total_std": total_std,
            "epistemic_std": np.sqrt(epistemic_var),
            "aleatoric_std": np.sqrt(aleatoric_var),
        })
        pred_df.to_csv(data_dir / f"ensemble_mlp_fold{fold_id}_predictions.csv", index=False)

        with open(data_dir / f"ensemble_mlp_fold{fold_id}_metrics.json", "w") as f:
            json.dump({
                "model": "ensemble_mlp", "target": args.target, "fold": fold_id, "r2": float(r2),
                "rmse": float(rmse), "mae": float(mae), "training_time_s": float(elapsed),
                "n_members": args.n_members,
                "coverage_1sigma": float(cov_1s),
                "coverage_2sigma": float(cov_2s),
                "mean_epistemic_std": float(np.mean(np.sqrt(epistemic_var))),
                "mean_aleatoric_std": float(np.mean(np.sqrt(aleatoric_var))),
            }, f, indent=2)

        # Save member weights
        for m, state in enumerate(member_states):
            torch.save({"state_dict": state, "input_dim": X_train.shape[1]},
                        data_dir / f"ensemble_mlp_fold{fold_id}_member{m}_model.pt")

    logger.info("Ensemble MLP training complete.")


if __name__ == "__main__":
    main()
