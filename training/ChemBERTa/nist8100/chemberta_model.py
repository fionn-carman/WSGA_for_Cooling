"""
ChemBERTa-2 property predictor with heteroscedastic regression head.

Fine-tunes DeepChem/ChemBERTa-77M-MTR (RoBERTa-based, 384-dim, 3 layers)
with CLS pooling and a Gaussian NLL loss (mean + logvar heads).

Extracted and simplified from testing/MLPredict/FOM1/TransformerEnsemble/transformer.py.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from typing import Optional

MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
BACKBONE_DIM = 384


class SMILESDataset(Dataset):
    """Dataset that tokenizes SMILES on-the-fly."""

    def __init__(self, smiles: list[str], targets: np.ndarray, tokenizer, max_length: int = 128):
        self.smiles = smiles
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target": self.targets[idx],
        }


class ChemBERTaPredictor(nn.Module):
    """
    ChemBERTa-2 backbone with heteroscedastic regression head.

    Architecture: SMILES -> ChemBERTa -> CLS pooling -> MLP -> (mean, logvar)
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)

        self.head = nn.Sequential(
            nn.Linear(BACKBONE_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token (first token)
        emb = outputs.last_hidden_state[:, 0, :]
        h = self.head(emb)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar


def gaussian_nll_loss(mean, logvar, target):
    """Gaussian negative log-likelihood with logvar clamping."""
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    var = torch.exp(logvar) + 1e-6
    return 0.5 * torch.mean(logvar + (target - mean) ** 2 / var)


def train_one_model(
    smiles_train: list[str],
    y_train: np.ndarray,
    smiles_val: list[str],
    y_val: np.ndarray,
    tokenizer,
    device: torch.device,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    lr_backbone: float = 2e-5,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-2,
    epochs: int = 100,
    batch_size: int = 16,
    patience: int = 15,
    max_length: int = 128,
    max_grad_norm: float = 1.0,
    trial=None,
    verbose: bool = True,
):
    """
    Train a single ChemBERTa model with early stopping.

    Args:
        trial: Optional Optuna trial for pruning (reports val R² every 10 epochs).

    Returns:
        (model, best_val_loss) with model restored to best checkpoint.
    """
    from sklearn.metrics import r2_score

    model = ChemBERTaPredictor(hidden_dim=hidden_dim, dropout=dropout).to(device)

    train_ds = SMILESDataset(smiles_train, y_train, tokenizer, max_length)
    val_ds = SMILESDataset(smiles_val, y_val, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Differential learning rates
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.head.parameters())
        + list(model.mean_head.parameters())
        + list(model.logvar_head.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].unsqueeze(1).to(device)

            optimizer.zero_grad()
            mean, logvar = model(input_ids, attention_mask)
            loss = gaussian_nll_loss(mean, logvar, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_means = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target"].unsqueeze(1).to(device)

                mean, logvar = model(input_ids, attention_mask)
                val_loss_sum += gaussian_nll_loss(mean, logvar, targets).item()
                val_batches += 1
                val_means.append(mean.cpu().numpy().squeeze(-1))
                val_targets.append(targets.cpu().numpy().squeeze(-1))

        avg_val_loss = val_loss_sum / val_batches
        scheduler.step(avg_val_loss)

        # Optuna pruning: report val R² every 10 epochs
        if trial is not None and (epoch + 1) % 10 == 0:
            import optuna
            val_preds = np.concatenate(val_means)
            val_true = np.concatenate(val_targets)
            val_r2 = r2_score(val_true, val_preds)
            trial.report(val_r2, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"      Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
            break

        if verbose and (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"      Epoch {epoch+1:>3d}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, lr_bb={current_lr:.2e}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, best_val_loss


def predict(model, smiles: list[str], tokenizer, device: torch.device,
            batch_size: int = 32, max_length: int = 128) -> np.ndarray:
    """Batch prediction returning point estimates (mean only, ignoring logvar)."""
    dummy_targets = np.zeros(len(smiles))
    ds = SMILESDataset(smiles, dummy_targets, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_means = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mean, _ = model(input_ids, attention_mask)
            all_means.append(mean.cpu().numpy().squeeze(-1))

    return np.concatenate(all_means)
