"""Two-phase REINVENT training: pre-training (teacher forcing) + REINFORCE fine-tuning.

Pre-training: Cross-entropy on NIST 8100 CHO SMILES corpus.
Fine-tuning: Standard REINVENT augmented likelihood (Olivecrona et al. 2017):

    L = (log P_prior(x) + sigma * R(x) - log P_agent(x))^2

The prior is frozen; only the agent is updated.  The loss drives the agent
to sample molecules whose log-probability under the augmented prior
(prior + reward) is high.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from .vocabulary import SMILESVocabulary


# ─────────────────────────────────────────────
# Dataset for pre-training
# ─────────────────────────────────────────────

class SMILESDataset(Dataset):
    """Tokenized SMILES dataset for teacher-forcing pre-training."""

    def __init__(self, smiles_list, vocab, max_len=80):
        self.encoded = []
        for smi in smiles_list:
            enc = vocab.encode(smi)
            if len(enc) <= max_len:
                self.encoded.append(enc)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long)


def collate_fn(batch):
    """Pad encoded sequences to equal length within a batch."""
    max_len = max(len(seq) for seq in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = seq
    return padded


# ─────────────────────────────────────────────
# Pre-training
# ─────────────────────────────────────────────

def pretrain(model, vocab, smiles_list, epochs=30, batch_size=256,
             lr=1e-3, device="cpu", log_fn=print):
    """Pre-train GRU on SMILES corpus via teacher forcing.

    After training, samples 1000 molecules and reports validity.

    Args:
        model: GRUModel instance (will be trained in-place).
        vocab: SMILESVocabulary.
        smiles_list: list of canonical SMILES for training.
        epochs: number of training epochs.
        batch_size: training batch size.
        lr: learning rate.
        device: torch device string.
        log_fn: callable for logging messages.

    Returns:
        model (same reference, trained in-place).
    """
    dataset = SMILESDataset(smiles_list, vocab)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vocab_size = len(vocab)

    log_fn(f"Pre-training on {len(dataset)} sequences, "
           f"{epochs} epochs, batch={batch_size}, lr={lr}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            batch = batch.to(device)
            logits, _ = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=vocab.pad_idx,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Periodic validation: sample and check validity
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                gen_smiles, _ = model.sample(vocab, batch_size=1000)
            n_valid = sum(
                1 for s in gen_smiles if Chem.MolFromSmiles(s) is not None
            )
            log_fn(f"  Epoch {epoch:3d}/{epochs}: loss={avg_loss:.4f}, "
                   f"validity={n_valid}/1000 ({n_valid/10:.1f}%)")

    # Final validity check
    model.eval()
    with torch.no_grad():
        final_smiles, _ = model.sample(vocab, batch_size=1000)
    n_valid = sum(1 for s in final_smiles if Chem.MolFromSmiles(s) is not None)
    log_fn(f"Pre-training complete. Final validity: "
           f"{n_valid}/1000 ({n_valid/10:.1f}%)")

    return model


# ─────────────────────────────────────────────
# Encoding helper
# ─────────────────────────────────────────────

def encode_batch(smiles_list, vocab, device="cpu"):
    """Encode a list of SMILES into a padded tensor.

    Returns:
        (B, L) long tensor on *device*, padded with vocab.pad_idx.
    """
    encoded = [vocab.encode(smi) for smi in smiles_list]
    max_len = max(len(e) for e in encoded) if encoded else 1
    padded = torch.full(
        (len(encoded), max_len), vocab.pad_idx, dtype=torch.long,
    )
    for i, enc in enumerate(encoded):
        padded[i, : len(enc)] = torch.tensor(enc, dtype=torch.long)
    return padded.to(device)


# ─────────────────────────────────────────────
# REINVENT fine-tuning
# ─────────────────────────────────────────────

class ReinventTrainer:
    """Standard REINVENT augmented-likelihood fine-tuning.

    Loss per batch:
        L = mean( (log P_prior(x) + sigma * R(x) - log P_agent(x))^2 )

    The prior is frozen.  Only the agent's weights are updated.
    """

    def __init__(self, prior, agent, vocab, reward_fn, *,
                 sigma=0.5, lr=5e-5, batch_size=128, max_len=80,
                 device="cpu"):
        self.prior = prior          # frozen
        self.agent = agent          # trainable
        self.vocab = vocab
        self.reward_fn = reward_fn
        self.sigma = sigma
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

        # Evaluation cache: canonical SMILES -> {FitnessScore, row_dict}
        self.eval_cache = {}

        # Tracking
        self.n_unique_evaluated = 0
        self.best_fitness = -float("inf")
        self.best_smiles = None
        self.steps_since_improvement = 0

    def step(self):
        """Execute one REINVENT fine-tuning step.

        Returns:
            dict with step metrics (reward_mean, loss, n_valid, n_new, etc.)
        """
        t0 = time.time()

        # 1. Sample SMILES from agent
        self.agent.eval()
        with torch.no_grad():
            smiles, _ = self.agent.sample(
                self.vocab, self.batch_size, self.max_len,
            )

        # 2. Canonicalize and identify new/cached molecules
        canonical = []
        parseable_mask = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                csmi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                canonical.append(csmi)
                parseable_mask.append(True)
            else:
                canonical.append(None)
                parseable_mask.append(False)

        new_smiles = []
        new_indices = []
        for i, csmi in enumerate(canonical):
            if csmi is not None and csmi not in self.eval_cache:
                new_smiles.append(csmi)
                new_indices.append(i)

        # 3. Evaluate only new molecules through WSGA pipeline
        if new_smiles:
            rewards_new, df_new = self.reward_fn.evaluate_batch(new_smiles)
            for j, csmi in enumerate(new_smiles):
                row_dict = df_new.iloc[j].to_dict() if df_new is not None else {}
                self.eval_cache[csmi] = {
                    "FitnessScore": float(rewards_new[j]),
                    "row": row_dict,
                }
            self.n_unique_evaluated += len(new_smiles)

        # 4. Assemble rewards for full batch (from cache)
        rewards = np.zeros(len(smiles))
        for i, csmi in enumerate(canonical):
            if csmi is not None and csmi in self.eval_cache:
                rewards[i] = self.eval_cache[csmi]["FitnessScore"]

        # 5. Compute REINVENT loss
        #    Encode sampled SMILES (use original strings, not canonical,
        #    to match the token sequence the model actually produced)
        encoded = encode_batch(smiles, self.vocab, self.device)
        rewards_t = torch.tensor(
            rewards, dtype=torch.float32, device=self.device,
        )

        self.prior.eval()
        with torch.no_grad():
            prior_lp = self.prior.log_prob_of_sequence(
                encoded, pad_idx=self.vocab.pad_idx,
            )

        self.agent.train()
        agent_lp = self.agent.log_prob_of_sequence(
            encoded, pad_idx=self.vocab.pad_idx,
        )

        augmented = prior_lp + self.sigma * rewards_t
        loss = ((augmented - agent_lp) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()

        # 6. Track convergence
        batch_best = float(rewards.max()) if rewards.max() > 0 else 0.0
        if batch_best > self.best_fitness + 0.01:
            self.best_fitness = batch_best
            # Find best SMILES
            best_idx = int(rewards.argmax())
            self.best_smiles = canonical[best_idx]
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

        n_valid = sum(parseable_mask)
        elapsed = time.time() - t0

        return {
            "reward_mean": float(rewards.mean()),
            "reward_max": float(rewards.max()),
            "loss": float(loss.item()),
            "n_valid": n_valid,
            "n_new": len(new_smiles),
            "n_unique_total": self.n_unique_evaluated,
            "best_fitness": self.best_fitness,
            "best_smiles": self.best_smiles,
            "steps_since_improvement": self.steps_since_improvement,
            "elapsed": elapsed,
        }

    def get_all_evaluated(self):
        """Return DataFrame of all unique evaluated molecules.

        Contains the full property rows from the WSGA evaluation pipeline.
        """
        rows = []
        for csmi, data in self.eval_cache.items():
            row = data["row"].copy() if data["row"] else {}
            row["SMILES"] = csmi
            row["FitnessScore"] = data["FitnessScore"]
            rows.append(row)
        import pandas as pd
        return pd.DataFrame(rows)
