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
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
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
# Scaffold Diversity Filter (REINVENT 2.0)
# ─────────────────────────────────────────────

class ScaffoldDiversityFilter:
    """Diversity filter that tracks molecular keys and zeroes rewards
    once a key has been rewarded max_per_key times.

    Supports two modes:
      - 'scaffold' (REINVENT 2.0): generic Murcko scaffold
      - 'brics': frozenset of BRICS fragment SMILES
    """

    def __init__(self, max_per_scaffold=25, mode="scaffold"):
        self.max_per_scaffold = max_per_scaffold
        self.mode = mode
        self.scaffold_counts = defaultdict(int)

    def _get_key(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return "unknown"
            if self.mode == "brics":
                frags = BRICS.BRICSDecompose(mol)
                return frozenset(frags) if frags else "unknown"
            else:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol))
                return Chem.MolToSmiles(scaffold)
        except Exception:
            return "unknown"

    def filter_rewards(self, smiles_list, rewards):
        """Zero out rewards for saturated keys.

        Args:
            smiles_list: list of canonical SMILES (may contain None).
            rewards: np.array of rewards.

        Returns:
            (filtered_rewards, n_filtered) — modified rewards array and count
            of molecules whose reward was zeroed.
        """
        filtered = rewards.copy()
        n_filtered = 0
        for i, smi in enumerate(smiles_list):
            if smi is None or filtered[i] <= 0:
                continue
            key = self._get_key(smi)
            if self.scaffold_counts[key] >= self.max_per_scaffold:
                filtered[i] = 0.0
                n_filtered += 1
            else:
                self.scaffold_counts[key] += 1
        return filtered, n_filtered

    @property
    def n_scaffolds(self):
        return len(self.scaffold_counts)

    @property
    def n_saturated(self):
        return sum(
            1 for c in self.scaffold_counts.values()
            if c >= self.max_per_scaffold
        )


# ─────────────────────────────────────────────
# Tanimoto Niching Filter
# ─────────────────────────────────────────────

class TanimotoNichingFilter:
    """Continuous diversity penalty based on Tanimoto similarity.

    Ports the WSGA niching mechanism (wsga_helper.apply_niching) into
    REINVENT's reward pipeline.  For each molecule, computes average
    Tanimoto similarity to the current top-K molecules in a reference
    set, then applies:

        reward *= exp(-alpha * max(0, avg_sim - tau)^2) * (1 - avg_sim)

    Unlike the scaffold filter (binary cutoff), this gives a smooth
    gradient that penalises molecules proportional to their similarity
    to already-discovered elites.
    """

    def __init__(self, tau=0.15, alpha=1000, radius=8, ref_size=100):
        self.tau = tau
        self.alpha = alpha
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=2048)
        self.ref_size = ref_size
        # Reference fingerprints: updated each step from eval_cache
        self._ref_fps = []

    def update_reference(self, eval_cache):
        """Rebuild reference set from top-K molecules in the eval cache."""
        items = [
            (csmi, data["FitnessScore"])
            for csmi, data in eval_cache.items()
            if data["FitnessScore"] > 0
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[: self.ref_size]

        self._ref_fps = []
        for csmi, _ in items:
            mol = Chem.MolFromSmiles(csmi)
            if mol:
                self._ref_fps.append(self.fpgen.GetFingerprint(mol))

    def apply_niching(self, smiles_list, rewards):
        """Apply Tanimoto niching penalty to rewards.

        Args:
            smiles_list: list of canonical SMILES (may contain None).
            rewards: np.array of rewards.

        Returns:
            (niched_rewards, mean_penalty) — modified rewards and
            average penalty applied (for logging).
        """
        if not self._ref_fps:
            return rewards.copy(), 0.0

        niched = rewards.copy()
        penalties = []
        for i, smi in enumerate(smiles_list):
            if smi is None or niched[i] <= 0:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = self.fpgen.GetFingerprint(mol)
            sims = [DataStructs.TanimotoSimilarity(fp, ref)
                    for ref in self._ref_fps]
            avg_sim = np.mean(sims) if sims else 0.0

            penalty = (np.exp(-self.alpha * max(0, avg_sim - self.tau) ** 2)
                       * (1 - avg_sim))
            niched[i] *= penalty
            penalties.append(penalty)

        mean_pen = np.mean(penalties) if penalties else 1.0
        return niched, mean_pen


# ─────────────────────────────────────────────
# Experience Replay Buffer (Augmented Memory)
# ─────────────────────────────────────────────

class ReplayBuffer:
    """Store top-performing molecules and replay during training.

    Implements scaffold-aware diversity control (max_per_scaffold) to prevent
    mode collapse into a single Murcko scaffold.
    """

    def __init__(self, max_size=100, max_per_scaffold=10):
        self.buffer = {}  # SMILES -> {reward, step, scaffold}
        self.max_size = max_size
        self.max_per_scaffold = max_per_scaffold

    def _get_scaffold(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return "unknown"
            scaffold = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol))
            return Chem.MolToSmiles(scaffold)
        except Exception:
            return "unknown"

    def add(self, smiles, reward, step):
        scaffold = self._get_scaffold(smiles)
        self.buffer[smiles] = {
            "reward": reward, "step": step, "scaffold": scaffold,
        }
        if len(self.buffer) > self.max_size:
            self._evict_lowest()
        self._scaffold_purge()

    def _evict_lowest(self):
        if not self.buffer:
            return
        worst_smi = min(self.buffer, key=lambda s: self.buffer[s]["reward"])
        del self.buffer[worst_smi]

    def _scaffold_purge(self):
        """Remove excess molecules from overrepresented scaffolds."""
        by_scaffold = defaultdict(list)
        for smi, data in self.buffer.items():
            by_scaffold[data["scaffold"]].append((smi, data["reward"]))

        for scaffold, mols in by_scaffold.items():
            if len(mols) > self.max_per_scaffold:
                mols.sort(key=lambda x: x[1], reverse=True)
                for smi, _ in mols[self.max_per_scaffold:]:
                    if smi in self.buffer:
                        del self.buffer[smi]

    def sample(self, n):
        """Sample n molecules weighted by reward. Returns list of (smiles, reward)."""
        if not self.buffer or n <= 0:
            return []
        items = list(self.buffer.items())
        weights = np.array([d["reward"] for _, d in items])
        weights = np.clip(weights, 0.01, None)
        weights /= weights.sum()
        n_sample = min(n, len(items))
        idx = np.random.choice(len(items), size=n_sample, replace=False, p=weights)
        return [(items[i][0], items[i][1]["reward"]) for i in idx]

    def __len__(self):
        return len(self.buffer)

    @property
    def n_scaffolds(self):
        scaffolds = set(d["scaffold"] for d in self.buffer.values())
        return len(scaffolds)


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
                 device="cpu",
                 diversity_filter=True, max_per_scaffold=25,
                 diversity_mode="scaffold",
                 replay_buffer_size=100, replay_max_per_scaffold=10,
                 replay_fraction=0.5,
                 tanimoto_niching=False, niching_tau=0.15,
                 niching_alpha=1000, niching_radius=8,
                 niching_ref_size=100):
        self.prior = prior          # frozen
        self.agent = agent          # trainable
        self.vocab = vocab
        self.reward_fn = reward_fn
        self.sigma = sigma
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device
        self.replay_fraction = replay_fraction

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

        # Evaluation cache: canonical SMILES -> {FitnessScore, row_dict}
        self.eval_cache = {}

        # Diversity mechanisms
        self.diversity_filter = (
            ScaffoldDiversityFilter(max_per_scaffold=max_per_scaffold,
                                    mode=diversity_mode)
            if diversity_filter else None
        )
        self.replay_buffer = (
            ReplayBuffer(max_size=replay_buffer_size,
                         max_per_scaffold=replay_max_per_scaffold)
            if replay_buffer_size > 0 else None
        )
        self.tanimoto_niching = (
            TanimotoNichingFilter(tau=niching_tau, alpha=niching_alpha,
                                  radius=niching_radius,
                                  ref_size=niching_ref_size)
            if tanimoto_niching else None
        )

        # Tracking
        self.step_count = 0
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
        self.step_count += 1

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

        # 4b. Apply scaffold diversity filter
        n_filtered = 0
        if self.diversity_filter is not None:
            rewards, n_filtered = self.diversity_filter.filter_rewards(
                canonical, rewards)

        # 4b2. Apply Tanimoto niching penalty
        niching_penalty = 1.0
        if self.tanimoto_niching is not None:
            self.tanimoto_niching.update_reference(self.eval_cache)
            rewards, niching_penalty = self.tanimoto_niching.apply_niching(
                canonical, rewards)

        # 4c. Add only post-filter rewarded molecules to replay buffer
        n_replay_used = 0
        if self.replay_buffer is not None:
            for i, csmi in enumerate(canonical):
                if csmi is not None and rewards[i] > 0:
                    self.replay_buffer.add(csmi, float(rewards[i]),
                                           self.step_count)

        # 5. Build combined loss batch: fresh SMILES + replay SMILES
        #    Encode sampled SMILES (use original strings, not canonical,
        #    to match the token sequence the model actually produced)
        fresh_encoded = encode_batch(smiles, self.vocab, self.device)
        fresh_rewards_t = torch.tensor(
            rewards, dtype=torch.float32, device=self.device,
        )

        # Get replay molecules and encode them
        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            n_replay_target = int(self.batch_size * self.replay_fraction)
            replay_samples = self.replay_buffer.sample(n_replay_target)
            n_replay_used = len(replay_samples)
        else:
            replay_samples = []

        if replay_samples:
            replay_smiles = [s for s, _ in replay_samples]
            replay_rewards = np.array([r for _, r in replay_samples])

            # Re-apply the diversity filter so replay cannot bypass scaffold caps.
            if self.diversity_filter is not None:
                replay_rewards, replay_filtered = (
                    self.diversity_filter.filter_rewards(
                        replay_smiles, replay_rewards
                    )
                )
                n_filtered += replay_filtered

            replay_encoded = encode_batch(
                replay_smiles, self.vocab, self.device)
            replay_rewards_t = torch.tensor(
                replay_rewards, dtype=torch.float32, device=self.device,
            )

            # Pad to same sequence length and concatenate
            max_len = max(fresh_encoded.shape[1], replay_encoded.shape[1])
            if fresh_encoded.shape[1] < max_len:
                pad = torch.full(
                    (fresh_encoded.shape[0], max_len - fresh_encoded.shape[1]),
                    self.vocab.pad_idx, dtype=torch.long, device=self.device,
                )
                fresh_encoded = torch.cat([fresh_encoded, pad], dim=1)
            if replay_encoded.shape[1] < max_len:
                pad = torch.full(
                    (replay_encoded.shape[0], max_len - replay_encoded.shape[1]),
                    self.vocab.pad_idx, dtype=torch.long, device=self.device,
                )
                replay_encoded = torch.cat([replay_encoded, pad], dim=1)

            all_encoded = torch.cat([fresh_encoded, replay_encoded], dim=0)
            all_rewards_t = torch.cat([fresh_rewards_t, replay_rewards_t], dim=0)
        else:
            all_encoded = fresh_encoded
            all_rewards_t = fresh_rewards_t

        # Compute augmented likelihood loss on combined batch
        self.prior.eval()
        with torch.no_grad():
            prior_lp = self.prior.log_prob_of_sequence(
                all_encoded, pad_idx=self.vocab.pad_idx,
            )

        self.agent.train()
        agent_lp = self.agent.log_prob_of_sequence(
            all_encoded, pad_idx=self.vocab.pad_idx,
        )

        augmented = prior_lp + self.sigma * all_rewards_t
        loss = ((augmented - agent_lp) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()

        # 6. Track convergence (use unfiltered rewards for best tracking)
        raw_rewards = np.zeros(len(smiles))
        for i, csmi in enumerate(canonical):
            if csmi is not None and csmi in self.eval_cache:
                raw_rewards[i] = self.eval_cache[csmi]["FitnessScore"]

        batch_best = float(raw_rewards.max()) if raw_rewards.max() > 0 else 0.0
        if batch_best > self.best_fitness + 0.01:
            self.best_fitness = batch_best
            best_idx = int(raw_rewards.argmax())
            self.best_smiles = canonical[best_idx]
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

        n_valid = sum(parseable_mask)
        elapsed = time.time() - t0

        metrics = {
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
            # Diversity metrics
            "n_filtered": n_filtered,
            "n_replay": n_replay_used,
            "n_scaffolds": (self.diversity_filter.n_scaffolds
                            if self.diversity_filter else 0),
            "n_saturated": (self.diversity_filter.n_saturated
                            if self.diversity_filter else 0),
            "niching_penalty": niching_penalty,
            "replay_size": len(self.replay_buffer)
                           if self.replay_buffer else 0,
        }
        return metrics

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
