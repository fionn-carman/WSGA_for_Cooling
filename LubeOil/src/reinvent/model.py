"""GRU prior/agent network for REINVENT.

Standard REINVENT architecture (Olivecrona et al. 2017):
    Embedding(vocab_size, 128) -> GRU(128, 512, 3 layers) -> Linear(512, vocab_size)

No value head (this is not PPO). The same class is used for both the frozen
prior and the trainable agent — they share architecture but differ in weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModel(nn.Module):
    """Autoregressive GRU for character-level SMILES generation."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512,
                 num_layers=3, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        """Forward pass.

        Args:
            x: (batch, seq_len) token indices.
            hidden: optional (num_layers, batch, hidden_dim) GRU state.

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: updated GRU hidden state.
        """
        emb = self.embedding(x)
        out, hidden = self.gru(emb, hidden)
        logits = self.output_layer(out)
        return logits, hidden

    def log_prob_of_sequence(self, encoded_batch, pad_idx=0):
        """Compute summed log-probability of encoded sequences.

        Args:
            encoded_batch: (B, L) tensor with SOS prefix and EOS suffix,
                           padded with pad_idx.

        Returns:
            (B,) tensor of per-sequence summed log-probabilities.
        """
        x = encoded_batch[:, :-1]       # inputs  (all but last token)
        targets = encoded_batch[:, 1:]   # targets (all but first token)

        logits, _ = self.forward(x)
        log_probs = F.log_softmax(logits, dim=-1)

        mask = (targets != pad_idx).float()
        token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return (token_lp * mask).sum(dim=1)

    @torch.no_grad()
    def sample(self, vocab, batch_size=128, max_len=80, temperature=1.0):
        """Sample SMILES autoregressively.

        Returns:
            smiles: list[str] of decoded SMILES.
            log_probs: (batch_size,) tensor of summed log-probabilities
                       (detached — for logging only, not for backprop).
        """
        device = next(self.parameters()).device
        hidden = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device,
        )
        input_tok = torch.full(
            (batch_size, 1), vocab.sos_idx, dtype=torch.long, device=device,
        )

        sequences = [[] for _ in range(batch_size)]
        total_lp = torch.zeros(batch_size, device=device)
        finished = [False] * batch_size

        for _ in range(max_len):
            logits, hidden = self.forward(input_tok, hidden)
            logits = logits[:, -1, :] / temperature
            logits = torch.clamp(logits, min=-50, max=50)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            tokens = dist.sample()
            log_probs_step = dist.log_prob(tokens)  # (batch_size,)

            for i in range(batch_size):
                if finished[i]:
                    continue
                tok = tokens[i].item()
                if tok == vocab.eos_idx:
                    finished[i] = True
                else:
                    sequences[i].append(tok)
                    total_lp[i] += log_probs_step[i]

            if all(finished):
                break
            input_tok = tokens.unsqueeze(1)

        smiles = [vocab.decode(seq) for seq in sequences]
        return smiles, total_lp
