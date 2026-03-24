"""Character-level SMILES vocabulary for REINVENT."""

import json

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"


class SMILESVocabulary:
    """Character-level tokenizer for SMILES strings.

    Special tokens: PAD=0, SOS=1, EOS=2.  All remaining tokens are
    single SMILES characters sorted alphabetically.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def build(self, smiles_list):
        """Build vocabulary from a list of SMILES strings."""
        chars = set()
        for smi in smiles_list:
            if isinstance(smi, str):
                chars.update(smi)
        special = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        all_tokens = special + sorted(chars)
        self.char2idx = {tok: i for i, tok in enumerate(all_tokens)}
        self.idx2char = {i: tok for i, tok in enumerate(all_tokens)}
        return self

    @property
    def pad_idx(self):
        return self.char2idx[PAD_TOKEN]

    @property
    def sos_idx(self):
        return self.char2idx[SOS_TOKEN]

    @property
    def eos_idx(self):
        return self.char2idx[EOS_TOKEN]

    def __len__(self):
        return len(self.char2idx)

    def encode(self, smi):
        """Encode SMILES string → list of token indices (with SOS/EOS)."""
        return (
            [self.sos_idx]
            + [self.char2idx.get(c, self.pad_idx) for c in smi]
            + [self.eos_idx]
        )

    def decode(self, indices):
        """Decode list of token indices → SMILES string."""
        chars = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx in (self.sos_idx, self.pad_idx):
                continue
            chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

    def save(self, path):
        """Save vocabulary to JSON file."""
        data = {
            "char2idx": self.char2idx,
            "idx2char": {str(k): v for k, v in self.idx2char.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load vocabulary from JSON file."""
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        vocab.char2idx = data["char2idx"]
        vocab.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        return vocab
