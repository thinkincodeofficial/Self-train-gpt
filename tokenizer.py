"""
tokenizer.py - turns text into numbers (token ids) and back.

Two tokenizers are supported:
  * CharTokenizer      - one token per character (used for Shakespeare, Version A)
  * TiktokenTokenizer  - GPT-2's BPE tokenizer via tiktoken (used for TinyStories, Version B)

The tokenizer's settings are saved as a small "meta" dict (meta.pkl) next to the
prepared data, and copied into every checkpoint, so sample.py can always rebuild
the exact same tokenizer the model was trained with.
"""

import pickle
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer: every unique character in the text gets an id."""

    kind = "char"

    def __init__(self, stoi, itos):
        self.stoi = dict(stoi)                               # "a" -> 0
        self.itos = {int(i): ch for i, ch in itos.items()}   # 0 -> "a"
        self.vocab_size = len(self.stoi)

    @classmethod
    def from_text(cls, text):
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    def encode(self, text):
        # Characters the model never saw are skipped instead of crashing.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)

    def to_meta(self):
        return {"kind": self.kind, "vocab_size": self.vocab_size,
                "stoi": self.stoi, "itos": self.itos}


class TiktokenTokenizer:
    """GPT-2 byte-pair-encoding tokenizer (50,257 tokens) from the tiktoken library."""

    kind = "tiktoken"

    def __init__(self, encoding_name="gpt2"):
        import tiktoken  # imported here so char-level runs don't need tiktoken
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        self.eot_token = self.enc.eot_token  # <|endoftext|>, separates stories

    def encode(self, text):
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def encode_batch(self, texts, num_threads=8):
        # Fast multi-threaded encoding, used by prepare.py for big datasets.
        return self.enc.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids])

    def to_meta(self):
        return {"kind": self.kind, "vocab_size": self.vocab_size,
                "encoding": self.encoding_name}


def load_tokenizer(meta):
    """Rebuild the right tokenizer from a meta dict."""
    if meta["kind"] == "char":
        return CharTokenizer(meta["stoi"], meta["itos"])
    if meta["kind"] == "tiktoken":
        return TiktokenTokenizer(meta["encoding"])
    raise ValueError(f"Unknown tokenizer kind: {meta['kind']}")


def encode_prompt(tokenizer, prompt):
    """Encode a prompt, never returning an empty list (the model needs >= 1 token to start)."""
    ids = tokenizer.encode(prompt) if prompt else []
    if ids:
        return ids
    if tokenizer.kind == "tiktoken":
        return [tokenizer.eot_token]           # "start of a new document"
    return [tokenizer.stoi.get("\n", 0)]       # start from a newline


def save_meta(meta, path):
    with open(Path(path), "wb") as f:
        pickle.dump(meta, f)


def load_meta(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)
