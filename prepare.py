"""
prepare.py - download text, tokenize it, and split it into train.bin / val.bin.

Usage:
  python prepare.py --dataset shakespeare                    # Version A (~1 MB, char-level)
  python prepare.py --dataset tinystories                    # Version B (GPT-2 tokens, ~2 GB download)
  python prepare.py --dataset tinystories --max_stories 500000   # smaller, faster subset
  python prepare.py --dataset textfile --input transcripts/ --name my_videos --tokenizer char

Output (in code/data/<name>/):
  train.bin  - training tokens, uint16, one after another
  val.bin    - validation tokens (held out, used to check the model isn't just memorising)
  meta.pkl   - tokenizer info (vocab size, char table / encoding name)
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np

from tokenizer import CharTokenizer, TiktokenTokenizer, save_meta

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HF_CACHE = DATA_DIR / "hf_cache"   # keep Hugging Face downloads on G:, not C:
SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")


def download(url, dest):
    """Download a file once; skip if it already exists."""
    dest = Path(dest)
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def write_tokens(ids, path):
    """Save a list of token ids as raw uint16 numbers. Returns how many were written."""
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
    return len(arr)


def split_text(text, val_fraction):
    """First (1 - val_fraction) of the text is train, the last part is validation."""
    cut = int(len(text) * (1 - val_fraction))
    return text[:cut], text[cut:]


def _finish(out_dir, tok, n_train, n_val):
    save_meta(tok.to_meta(), out_dir / "meta.pkl")
    print(f"\nVocab size : {tok.vocab_size:,}")
    print(f"Train      : {n_train:,} tokens")
    print(f"Val        : {n_val:,} tokens")
    print(f"Written to : {out_dir}")


def prepare_char_text(text, out_dir, val_fraction):
    """Character-level: build a vocab from the text, encode, split, save."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = CharTokenizer.from_text(text)
    assert tok.vocab_size < 65535, "Too many unique characters for uint16"
    train_text, val_text = split_text(text, val_fraction)
    n_train = write_tokens(tok.encode(train_text), out_dir / "train.bin")
    n_val = write_tokens(tok.encode(val_text), out_dir / "val.bin")
    _finish(out_dir, tok, n_train, n_val)


def prepare_bpe_text(text, out_dir, val_fraction):
    """GPT-2 BPE: encode with tiktoken, split, save."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = TiktokenTokenizer("gpt2")
    train_text, val_text = split_text(text, val_fraction)
    n_train = write_tokens(tok.encode(train_text), out_dir / "train.bin")
    n_val = write_tokens(tok.encode(val_text), out_dir / "val.bin")
    _finish(out_dir, tok, n_train, n_val)


def read_text_source(path):
    """Read one .txt file, or every .txt file in a folder (joined with blank lines)."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.rglob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files in {p}")
        print(f"Reading {len(files)} text files from {p}")
        return "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
    return p.read_text(encoding="utf-8", errors="ignore")


def encode_stories_to_bin(split, tok, path, batch_size, num_threads):
    """
    Encode a Hugging Face split of stories straight to disk in batches, so the whole
    dataset never has to sit in RAM as Python lists. Each story ends with <|endoftext|>.
    """
    total = 0
    n = len(split)
    with open(path, "wb") as f:
        for start in range(0, n, batch_size):
            texts = split[start:start + batch_size]["text"]
            flat = []
            for ids in tok.encode_batch(texts, num_threads=num_threads):
                flat.extend(ids)
                flat.append(tok.eot_token)
            np.array(flat, dtype=np.uint16).tofile(f)
            total += len(flat)
            done = min(start + batch_size, n)
            if (start // batch_size) % 50 == 0 or done == n:
                print(f"  {path.name}: {done:,}/{n:,} stories, {total:,} tokens")
    return total


def prepare_shakespeare(args):
    out_dir = DATA_DIR / "shakespeare_char"
    raw = out_dir / "input.txt"
    download(SHAKESPEARE_URL, raw)
    text = raw.read_text(encoding="utf-8")
    print(f"Characters in dataset: {len(text):,}")
    prepare_char_text(text, out_dir, args.val_fraction)


def prepare_tinystories(args):
    from datasets import load_dataset  # only needed for this dataset
    out_dir = DATA_DIR / "tinystories"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading roneneldan/TinyStories (first time downloads ~2 GB) ...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir=str(HF_CACHE))
    train = ds["train"]
    if args.max_stories:
        train = train.select(range(min(args.max_stories, len(train))))
    tok = TiktokenTokenizer("gpt2")
    n_train = encode_stories_to_bin(train, tok, out_dir / "train.bin",
                                    args.batch_size, args.num_threads)
    n_val = encode_stories_to_bin(ds["validation"], tok, out_dir / "val.bin",
                                  args.batch_size, args.num_threads)
    _finish(out_dir, tok, n_train, n_val)


def prepare_textfile(args):
    if not args.input:
        raise SystemExit("--input is required for --dataset textfile")
    out_dir = DATA_DIR / args.name
    text = read_text_source(args.input)
    print(f"Characters in dataset: {len(text):,}")
    if args.tokenizer == "char":
        prepare_char_text(text, out_dir, args.val_fraction)
    else:
        prepare_bpe_text(text, out_dir, args.val_fraction)


def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset for train.py")
    parser.add_argument("--dataset", choices=["shakespeare", "tinystories", "textfile"],
                        default="shakespeare")
    parser.add_argument("--input", help="textfile: a .txt file or a folder of .txt files")
    parser.add_argument("--name", default="custom", help="textfile: output folder name")
    parser.add_argument("--tokenizer", choices=["char", "gpt2"], default="char",
                        help="textfile: char-level or GPT-2 BPE")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_stories", type=int, default=0,
                        help="tinystories: use only the first N training stories (0 = all)")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="tinystories: stories encoded per batch")
    parser.add_argument("--num_threads", type=int, default=8,
                        help="tinystories: tokenizer threads")
    args = parser.parse_args()

    if args.dataset == "shakespeare":
        prepare_shakespeare(args)
    elif args.dataset == "tinystories":
        prepare_tinystories(args)
    else:
        prepare_textfile(args)


if __name__ == "__main__":
    main()
