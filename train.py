"""
train.py - train the GPT from scratch.

Usage:
  python train.py --config configs/shakespeare_char.json          # Version A
  python train.py --config configs/tinystories.json               # Version B
  python train.py --config configs/shakespeare_char.json --max_iters 200 --run_name quick_test
  python train.py --config configs/tinystories.json --init_from resume   # continue after a stop

Any setting in TrainConfig can be overridden on the command line: --name value.

Writes:
  checkpoints/<run_name>/ckpt_best.pt     lowest validation loss so far
  checkpoints/<run_name>/ckpt_latest.pt   most recent (used by --init_from resume)
  checkpoints/<run_name>/config.json      the settings used
  samples/<run_name>/step_XXXXX.txt       generated text at chosen steps (for the montage)
  samples/<run_name>/all_samples.md       every sample in one file
  logs/<run_name>/train_log.csv           loss every log_interval steps
  logs/<run_name>/eval_log.csv            train/val loss + GPU temp/usage at every eval
"""

import argparse
import csv
import json
import math
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import encode_prompt, load_meta, load_tokenizer

ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- config

@dataclass
class TrainConfig:
    run_name: str = "shakespeare_char"     # folder name for checkpoints / samples / logs
    dataset: str = "shakespeare_char"      # folder inside data/
    init_from: str = "scratch"             # "scratch" or "resume"
    # model
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.2
    bias: bool = False
    # optimisation
    batch_size: int = 64
    grad_accum_steps: int = 1              # >1 simulates a bigger batch without more VRAM
    max_iters: int = 5000
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0                 # 0 disables
    # evaluation / logging
    eval_interval: int = 250
    eval_iters: int = 200
    log_interval: int = 10
    # samples for the video montage
    sample_steps: list = field(default_factory=lambda: [0, 500, 2000, 5000])
    sample_interval: int = 0               # also sample every N steps (0 = only sample_steps)
    sample_prompt: str = "\n"
    sample_tokens: int = 300
    sample_temperature: float = 0.8
    sample_top_k: int = 200
    # system
    device: str = "auto"                   # "auto", "cuda", "cpu"
    dtype: str = "auto"                    # "auto", "bfloat16", "float16", "float32"
    compile: bool = False                  # torch.compile needs Triton (tricky on Windows)
    seed: int = 1337


def _str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("1", "true", "yes", "y"):
        return True
    if s.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {s}")


def _int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def parse_config():
    """Defaults -> values from --config JSON -> command-line overrides."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", help="JSON file with settings")
    known, _ = pre.parse_known_args()

    cfg = TrainConfig()
    names = {f.name for f in fields(TrainConfig)}
    if known.config:
        path = Path(known.config)
        if not path.exists():
            path = ROOT / known.config
        with open(path, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if k not in names:
                    raise SystemExit(f"Unknown setting in {path}: {k}")
                setattr(cfg, k, v)

    parser = argparse.ArgumentParser(parents=[pre], description="Train a tiny GPT")
    for f in fields(TrainConfig):
        default = getattr(cfg, f.name)
        conv = {bool: _str2bool, list: _int_list}.get(f.type, f.type)
        parser.add_argument(f"--{f.name}", type=conv, default=default)
    args = parser.parse_args()
    for f in fields(TrainConfig):
        setattr(cfg, f.name, getattr(args, f.name))
    return cfg


# ----------------------------------------------------------------------------- device / precision

def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def resolve_dtype(name, device_type):
    """bf16 on the RTX 4060 (Ada supports it); fp16 on older GPUs; fp32 on CPU."""
    if name != "auto":
        return name
    if device_type == "cuda":
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


def make_autocast(device_type, dtype_name):
    """Mixed precision: most maths in 16-bit (faster, half the memory), weights stay 32-bit."""
    if dtype_name == "float32":
        return nullcontext()
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return torch.autocast(device_type=device_type, dtype=ptdtype)


# ----------------------------------------------------------------------------- data

class BinDataset:
    """Grabs random chunks of tokens from train.bin / val.bin."""

    def __init__(self, data_dir, block_size, batch_size, device):
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        for split in ("train", "val"):
            path = self.data_dir / f"{split}.bin"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing. Run prepare.py first.")
            n = os.path.getsize(path) // 2   # uint16 = 2 bytes per token
            if n <= block_size + 1:
                raise ValueError(f"{path} has only {n} tokens; need more than block_size={block_size}")

    def get_batch(self, split):
        # Re-open the memmap each call: avoids a known memory-leak with long-lived memmaps.
        data = np.memmap(self.data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if self.device.startswith("cuda"):
            # Pinned memory + non_blocking lets the copy to the GPU overlap with compute.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


# ----------------------------------------------------------------------------- helpers

def get_lr(it, cfg):
    """Linear warmup, then cosine decay from learning_rate down to min_lr."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))   # goes 1 -> 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, data, eval_iters, ctx):
    """Average loss over many random batches, for a less noisy number."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def gpu_stats():
    """Ask nvidia-smi for temperature, usage, memory and power (nice to show on camera)."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
        keys = ["gpu_temp_c", "gpu_util_pct", "gpu_mem_mb", "gpu_power_w"]
        stats = {}
        for key, val in zip(keys, parts):
            try:
                stats[key] = float(val)
            except ValueError:
                stats[key] = ""   # laptops often report power as [N/A]
        return stats
    except Exception:
        return {}


def generate_sample(model, tokenizer, cfg, device, ctx):
    """Write some text with the current model so we can watch it improve."""
    model.eval()
    ids = encode_prompt(tokenizer, cfg.sample_prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    stop = getattr(tokenizer, "eot_token", None)
    with torch.no_grad(), ctx:
        y = model.generate(x, cfg.sample_tokens, temperature=cfg.sample_temperature,
                           top_k=cfg.sample_top_k, stop_token=stop)
    model.train()
    return tokenizer.decode(y[0].tolist())


class RunLogger:
    """Writes CSV logs and sample text files for one run."""

    TRAIN_COLS = ["step", "loss", "lr", "ms_per_step", "tokens_per_sec"]
    EVAL_COLS = ["step", "train_loss", "val_loss", "lr", "elapsed_min",
                 "gpu_temp_c", "gpu_util_pct", "gpu_mem_mb", "gpu_power_w"]

    def __init__(self, log_dir, sample_dir, resume):
        self.log_dir = Path(log_dir)
        self.sample_dir = Path(sample_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.train_path = self.log_dir / "train_log.csv"
        self.eval_path = self.log_dir / "eval_log.csv"
        for path, cols in ((self.train_path, self.TRAIN_COLS), (self.eval_path, self.EVAL_COLS)):
            if not resume or not path.exists():
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(cols)

    def _append(self, path, cols, row):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in cols])

    def log_step(self, **row):
        self._append(self.train_path, self.TRAIN_COLS, row)

    def log_eval(self, **row):
        self._append(self.eval_path, self.EVAL_COLS, row)

    def save_sample(self, step, text, val_loss=None):
        (self.sample_dir / f"step_{step:05d}.txt").write_text(text, encoding="utf-8")
        loss_txt = f" (val loss {val_loss:.3f})" if val_loss is not None else ""
        with open(self.sample_dir / "all_samples.md", "a", encoding="utf-8") as f:
            f.write(f"\n## Step {step}{loss_txt}\n\n```\n{text}\n```\n")


def save_checkpoint(path, model, optimizer, scaler, model_args, iter_num, best_val_loss, cfg, meta):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": asdict(cfg),
        "meta": meta,             # tokenizer info travels with the model
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)         # atomic: a crash mid-save never corrupts the old file


def strip_compile_prefix(state_dict):
    """torch.compile renames weights to '_orig_mod.xxx'; undo that."""
    prefix = "_orig_mod."
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


# ----------------------------------------------------------------------------- main

def main():
    cfg = parse_config()
    torch.manual_seed(cfg.seed)

    device = resolve_device(cfg.device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype_name = resolve_dtype(cfg.dtype, device_type)
    ctx = make_autocast(device_type, dtype_name)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device} | precision: {dtype_name}")

    data_dir = ROOT / "data" / cfg.dataset
    meta = load_meta(data_dir / "meta.pkl")
    tokenizer = load_tokenizer(meta)
    data = BinDataset(data_dir, cfg.block_size, cfg.batch_size, device)

    ckpt_dir = ROOT / "checkpoints" / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume = cfg.init_from == "resume"

    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                      block_size=cfg.block_size, dropout=cfg.dropout, bias=cfg.bias,
                      vocab_size=meta["vocab_size"])
    iter_num, best_val_loss = 0, float("inf")
    ckpt = None
    if resume:
        ckpt_path = ckpt_dir / "ckpt_latest.pt"
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_args = ckpt["model_args"]       # architecture must match the saved one
        model = GPT(GPTConfig(**model_args))
        model.load_state_dict(strip_compile_prefix(ckpt["model"]))
        iter_num = ckpt["iter_num"]
        best_val_loss = ckpt["best_val_loss"]
    else:
        model = GPT(GPTConfig(**model_args))
    model.to(device)

    print(f"Parameters: {model.num_parameters() / 1e6:.2f}M total, "
          f"{model.num_parameters(non_embedding=True) / 1e6:.2f}M without embeddings")

    # GradScaler only does something for float16 (bf16 doesn't need loss scaling).
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype_name == "float16"))
    optimizer = model.configure_optimizer(cfg.weight_decay, cfg.learning_rate,
                                          (cfg.beta1, cfg.beta2), device_type)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        ckpt = None   # free memory

    raw_model = model
    if cfg.compile:
        print("Compiling model (first step will be slow) ...")
        model = torch.compile(model)

    logger = RunLogger(ROOT / "logs" / cfg.run_name, ROOT / "samples" / cfg.run_name, resume)
    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    tokens_per_iter = cfg.batch_size * cfg.grad_accum_steps * cfg.block_size
    print(f"Tokens per step: {tokens_per_iter:,} | steps: {cfg.max_iters:,}")

    def checkpoint(name):
        save_checkpoint(ckpt_dir / name, raw_model, optimizer, scaler, model_args,
                        iter_num, best_val_loss, cfg, meta)

    sample_steps = set(cfg.sample_steps)
    start_time = time.time()
    t0 = time.time()
    step_times = []
    x, y = data.get_batch("train")

    try:
        while True:
            lr = get_lr(iter_num, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # --- evaluate + checkpoint
            val_loss = None
            if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters:
                losses = estimate_loss(model, data, cfg.eval_iters, ctx)
                val_loss = losses["val"]
                stats = gpu_stats()
                elapsed = (time.time() - start_time) / 60
                temp = f" | GPU {stats['gpu_temp_c']:.0f}C" if isinstance(stats.get("gpu_temp_c"), float) else ""
                print(f"[eval] step {iter_num}: train {losses['train']:.4f}, val {val_loss:.4f}"
                      f" | {elapsed:.1f} min{temp}")
                logger.log_eval(step=iter_num, train_loss=round(losses["train"], 4),
                                val_loss=round(val_loss, 4), lr=lr, elapsed_min=round(elapsed, 2), **stats)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if iter_num > 0:
                        checkpoint("ckpt_best.pt")
                        print(f"  saved ckpt_best.pt (val {best_val_loss:.4f})")
                if iter_num > 0:
                    checkpoint("ckpt_latest.pt")

            # --- sample text for the montage
            if iter_num in sample_steps or (cfg.sample_interval and iter_num % cfg.sample_interval == 0):
                text = generate_sample(raw_model, tokenizer, cfg, device, ctx)
                logger.save_sample(iter_num, text, val_loss)
                print(f"--- sample @ step {iter_num} ---\n{text[:400]}\n---")

            if iter_num >= cfg.max_iters:
                break

            # --- one optimisation step (possibly several micro-batches)
            for _ in range(cfg.grad_accum_steps):
                with ctx:
                    _, loss = model(x, y)
                    loss = loss / cfg.grad_accum_steps
                x, y = data.get_batch("train")      # fetch next batch while the GPU works
                scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # --- timing + logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            step_times.append(dt)
            iter_num += 1
            if iter_num % cfg.log_interval == 0:
                lossf = loss.item() * cfg.grad_accum_steps   # .item() waits for the GPU
                recent = step_times[-cfg.log_interval:]
                avg = sum(recent) / len(recent)
                eta_min = (cfg.max_iters - iter_num) * avg / 60
                print(f"step {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, {avg * 1000:.0f} ms/step,"
                      f" ETA {eta_min:.1f} min")
                logger.log_step(step=iter_num, loss=round(lossf, 4), lr=lr,
                                ms_per_step=round(avg * 1000, 1),
                                tokens_per_sec=round(tokens_per_iter / avg))
                step_times = step_times[-cfg.log_interval:]
    except KeyboardInterrupt:
        print("\nStopped by user. Saving ckpt_latest.pt ...")
        checkpoint("ckpt_latest.pt")
        print(f"Resume with: python train.py --config <same config> --init_from resume")
        return

    print(f"\nDone. Best val loss {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Samples:     {ROOT / 'samples' / cfg.run_name}")
    if device_type == "cuda":
        print(f"Peak VRAM:   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
