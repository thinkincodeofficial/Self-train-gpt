"""
sample.py - load a trained checkpoint and generate text from a prompt.

Usage:
  python sample.py --run shakespeare_char --prompt "ROMEO:"
  python sample.py --run tinystories --prompt "Once upon a time" --num_samples 5 --out samples/best_of.txt
  python sample.py --ckpt checkpoints/tinystories/ckpt_latest.pt --temperature 1.2
  python sample.py --run tinystories --prompt_file my_prompt.txt --no_stream
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import encode_prompt, load_tokenizer

ROOT = Path(__file__).resolve().parent


def load_model(ckpt_path, device):
    """Rebuild the model and tokenizer exactly as they were saved."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    prefix = "_orig_mod."
    state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    tokenizer = load_tokenizer(ckpt["meta"])
    info = {"iter_num": ckpt.get("iter_num"), "best_val_loss": ckpt.get("best_val_loss")}
    return model, tokenizer, info


def stream_text(model, tokenizer, ids, args, device, ctx):
    """Print tokens as they are generated (looks great on camera). Returns the full text."""
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    stop = getattr(tokenizer, "eot_token", None) if args.stop_at_eot else None
    new_ids, printed = [], ""
    with ctx:
        for next_id in model.generate_stream(x, args.max_new_tokens, args.temperature,
                                             args.top_k, stop_token=stop):
            new_ids.append(next_id.item())
            text = tokenizer.decode(new_ids)
            # BPE can split one emoji/accent over several tokens; wait until it's complete.
            if text.endswith("�"):
                continue
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
    print()
    return tokenizer.decode(ids + new_ids)


def batch_text(model, tokenizer, ids, args, device, ctx):
    """Generate everything first, then return it."""
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    stop = getattr(tokenizer, "eot_token", None) if args.stop_at_eot else None
    with torch.no_grad(), ctx:
        y = model.generate(x, args.max_new_tokens, args.temperature, args.top_k, stop_token=stop)
    return tokenizer.decode(y[0].tolist())


def main():
    p = argparse.ArgumentParser(description="Generate text with a trained tiny GPT")
    p.add_argument("--run", default="shakespeare_char", help="run name inside checkpoints/")
    p.add_argument("--which", choices=["best", "latest"], default="best")
    p.add_argument("--ckpt", help="explicit checkpoint path (overrides --run/--which)")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--prompt_file", help="read the prompt from a text file")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = always pick the top token")
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_stream", action="store_true", help="don't print token by token")
    p.add_argument("--no_stop_at_eot", dest="stop_at_eot", action="store_false",
                   help="keep going past <|endoftext|> (BPE models only)")
    p.add_argument("--out", help="also save all samples to this file")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if device_type == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dt)
    else:
        ctx = nullcontext()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / args.run / f"ckpt_{args.which}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    model, tokenizer, info = load_model(ckpt_path, device)
    print(f"Loaded {ckpt_path} (step {info['iter_num']}, best val loss {info['best_val_loss']:.4f})\n")

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    prompt = prompt.replace("\\n", "\n")   # let "--prompt 'Hi\nthere'" work from the shell
    ids = encode_prompt(tokenizer, prompt)

    results = []
    for i in range(args.num_samples):
        print(f"===== sample {i + 1}/{args.num_samples} =====")
        if args.no_stream:
            text = batch_text(model, tokenizer, ids, args, device, ctx)
            print(text)
        else:
            sys.stdout.write(prompt)
            text = stream_text(model, tokenizer, ids, args, device, ctx)
        results.append(text)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n\n=====\n\n".join(results), encoding="utf-8")
        print(f"\nSaved {len(results)} samples to {out}")


if __name__ == "__main__":
    main()
