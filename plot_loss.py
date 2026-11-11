"""
plot_loss.py - turn the training logs into a chart for the video.

Usage:
  python plot_loss.py --run shakespeare_char
  python plot_loss.py --run tinystories --dark --out loss_dark.png
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    """Read a CSV into a list of dicts with numbers converted to float (blanks stay None)."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else None) for k, v in row.items()})
    return rows


def ema(values, alpha=0.1):
    """Exponential moving average: smooths the noisy per-step loss line."""
    out, avg = [], None
    for v in values:
        avg = v if avg is None else alpha * v + (1 - alpha) * avg
        out.append(avg)
    return out


def main():
    p = argparse.ArgumentParser(description="Plot training loss")
    p.add_argument("--run", default="shakespeare_char")
    p.add_argument("--out", help="output image (default logs/<run>/loss.png)")
    p.add_argument("--dark", action="store_true", help="dark background for the video")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # save to file, no window
    import matplotlib.pyplot as plt

    log_dir = ROOT / "logs" / args.run
    train = read_csv(log_dir / "train_log.csv")
    evals = read_csv(log_dir / "eval_log.csv")
    if not train and not evals:
        raise SystemExit(f"No logs in {log_dir}. Train first.")

    if args.dark:
        plt.style.use("dark_background")
    has_temp = any(r.get("gpu_temp_c") is not None for r in evals)
    fig, axes = plt.subplots(2 if has_temp else 1, 1, figsize=(12, 8 if has_temp else 6),
                             sharex=True, squeeze=False)
    ax = axes[0][0]

    if train:
        steps = [r["step"] for r in train]
        losses = [r["loss"] for r in train]
        ax.plot(steps, losses, alpha=0.25, linewidth=1, label="train loss (raw)")
        ax.plot(steps, ema(losses), linewidth=2, label="train loss (smoothed)")
    if evals:
        ax.plot([r["step"] for r in evals], [r["val_loss"] for r in evals],
                "o-", linewidth=2, label="validation loss")
    ax.set_ylabel("loss (lower = better)")
    ax.set_title(f"Training: {args.run}")
    ax.grid(alpha=0.3)
    ax.legend()

    if has_temp:
        t = [r for r in evals if r.get("gpu_temp_c") is not None]
        axes[1][0].plot([r["step"] for r in t], [r["gpu_temp_c"] for r in t], "o-", color="tab:red")
        axes[1][0].set_ylabel("GPU temp (C)")
        axes[1][0].grid(alpha=0.3)
    axes[-1][0].set_xlabel("step")

    out = Path(args.out) if args.out else log_dir / "loss.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
