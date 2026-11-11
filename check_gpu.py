"""
check_gpu.py - setup step 4: confirm PyTorch can see the RTX 4060.

Usage:
  python check_gpu.py
"""

import sys
import time


def main():
    print(f"Python : {sys.version.split()[0]}")
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is not installed. Use the selector at pytorch.org (Windows, Pip, CUDA).")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {ok}")
    if not ok:
        print("\nNo GPU found. You probably installed the CPU-only build.\n"
              "Uninstall torch and reinstall the CUDA build from pytorch.org.")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
    print(f"bf16   : {torch.cuda.is_bf16_supported()}")

    # Tiny speed test: a few big matrix multiplies in bf16.
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b                          # warm-up
    torch.cuda.synchronize()
    t0 = time.time()
    n = 20
    for _ in range(n):
        a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    tflops = 2 * 4096 ** 3 * n / dt / 1e12
    print(f"Matmul : {tflops:.1f} TFLOPS (bf16)")
    print("\nAll good. Plug in the charger and set performance mode before training.")


if __name__ == "__main__":
    main()
