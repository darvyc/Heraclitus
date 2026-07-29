"""Minimal latency, memory, and numerical-health benchmark for Heraclitus."""
from __future__ import annotations

import argparse
import time

import torch

from heraclitus import HeraclitusConfig, HeraclitusParameter


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--state-size", type=int, default=64)
    parser.add_argument("--num-shadows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    module = HeraclitusParameter(
        HeraclitusConfig(
            hidden_size=args.hidden_size,
            state_size=args.state_size,
            num_shadows=args.num_shadows,
        )
    ).to(device).eval()
    hidden = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device=device,
    )

    with torch.inference_mode():
        for _ in range(3):
            module(hidden)
        synchronise(device)
        started = time.perf_counter()
        for _ in range(args.iterations):
            result = module.forward_with_state(hidden)
        synchronise(device)
        elapsed = time.perf_counter() - started

    tokens = args.batch_size * args.sequence_length * args.iterations
    print(f"device={device}")
    print(f"parameters={sum(p.numel() for p in module.parameters()):,}")
    print(f"tokens_per_second={tokens / elapsed:,.2f}")
    print(f"milliseconds_per_iteration={1000.0 * elapsed / args.iterations:,.3f}")
    print(f"innovation_rms={float(result.diagnostics.innovation_rms):.6f}")
    print(f"surprise_mean={float(result.diagnostics.surprise_mean):.6f}")
    print(f"effective_shadows={float(result.diagnostics.effective_shadows):.6f}")
    print(f"residual_ratio={float(result.diagnostics.residual_ratio):.6f}")
    if device.type == "cuda":
        print(f"peak_cuda_memory_bytes={torch.cuda.max_memory_allocated(device):,}")


if __name__ == "__main__":
    main()
