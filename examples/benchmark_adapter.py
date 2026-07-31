"""Latency, state-size, and numerical-health benchmark for Heraclitus 3."""
from __future__ import annotations

import argparse
import time

import torch

from heraclitus import HeraclitusAdapter, HeraclitusConfig


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return values[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--state-size", type=int, default=128)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--write-topk", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--activate-output", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    module = HeraclitusAdapter(
        HeraclitusConfig(
            hidden_size=args.hidden_size,
            state_size=args.state_size,
            memory_slots=args.memory_slots,
            num_heads=args.num_heads,
            write_topk=args.write_topk,
        )
    ).to(device=device, dtype=dtype).eval()
    if args.activate_output:
        with torch.no_grad():
            module.output.weight.normal_(std=0.01)

    hidden = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for _ in range(2):
            module.forward_with_state(hidden)
        synchronise(device)
        started = time.perf_counter()
        for _ in range(args.iterations):
            result = module.forward_with_state(hidden)
        synchronise(device)
        elapsed = time.perf_counter() - started

    tokens = args.batch_size * args.sequence_length * args.iterations
    state_bytes = (
        result.state.memory.numel() * result.state.memory.element_size()
        + result.state.usage.numel() * result.state.usage.element_size()
        + result.state.steps.numel() * result.state.steps.element_size()
    )
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"parameters={sum(parameter.numel() for parameter in module.parameters()):,}")
    print(f"state_bytes_per_batch={state_bytes:,}")
    print(f"tokens_per_second={tokens / elapsed:,.2f}")
    print(f"milliseconds_per_iteration={1000.0 * elapsed / args.iterations:,.3f}")
    maximum_ratio = float(result.diagnostics.maximum_residual_ratio.detach())
    effective_slots = float(result.diagnostics.effective_slots.detach())
    usage_maximum = float(result.diagnostics.usage_maximum.detach())
    print(f"maximum_residual_ratio={maximum_ratio:.6f}")
    print(f"effective_slots={effective_slots:.6f}")
    print(f"usage_maximum={usage_maximum:.6f}")
    if device.type == "cuda":
        print(f"peak_cuda_memory_bytes={torch.cuda.max_memory_allocated(device):,}")


if __name__ == "__main__":
    main()
