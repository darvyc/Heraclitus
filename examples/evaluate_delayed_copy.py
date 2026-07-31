"""Small delayed-copy evaluation harness for Heraclitus 3.

This is a reproducible diagnostic, not a published benchmark result. It trains a
fresh adapter and classifier to recover a symbol after intervening filler tokens.
"""
from __future__ import annotations

import argparse
import random

import torch
from torch import Tensor, nn

from heraclitus import HeraclitusAdapter, HeraclitusConfig


def make_batch(
    batch_size: int,
    delay: int,
    hidden_size: int,
    vocabulary_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    labels = torch.randint(vocabulary_size, (batch_size,), device=device)
    hidden = torch.zeros(batch_size, delay + 2, hidden_size, device=device)
    hidden[torch.arange(batch_size, device=device), 0, labels] = 1.0
    hidden[:, 0, vocabulary_size] = 1.0
    hidden[:, 1:-1] = 0.02 * torch.randn(
        batch_size, delay, hidden_size, device=device
    )
    hidden[:, -1, vocabulary_size + 1] = 1.0
    return hidden, labels


def accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--delay", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--vocabulary-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.vocabulary_size + 2 > args.hidden_size:
        raise ValueError("hidden_size must be at least vocabulary_size + 2")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    adapter = HeraclitusAdapter(
        HeraclitusConfig(
            hidden_size=args.hidden_size,
            state_size=16,
            memory_slots=4,
            num_heads=4,
            write_topk=1,
            max_residual_ratio=0.25,
        )
    ).to(device)
    classifier = nn.Linear(args.hidden_size, args.vocabulary_size).to(device)
    optimiser = torch.optim.AdamW(
        list(adapter.parameters()) + list(classifier.parameters()),
        lr=3e-3,
    )

    for step in range(1, args.steps + 1):
        hidden, labels = make_batch(
            args.batch_size,
            args.delay,
            args.hidden_size,
            args.vocabulary_size,
            device,
        )
        result = adapter.forward_with_state(hidden, detach_state=False)
        logits = classifier(result.hidden_states[:, -1])
        loss = nn.functional.cross_entropy(logits, labels) + result.regularization_loss()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        if step in {1, args.steps} or step % 100 == 0:
            print(
                f"step={step} loss={float(loss.detach()):.4f} "
                f"accuracy={accuracy(logits, labels):.3f}"
            )

    with torch.inference_mode():
        hidden, labels = make_batch(
            1024,
            args.delay,
            args.hidden_size,
            args.vocabulary_size,
            device,
        )
        adapted = adapter.forward_with_state(hidden).hidden_states[:, -1]
        adapter_accuracy = accuracy(classifier(adapted), labels)
        no_memory_accuracy = accuracy(classifier(hidden[:, -1]), labels)

    print(f"validation_adapter_accuracy={adapter_accuracy:.3f}")
    print(f"validation_no_memory_accuracy={no_memory_accuracy:.3f}")
    print(f"chance_accuracy={1.0 / args.vocabulary_size:.3f}")


if __name__ == "__main__":
    main()
