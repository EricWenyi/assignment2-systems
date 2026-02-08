"""
End-to-end benchmarking script for the Transformer model.

Measures wall-clock time for forward and backward passes.

Usage examples:
    # Forward only, default hyperparams
    python benchmark/benchmark.py --mode forward

    # Forward + backward
    python benchmark/benchmark.py --mode both

    # Custom model size
    python benchmark/benchmark.py --num_layers 12 --d_model 768 --num_heads 12 --d_ff 3072

    # Adjust warmup and timing steps
    python benchmark/benchmark.py --warmup_steps 5 --num_steps 20
"""

import argparse
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Transformer forward/backward passes")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Data hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)

    # Benchmarking parameters
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["forward", "both"],
        default="both",
        help="'forward' for forward pass only, 'both' for forward + backward",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(
        f"Config: layers={args.num_layers}, d_model={args.d_model}, "
        f"heads={args.num_heads}, d_ff={args.d_ff}, ctx={args.context_length}"
    )
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: {args.mode}")
    print(f"Warmup steps: {args.warmup_steps}, Timed steps: {args.num_steps}")
    print()

    # Generate random batch of data
    input_ids = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=device
    )
    # Targets shifted by one (standard LM setup)
    targets = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=device
    )

    def forward_step():
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, args.vocab_size), targets.view(-1)
        )
        return loss

    def forward_backward_step():
        model.zero_grad(set_to_none=True)
        loss = forward_step()
        loss.backward()
        return loss

    step_fn = forward_step if args.mode == "forward" else forward_backward_step

    # Warm-up
    print("Running warm-up steps...")
    for _ in range(args.warmup_steps):
        step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed steps
    print("Running timed steps...")
    step_times = []
    for _ in range(args.num_steps):
        start = timeit.default_timer()
        step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        step_times.append(end - start)

    # Report results
    mean_time = sum(step_times) / len(step_times)
    min_time = min(step_times)
    max_time = max(step_times)
    print(f"\n--- Results ({args.mode}) ---")
    print(f"Mean step time: {mean_time * 1000:.2f} ms")
    print(f"Min  step time: {min_time * 1000:.2f} ms")
    print(f"Max  step time: {max_time * 1000:.2f} ms")
    print(f"All step times (ms): {[f'{t * 1000:.2f}' for t in step_times]}")


if __name__ == "__main__":
    main()
