"""
Profiling script for nsys. Runs a single (model_size, context_length) config
with NVTX annotations for warmup, forward, backward, and optimizer phases.

Usage:
    uv run nsys profile -o profiles/profile_small_128 --force-overwrite true \
        python cs336_systems/benchmark/benchmark_profile.py --model_size small --context_length 128
"""

import argparse

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
}

VOCAB_SIZE = 10000
BATCH_SIZE = 4


def main():
    parser = argparse.ArgumentParser(description="Profile transformer with nsys/NVTX")
    parser.add_argument("--model_size", type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=5)
    args = parser.parse_args()

    batch_size = args.batch_size
    device = torch.device("cuda")
    config = MODEL_CONFIGS[args.model_size]

    print(f"Profiling {args.model_size} with context_length={args.context_length}, batch_size={batch_size}")

    try:
        model = BasicsTransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=args.context_length,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=10000.0,
        ).to(device)
    except torch.cuda.OutOfMemoryError:
        print(f"OOM creating model {args.model_size} — skipping")
        return

    optimizer = AdamW(model.parameters(), lr=1e-4)

    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, args.context_length), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (batch_size, args.context_length), device=device)

    # Warmup
    torch.cuda.nvtx.range_push("warmup")
    try:
        for _ in range(args.warmup_steps):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), targets.view(-1)
            )
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.nvtx.range_pop()
        print(f"OOM during warmup for {args.model_size} — skipping")
        return
    torch.cuda.nvtx.range_pop()  # warmup

    # Measured steps
    torch.cuda.nvtx.range_push("measured")
    for step in range(args.num_steps):
        torch.cuda.nvtx.range_push(f"step_{step}")

        optimizer.zero_grad()

        torch.cuda.nvtx.range_push("forward")
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, VOCAB_SIZE), targets.view(-1)
        )
        torch.cuda.nvtx.range_pop()  # forward

        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()  # backward

        torch.cuda.nvtx.range_push("optimizer")
        optimizer.step()
        torch.cuda.nvtx.range_pop()  # optimizer

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()  # step_N

    torch.cuda.nvtx.range_pop()  # measured

    print("Profiling complete.")


if __name__ == "__main__":
    main()
