"""
Benchmark all model configurations from Table 1 and generate a results table.

Usage:
    uv run python cs336_systems/benchmark/benchmark_all.py
    uv run python cs336_systems/benchmark/benchmark_all.py --context_length 256
"""

import argparse
import timeit
import statistics

import torch
import pandas as pd

from cs336_basics.model import BasicsTransformerLM

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

VOCAB_SIZE = 10000
BATCH_SIZE = 4


def benchmark_config(
    name: str,
    config: dict,
    context_length: int,
    warmup_steps: int,
    num_steps: int,
    device: torch.device,
) -> dict:
    """Benchmark a single model configuration, returns timing results."""
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)

    def forward_step():
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, VOCAB_SIZE), targets.view(-1)
        )
        return loss

    def forward_backward_step():
        model.zero_grad(set_to_none=True)
        loss = forward_step()
        loss.backward()
        return loss

    results = {"size": name, "params": num_params}

    for mode, step_fn in [("forward", forward_step), ("both", forward_backward_step)]:
        # Warm-up
        for _ in range(warmup_steps):
            step_fn()
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timed steps
        times = []
        for _ in range(num_steps):
            start = timeit.default_timer()
            step_fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)

        mean_ms = statistics.mean(times) * 1000
        std_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
        results[f"{mode}_mean_ms"] = mean_ms
        results[f"{mode}_std_ms"] = std_ms

    # Free memory
    del model, input_ids, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Context length: {args.context_length}, Batch size: {BATCH_SIZE}")
    print(f"Warmup: {args.warmup_steps}, Measurement steps: {args.num_steps}")
    print()

    rows = []
    for name, config in MODEL_CONFIGS.items():
        print(f"Benchmarking {name} ...")
        result = benchmark_config(
            name, config, args.context_length, args.warmup_steps, args.num_steps, device
        )
        rows.append(result)
        print(
            f"  Forward: {result['forward_mean_ms']:.2f} +/- {result['forward_std_ms']:.2f} ms | "
            f"Forward+Backward: {result['both_mean_ms']:.2f} +/- {result['both_std_ms']:.2f} ms"
        )

    df = pd.DataFrame(rows)
    df["params"] = df["params"].apply(lambda x: f"{x:,}")
    df = df.rename(columns={
        "size": "Size",
        "params": "Parameters",
        "forward_mean_ms": "Forward Mean (ms)",
        "forward_std_ms": "Forward Std (ms)",
        "both_mean_ms": "Fwd+Bwd Mean (ms)",
        "both_std_ms": "Fwd+Bwd Std (ms)",
    })

    print("\n" + "=" * 80)
    print(df.to_markdown(index=False, floatfmt=".2f"))
    print("=" * 80)


if __name__ == "__main__":
    main()
