"""
Run nsys profiling for all (model_size, context_length) combos and produce a summary table.
Tries batch_size=4 first; if OOM, retries with batch_size=1.

Usage:
    uv run python cs336_systems/benchmark/profile_all.py
"""

import subprocess
import os

MODEL_SIZES = ["small", "medium", "large", "xl"]
CONTEXT_LENGTHS = [128, 256, 512, 1024]
PROFILES_DIR = "profiles"
WARMUP_STEPS = 3
NUM_STEPS = 5


def run_profile(model_size, context_length, batch_size):
    """Run nsys profile for a single config. Returns path to .nsys-rep or None on failure."""
    profile_name = f"{PROFILES_DIR}/profile_{model_size}_{context_length}"
    rep_file = f"{profile_name}.nsys-rep"

    cmd = [
        "nsys", "profile",
        "-o", profile_name,
        "--force-overwrite", "true",
        "--capture-range=none",
        "python", "cs336_systems/benchmark/benchmark_profile.py",
        "--model_size", model_size,
        "--context_length", str(context_length),
        "--batch_size", str(batch_size),
        "--warmup_steps", str(WARMUP_STEPS),
        "--num_steps", str(NUM_STEPS),
    ]

    print(f"  Profiling {model_size} ctx={context_length} bs={batch_size} ...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if "OOM" in result.stdout or "OutOfMemoryError" in result.stderr:
            print(f"    OOM with bs={batch_size}")
            return None
        if result.returncode != 0:
            print(f"    Failed (rc={result.returncode})")
            if result.stderr:
                # Check for OOM in stderr
                if "OutOfMemory" in result.stderr or "out of memory" in result.stderr.lower():
                    print(f"    OOM detected in stderr")
                    return None
                print(f"    stderr: {result.stderr[-200:]}")
            return None
        if os.path.exists(rep_file):
            print(f"    Done -> {rep_file}")
            return rep_file
        else:
            print(f"    No output file generated")
            return None
    except subprocess.TimeoutExpired:
        print(f"    Timeout — skipping")
        return None


def parse_nvtx_stats(rep_file):
    """Parse nsys stats to extract forward/backward/optimizer times."""
    cmd = [
        "nsys", "stats", rep_file,
        "--report", "nvtx_pushpop_sum",
        "--format", "csv",
        "--force-export", "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None

    times = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        range_name = parts[-1].strip().strip('"').strip(":")
        if range_name in ("forward", "backward", "optimizer"):
            try:
                avg_ns = float(parts[3].strip().strip('"'))
                times[range_name] = avg_ns / 1e6  # convert to ms
            except (ValueError, IndexError):
                pass
    return times


def main():
    os.makedirs(PROFILES_DIR, exist_ok=True)

    results = []
    for model_size in MODEL_SIZES:
        for ctx_len in CONTEXT_LENGTHS:
            # Try batch_size=4 first, then fall back to 1
            rep_file = run_profile(model_size, ctx_len, batch_size=4)
            batch_used = 4
            if rep_file is None:
                rep_file = run_profile(model_size, ctx_len, batch_size=1)
                batch_used = 1

            if rep_file is None:
                results.append({
                    "model": model_size,
                    "ctx_len": ctx_len,
                    "batch_size": "-",
                    "forward": "OOM",
                    "backward": "OOM",
                    "optimizer": "OOM",
                    "total": "OOM",
                })
            else:
                times = parse_nvtx_stats(rep_file)
                if times and "forward" in times:
                    total = sum(times.get(k, 0) for k in ("forward", "backward", "optimizer"))
                    results.append({
                        "model": model_size,
                        "ctx_len": ctx_len,
                        "batch_size": str(batch_used),
                        "forward": f"{times.get('forward', 0):.2f}",
                        "backward": f"{times.get('backward', 0):.2f}",
                        "optimizer": f"{times.get('optimizer', 0):.2f}",
                        "total": f"{total:.2f}",
                    })
                else:
                    results.append({
                        "model": model_size,
                        "ctx_len": ctx_len,
                        "batch_size": str(batch_used),
                        "forward": "parse error",
                        "backward": "parse error",
                        "optimizer": "parse error",
                        "total": "parse error",
                    })

    # Print table
    print("\n" + "=" * 100)
    print(f"{'Model':<8} {'Ctx Len':>8} {'BS':>4} {'Forward (ms)':>14} {'Backward (ms)':>15} {'Optimizer (ms)':>16} {'Total (ms)':>12}")
    print("-" * 100)
    for r in results:
        print(f"{r['model']:<8} {r['ctx_len']:>8} {r['batch_size']:>4} {r['forward']:>14} {r['backward']:>15} {r['optimizer']:>16} {r['total']:>12}")
    print("=" * 100)

    # Markdown table
    print("\n### Markdown Table\n")
    print("| Model | Context Length | Batch Size | Forward (ms) | Backward (ms) | Optimizer (ms) | Total (ms) |")
    print("|-------|---------------|-----------|-------------|--------------|---------------|------------|")
    for r in results:
        print(f"| {r['model']} | {r['ctx_len']} | {r['batch_size']} | {r['forward']} | {r['backward']} | {r['optimizer']} | {r['total']} |")


if __name__ == "__main__":
    main()
