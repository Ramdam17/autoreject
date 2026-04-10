"""Benchmark batched consensus scoring vs current loop approach.

Compares:
- Current: 11 separate (indexing + mean + RMSE) per fold
- Batched: 1 einsum + 1 RMSE per fold
- Also compares fast_median (topk) vs _torch_median (sort)

Usage
-----
    python -m autoreject.benchmarks.bench_scoring
    python -m autoreject.benchmarks.bench_scoring --profile realistic
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import argparse
import logging
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROFILES = {
    "medium": {"n_epochs": 200, "n_channels": 64, "n_times": 1000},
    "realistic": {"n_epochs": 400, "n_channels": 128, "n_times": 1500},
}


def _sync(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def bench_current_scoring(X_gpu, bad_sensor_counts_train, train_t,
                          consensus_values, n_channels, median_gpu,
                          optimizer, device, n_repeats=5):
    """Current approach: loop over consensus values."""
    import torch

    times = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()

        X_train = X_gpu[train_t]
        scores = torch.zeros(len(consensus_values), device=device)

        sorted_idx = np.argsort(bad_sensor_counts_train)[::-1]
        sorted_counts = bad_sensor_counts_train[sorted_idx]

        for c_idx, this_consensus in enumerate(consensus_values):
            n_consensus_ch = this_consensus * n_channels
            bad_epochs = np.zeros(len(bad_sensor_counts_train), dtype=bool)
            if np.max(sorted_counts) >= n_consensus_ch:
                n_drop = np.sum(sorted_counts >= n_consensus_ch)
                bad_epochs[sorted_idx[:n_drop]] = True

            good_mask = ~bad_epochs
            n_good = good_mask.sum()

            if n_good == 0:
                scores[c_idx] = float('-inf')
                continue

            good_idx = np.nonzero(good_mask)[0]
            good_idx_t = torch.tensor(good_idx, device=device)

            X_good = X_train[good_idx_t]
            mean_gpu = X_good.mean(dim=0)

            sq_diff = (median_gpu - mean_gpu) ** 2
            scores[c_idx] = -sq_diff.mean().sqrt()

        _sync(device)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    return np.median(times)


def bench_batched_scoring(X_gpu, bad_sensor_counts_train, train_t,
                          consensus_values, n_channels, median_gpu,
                          optimizer, device, n_repeats=5):
    """Batched approach: weight matrix + einsum."""
    import torch
    from autoreject.kernels.batched_scoring import (
        build_consensus_weights,
        batched_consensus_score,
    )

    times = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()

        X_train = X_gpu[train_t]

        # Build weights on CPU (fast)
        weights, valid_mask = build_consensus_weights(
            bad_sensor_counts_train, consensus_values, n_channels, None
        )
        weights_gpu = torch.tensor(weights, device=device)

        # Batched scoring
        scores_np = batched_consensus_score(
            X_train, median_gpu, weights_gpu, valid_mask, torch
        )

        _sync(device)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    return np.median(times)


def bench_median_sort(X_test_gpu, device, n_repeats=10):
    """Current: _torch_median (full sort)."""
    from autoreject.gpu_pipeline import _torch_median

    times = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()
        _torch_median(X_test_gpu, dim=0)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def bench_median_topk(X_test_gpu, device, n_repeats=10):
    """New: fast_median (topk)."""
    import torch
    from autoreject.kernels.batched_scoring import fast_median

    times = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()
        fast_median(X_test_gpu, dim=0, torch_module=torch)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def main():
    import torch
    from autoreject.gpu_pipeline import GPUThresholdOptimizer, _torch_median

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="realistic",
                        choices=list(PROFILES.keys()))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    params = PROFILES[args.profile]
    n_epochs = params["n_epochs"]
    n_channels = params["n_channels"]
    n_times = params["n_times"]

    rng = np.random.RandomState(42)
    optimizer = GPUThresholdOptimizer(device=device)

    # Generate data
    X = rng.randn(n_epochs, n_channels, n_times).astype(np.float32) * 1e-5
    X_gpu = optimizer._to_tensor(X)

    bad_sensor_counts = rng.randint(0, n_channels // 4, size=n_epochs)
    consensus_values = np.linspace(0.0, 1.0, 11)

    # Simulate one fold
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=10)
    train, test = list(cv.split(np.zeros(n_epochs)))[0]

    train_t = torch.tensor(train, device=device)
    test_t = torch.tensor(test, device=device)

    X_test = X_gpu[test_t]
    median_gpu = _torch_median(X_test, dim=0)
    bad_counts_train = bad_sensor_counts[train]

    print()
    print("=" * 60)
    print(f"  Scoring Optimization Benchmark")
    print(f"  Profile: {args.profile} ({n_epochs}×{n_channels}×{n_times})")
    print(f"  Device: {device}")
    print("=" * 60)

    # --- Scoring benchmark ---
    print()
    print("--- Consensus Scoring (per fold) ---")

    # Warmup
    bench_current_scoring(X_gpu, bad_counts_train, train_t, consensus_values,
                          n_channels, median_gpu, optimizer, device, n_repeats=2)
    bench_batched_scoring(X_gpu, bad_counts_train, train_t, consensus_values,
                          n_channels, median_gpu, optimizer, device, n_repeats=2)

    t_current = bench_current_scoring(
        X_gpu, bad_counts_train, train_t, consensus_values,
        n_channels, median_gpu, optimizer, device
    )
    t_batched = bench_batched_scoring(
        X_gpu, bad_counts_train, train_t, consensus_values,
        n_channels, median_gpu, optimizer, device
    )

    print(f"  Current (loop):  {t_current:>8.1f} ms/fold")
    print(f"  Batched (einsum): {t_batched:>7.1f} ms/fold")
    print(f"  Speedup:         {t_current / t_batched:>8.1f}x")

    # --- Median benchmark ---
    print()
    print("--- Median Computation (per fold) ---")

    # Warmup
    bench_median_sort(X_test, device, n_repeats=2)
    bench_median_topk(X_test, device, n_repeats=2)

    t_sort = bench_median_sort(X_test, device)
    t_topk = bench_median_topk(X_test, device)

    print(f"  Sort-based:    {t_sort:>8.1f} ms")
    print(f"  TopK-based:    {t_topk:>8.1f} ms")
    print(f"  Speedup:       {t_sort / t_topk:>8.1f}x")

    # --- Verify correctness ---
    print()
    print("--- Correctness ---")
    from autoreject.kernels.batched_scoring import fast_median

    med_sort = _torch_median(X_test, dim=0)
    med_topk = fast_median(X_test, dim=0, torch_module=torch)

    diff = (med_sort - med_topk).abs().max().item()
    print(f"  Median max diff: {diff:.2e}")
    print(f"  Median match:    {'OK' if diff < 1e-5 else 'FAIL'}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
