"""Benchmark batched interpolation vs per-epoch loop.

Usage
-----
    python -m autoreject.benchmarks.bench_interpolation
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import argparse
import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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


def make_interp_data(n_epochs, n_picks, n_times, device, rng):
    """Generate interpolation test data and cache."""
    import torch
    from autoreject.gpu_interpolation import _calc_g_torch

    data = rng.randn(n_epochs, n_picks, n_times).astype(np.float32) * 1e-5
    data_gpu = torch.tensor(data, dtype=torch.float32, device=device)

    # Random positions on unit sphere
    pos = rng.randn(n_picks, 3).astype(np.float64)
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)

    pos_t = torch.tensor(pos, dtype=torch.float64, device="cpu")
    cosang_all = pos_t @ pos_t.T
    G_all = _calc_g_torch(cosang_all)

    # Generate bad channel patterns (~10% bad per epoch, limited variety)
    interp_channels = []
    n_patterns = min(30, n_picks // 3)
    patterns = []
    for _ in range(n_patterns):
        n_bad = rng.randint(1, max(2, n_picks // 10))
        pat = sorted(rng.choice(n_picks, n_bad, replace=False).tolist())
        patterns.append(pat)

    for _ in range(n_epochs):
        interp_channels.append(patterns[rng.randint(0, len(patterns))])

    # Build interp_cache
    interp_cache = {}
    compute_dtype = torch.float64
    data_dtype = torch.float32

    for bad_ch_indices in interp_channels:
        cache_key = tuple(sorted(bad_ch_indices))
        if cache_key in interp_cache:
            continue

        goods_mask = np.ones(n_picks, dtype=bool)
        for idx in bad_ch_indices:
            goods_mask[idx] = False

        good_idx = np.where(goods_mask)[0]
        bad_idx = np.where(~goods_mask)[0]

        good_idx_t = torch.tensor(good_idx, dtype=torch.long)
        bad_idx_t = torch.tensor(bad_idx, dtype=torch.long)

        G_from = G_all[good_idx_t][:, good_idx_t]
        G_to_from = G_all[bad_idx_t][:, good_idx_t]

        n_from = len(good_idx)
        G_from_reg = G_from + 1e-5 * torch.eye(n_from, dtype=compute_dtype)

        ones_col = torch.ones((n_from, 1), dtype=compute_dtype)
        ones_row = torch.ones((1, n_from), dtype=compute_dtype)
        zero = torch.zeros((1, 1), dtype=compute_dtype)

        C = torch.cat([
            torch.cat([G_from_reg, ones_col], dim=1),
            torch.cat([ones_row, zero], dim=1),
        ], dim=0)

        C_inv = torch.linalg.pinv(C)

        n_bad = len(bad_idx)
        ones_to = torch.ones((n_bad, 1), dtype=compute_dtype)
        interpolation = (
            torch.cat([G_to_from, ones_to], dim=1) @ C_inv[:, :-1]
        )
        interpolation = interpolation.to(device=device, dtype=data_dtype)

        interp_cache[cache_key] = (interpolation, good_idx, bad_idx)

    n_unique = len(interp_cache)
    logger.info("Generated %d epochs with %d unique bad-channel patterns",
                n_epochs, n_unique)

    return data_gpu, interp_channels, interp_cache


def bench_current_interp(data_gpu, interp_channels, interp_cache, device,
                         n_repeats=5):
    """Current: per-epoch matmul loop."""
    import torch

    times = []
    for _ in range(n_repeats):
        data = data_gpu.clone()
        _sync(device)
        t0 = time.perf_counter()

        for epoch_idx, bad_ch_indices in enumerate(interp_channels):
            if len(bad_ch_indices) == 0:
                continue
            cache_key = tuple(sorted(bad_ch_indices))
            interpolation, good_idx, bad_idx = interp_cache[cache_key]

            good_data = data[epoch_idx, good_idx, :]
            interpolated = interpolation @ good_data
            data[epoch_idx, bad_idx, :] = interpolated

        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times)


def bench_batched_interp(data_gpu, interp_channels, interp_cache, device,
                         n_repeats=5):
    """Batched: group by pattern + bmm."""
    import torch
    from autoreject.kernels.batched_interpolation import (
        batched_interpolate_epochs,
    )

    times = []
    for _ in range(n_repeats):
        data = data_gpu.clone()
        _sync(device)
        t0 = time.perf_counter()

        batched_interpolate_epochs(data, interp_channels, interp_cache, torch)

        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times)


def main():
    import torch

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
    data_gpu, interp_channels, interp_cache = make_interp_data(
        n_epochs, n_channels, n_times, device, rng
    )

    print()
    print("=" * 55)
    print(f"  Interpolation Benchmark")
    print(f"  Profile: {args.profile} ({n_epochs}×{n_channels}×{n_times})")
    print(f"  Device: {device}")
    print("=" * 55)

    # Warmup
    bench_current_interp(data_gpu, interp_channels, interp_cache, device, 2)
    bench_batched_interp(data_gpu, interp_channels, interp_cache, device, 2)

    t_current = bench_current_interp(
        data_gpu, interp_channels, interp_cache, device
    )
    t_batched = bench_batched_interp(
        data_gpu, interp_channels, interp_cache, device
    )

    print()
    print(f"  Current (per-epoch loop): {t_current:>8.1f} ms")
    print(f"  Batched (group + bmm):    {t_batched:>8.1f} ms")
    print(f"  Speedup:                  {t_current / t_batched:>8.1f}x")

    # Verify correctness
    data1 = data_gpu.clone()
    for epoch_idx, bad_ch_indices in enumerate(interp_channels):
        if len(bad_ch_indices) == 0:
            continue
        cache_key = tuple(sorted(bad_ch_indices))
        interpolation, good_idx, bad_idx = interp_cache[cache_key]
        good_data = data1[epoch_idx, good_idx, :]
        data1[epoch_idx, bad_idx, :] = interpolation @ good_data

    data2 = data_gpu.clone()
    from autoreject.kernels.batched_interpolation import batched_interpolate_epochs
    batched_interpolate_epochs(data2, interp_channels, interp_cache, torch)

    diff = (data1 - data2).abs().max().item()
    print(f"\n  Correctness max diff: {diff:.2e}")
    print(f"  Match: {'OK' if diff < 1e-5 else 'FAIL'}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
