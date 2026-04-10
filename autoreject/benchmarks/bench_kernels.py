"""Benchmark custom kernels vs PyTorch backend.

Compares Metal/CUDA fused threshold-CV kernel against
GPUThresholdOptimizer.batched_all_channels_cv_loss_parallel().

Usage
-----
    python -m autoreject.benchmarks.bench_kernels
    python -m autoreject.benchmarks.bench_kernels --profile realistic
    python -m autoreject.benchmarks.bench_kernels --all
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Benchmark profiles: n_train = 80% of n_epochs
PROFILES = {
    "small": {"n_train": 40, "n_channels": 32, "n_times": 500},
    "medium": {"n_train": 160, "n_channels": 64, "n_times": 1000},
    "realistic": {"n_train": 320, "n_channels": 128, "n_times": 1500},
    "large": {"n_train": 640, "n_channels": 256, "n_times": 2000},
}


def generate_bench_data(n_train, n_channels, n_times, n_test=None,
                        random_state=42):
    """Generate benchmark data matching one CV fold."""
    rng = np.random.RandomState(random_state)

    if n_test is None:
        n_test = max(1, n_train // 4)

    # Training data
    data_train = rng.randn(n_train, n_channels, n_times).astype(np.float64)
    data_train *= 1e-5
    channel_scales = rng.uniform(0.5, 2.0, size=n_channels)
    data_train *= channel_scales[np.newaxis, :, np.newaxis]

    # Add artifacts
    n_bad = max(1, n_train // 10)
    bad_idx = rng.choice(n_train, n_bad, replace=False)
    data_train[bad_idx] *= rng.uniform(2.0, 5.0, size=(n_bad, 1, 1))

    # PTP
    ptp_train = data_train.max(axis=-1) - data_train.min(axis=-1)

    # Thresholds: sorted PTPs per channel (n_thresh = n_train)
    n_thresh = n_train
    threshes_all = np.zeros((n_channels, n_thresh), dtype=np.float64)
    for ch in range(n_channels):
        threshes_all[ch] = np.sort(ptp_train[:, ch])

    # Test median
    test_data = rng.randn(n_test, n_channels, n_times) * 1e-5
    test_data *= channel_scales[np.newaxis, :, np.newaxis]
    median_test = np.median(test_data, axis=0)

    return data_train, ptp_train, threshes_all, median_test


def _sync_device(device):
    """Synchronize GPU for accurate timing."""
    try:
        import torch
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


# =========================================================================
# Backend runners
# =========================================================================


def bench_numpy(data_train, ptp_train, threshes_all, median_test):
    """Benchmark NumPy reference (CPU, float64)."""
    from autoreject.tests.test_kernels import numpy_batched_cv_loss

    start = time.perf_counter()
    result = numpy_batched_cv_loss(data_train, ptp_train, threshes_all,
                                   median_test)
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, result


def bench_torch(data_train, ptp_train, threshes_all, median_test, device):
    """Benchmark PyTorch batched_all_channels_cv_loss_parallel."""
    from autoreject.gpu_pipeline import GPUThresholdOptimizer, _torch_median

    optimizer = GPUThresholdOptimizer(device=device)

    # Transfer to GPU
    data_gpu = optimizer._to_tensor(data_train)
    ptp_gpu = optimizer._to_tensor(ptp_train)
    thresh_gpu = optimizer._to_tensor(threshes_all)

    # Pre-compute test median on GPU (matching real pipeline flow)
    median_gpu = optimizer._to_tensor(median_test)

    # Build a single fake fold: train=all data, test median pre-computed
    n_train = data_train.shape[0]
    train_idx = np.arange(n_train)
    test_idx = np.arange(max(1, n_train // 4))
    cv_splits = [(train_idx, test_idx)]

    _sync_device(device)
    start = time.perf_counter()

    # This is the operation we're benchmarking
    result_gpu = optimizer.batched_all_channels_cv_loss_parallel(
        data_gpu, ptp_gpu, thresh_gpu, cv_splits
    )

    _sync_device(device)
    elapsed = (time.perf_counter() - start) * 1000

    result = result_gpu.cpu().numpy()
    optimizer.clear_cache()
    return elapsed, result


def bench_metal(data_train, ptp_train, threshes_all, median_test):
    """Benchmark Metal fused kernel."""
    from autoreject.kernels.metal_thresh_cv import metal_batched_cv_loss

    # Convert to float32 for Metal
    data_f32 = data_train.astype(np.float32)
    ptp_f32 = ptp_train.astype(np.float32)
    thresh_f32 = threshes_all.astype(np.float32)
    median_f32 = median_test.astype(np.float32)

    start = time.perf_counter()
    result = metal_batched_cv_loss(data_f32, ptp_f32, thresh_f32, median_f32)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, result


def bench_cuda(data_train, ptp_train, threshes_all, median_test):
    """Benchmark CUDA fused kernel."""
    from autoreject.kernels.cuda_thresh_cv import cuda_batched_cv_loss

    start = time.perf_counter()
    result = cuda_batched_cv_loss(data_train, ptp_train, threshes_all,
                                  median_test)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, result


# =========================================================================
# Main benchmark
# =========================================================================


def run_benchmark(profile_name, device, n_warmup=2, n_repeats=5):
    """Run benchmark for all available backends."""
    params = PROFILES[profile_name]
    n_train = params["n_train"]
    n_channels = params["n_channels"]
    n_times = params["n_times"]
    n_thresh = n_train  # = n_epochs per channel

    data_mb = n_train * n_channels * n_times * 4 / (1024 * 1024)

    print()
    print("=" * 70)
    print(f"  Threshold-CV Kernel Benchmark")
    print(f"  Profile: {profile_name} ({n_train} train × {n_channels} ch "
          f"× {n_times} times)")
    print(f"  Grid: {n_channels} × {n_thresh} = "
          f"{n_channels * n_thresh:,} threadgroups")
    print(f"  Data: {data_mb:.1f} MB (f32)")
    print(f"  Device: {device}")
    print(f"  Warmup: {n_warmup}, Repeats: {n_repeats}")
    print("=" * 70)

    data, ptp, thresh, median = generate_bench_data(
        n_train, n_channels, n_times
    )

    results = {}

    # --- NumPy baseline ---
    logger.info("Benchmarking NumPy (CPU, float64)...")
    times = []
    for i in range(n_warmup + n_repeats):
        elapsed, ref_result = bench_numpy(data, ptp, thresh, median)
        if i >= n_warmup:
            times.append(elapsed)
    results["numpy"] = np.median(times)

    # --- PyTorch ---
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch and device != "cpu":
        logger.info("Benchmarking PyTorch (%s)...", device)
        times = []
        for i in range(n_warmup + n_repeats):
            elapsed, torch_result = bench_torch(data, ptp, thresh, median,
                                                device)
            if i >= n_warmup:
                times.append(elapsed)
        results["torch"] = np.median(times)

    # --- Metal kernel ---
    from autoreject.kernels import METAL_AVAILABLE
    if METAL_AVAILABLE:
        logger.info("Benchmarking Metal kernel...")
        times = []
        for i in range(n_warmup + n_repeats):
            elapsed, metal_result = bench_metal(data, ptp, thresh, median)
            if i >= n_warmup:
                times.append(elapsed)
        results["metal"] = np.median(times)

    # --- CUDA kernel ---
    from autoreject.kernels import CUPY_AVAILABLE
    if CUPY_AVAILABLE:
        logger.info("Benchmarking CUDA kernel...")
        times = []
        for i in range(n_warmup + n_repeats):
            elapsed, cuda_result = bench_cuda(data, ptp, thresh, median)
            if i >= n_warmup:
                times.append(elapsed)
        results["cuda"] = np.median(times)

    # --- Print results ---
    print()
    print(f"{'Backend':<20} {'Time (ms)':>10} {'vs NumPy':>10} "
          f"{'vs PyTorch':>12}")
    print("-" * 54)

    numpy_time = results.get("numpy", 1)
    torch_time = results.get("torch", None)

    for name, ms in sorted(results.items(), key=lambda x: x[1]):
        vs_numpy = f"{numpy_time / ms:.1f}x"
        if torch_time is not None and name != "torch":
            vs_torch = f"{torch_time / ms:.1f}x"
        elif name == "torch":
            vs_torch = "1.0x"
        else:
            vs_torch = "-"
        print(f"  {name:<18} {ms:>9.1f} {vs_numpy:>10} {vs_torch:>12}")

    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark custom kernels vs PyTorch",
    )
    parser.add_argument(
        "--profile", type=str, default="medium",
        choices=list(PROFILES.keys()),
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    # Detect device
    device = args.device
    if device is None:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and \
                    torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    profiles = list(PROFILES.keys()) if args.all else [args.profile]
    all_results = []

    for profile_name in profiles:
        try:
            results = run_benchmark(
                profile_name, device,
                n_warmup=args.warmup, n_repeats=args.repeats,
            )
            all_results.append((profile_name, results))
        except Exception as e:
            logger.error("Profile '%s' failed: %s", profile_name, e)
            import traceback
            traceback.print_exc()

    # Save CSV
    if all_results and args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            backends = sorted(set(
                k for _, r in all_results for k in r.keys()
            ))
            writer.writerow(["profile"] + backends)
            for name, results in all_results:
                row = [name] + [f"{results.get(b, 0):.2f}" for b in backends]
                writer.writerow(row)
        logger.info("Saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
