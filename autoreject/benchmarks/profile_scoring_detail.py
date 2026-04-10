"""Detailed profiling of the cv_scoring_loop to identify sub-bottlenecks.

Decomposes the scoring loop into:
- _get_bad_epochs() calls (CPU)
- tensor creation/indexing overhead
- GPU mean computation
- GPU RMSE computation
- GPU sync overhead

Usage
-----
    python -m autoreject.benchmarks.profile_scoring_detail
    python -m autoreject.benchmarks.profile_scoring_detail --profile realistic
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


def run_detail_profile(profile_name, device="mps"):
    import torch
    from autoreject.gpu_pipeline import GPUThresholdOptimizer, _torch_median

    params = PROFILES[profile_name]
    n_epochs = params["n_epochs"]
    n_channels = params["n_channels"]
    n_times = params["n_times"]

    rng = np.random.RandomState(42)

    # Simulate data on GPU
    optimizer = GPUThresholdOptimizer(device=device)
    X = rng.randn(n_epochs, n_channels, n_times).astype(np.float32) * 1e-5
    X_gpu = optimizer._to_tensor(X)

    # Simulate bad_sensor_counts
    bad_sensor_counts = rng.randint(0, n_channels // 4, size=n_epochs)

    # CV splits
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=10)
    cv_splits = list(cv.split(np.zeros(n_epochs)))

    consensus_values = np.linspace(0.0, 1.0, 11)
    n_folds = len(cv_splits)
    n_consensus = len(consensus_values)

    # Timing accumulators
    t_median = 0.0
    t_get_bad = 0.0
    t_tensor_create = 0.0
    t_indexing = 0.0
    t_mean = 0.0
    t_rmse = 0.0
    t_sync = 0.0
    n_iterations = 0

    logger.info("Profile: %s (%d epochs × %d ch × %d times)",
                profile_name, n_epochs, n_channels, n_times)
    logger.info("Loop: %d folds × %d consensus = %d iterations",
                n_folds, n_consensus, n_folds * n_consensus)

    for fold, (train, test) in enumerate(cv_splits):
        train_t = optimizer.torch.tensor(train, device=optimizer.device)
        test_t = optimizer.torch.tensor(test, device=optimizer.device)

        # Median
        _sync(device)
        t0 = time.perf_counter()
        X_test = X_gpu[test_t]
        median_X = _torch_median(X_test, dim=0)
        _sync(device)
        t_median += time.perf_counter() - t0

        for idx, this_consensus in enumerate(consensus_values):
            # _get_bad_epochs equivalent (CPU)
            t0 = time.perf_counter()
            n_consensus_ch = this_consensus * n_channels
            sorted_idx = np.argsort(bad_sensor_counts[train])[::-1]
            sorted_counts = np.sort(bad_sensor_counts[train])[::-1]
            bad_epochs = np.zeros(len(train), dtype=bool)
            if np.max(sorted_counts) >= n_consensus_ch:
                n_drop = np.sum(sorted_counts >= n_consensus_ch)
                bad_epochs[sorted_idx[:n_drop]] = True
            good_epochs_idx = np.nonzero(~bad_epochs)[0]
            t_get_bad += time.perf_counter() - t0

            if len(good_epochs_idx) == 0:
                continue

            # Tensor creation
            t0 = time.perf_counter()
            good_idx_t = optimizer.torch.tensor(
                good_epochs_idx, device=optimizer.device
            )
            t_tensor_create += time.perf_counter() - t0

            # GPU indexing
            _sync(device)
            t0 = time.perf_counter()
            X_train = X_gpu[train_t]
            X_good = X_train[good_idx_t]
            _sync(device)
            t_indexing += time.perf_counter() - t0

            # GPU mean
            _sync(device)
            t0 = time.perf_counter()
            mean_gpu = X_good.mean(dim=0)
            _sync(device)
            t_mean += time.perf_counter() - t0

            # GPU RMSE
            _sync(device)
            t0 = time.perf_counter()
            sq_diff = (median_X - mean_gpu) ** 2
            score = sq_diff.mean().sqrt()
            _sync(device)
            t_rmse += time.perf_counter() - t0

            n_iterations += 1

    # One final sync
    _sync(device)
    t0 = time.perf_counter()
    _ = score.cpu().numpy()
    t_sync = time.perf_counter() - t0

    total = t_median + t_get_bad + t_tensor_create + t_indexing + t_mean + t_rmse

    print()
    print("=" * 65)
    print(f"  CV Scoring Loop — Detailed Breakdown ({n_iterations} iterations)")
    print(f"  Profile: {profile_name} | Device: {device}")
    print("=" * 65)
    print()
    print(f"{'Sub-operation':<30} {'Total (ms)':>10} {'Per-iter (ms)':>14} {'%':>8}")
    print("-" * 64)

    items = [
        ("median (torch.sort)", t_median),
        ("_get_bad_epochs (CPU)", t_get_bad),
        ("tensor creation", t_tensor_create),
        ("GPU indexing", t_indexing),
        ("GPU mean", t_mean),
        ("GPU RMSE", t_rmse),
    ]

    for name, t in sorted(items, key=lambda x: -x[1]):
        ms = t * 1000
        per_iter = ms / max(n_iterations, 1)
        pct = ms / (total * 1000) * 100 if total > 0 else 0
        print(f"  {name:<28} {ms:>9.1f} {per_iter:>13.3f} {pct:>7.1f}%")

    print("-" * 64)
    print(f"  {'TOTAL':<28} {total * 1000:>9.1f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="realistic",
                        choices=list(PROFILES.keys()))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        try:
            import torch
            if hasattr(torch.backends, "mps") and \
                    torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    run_detail_profile(args.profile, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
