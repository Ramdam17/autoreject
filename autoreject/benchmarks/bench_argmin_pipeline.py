"""Benchmark GPU-only argmin threshold pipeline vs current bayes_opt pipeline.

Measures the FULL compute_all_thresholds_gpu flow, not just the selection step.

Usage
-----
    python -m autoreject.benchmarks.bench_argmin_pipeline
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    import torch
    from autoreject.gpu_pipeline import GPUThresholdOptimizer
    from autoreject.kernels.gpu_argmin_thresh import (
        compute_all_thresholds_gpu_argmin,
    )
    from autoreject.benchmarks.profile_pipeline import generate_synthetic_epochs
    from autoreject.utils import _handle_picks, _GDKW
    from sklearn.model_selection import StratifiedShuffleSplit

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    for profile_name, n_ep, n_ch, n_t in [
        ("medium", 200, 64, 1000),
        ("realistic", 400, 128, 1500),
    ]:
        logger.info("Profile: %s (%d×%d×%d)", profile_name, n_ep, n_ch, n_t)

        epochs = generate_synthetic_epochs(n_ep, n_ch, n_t)
        picks = _handle_picks(info=epochs.info, picks=None)
        data = epochs.get_data(**_GDKW)
        y = np.ones((n_ep,))

        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        cv_splits = list(cv.split(data, y))

        optimizer = GPUThresholdOptimizer(device=device)
        data_picked = data[:, picks, :]
        data_gpu = optimizer._to_tensor(data_picked)

        def sync():
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()

        # --- Current: compute_all_thresholds_gpu (with bayes_opt) ---
        times_current = []
        for i in range(5):
            sync()
            t0 = time.perf_counter()
            result_current = optimizer.compute_all_thresholds_gpu(
                data_gpu, picks, cv_splits, y,
                method="bayesian_optimization", random_state=42,
            )
            sync()
            times_current.append((time.perf_counter() - t0) * 1000)

        # --- New: GPU-only argmin ---
        times_argmin = []
        for i in range(5):
            sync()
            t0 = time.perf_counter()
            result_argmin = compute_all_thresholds_gpu_argmin(
                optimizer, data_gpu, cv_splits,
            )
            sync()
            times_argmin.append((time.perf_counter() - t0) * 1000)

        t_current = np.median(times_current)
        t_argmin = np.median(times_argmin)

        # Compare thresholds
        match_rate = (result_current == result_argmin).mean() * 100
        rel_diff = np.abs(result_current - result_argmin) / (
            np.abs(result_current) + 1e-20
        )

        print()
        print(f"  === {profile_name} ({n_ep}×{n_ch}×{n_t}) ===")
        print(f"  Current (bayes_opt): {t_current:>8.1f} ms")
        print(f"  GPU argmin:          {t_argmin:>8.1f} ms")
        print(f"  Speedup:             {t_current / t_argmin:>8.1f}x")
        print(f"  Threshold match:     {match_rate:>7.1f}%")
        print(f"  Mean rel diff:       {rel_diff.mean() * 100:>7.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
