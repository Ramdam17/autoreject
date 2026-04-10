"""Profile autoreject GPU pipeline to identify kernel optimization candidates.

This script profiles AutoReject.fit() at multiple data sizes, decomposing
wall time into individual operations to identify where custom Metal/CUDA
kernels could improve performance over the existing PyTorch backend.

Usage
-----
    python -m autoreject.benchmarks.profile_pipeline
    python -m autoreject.benchmarks.profile_pipeline --profile medium
    python -m autoreject.benchmarks.profile_pipeline --profile realistic --device mps
    python -m autoreject.benchmarks.profile_pipeline --all

References
----------
Jas et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG
data. NeuroImage, 159, 417-429.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import argparse
import csv
import logging
import os
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# Data size profiles
# =========================================================================

PROFILES = {
    "small": {"n_epochs": 50, "n_channels": 32, "n_times": 500},
    "medium": {"n_epochs": 200, "n_channels": 64, "n_times": 1000},
    "realistic": {"n_epochs": 400, "n_channels": 128, "n_times": 1500},
    "large": {"n_epochs": 800, "n_channels": 256, "n_times": 2000},
}


# =========================================================================
# Timer infrastructure
# =========================================================================


class GPUTimer:
    """Timer that handles GPU synchronization properly.

    For accurate GPU timing, we must synchronize before starting and
    stopping the timer. On CUDA we use torch.cuda.Event for precise
    kernel-level timing. On MPS we use explicit synchronization +
    time.perf_counter.
    """

    def __init__(self, device=None):
        self.device = device
        self._torch = None
        self._records = OrderedDict()
        self._stack = []

        if device is not None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                pass

    def _sync(self):
        """Synchronize GPU to ensure accurate timing."""
        if self._torch is None:
            return
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        elif self.device == "mps":
            self._torch.mps.synchronize()

    @contextmanager
    def time(self, name):
        """Context manager to time an operation with GPU sync."""
        self._sync()
        start = time.perf_counter()
        self._stack.append(name)
        yield
        self._sync()
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._stack.pop()

        if name not in self._records:
            self._records[name] = []
        self._records[name].append(elapsed_ms)

    def get_results(self):
        """Return timing results as dict of {name: mean_ms}."""
        return {
            name: np.mean(times) for name, times in self._records.items()
        }

    def reset(self):
        """Clear all recorded timings."""
        self._records.clear()


# =========================================================================
# Synthetic EEG data generation
# =========================================================================


def generate_synthetic_epochs(n_epochs, n_channels, n_times, sfreq=256.0,
                              random_state=42):
    """Generate synthetic MNE Epochs with realistic channel montage.

    Creates EEG-like data using a standard 10-20 montage (or extended)
    with white noise. The epochs have proper channel positions, which is
    required for spherical spline interpolation.

    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of EEG channels. Will use standard_1020 montage up to 94
        channels, or standard_1005 for more.
    n_times : int
        Number of time samples per epoch.
    sfreq : float
        Sampling frequency in Hz.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    epochs : mne.Epochs
        Synthetic epochs with montage set.
    """
    import mne

    rng = np.random.RandomState(random_state)

    # Choose montage based on channel count
    if n_channels <= 94:
        montage = mne.channels.make_standard_montage("standard_1020")
    else:
        montage = mne.channels.make_standard_montage("standard_1005")

    available_channels = montage.ch_names
    if n_channels > len(available_channels):
        raise ValueError(
            f"Requested {n_channels} channels, but montage only has "
            f"{len(available_channels)}. Max supported: {len(available_channels)}"
        )

    ch_names = available_channels[:n_channels]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Generate EEG-like data: pink noise (1/f) + some artifacts
    # Shape: (n_epochs, n_channels, n_times)
    data = rng.randn(n_epochs, n_channels, n_times) * 1e-5  # ~10 µV scale

    # Add some channel-specific variance to create realistic PTP distribution
    channel_scales = rng.uniform(0.5, 2.0, size=n_channels)
    data *= channel_scales[np.newaxis, :, np.newaxis]

    # Add a few "bad" epochs with higher amplitude (artifact-like)
    n_bad = max(1, n_epochs // 10)
    bad_epoch_idx = rng.choice(n_epochs, n_bad, replace=False)
    data[bad_epoch_idx] *= rng.uniform(2.0, 5.0, size=(n_bad, 1, 1))

    # Create events
    events = np.column_stack([
        np.arange(0, n_epochs * n_times, n_times),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])

    epochs = mne.EpochsArray(data, info, events=events, tmin=0.0)
    epochs.set_montage(montage)

    return epochs


# =========================================================================
# Profiling: individual operation timing
# =========================================================================


def profile_compute_thresholds(epochs, timer, device):
    """Profile the threshold computation pipeline.

    Instruments: PTP, batched CV loss (the main hotspot), Bayesian opt.
    """
    from autoreject.gpu_pipeline import (
        GPUThresholdOptimizer,
        _torch_median,
    )
    from autoreject.utils import _handle_picks, _check_data, _GDKW
    from sklearn.model_selection import StratifiedShuffleSplit

    picks = _handle_picks(info=epochs.info, picks=None)

    n_epochs = len(epochs)
    data = epochs.get_data(**_GDKW)
    y = np.ones((n_epochs,))

    # Create CV splits
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    cv_splits = list(cv.split(data, y))

    optimizer = GPUThresholdOptimizer(device=device)

    # --- Transfer to GPU ---
    data_picked = data[:, picks, :]
    with timer.time("data_transfer_to_gpu"):
        data_gpu = optimizer._to_tensor(data_picked)

    # --- PTP computation ---
    with timer.time("ptp_computation"):
        ptp_all = (
            data_gpu.max(dim=-1).values - data_gpu.min(dim=-1).values
        )

    # --- Build thresholds tensor ---
    ptp_all_np = ptp_all.cpu().numpy()
    n_channels = len(picks)
    with timer.time("threshold_tensor_build"):
        threshes_all_np = np.zeros((n_channels, n_epochs))
        for ch_idx in range(n_channels):
            threshes_all_np[ch_idx] = np.sort(ptp_all_np[:, ch_idx])
        threshes_all = optimizer._to_tensor(threshes_all_np)

    # --- Batched CV loss (THE hotspot) ---
    with timer.time("batched_cv_loss"):
        all_losses = optimizer.batched_all_channels_cv_loss_parallel(
            data_gpu, ptp_all, threshes_all, cv_splits
        )

    # --- Bayesian optimization (CPU) ---
    all_losses_np = all_losses.cpu().numpy()
    with timer.time("bayesian_optimization"):
        from autoreject.bayesopt import bayes_opt, expected_improvement

        best_thresholds = np.zeros(n_channels)
        for ch_idx in range(n_channels):
            all_threshes = threshes_all_np[ch_idx]
            losses_np = all_losses_np[ch_idx]
            loss_cache = {
                thresh: loss for thresh, loss in zip(all_threshes, losses_np)
            }

            def cached_loss_func(thresh, cache=loss_cache, threshes=all_threshes):
                idx = np.where(thresh - threshes >= 0)[0][-1]
                return cache[threshes[idx]]

            n_epochs_thresh = len(all_threshes)
            idx = np.concatenate((
                np.linspace(0, n_epochs_thresh, 40, endpoint=False, dtype=int),
                [n_epochs_thresh - 1],
            ))
            idx = np.unique(idx)
            initial_x = all_threshes[idx]

            best_thresh, _ = bayes_opt(
                cached_loss_func, initial_x, all_threshes,
                expected_improvement, max_iter=10, debug=False, random_state=42,
            )
            best_thresholds[ch_idx] = best_thresh

    optimizer.clear_cache()
    return best_thresholds


def profile_legendre_interpolation(epochs, timer, device):
    """Profile Legendre G-matrix computation and interpolation matrix build.

    Instruments: Legendre polynomial eval, G-matrix, pinv, matrix apply.
    """
    import torch
    from autoreject.gpu_interpolation import (
        _calc_g_torch,
        legval_torch,
        _get_loocv_interp_matrices,
    )
    from autoreject.utils import _handle_picks

    picks = _handle_picks(info=epochs.info, picks=None)
    pos = epochs._get_channel_positions(picks)

    # Determine compute strategy
    use_cuda = device == "cuda"
    if use_cuda:
        compute_device = device
        compute_dtype = torch.float64
        data_dtype = torch.float64
    else:
        compute_device = "cpu"
        compute_dtype = torch.float64
        data_dtype = torch.float32 if device == "mps" else torch.float64

    # --- Position normalization ---
    with timer.time("position_normalization"):
        pos_t = torch.tensor(pos, dtype=compute_dtype, device=compute_device)
        norms = torch.norm(pos_t, dim=1, keepdim=True)
        pos_t = pos_t / norms

    # --- Cosine angle matrix ---
    with timer.time("cosine_angle_matrix"):
        cosang_all = pos_t @ pos_t.T

    # --- Legendre / G-matrix (THE Legendre hotspot) ---
    with timer.time("legendre_g_matrix"):
        G_all = _calc_g_torch(cosang_all)

    # --- LOOCV interpolation matrices (includes pinv) ---
    # Clear cache to force recomputation
    from autoreject.gpu_interpolation import _LOOCV_INTERP_CACHE
    _LOOCV_INTERP_CACHE.clear()

    with timer.time("loocv_interp_matrices"):
        interp_matrices = _get_loocv_interp_matrices(
            pos, picks, device, compute_device, compute_dtype, data_dtype
        )

    return interp_matrices


def profile_epoch_interpolation(epochs, timer, device):
    """Profile per-epoch interpolation application.

    Instruments: the per-epoch loop that applies interpolation matrices.
    """
    import torch
    from autoreject.gpu_interpolation import _calc_g_torch
    from autoreject.utils import _handle_picks, _GDKW

    picks = _handle_picks(info=epochs.info, picks=None)
    n_epochs = len(epochs)
    n_picks = len(picks)
    picks = np.asarray(picks)

    # Setup
    use_cuda = device == "cuda"
    if use_cuda:
        compute_device = device
        compute_dtype = torch.float64
        data_dtype = torch.float64
    else:
        compute_device = "cpu"
        compute_dtype = torch.float64
        data_dtype = torch.float32 if device == "mps" else torch.float64

    X_full = epochs.get_data(**_GDKW)
    X_gpu = torch.tensor(X_full, dtype=data_dtype, device=device)

    pos = epochs._get_channel_positions(picks)
    pos_t = torch.tensor(pos, dtype=compute_dtype, device=compute_device)
    norms = torch.norm(pos_t, dim=1, keepdim=True)
    pos_t = pos_t / norms
    cosang_all = pos_t @ pos_t.T
    G_all = _calc_g_torch(cosang_all)

    # Generate realistic bad channel patterns
    # ~10% of channels bad per epoch, varying across epochs
    rng = np.random.RandomState(42)
    interp_channels = []
    for _ in range(n_epochs):
        n_bad = max(1, rng.randint(1, max(2, n_picks // 10)))
        bad_chs = sorted(rng.choice(n_picks, n_bad, replace=False).tolist())
        interp_channels.append(bad_chs)

    # --- Per-epoch interpolation loop ---
    data_gpu = X_gpu[:, picks, :].clone()
    interp_cache = {}

    with timer.time("per_epoch_interpolation"):
        for epoch_idx, bad_ch_indices in enumerate(interp_channels):
            if len(bad_ch_indices) == 0:
                continue

            cache_key = tuple(sorted(bad_ch_indices))

            if cache_key not in interp_cache:
                goods_mask = np.ones(n_picks, dtype=bool)
                for bad_idx in bad_ch_indices:
                    goods_mask[bad_idx] = False

                good_idx_in_picks = np.where(goods_mask)[0]
                bad_idx_in_picks = np.where(~goods_mask)[0]

                good_idx_t = torch.tensor(
                    good_idx_in_picks, device=compute_device, dtype=torch.long
                )
                bad_idx_t = torch.tensor(
                    bad_idx_in_picks, device=compute_device, dtype=torch.long
                )

                G_from = G_all[good_idx_t][:, good_idx_t]
                G_to_from = G_all[bad_idx_t][:, good_idx_t]

                n_from = len(good_idx_in_picks)
                G_from_reg = G_from + 1e-5 * torch.eye(
                    n_from, device=compute_device, dtype=compute_dtype
                )

                ones_col = torch.ones(
                    (n_from, 1), device=compute_device, dtype=compute_dtype
                )
                ones_row = torch.ones(
                    (1, n_from), device=compute_device, dtype=compute_dtype
                )
                zero = torch.zeros(
                    (1, 1), device=compute_device, dtype=compute_dtype
                )

                C = torch.cat([
                    torch.cat([G_from_reg, ones_col], dim=1),
                    torch.cat([ones_row, zero], dim=1),
                ], dim=0)

                C_inv = torch.linalg.pinv(C)

                n_bad = len(bad_idx_in_picks)
                ones_to = torch.ones(
                    (n_bad, 1), device=compute_device, dtype=compute_dtype
                )
                interpolation = (
                    torch.cat([G_to_from, ones_to], dim=1) @ C_inv[:, :-1]
                )
                interpolation = interpolation.to(device=device, dtype=data_dtype)

                interp_cache[cache_key] = (
                    interpolation, good_idx_in_picks, bad_idx_in_picks,
                )

            interpolation, good_idx, bad_idx = interp_cache[cache_key]
            good_data = data_gpu[epoch_idx, good_idx, :]
            interpolated = interpolation @ good_data
            data_gpu[epoch_idx, bad_idx, :] = interpolated


def profile_cv_scoring(epochs, timer, device):
    """Profile the consensus × fold scoring loop.

    Instruments: median computation, per-consensus scoring.
    """
    import torch
    from autoreject.gpu_pipeline import GPUThresholdOptimizer, _torch_median
    from autoreject.utils import _handle_picks, _GDKW

    picks = _handle_picks(info=epochs.info, picks=None)
    n_epochs = len(epochs)

    optimizer = GPUThresholdOptimizer(device=device)
    X_full = epochs.get_data(**_GDKW)
    X_gpu = optimizer._to_tensor(X_full)
    picks_t = optimizer.torch.tensor(picks, device=optimizer.device)
    X_picks_gpu = X_gpu[:, picks_t, :]

    # Simulate CV splits
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=10)
    cv_splits = list(cv.split(np.zeros(n_epochs)))

    consensus_values = np.linspace(0.0, 1.0, 11)
    n_folds = len(cv_splits)

    # --- Median computation ---
    with timer.time("cv_median_computation"):
        for train, test in cv_splits:
            test_t = optimizer.torch.tensor(test, device=optimizer.device)
            X_test = X_picks_gpu[test_t]
            _torch_median(X_test, dim=0)

    # --- Scoring loop ---
    with timer.time("cv_scoring_loop"):
        for fold, (train, test) in enumerate(cv_splits):
            train_t = optimizer.torch.tensor(train, device=optimizer.device)
            test_t = optimizer.torch.tensor(test, device=optimizer.device)

            X_test = X_picks_gpu[test_t]
            median_X = _torch_median(X_test, dim=0)
            X_train = X_picks_gpu[train_t]

            for this_consensus in consensus_values:
                # Simulate selecting good epochs (random subset)
                n_good = max(1, int(len(train) * (1 - this_consensus * 0.1)))
                good_idx_t = train_t[:n_good]

                X_good = X_picks_gpu[good_idx_t]
                mean_gpu = X_good.mean(dim=0)

                sq_diff = (median_X - mean_gpu) ** 2
                sq_diff.mean().sqrt()

    optimizer.clear_cache()


# =========================================================================
# GPU memory tracking
# =========================================================================


def get_gpu_memory_mb(device):
    """Get current GPU memory usage in MB."""
    try:
        import torch
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        elif device == "mps":
            # MPS doesn't have a direct memory query
            # Return allocated driver memory if available
            if hasattr(torch.mps, "current_allocated_memory"):
                return torch.mps.current_allocated_memory() / (1024 * 1024)
            if hasattr(torch.mps, "driver_allocated_memory"):
                return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        pass
    return float("nan")


def reset_gpu_memory(device):
    """Reset GPU memory tracking."""
    try:
        import torch
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        elif device == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    except Exception:
        pass


# =========================================================================
# Main profiling orchestration
# =========================================================================


def run_profile(profile_name, device, n_repeats=3):
    """Run full profiling for a given data size profile.

    Parameters
    ----------
    profile_name : str
        One of 'small', 'medium', 'realistic', 'large'.
    device : str
        GPU device ('mps', 'cuda', or 'cpu').
    n_repeats : int
        Number of repeats for averaging.

    Returns
    -------
    results : dict
        Timing results {operation_name: mean_ms}.
    """
    params = PROFILES[profile_name]
    n_epochs = params["n_epochs"]
    n_channels = params["n_channels"]
    n_times = params["n_times"]
    data_mb = n_epochs * n_channels * n_times * 4 / (1024 * 1024)

    logger.info(
        "Profile: %s | %d epochs × %d channels × %d times (%.1f MB f32)",
        profile_name, n_epochs, n_channels, n_times, data_mb,
    )
    logger.info("Device: %s", device)
    logger.info("Generating synthetic EEG data...")

    epochs = generate_synthetic_epochs(n_epochs, n_channels, n_times)

    timer = GPUTimer(device=device if device != "cpu" else None)

    for repeat in range(n_repeats):
        logger.info("--- Repeat %d/%d ---", repeat + 1, n_repeats)
        reset_gpu_memory(device)

        # Phase 1: Threshold computation
        logger.info("  Profiling threshold computation...")
        profile_compute_thresholds(epochs, timer, device)

        # Phase 2: Legendre / interpolation matrices
        logger.info("  Profiling Legendre / interpolation matrices...")
        profile_legendre_interpolation(epochs, timer, device)

        # Phase 3: Per-epoch interpolation
        logger.info("  Profiling per-epoch interpolation...")
        profile_epoch_interpolation(epochs, timer, device)

        # Phase 4: CV scoring loop
        logger.info("  Profiling CV scoring loop...")
        profile_cv_scoring(epochs, timer, device)

    results = timer.get_results()
    peak_mem = get_gpu_memory_mb(device)

    return results, peak_mem, params


def print_report(results, peak_mem, params, profile_name, device):
    """Print a formatted profiling report."""
    total = sum(results.values())

    print()
    print("=" * 70)
    print(f"  AutoReject Pipeline Profile")
    print(f"  Device: {device}")
    print(
        f"  Data: {params['n_epochs']} epochs × {params['n_channels']} "
        f"channels × {params['n_times']} times"
    )
    print(f"  Profile: {profile_name}")
    print("=" * 70)
    print()
    print(f"{'Operation':<40} {'Time (ms)':>10} {'% Total':>10}")
    print("-" * 62)

    for name, ms in sorted(results.items(), key=lambda x: -x[1]):
        pct = ms / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {name:<38} {ms:>9.1f} {pct:>8.1f}%  {bar}")

    print("-" * 62)
    print(f"  {'TOTAL':<38} {total:>9.1f} {'100.0':>8s}%")
    if not np.isnan(peak_mem):
        print(f"  {'Peak GPU memory':<38} {peak_mem:>8.1f} MB")
    print()

    # Highlight kernel candidates
    print("  Kernel candidates (>10% of total):")
    for name, ms in sorted(results.items(), key=lambda x: -x[1]):
        pct = ms / total * 100 if total > 0 else 0
        if pct > 10:
            print(f"    -> {name}: {ms:.1f} ms ({pct:.1f}%)")
    print()


def save_results_csv(all_results, output_path):
    """Save profiling results to CSV."""
    if not all_results:
        return

    # Collect all operation names
    all_ops = set()
    for _, results, _, _, _ in all_results:
        all_ops.update(results.keys())
    all_ops = sorted(all_ops)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["profile", "device", "peak_mem_mb"] + all_ops + ["total_ms"]
        writer.writerow(header)

        for profile_name, results, peak_mem, params, device in all_results:
            total = sum(results.values())
            row = [profile_name, device, f"{peak_mem:.1f}"]
            for op in all_ops:
                row.append(f"{results.get(op, 0.0):.2f}")
            row.append(f"{total:.2f}")
            writer.writerow(row)

    logger.info("Results saved to %s", output_path)


# =========================================================================
# CLI
# =========================================================================


def _detect_device():
    """Auto-detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Profile autoreject GPU pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile", type=str, default="medium",
        choices=list(PROFILES.keys()),
        help="Data size profile (default: medium)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all profiles",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to profile (auto-detected if not set)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of repeats for averaging (default: 3)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: autoreject/benchmarks/profile_results.csv)",
    )

    args = parser.parse_args()

    device = args.device or _detect_device()
    logger.info("Using device: %s", device)

    if device == "cpu":
        logger.warning(
            "No GPU detected. Profiling on CPU — GPU timing columns will "
            "show CPU performance."
        )

    profiles_to_run = list(PROFILES.keys()) if args.all else [args.profile]
    all_results = []

    for profile_name in profiles_to_run:
        try:
            results, peak_mem, params = run_profile(
                profile_name, device, n_repeats=args.repeats
            )
            print_report(results, peak_mem, params, profile_name, device)
            all_results.append((profile_name, results, peak_mem, params, device))
        except Exception as e:
            logger.error("Profile '%s' failed: %s", profile_name, e)
            import traceback
            traceback.print_exc()

    # Save CSV
    if all_results:
        output_path = args.output or str(
            Path(__file__).parent / "profile_results.csv"
        )
        save_results_csv(all_results, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
