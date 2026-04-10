"""Compare argmin vs bayes_opt threshold selection across multiple seeds.

Tests whether replacing Bayesian optimization with simple argmin on
pre-computed losses produces results within the same variance envelope
as the existing GPU pipeline.

Uses real MNE sample data (same as existing benchmarks) for validity.

Usage
-----
    python -m autoreject.benchmarks.test_argmin_vs_bayesopt
    python -m autoreject.benchmarks.test_argmin_vs_bayesopt --n-seeds 20
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


def get_sample_epochs(n_channels=32):
    """Load MNE sample data and create epochs.

    Falls back to synthetic data if sample data is not available.
    """
    import mne

    try:
        sample_path = mne.datasets.sample.data_path()
        raw_fname = sample_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        raw.pick("eeg")

        # Limit channels if needed
        if n_channels < len(raw.ch_names):
            raw.pick(raw.ch_names[:n_channels])

        events = mne.find_events(raw, stim_channel="STI 014")
        epochs = mne.Epochs(
            raw, events, event_id=[1, 2, 3, 4],
            tmin=-0.2, tmax=0.5, baseline=None, preload=True,
        )
        logger.info("Loaded MNE sample data: %d epochs, %d channels",
                     len(epochs), len(epochs.ch_names))
        return epochs

    except Exception as e:
        logger.warning("Could not load sample data (%s), using synthetic", e)
        from autoreject.benchmarks.profile_pipeline import (
            generate_synthetic_epochs,
        )
        return generate_synthetic_epochs(200, n_channels, 500)


def run_gpu_pipeline_with_method(epochs, method, seed, device):
    """Run the GPU threshold computation with a specific method.

    Parameters
    ----------
    epochs : mne.Epochs
    method : str
        'bayesian_optimization' or 'argmin'
    seed : int
    device : str

    Returns
    -------
    threshes : dict
        Channel name -> threshold
    consensus : float
    n_interpolate : int
    elapsed_ms : float
    """
    from autoreject.gpu_pipeline import GPUThresholdOptimizer, _torch_median
    from autoreject.utils import _handle_picks, _check_data, _GDKW
    from autoreject.bayesopt import bayes_opt, expected_improvement
    from sklearn.model_selection import StratifiedShuffleSplit

    picks = _handle_picks(info=epochs.info, picks=None)
    n_epochs = len(epochs)
    data = epochs.get_data(**_GDKW)
    y = np.ones((n_epochs,))

    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    cv_splits = list(cv.split(data, y))

    optimizer = GPUThresholdOptimizer(device=device)

    data_picked = data[:, picks, :]
    data_gpu = optimizer._to_tensor(data_picked)

    n_channels = len(picks)

    t0 = time.perf_counter()

    # PTP
    ptp_all = (
        data_gpu.max(dim=-1).values - data_gpu.min(dim=-1).values
    )
    ptp_all_np = ptp_all.cpu().numpy()

    # Build thresholds
    threshes_all_np = np.zeros((n_channels, n_epochs))
    for ch_idx in range(n_channels):
        threshes_all_np[ch_idx] = np.sort(ptp_all_np[:, ch_idx])
    threshes_all = optimizer._to_tensor(threshes_all_np)

    # Compute ALL CV losses
    all_losses = optimizer.batched_all_channels_cv_loss_parallel(
        data_gpu, ptp_all, threshes_all, cv_splits
    )
    all_losses_np = all_losses.cpu().numpy()

    # Threshold selection
    best_thresholds = np.zeros(n_channels)
    ch_names = epochs.ch_names

    if method == "argmin":
        # Simple argmin over pre-computed losses
        best_idx = np.argmin(all_losses_np, axis=1)
        for ch_idx in range(n_channels):
            best_thresholds[ch_idx] = threshes_all_np[ch_idx, best_idx[ch_idx]]
    else:
        # Bayesian optimization (current approach)
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
                expected_improvement, max_iter=10, debug=False,
                random_state=seed,
            )
            best_thresholds[ch_idx] = best_thresh

    elapsed_ms = (time.perf_counter() - t0) * 1000

    threshes = {}
    for i, pick in enumerate(picks):
        threshes[ch_names[pick]] = best_thresholds[i]

    optimizer.clear_cache()
    return threshes, elapsed_ms


def main():
    parser = argparse.ArgumentParser(
        description="Compare argmin vs bayes_opt threshold selection",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-channels", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Detect device
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

    logger.info("Device: %s", device)
    logger.info("Seeds: %d", args.n_seeds)

    epochs = get_sample_epochs(args.n_channels)

    seeds = list(range(42, 42 + args.n_seeds))

    # Collect results
    bayes_thresholds = []
    argmin_thresholds = []
    bayes_times = []
    argmin_times = []

    for seed in seeds:
        logger.info("Seed %d — running bayes_opt...", seed)
        t_bayes, t_ms_bayes = run_gpu_pipeline_with_method(
            epochs, "bayesian_optimization", seed, device
        )
        bayes_thresholds.append(t_bayes)
        bayes_times.append(t_ms_bayes)

        logger.info("Seed %d — running argmin...", seed)
        t_argmin, t_ms_argmin = run_gpu_pipeline_with_method(
            epochs, "argmin", seed, device
        )
        argmin_thresholds.append(t_argmin)
        argmin_times.append(t_ms_argmin)

    # Analysis
    ch_names = sorted(bayes_thresholds[0].keys())
    n_channels = len(ch_names)

    # Per-channel threshold comparison
    bayes_matrix = np.array([
        [t[ch] for ch in ch_names] for t in bayes_thresholds
    ])  # (n_seeds, n_channels)
    argmin_matrix = np.array([
        [t[ch] for ch in ch_names] for t in argmin_thresholds
    ])  # (n_seeds, n_channels)

    # Match rate: how often do they pick the same threshold?
    exact_match = (bayes_matrix == argmin_matrix).mean() * 100

    # Relative difference
    rel_diff = np.abs(bayes_matrix - argmin_matrix) / (
        np.abs(bayes_matrix) + 1e-20
    )
    mean_rel_diff = rel_diff.mean() * 100

    # Variance comparison
    bayes_cv = bayes_matrix.std(axis=0) / (bayes_matrix.mean(axis=0) + 1e-20)
    argmin_cv = argmin_matrix.std(axis=0) / (argmin_matrix.mean(axis=0) + 1e-20)

    # Loss comparison: what's the actual CV loss at the chosen threshold?
    # (We can't easily recompute this here, but we can compare thresholds)

    # Timing
    mean_bayes_ms = np.mean(bayes_times)
    mean_argmin_ms = np.mean(argmin_times)

    print()
    print("=" * 65)
    print(f"  Argmin vs Bayes Opt — {args.n_seeds} seeds, "
          f"{n_channels} channels")
    print(f"  Device: {device}")
    print("=" * 65)

    print()
    print("--- Threshold Agreement ---")
    print(f"  Exact match rate:          {exact_match:>6.1f}%")
    print(f"  Mean relative difference:  {mean_rel_diff:>6.2f}%")

    print()
    print("--- Cross-seed Variance (coefficient of variation) ---")
    print(f"  Bayes opt mean CV:   {bayes_cv.mean():>8.4f} "
          f"(std: {bayes_cv.std():.4f})")
    print(f"  Argmin mean CV:      {argmin_cv.mean():>8.4f} "
          f"(std: {argmin_cv.std():.4f})")
    print(f"  Ratio (argmin/bayes): {argmin_cv.mean() / (bayes_cv.mean() + 1e-20):.2f}")

    print()
    print("--- Timing ---")
    print(f"  Bayes opt:   {mean_bayes_ms:>8.1f} ms (mean over seeds)")
    print(f"  Argmin:      {mean_argmin_ms:>8.1f} ms (mean over seeds)")
    print(f"  Speedup:     {mean_bayes_ms / mean_argmin_ms:>8.1f}x")

    print()
    print("--- Per-channel Detail (first 5 channels) ---")
    print(f"  {'Channel':<10} {'Bayes mean':>12} {'Bayes std':>12} "
          f"{'Argmin mean':>12} {'Argmin std':>12} {'Match%':>8}")
    print("  " + "-" * 68)
    for ch in ch_names[:5]:
        ch_idx = ch_names.index(ch)
        b_mean = bayes_matrix[:, ch_idx].mean()
        b_std = bayes_matrix[:, ch_idx].std()
        a_mean = argmin_matrix[:, ch_idx].mean()
        a_std = argmin_matrix[:, ch_idx].std()
        match = (bayes_matrix[:, ch_idx] == argmin_matrix[:, ch_idx]).mean() * 100
        print(f"  {ch:<10} {b_mean:>12.2e} {b_std:>12.2e} "
              f"{a_mean:>12.2e} {a_std:>12.2e} {match:>7.0f}%")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
