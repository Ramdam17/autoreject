#!/usr/bin/env python
"""Benchmark script to measure performance improvements.

Usage:
    python tools/benchmark.py [--backend numpy|numba|torch|jax] [--n-jobs N]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add the autoreject package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_synthetic_epochs(n_epochs=50, n_channels=64, n_times=512,
                           sfreq=256, seed=42):
    """Create synthetic test epochs.
    
    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of channels.
    n_times : int
        Number of time points.
    sfreq : float
        Sampling frequency.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    epochs : mne.Epochs
        Synthetic epochs object.
    """
    import mne
    
    np.random.seed(seed)
    
    # Create channel info
    ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create montage with spherical positions
    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    phi = np.linspace(np.pi/4, 3*np.pi/4, n_channels)
    radius = 0.09
    
    pos = np.column_stack([
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi)
    ])
    
    ch_pos = {ch: p for ch, p in zip(ch_names, pos)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage)
    
    # Generate synthetic data with some artifacts
    data = np.random.randn(n_epochs, n_channels, n_times) * 20e-6
    common_signal = np.random.randn(n_epochs, 1, n_times) * 5e-6
    data += common_signal
    
    # Add some bad epochs
    bad_epochs = np.random.choice(n_epochs, size=n_epochs // 10, replace=False)
    for idx in bad_epochs:
        data[idx] *= 3.0
    
    # Add some bad channels in specific epochs
    for _ in range(n_epochs // 5):
        epoch_idx = np.random.randint(n_epochs)
        ch_idx = np.random.randint(n_channels)
        data[epoch_idx, ch_idx, :] += np.random.randn(n_times) * 100e-6
    
    events = np.column_stack([
        np.arange(0, n_epochs * n_times, n_times),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int)
    ])
    
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, verbose=False)
    return epochs


def benchmark_vote_bad_epochs(epochs, picks, n_runs=5, backend_name='numpy'):
    """Benchmark _vote_bad_epochs function.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    picks : array-like
        Channel indices.
    n_runs : int
        Number of runs for averaging.
    backend_name : str
        Backend to use.
    
    Returns
    -------
    mean_time : float
        Mean execution time in seconds.
    std_time : float
        Standard deviation of execution time.
    """
    from autoreject.autoreject import _AutoReject, _compute_thresholds
    from autoreject.backends import get_backend, clear_backend_cache
    
    # Force backend
    os.environ['AUTOREJECT_BACKEND'] = backend_name
    clear_backend_cache()
    backend = get_backend()
    
    # Pre-compute thresholds
    threshes = _compute_thresholds(
        epochs, method='bayesian_optimization',
        random_state=42, picks=picks, augment=False,
        verbose=False, n_jobs=1
    )
    
    ar = _AutoReject(n_interpolate=4, consensus=0.5, picks=picks, verbose=False)
    ar.threshes_ = threshes
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        ar._vote_bad_epochs(epochs, picks=picks, backend=backend)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)


def benchmark_ransac(epochs, picks, n_runs=3, backend_name='numpy'):
    """Benchmark RANSAC.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    picks : array-like
        Channel indices.
    n_runs : int
        Number of runs for averaging.
    backend_name : str
        Backend to use.
    
    Returns
    -------
    mean_time : float
        Mean execution time in seconds.
    std_time : float
        Standard deviation of execution time.
    """
    from autoreject import Ransac
    from autoreject.backends import clear_backend_cache
    
    # Force backend
    os.environ['AUTOREJECT_BACKEND'] = backend_name
    clear_backend_cache()
    
    times = []
    for _ in range(n_runs):
        ransac = Ransac(
            n_resample=25,
            min_channels=0.25,
            min_corr=0.75,
            unbroken_time=0.4,
            n_jobs=1,
            random_state=42,
            picks=picks,
            verbose=False
        )
        start = time.perf_counter()
        ransac.fit(epochs)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)


def benchmark_interpolation(epochs, picks, n_runs=3, n_jobs=1, backend_name='numpy'):
    """Benchmark epoch interpolation.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    picks : array-like
        Channel indices.
    n_runs : int
        Number of runs for averaging.
    n_jobs : int
        Number of parallel jobs.
    backend_name : str
        Backend to use.
    
    Returns
    -------
    mean_time : float
        Mean execution time in seconds.
    std_time : float
        Standard deviation of execution time.
    """
    from autoreject.autoreject import (
        _AutoReject, _compute_thresholds, _interpolate_bad_epochs, _get_interp_chs
    )
    from autoreject.backends import clear_backend_cache
    
    # Force backend
    os.environ['AUTOREJECT_BACKEND'] = backend_name
    clear_backend_cache()
    
    # Pre-compute thresholds and labels
    threshes = _compute_thresholds(
        epochs, method='bayesian_optimization',
        random_state=42, picks=picks, augment=False,
        verbose=False, n_jobs=1
    )
    
    ar = _AutoReject(n_interpolate=4, consensus=0.5, picks=picks, verbose=False)
    ar.threshes_ = threshes
    ar.consensus_ = {'eeg': 0.5}
    ar.n_interpolate_ = {'eeg': 4}
    
    labels, _ = ar._vote_bad_epochs(epochs, picks=picks)
    labels_interp = ar._get_epochs_interpolation(
        epochs, labels=labels, picks=picks, n_interpolate=4
    )
    interp_channels = _get_interp_chs(labels_interp, epochs.ch_names, picks)
    
    times = []
    for _ in range(n_runs):
        epochs_copy = epochs.copy()
        start = time.perf_counter()
        _interpolate_bad_epochs(
            epochs_copy, interp_channels=interp_channels,
            picks=picks, dots=None, verbose=False, n_jobs=n_jobs
        )
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)


def print_results(name, times, baseline_times=None):
    """Print benchmark results.
    
    Parameters
    ----------
    name : str
        Name of the benchmark.
    times : dict
        Dictionary of {backend: (mean, std)}.
    baseline_times : tuple | None
        Baseline (mean, std) for speedup calculation.
    """
    print(f"\n{name}")
    print("=" * 60)
    
    for backend, (mean, std) in times.items():
        speedup = ""
        if baseline_times is not None:
            baseline_mean = baseline_times[0]
            if mean > 0:
                speedup = f" (speedup: {baseline_mean / mean:.2f}x)"
        print(f"  {backend:12s}: {mean*1000:8.2f} ms ± {std*1000:6.2f} ms{speedup}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark autoreject performance')
    parser.add_argument('--backend', type=str, default=None,
                        choices=['numpy', 'numba', 'torch', 'jax', 'all'],
                        help='Backend to benchmark (default: all available)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--n-channels', type=int, default=64,
                        help='Number of channels (default: 64)')
    args = parser.parse_args()
    
    # Import after parsing args to avoid slow import if just showing help
    import mne
    mne.set_log_level('ERROR')
    
    from autoreject.backends import get_backend_names, detect_hardware
    from autoreject.utils import _handle_picks
    
    print("=" * 60)
    print("Autoreject Performance Benchmark")
    print("=" * 60)
    
    # Detect hardware
    hw = detect_hardware()
    print(f"\nHardware detected:")
    print(f"  CPU cores: {os.cpu_count()}")
    if hw.get('cuda'):
        print(f"  CUDA: available ({hw.get('cuda_device', 'unknown')} devices)")
    if hw.get('mps'):
        print("  MPS (Apple Silicon): available")
    
    # Get available backends
    backends = get_backend_names()
    print(f"\nAvailable backends: {', '.join(backends)}")
    
    if args.backend is not None and args.backend != 'all':
        backends = [args.backend]
    
    # Create test data
    print(f"\nCreating synthetic epochs ({args.n_epochs} epochs, "
          f"{args.n_channels} channels)...")
    epochs = create_synthetic_epochs(
        n_epochs=args.n_epochs,
        n_channels=args.n_channels
    )
    picks = _handle_picks(epochs.info, picks=None)
    
    # Benchmark vote_bad_epochs
    print("\nBenchmarking _vote_bad_epochs...")
    vote_times = {}
    for backend in backends:
        try:
            mean, std = benchmark_vote_bad_epochs(epochs, picks, backend_name=backend)
            vote_times[backend] = (mean, std)
        except Exception as e:
            print(f"  {backend}: Error - {e}")
    
    baseline = vote_times.get('numpy')
    print_results("_vote_bad_epochs (peak-to-peak computation)", vote_times, baseline)
    
    # Benchmark RANSAC
    print("\nBenchmarking RANSAC...")
    ransac_times = {}
    for backend in backends:
        try:
            mean, std = benchmark_ransac(epochs, picks, backend_name=backend)
            ransac_times[backend] = (mean, std)
        except Exception as e:
            print(f"  {backend}: Error - {e}")
    
    baseline = ransac_times.get('numpy')
    print_results("RANSAC.fit()", ransac_times, baseline)
    
    # Benchmark interpolation with different n_jobs
    print("\nBenchmarking _interpolate_bad_epochs...")
    for n_jobs in [1, 2, 4, -1]:
        if n_jobs == -1:
            n_jobs_str = "all CPUs"
        else:
            n_jobs_str = f"{n_jobs} job(s)"
        print(f"\n  n_jobs={n_jobs} ({n_jobs_str}):")
        
        interp_times = {}
        for backend in ['numpy']:  # Interpolation uses MNE, not our backends
            try:
                mean, std = benchmark_interpolation(
                    epochs, picks, n_jobs=n_jobs, backend_name=backend
                )
                interp_times[f"n_jobs={n_jobs}"] = (mean, std)
            except Exception as e:
                print(f"    Error - {e}")
        
        if n_jobs == 1:
            baseline_interp = interp_times.get("n_jobs=1")
        else:
            baseline_interp = None
        
        for key, (mean, std) in interp_times.items():
            speedup = ""
            if baseline_interp is not None and n_jobs != 1:
                if mean > 0:
                    speedup = f" (speedup: {baseline_interp[0] / mean:.2f}x)"
            print(f"    {mean*1000:8.2f} ms ± {std*1000:6.2f} ms{speedup}")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == '__main__':
    main()
