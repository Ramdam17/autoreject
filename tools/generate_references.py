#!/usr/bin/env python
"""Generate reference outputs for retrocompatibility tests.

This script generates deterministic reference outputs from the current
(unoptimized) implementation of autoreject. These references are used
to validate that performance optimizations do not change numerical results.

Usage:
    python tools/generate_references.py

The script creates .npz files in autoreject/tests/references/
"""

# Authors: autoreject contributors

import os
import sys
from pathlib import Path

import numpy as np
import mne

# Add the package to path for development
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from autoreject import AutoReject, Ransac, compute_thresholds
from autoreject.autoreject import (
    _AutoReject, _compute_thresholds,
    _interpolate_bad_epochs, _get_interp_chs, _run_local_reject_cv
)
from autoreject.utils import _handle_picks, _GDKW

# Output directory
REFERENCES_DIR = PACKAGE_ROOT / "autoreject" / "tests" / "references"


def create_deterministic_epochs(n_epochs=30, n_channels=32, n_times=256,
                                 sfreq=256, seed=42):
    """Create deterministic synthetic epochs for testing.
    
    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of EEG channels.
    n_times : int
        Number of time points per epoch.
    sfreq : float
        Sampling frequency.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    epochs : mne.EpochsArray
        Synthetic epochs with reproducible data.
    """
    np.random.seed(seed)
    
    # Create channel info - use 1-based naming to avoid EEG000
    ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create custom montage with spherical positions
    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    phi = np.linspace(np.pi/4, 3*np.pi/4, n_channels)
    radius = 0.09  # head radius in meters
    
    pos = np.column_stack([
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi)
    ])
    
    ch_pos = {ch: p for ch, p in zip(ch_names, pos)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage)
    
    # Generate synthetic data with realistic properties
    # Base signal: random noise with some structure
    data = np.random.randn(n_epochs, n_channels, n_times) * 20e-6  # 20 µV
    
    # Add some correlated structure (simulating brain activity)
    common_signal = np.random.randn(n_epochs, 1, n_times) * 5e-6
    data += common_signal
    
    # Add some bad epochs (higher amplitude)
    bad_epoch_indices = [5, 12, 22]
    for idx in bad_epoch_indices:
        data[idx] *= 3.0
    
    # Add some bad channels in specific epochs
    data[3, 10, :] += np.random.randn(n_times) * 100e-6  # artifact
    data[8, 5, :] += np.random.randn(n_times) * 80e-6
    data[15, 20, :] += np.random.randn(n_times) * 90e-6
    data[18, 15, :] += np.random.randn(n_times) * 120e-6
    
    # Create events
    events = np.column_stack([
        np.arange(0, n_epochs * n_times, n_times),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int)
    ])
    
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, verbose=False)
    return epochs


def generate_vote_bad_epochs_reference(epochs, picks, output_path):
    """Generate reference for _vote_bad_epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    picks : array-like
        Channel indices.
    output_path : Path
        Where to save the reference.
    """
    print("Generating reference for _vote_bad_epochs...")
    
    # First compute thresholds (needed for voting)
    np.random.seed(42)
    threshes = _compute_thresholds(
        epochs, method='bayesian_optimization',
        random_state=42, picks=picks, augment=False, verbose=False, n_jobs=1
    )
    
    # Create _AutoReject instance and set thresholds
    ar = _AutoReject(n_interpolate=4, consensus=0.5, picks=picks, verbose=False)
    ar.threshes_ = threshes
    
    # Call _vote_bad_epochs
    labels, bad_sensor_counts = ar._vote_bad_epochs(epochs, picks=picks)
    
    # Save reference
    np.savez_compressed(
        output_path,
        labels=labels,
        bad_sensor_counts=bad_sensor_counts,
        threshes_keys=list(threshes.keys()),
        threshes_values=list(threshes.values()),
        picks=picks,
        n_epochs=len(epochs),
        n_channels=len(epochs.ch_names)
    )
    print(f"  Saved to {output_path}")
    return threshes


def generate_compute_thresholds_reference(epochs, picks, output_path):
    """Generate reference for _compute_thresholds.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    picks : array-like
        Channel indices.
    output_path : Path
        Where to save the reference.
    """
    print("Generating reference for _compute_thresholds...")
    
    np.random.seed(42)
    
    # Test with bayesian_optimization
    threshes_bayes = _compute_thresholds(
        epochs, method='bayesian_optimization',
        random_state=42, picks=picks, augment=False, verbose=False, n_jobs=1
    )
    
    # Test with random_search
    threshes_random = _compute_thresholds(
        epochs, method='random_search',
        random_state=42, picks=picks, augment=False, verbose=False, n_jobs=1
    )
    
    # Save reference
    np.savez_compressed(
        output_path,
        bayes_keys=list(threshes_bayes.keys()),
        bayes_values=list(threshes_bayes.values()),
        random_keys=list(threshes_random.keys()),
        random_values=list(threshes_random.values()),
        picks=picks
    )
    print(f"  Saved to {output_path}")


def generate_ransac_reference(epochs, picks, output_path):
    """Generate reference for Ransac.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    picks : array-like
        Channel indices.
    output_path : Path
        Where to save the reference.
    """
    print("Generating reference for Ransac...")
    
    # Use fixed random state for reproducibility
    ransac = Ransac(
        n_resample=25,  # Reduced for faster testing
        min_channels=0.25,
        min_corr=0.75,
        unbroken_time=0.4,
        n_jobs=1,
        random_state=42,
        picks=picks,
        verbose=False
    )
    
    ransac.fit(epochs)
    
    # Save reference
    np.savez_compressed(
        output_path,
        corr_=ransac.corr_,
        bad_log=ransac.bad_log,
        bad_chs_=ransac.bad_chs_,
        mappings_shape=ransac.mappings_.shape,
        picks=picks
    )
    print(f"  Saved to {output_path}")


def generate_interpolate_epochs_reference(epochs, picks, threshes, output_path):
    """Generate reference for _interpolate_bad_epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    picks : array-like
        Channel indices.
    threshes : dict
        Pre-computed thresholds.
    output_path : Path
        Where to save the reference.
    """
    print("Generating reference for _interpolate_bad_epochs...")
    
    # Create _AutoReject instance
    ar = _AutoReject(n_interpolate=4, consensus=0.5, picks=picks, verbose=False)
    ar.threshes_ = threshes
    ar.consensus_ = {'eeg': 0.5}
    ar.n_interpolate_ = {'eeg': 4}
    
    # Get labels
    labels, bad_sensor_counts = ar._vote_bad_epochs(epochs, picks=picks)
    
    # Get interpolation labels
    labels_interp = ar._get_epochs_interpolation(
        epochs, labels=labels, picks=picks, n_interpolate=4
    )
    
    # Get channels to interpolate
    interp_channels = _get_interp_chs(labels_interp, epochs.ch_names, picks)
    
    # Create a copy and interpolate
    epochs_copy = epochs.copy()
    _interpolate_bad_epochs(
        epochs_copy, interp_channels=interp_channels,
        picks=picks, dots=None, verbose=False
    )
    
    # Save a sample of the interpolated data (first 5 epochs, first 10 channels)
    data_sample = epochs_copy.get_data(**_GDKW)[:5, :10, :]
    
    # Save reference
    np.savez_compressed(
        output_path,
        labels_interp=labels_interp,
        interp_channels_lengths=[len(ic) for ic in interp_channels],
        data_sample=data_sample,
        picks=picks
    )
    print(f"  Saved to {output_path}")


def generate_local_reject_cv_reference(epochs, picks, output_path):
    """Generate reference for _run_local_reject_cv.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    picks : array-like
        Channel indices.
    output_path : Path
        Where to save the reference.
    """
    print("Generating reference for _run_local_reject_cv...")
    
    from functools import partial
    from sklearn.model_selection import KFold
    
    np.random.seed(42)
    
    # Create threshold function
    thresh_func = partial(
        _compute_thresholds, n_jobs=1,
        method='bayesian_optimization',
        random_state=42, dots=None
    )
    
    # Small parameter grid for testing
    n_interpolate = np.array([1, 4])
    consensus = np.array([0.2, 0.5, 0.8])
    cv = KFold(n_splits=3, shuffle=False)
    
    local_reject, loss = _run_local_reject_cv(
        epochs, thresh_func, picks, n_interpolate, cv,
        consensus, dots=None, verbose=False
    )
    
    # Save reference
    np.savez_compressed(
        output_path,
        loss=loss,
        n_interpolate=n_interpolate,
        consensus=consensus,
        threshes_keys=list(local_reject.threshes_.keys()),
        threshes_values=list(local_reject.threshes_.values()),
        picks=picks
    )
    print(f"  Saved to {output_path}")


def main():
    """Generate all reference files."""
    print("=" * 60)
    print("Generating reference data for retrocompatibility tests")
    print("=" * 60)
    print()
    
    # Ensure output directory exists
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create deterministic test data
    print("Creating deterministic test epochs...")
    epochs = create_deterministic_epochs(
        n_epochs=30,
        n_channels=32,
        n_times=256,
        sfreq=256,
        seed=42
    )
    picks = _handle_picks(epochs.info, picks=None)
    print(f"  Created {len(epochs)} epochs with {len(picks)} channels")
    print()
    
    # Generate references
    threshes = generate_vote_bad_epochs_reference(
        epochs, picks, REFERENCES_DIR / "vote_bad_epochs_v1.npz"
    )
    
    generate_compute_thresholds_reference(
        epochs, picks, REFERENCES_DIR / "compute_thresholds_v1.npz"
    )
    
    generate_ransac_reference(
        epochs, picks, REFERENCES_DIR / "ransac_v1.npz"
    )
    
    generate_interpolate_epochs_reference(
        epochs, picks, threshes, REFERENCES_DIR / "interpolate_epochs_v1.npz"
    )
    
    generate_local_reject_cv_reference(
        epochs, picks, REFERENCES_DIR / "local_reject_cv_v1.npz"
    )
    
    print()
    print("=" * 60)
    print("Reference generation complete!")
    print(f"Files saved to: {REFERENCES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
