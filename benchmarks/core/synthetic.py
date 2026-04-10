"""Synthetic EEG data generation for benchmarks.

Generates realistic EEG-like data with proper channel montages,
configurable artifact rates, and reproducible random states.

Unifies the generators from:
- ``autoreject/benchmarks/profile_pipeline.py``
- old ``benchmarks/run_single.py`` (commit 7409d3c)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_epochs(n_channels: int, sfreq: float = 500,
                    epoch_duration: float = 2.0,
                    recording_duration_min: float = 10,
                    artifact_pct: float = 0.3,
                    random_state: int = 42) -> tuple[Any, dict]:
    """Generate synthetic MNE Epochs for benchmarking.

    Creates EEG-like data with a proper standard montage, realistic
    amplitude distribution, and injected artifact epochs.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels. Supports up to ~340 (standard_1005 montage).
    sfreq : float
        Sampling frequency in Hz.
    epoch_duration : float
        Duration of each epoch in seconds.
    recording_duration_min : float
        Total recording duration in minutes. Determines n_epochs.
    artifact_pct : float
        Fraction of epochs with injected artifacts (higher amplitude).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    epochs : mne.Epochs
        Synthetic epochs with montage.
    metadata : dict
        Data characteristics: n_epochs, n_channels, n_times, etc.
    """
    import mne

    rng = np.random.RandomState(random_state)

    # Compute n_epochs from recording duration
    n_times = int(epoch_duration * sfreq)
    total_samples = int(recording_duration_min * 60 * sfreq)
    n_epochs = total_samples // n_times

    # Choose montage
    if n_channels <= 94:
        montage = mne.channels.make_standard_montage("standard_1020")
    else:
        montage = mne.channels.make_standard_montage("standard_1005")

    available = montage.ch_names
    if n_channels > len(available):
        raise ValueError(
            f"Requested {n_channels} channels, montage has {len(available)}."
        )

    ch_names = available[:n_channels]
    info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels,
    )

    # Generate data: pink noise ~ 10 µV scale
    data = rng.randn(n_epochs, n_channels, n_times).astype(np.float64)
    data *= 1e-5

    # Channel-specific variance
    channel_scales = rng.uniform(0.5, 2.0, size=n_channels)
    data *= channel_scales[np.newaxis, :, np.newaxis]

    # Inject artifacts
    n_bad = max(1, int(n_epochs * artifact_pct))
    bad_idx = rng.choice(n_epochs, n_bad, replace=False)
    data[bad_idx] *= rng.uniform(2.0, 5.0, size=(n_bad, 1, 1))

    # Create events
    events = np.column_stack([
        np.arange(0, n_epochs * n_times, n_times),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])

    epochs = mne.EpochsArray(data, info, events=events, tmin=0.0)
    epochs.set_montage(montage)

    metadata = {
        "source": "synthetic",
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "n_times": n_times,
        "sfreq": sfreq,
        "epoch_duration": epoch_duration,
        "recording_duration_min": recording_duration_min,
        "artifact_pct": artifact_pct,
        "data_mb": n_epochs * n_channels * n_times * 8 / (1024 * 1024),
    }

    return epochs, metadata
