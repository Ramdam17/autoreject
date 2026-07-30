"""Deterministic epochs used by the reference fixtures. No side effects.

Extracted so that more than one test module can build the reference epochs
without importing ``test_retrocompat``, which sets
``AUTOREJECT_BACKEND=numpy`` in the environment at module import time and
only restores it in a session-scoped fixture. Importing that module for a
helper silently forces the NumPy backend for the whole pytest session --
which in turn makes ``should_use_gpu`` return ``(False, 'cpu')``
unconditionally, so any test meaning to exercise the GPU path quietly
stops doing so while still passing.

This module must stay free of import-time side effects.

The generator must produce data byte-identical to
``tools/generate_references.py``, otherwise the ``references/*.npz``
fixtures no longer describe the same input.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

from pathlib import Path

import numpy as np
import mne

REFERENCES_DIR = Path(__file__).parent / "references"


def create_reference_epochs(n_epochs=30, n_channels=32, n_times=256,
                            sfreq=256, seed=42):
    """Create deterministic test epochs matching the reference data.

    Uses an isolated ``RandomState`` so the result does not depend on what
    other tests have done to the global NumPy RNG.

    Parameters
    ----------
    n_epochs, n_channels, n_times : int
        Shape of the generated data. The defaults are the values the
        ``references/*.npz`` fixtures were generated with -- changing them
        invalidates the comparison against those fixtures.
    sfreq : float
        Sampling frequency, Hz.
    seed : int
        Seed for the isolated RandomState.

    Returns
    -------
    mne.EpochsArray
        Shape (n_epochs, n_channels, n_times), units V, ~20 uV amplitude,
        with spherical channel positions so interpolation is well defined.
    """
    rng = np.random.RandomState(seed)

    # 1-based naming to avoid EEG000
    ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                           ch_types=['eeg'] * n_channels)

    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    phi = np.linspace(np.pi / 4, 3 * np.pi / 4, n_channels)
    radius = 0.09
    pos = np.column_stack([
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi),
    ])
    info.set_montage(mne.channels.make_dig_montage(
        ch_pos={ch: p for ch, p in zip(ch_names, pos)}, coord_frame='head'))

    data = rng.randn(n_epochs, n_channels, n_times) * 20e-6
    data += rng.randn(n_epochs, 1, n_times) * 5e-6      # common signal

    for idx in [5, 12, 22]:                              # bad epochs
        data[idx] *= 3.0

    data[3, 10, :] += rng.randn(n_times) * 100e-6        # bad channel/epoch
    data[8, 5, :] += rng.randn(n_times) * 80e-6
    data[15, 20, :] += rng.randn(n_times) * 90e-6
    data[18, 15, :] += rng.randn(n_times) * 120e-6

    events = np.column_stack([
        np.arange(0, n_epochs * n_times, n_times),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])
    return mne.EpochsArray(data, info, events=events, tmin=0, verbose=False)


def load_reference(name):
    """Load a reference ``.npz`` fixture by name (without extension).

    Returns
    -------
    dict
        Contents of the archive, or ``None`` if the file is absent so the
        caller can skip.
    """
    path = REFERENCES_DIR / f"{name}.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))
