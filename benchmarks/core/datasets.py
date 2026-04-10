"""Real dataset loading for benchmarks.

Handles download, caching, and epoching of 3 reference datasets:

- **MNE Sample** (auditory-visual, ~60 EEG channels)
- **ds002778** (Parkinson's resting-state, 32ch Biosemi) [1]_
- **ds000117** (Face recognition, ~60 EEG channels) [2]_

Each loader returns ``(epochs, metadata)`` for consistent interface.

References
----------
.. [1] Rockhill, A. P. et al. (2020). OpenNeuro ds002778: EEG dataset of
       Parkinson's disease patients during resting state.
.. [2] Wakeman, D. G., & Henson, R. N. (2015). A multi-subject, multi-modal
       human neuroimaging dataset. Scientific Data, 2, 150001.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def load_dataset(config: dict) -> tuple[Any, dict]:
    """Load a dataset based on config source field.

    Parameters
    ----------
    config : dict
        Benchmark config with 'source' key.

    Returns
    -------
    epochs : mne.Epochs
    metadata : dict
    """
    source = config["source"]

    if source == "synthetic":
        from .synthetic import generate_epochs
        return generate_epochs(
            n_channels=config["n_channels"],
            sfreq=config.get("sfreq", 500),
            epoch_duration=config.get("epoch_duration", 2.0),
            recording_duration_min=config.get("recording_duration_min", 10),
            artifact_pct=config.get("artifact_pct", 0.3),
            random_state=config.get("random_state", 42),
        )
    elif source == "mne_sample":
        return load_mne_sample(config)
    elif source == "ds002778":
        return load_ds002778(config)
    elif source == "ds000117":
        return load_ds000117(config)
    else:
        raise ValueError(f"Unknown data source: {source}")


def load_mne_sample(config: dict) -> tuple[Any, dict]:
    """Load MNE sample dataset, EEG channels only.

    Tries the testing (truncated) dataset first, falls back to full sample.
    """
    import mne

    logger.info("Loading MNE sample dataset...")

    # Try truncated testing data first (faster download, ~400MB)
    try:
        data_path = mne.datasets.testing.data_path(download=True)
        raw_fname = (
            data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
        )
        if not raw_fname.exists():
            raise FileNotFoundError(raw_fname)
        raw = mne.io.read_raw_fif(str(raw_fname), preload=True)
        logger.info("  Using truncated testing dataset")
    except Exception:
        # Fall back to full sample data
        logger.info("  Testing data unavailable, trying full sample...")
        data_path = mne.datasets.sample.data_path(download=True)
        raw_fname = (
            data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
        )
        raw = mne.io.read_raw_fif(str(raw_fname), preload=True)

    # Pick EEG only
    raw.pick("eeg")

    # Find events
    events = mne.find_events(raw, stim_channel="STI 014")

    # Create epochs
    epochs = mne.Epochs(
        raw, events, event_id=None,
        tmin=-0.2, tmax=0.5, baseline=(None, 0),
        preload=True,
    )

    metadata = {
        "source": "mne_sample",
        "description": "MNE sample auditory-visual, EEG only",
        "n_epochs": len(epochs),
        "n_channels": len(epochs.ch_names),
        "n_times": epochs.get_data().shape[-1],
        "sfreq": epochs.info["sfreq"],
    }
    logger.info("  Loaded: %d epochs, %d channels", len(epochs),
                len(epochs.ch_names))

    return epochs, metadata


def load_ds002778(config: dict) -> tuple[Any, dict]:
    """Load OpenNeuro ds002778: Parkinson's resting-state EEG.

    32-channel Biosemi, fixed-length 3s epochs.
    """
    import mne

    logger.info("Loading ds002778 (Parkinson's resting-state)...")

    subject = config.get("subject", "pd14")
    session = config.get("session", "off")
    epoch_duration = config.get("epoch_duration", 3.0)

    # Determine cache directory
    repo_root = Path(__file__).parent.parent.parent
    target_dir = repo_root / "examples" / "ds002778"

    bdf_path = (
        target_dir / f"sub-{subject}" / f"ses-{session}" / "eeg"
        / f"sub-{subject}_ses-{session}_task-rest_eeg.bdf"
    )

    if not bdf_path.exists():
        logger.info("  Downloading from OpenNeuro...")
        import openneuro
        openneuro.download(
            dataset="ds002778",
            target_dir=str(target_dir),
            include=[f"sub-{subject}/ses-{session}/"],
        )

    raw = mne.io.read_raw_bdf(str(bdf_path), preload=True)

    # Set montage and pick EEG
    montage = mne.channels.make_standard_montage("biosemi32")
    montage_ch_names = set(montage.ch_names)
    eeg_picks = [ch for ch in raw.ch_names if ch in montage_ch_names]
    raw.pick(eeg_picks)
    raw.set_montage(montage)

    # Make fixed-length epochs
    epochs = mne.make_fixed_length_epochs(
        raw, duration=epoch_duration, preload=True,
    )

    metadata = {
        "source": "ds002778",
        "description": f"Parkinson resting-state sub-{subject} ses-{session}",
        "n_epochs": len(epochs),
        "n_channels": len(epochs.ch_names),
        "n_times": epochs.get_data().shape[-1],
        "sfreq": epochs.info["sfreq"],
    }
    logger.info("  Loaded: %d epochs, %d channels", len(epochs),
                len(epochs.ch_names))

    return epochs, metadata


def load_ds000117(config: dict) -> tuple[Any, dict]:
    """Load OpenNeuro ds000117: Face recognition MEG/EEG.

    Uses EEG channels from multiple runs, concatenated.
    """
    import mne

    logger.info("Loading ds000117 (Face recognition)...")

    subject = str(config.get("subject", "16"))
    runs = config.get("runs", [3, 4, 5, 6])
    tmin = config.get("tmin", -0.2)
    tmax = config.get("tmax", 0.8)

    repo_root = Path(__file__).parent.parent.parent
    target_dir = repo_root / "examples" / "ds000117"

    sub_dir = target_dir / f"sub-{subject}" / "ses-meg" / "meg"

    # Check if first run file exists
    first_run = (
        sub_dir
        / f"sub-{subject}_ses-meg_task-facerecognition_run-{runs[0]:02d}_meg.fif"
    )

    if not first_run.exists():
        logger.info("  Downloading from OpenNeuro...")
        import openneuro
        openneuro.download(
            dataset="ds000117",
            target_dir=str(target_dir),
            include=[f"sub-{subject}/ses-meg/"],
        )

    all_epochs = []
    for run in runs:
        run_fname = (
            sub_dir
            / f"sub-{subject}_ses-meg_task-facerecognition_run-{run:02d}_meg.fif"
        )
        if not run_fname.exists():
            logger.warning("  Run %d not found, skipping", run)
            continue

        raw = mne.io.read_raw_fif(str(run_fname), preload=True)

        # Pick EEG and remap special channels
        eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
        raw.pick(eeg_picks)

        # Rename special channels
        rename_map = {}
        if "EEG061" in raw.ch_names:
            rename_map["EEG061"] = "EOG061"
        if "EEG062" in raw.ch_names:
            rename_map["EEG062"] = "EOG062"
        if "EEG063" in raw.ch_names:
            rename_map["EEG063"] = "ECG063"
        if "EEG064" in raw.ch_names:
            rename_map["EEG064"] = "MISC064"

        if rename_map:
            raw.rename_channels(rename_map)
            for ch_name, ch_type in [
                ("EOG061", "eog"), ("EOG062", "eog"),
                ("ECG063", "ecg"), ("MISC064", "misc"),
            ]:
                if ch_name in raw.ch_names:
                    raw.set_channel_types({ch_name: ch_type})

        # Pick only EEG after renaming
        raw.pick("eeg")

        # Filter
        raw.filter(1.0, 40.0)

        # Find events and epoch
        events = mne.find_events(raw, stim_channel="STI101",
                                 min_duration=0.002)
        if len(events) == 0:
            logger.warning("  No events in run %d, skipping", run)
            continue

        epochs = mne.Epochs(
            raw, events, tmin=tmin, tmax=tmax,
            baseline=None, preload=True, decim=4,
        )
        all_epochs.append(epochs)
        logger.info("  Run %d: %d epochs", run, len(epochs))

    if not all_epochs:
        raise RuntimeError("No epochs loaded from ds000117")

    epochs = mne.concatenate_epochs(all_epochs)

    metadata = {
        "source": "ds000117",
        "description": f"Face recognition sub-{subject}, runs {runs}",
        "n_epochs": len(epochs),
        "n_channels": len(epochs.ch_names),
        "n_times": epochs.get_data().shape[-1],
        "sfreq": epochs.info["sfreq"],
    }
    logger.info("  Total: %d epochs, %d channels", len(epochs),
                len(epochs.ch_names))

    return epochs, metadata
