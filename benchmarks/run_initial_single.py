
import argparse
import yaml
import time
import os
import numpy as np

# Import autoreject from the repo (assume initial API)
from autoreject import AutoReject

def create_synthetic_data_mne(n_channels, sfreq, epoch_duration, recording_duration, artifact_pct, random_state=42):
    import mne
    np.random.seed(random_state)
    n_times = int(epoch_duration * sfreq)
    n_epochs = int(recording_duration * 60 / epoch_duration)
    # Montage selection
    if n_channels <= 32:
        montage_name = 'standard_1020'
    elif n_channels <= 64:
        montage_name = 'standard_1005'
    elif n_channels <= 128:
        montage_name = 'GSN-HydroCel-128'
    else:
        montage_name = 'GSN-HydroCel-256'
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        ch_names = montage.ch_names[:n_channels]
    except Exception:
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        montage = None
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    if montage is not None:
        try:
            info.set_montage(montage)
        except Exception:
            pass
    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-5
    t = np.arange(n_times) / sfreq
    for epoch_idx in range(n_epochs):
        freq = np.random.uniform(8, 12)
        phase = np.random.uniform(0, 2 * np.pi)
        alpha = np.sin(2 * np.pi * freq * t + phase) * 2e-5
        data[epoch_idx] += alpha
    n_bad_epochs = int(n_epochs * artifact_pct)
    artifact_epochs = np.random.choice(n_epochs, n_bad_epochs, replace=False)
    for epoch_idx in artifact_epochs:
        n_bad_chs = np.random.randint(1, max(2, int(n_channels * 0.2)))
        bad_chs = np.random.choice(n_channels, n_bad_chs, replace=False)
        artifact_amplitude = np.random.uniform(3, 10) * 1e-4
        data[epoch_idx, bad_chs, :] += (
            np.random.randn(n_bad_chs, n_times) * artifact_amplitude
        )
    epochs = mne.EpochsArray(data, info, verbose=False)
    return epochs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    subset_configs = config.get('subset_configs', [])
    cfg = next((c for c in subset_configs if c['name'] == args.config), None)
    if cfg is None:
        print(f"Config {args.config} not found.")
        return

    print(f"Running config: {cfg['name']}")
    print(cfg)
    channels = cfg['channels']
    sfreq = cfg['sfreq']
    epoch_duration = cfg['epoch_duration']
    recording_duration = cfg['recording_duration']
    artifact_pct = cfg.get('artifact_pct', 0.3)

    print("Creating synthetic MNE data...")
    epochs = create_synthetic_data_mne(channels, sfreq, epoch_duration, recording_duration, artifact_pct)
    print(f"Epochs shape: {epochs.get_data().shape}")

    print("Running AutoReject (CPU only)...")
    start = time.time()
    ar = AutoReject()
    ar.fit(epochs)
    consensus = getattr(ar, 'consensus_', None)
    n_interpolate = getattr(ar, 'n_interpolate_', None)
    elapsed = time.time() - start
    print(f"CPU time: {elapsed:.2f}s")
    print(f"Estimated consensus: {consensus}")
    print(f"Estimated n_interpolate: {n_interpolate}")

if __name__ == '__main__':
    main()
