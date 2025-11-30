import argparse
import yaml
import time
import os
import numpy as np

# Import autoreject from the repo (assume initial API)
from autoreject import AutoReject

def create_synthetic_data(channels, sfreq, epoch_duration, recording_duration, artifact_pct):
    n_epochs = int(recording_duration * sfreq / epoch_duration)
    n_times = int(epoch_duration * sfreq)
    rng = np.random.RandomState(42)
    data = rng.randn(n_epochs, channels, n_times)
    # Add artifacts
    n_artifacts = int(n_epochs * artifact_pct)
    for i in range(n_artifacts):
        data[i] += rng.normal(10, 5, size=(channels, n_times))
    return data

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

    print("Creating synthetic data...")
    data = create_synthetic_data(channels, sfreq, epoch_duration, recording_duration, artifact_pct)
    print(f"Data shape: {data.shape}")

    print("Running AutoReject (CPU only)...")
    start = time.time()
    ar = AutoReject()
    ar.fit(data)
    consensus = getattr(ar, 'consensus_', None)
    n_interpolate = getattr(ar, 'n_interpolate_', None)
    elapsed = time.time() - start
    print(f"CPU time: {elapsed:.2f}s")
    print(f"Estimated consensus: {consensus}")
    print(f"Estimated n_interpolate: {n_interpolate}")

if __name__ == '__main__':
    main()
