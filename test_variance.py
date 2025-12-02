import os
import importlib
import numpy as np
import mne

mne.set_log_level('ERROR')

def generate_data(seed):
    np.random.seed(seed)
    n_channels = 128
    n_epochs = 300
    n_times = 1000
    sfreq = 500
    artifact_pct = 0.2

    montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    ch_names = montage.ch_names[:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg']*n_channels)
    info.set_montage(montage)

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
        data[epoch_idx, bad_chs, :] += np.random.randn(n_bad_chs, n_times) * artifact_amplitude

    return mne.EpochsArray(data, info)

def run_ar(epochs, seed, backend):
    os.environ['AUTOREJECT_BACKEND'] = backend
    import autoreject
    importlib.reload(autoreject.backends)
    importlib.reload(autoreject.autoreject)
    importlib.reload(autoreject)
    from autoreject import AutoReject
    
    ar = AutoReject(n_interpolate=[1,2,4,8,12,16], consensus=[0.1,0.2,0.3,0.4,0.5], 
                    cv=10, random_state=seed, verbose=False)
    ar.fit(epochs.copy())
    return ar.consensus_['eeg'], ar.n_interpolate_['eeg'], ar.loss_['eeg'].mean(axis=2)

# Run CPU with 10 different seeds
print('Running CPU with 10 different seeds...')
seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
cpu_results = []

for seed in seeds:
    print(f'  CPU seed {seed}...', end=' ', flush=True)
    epochs = generate_data(seed)
    consensus, n_interp, loss = run_ar(epochs, seed, 'numpy')
    cpu_results.append({'seed': seed, 'consensus': consensus, 'n_interp': n_interp, 'loss': loss})
    print(f'consensus={consensus}, n_interp={n_interp}')

# Run GPU with first seed
print('\nRunning GPU with seed 42...')
epochs = generate_data(42)
gpu_consensus, gpu_n_interp, gpu_loss = run_ar(epochs, 42, 'torch')
print(f'  GPU: consensus={gpu_consensus}, n_interp={gpu_n_interp}')

# Analysis
cpu_42 = cpu_results[0]
print('\n' + '='*70)
print('RÉSULTATS')
print('='*70)

print(f'\nCPU consensus across seeds: {[r["consensus"] for r in cpu_results]}')
print(f'CPU n_interp across seeds: {[r["n_interp"] for r in cpu_results]}')

print(f'\nSeed 42: CPU={cpu_42["consensus"]}, GPU={gpu_consensus}')
print(f'Same? {cpu_42["consensus"] == gpu_consensus}')

# Loss comparison
mask = ~(np.isinf(cpu_42['loss']) | np.isinf(gpu_loss))
loss_diff = np.abs(gpu_loss - cpu_42['loss'])[mask].max()

cpu_losses = np.array([r['loss'] for r in cpu_results])
cpu_std = np.nanstd(cpu_losses, axis=0)[mask].max()

print(f'\nMax GPU-CPU diff: {loss_diff:.6e}')
print(f'Max CPU std across seeds: {cpu_std:.6e}')
print(f'\nGPU diff < CPU variance? {loss_diff < cpu_std}')