#!/usr/bin/env python
"""Debug script to find exact source of CPU/GPU CV difference - Full comparison."""

import numpy as np
import torch
import mne

mne.set_log_level('ERROR')

from autoreject.autoreject import (
    _AutoReject, _run_local_reject_cv, _get_interp_chs,
    _interpolate_bad_epochs, _slicemean, _pbar, _GDKW
)
from autoreject.gpu_pipeline import run_local_reject_cv_gpu_batch
from sklearn.model_selection import KFold

np.random.seed(42)
torch.manual_seed(42)

# Create simple test data
n_epochs, n_channels, n_times = 30, 32, 200
sfreq = 200.0

montage = mne.channels.make_standard_montage('standard_1020')
ch_names_32 = montage.ch_names[:n_channels]
info = mne.create_info(ch_names=ch_names_32, sfreq=sfreq, ch_types=['eeg'] * n_channels)

data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
epochs = mne.EpochsArray(data, info)
epochs.set_montage(montage)

picks = mne.pick_types(epochs.info, eeg=True)

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def thresh_func(data, dots=None, picks=None, verbose=None):
    threshes = {}
    if picks is None:
        picks = list(range(data.get_data().shape[1]))
    epoch_data = data.get_data()
    ch_names_list = data.info['ch_names']
    for ch_idx in picks:
        ch_data = epoch_data[:, ch_idx, :]
        ptp = np.ptp(ch_data, axis=-1)
        thresh = np.median(ptp) + 2 * np.std(ptp)
        threshes[ch_names_list[ch_idx]] = thresh
    return threshes

# Parameters - same as test
n_interpolate = np.array([1, 4, 8])
consensus = np.array([0.1, 0.5, 0.9])
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 70)
print("FULL CV COMPARISON")
print("=" * 70)

# Run CPU
print("\nRunning CPU...")
local_reject_cpu, loss_cpu = _run_local_reject_cv(
    epochs, thresh_func, np.array(picks), 
    n_interpolate, cv, consensus,
    dots=None, verbose=False, n_jobs=1
)

# Run GPU
print("Running GPU...")
local_reject_gpu, loss_gpu = run_local_reject_cv_gpu_batch(
    epochs, thresh_func, np.array(picks),
    n_interpolate, cv, consensus,
    dots=None, verbose=False, n_jobs=1, device=device
)

print("\n" + "=" * 70)
print("LOSS COMPARISON")
print("=" * 70)

print(f"\nLoss shapes: CPU={loss_cpu.shape}, GPU={loss_gpu.shape}")

# Compare cell by cell
print("\n--- Loss values by (consensus, n_interp, fold) ---")
for i, cons in enumerate(consensus):
    for j, n_int in enumerate(n_interpolate):
        print(f"\nconsensus={cons}, n_interp={n_int}:")
        for f in range(5):
            cpu_val = loss_cpu[i, j, f]
            gpu_val = loss_gpu[i, j, f]
            diff = abs(cpu_val - gpu_val)
            match = "✅" if diff < 1e-10 else "⚠️" if diff < 1e-5 else "❌"
            if np.isinf(cpu_val) and np.isinf(gpu_val):
                match = "✅ (both inf)"
            print(f"  fold {f}: CPU={cpu_val:.10e}, GPU={gpu_val:.10e}, diff={diff:.2e} {match}")

# Mean over folds
print("\n--- Mean loss over folds ---")
loss_cpu_mean = np.mean(np.where(np.isinf(loss_cpu), 1e10, loss_cpu), axis=2)
loss_gpu_mean = np.mean(np.where(np.isinf(loss_gpu), 1e10, loss_gpu), axis=2)

print("\nCPU mean loss:")
print(loss_cpu_mean)
print("\nGPU mean loss:")
print(loss_gpu_mean)

print("\n--- Best parameters ---")
best_cpu = np.unravel_index(np.argmin(loss_cpu_mean), loss_cpu_mean.shape)
best_gpu = np.unravel_index(np.argmin(loss_gpu_mean), loss_gpu_mean.shape)

print(f"CPU: consensus={consensus[best_cpu[0]]}, n_interp={n_interpolate[best_cpu[1]]}, loss={loss_cpu_mean[best_cpu]:.10e}")
print(f"GPU: consensus={consensus[best_gpu[0]]}, n_interp={n_interpolate[best_gpu[1]]}, loss={loss_gpu_mean[best_gpu]:.10e}")

# Show all losses near the minimum
print("\n--- All losses near minimum ---")
min_val = min(loss_cpu_mean.min(), loss_gpu_mean.min())
for i, cons in enumerate(consensus):
    for j, n_int in enumerate(n_interpolate):
        cpu_val = loss_cpu_mean[i, j]
        gpu_val = loss_gpu_mean[i, j]
        if cpu_val < min_val * 1.001 or gpu_val < min_val * 1.001:
            print(f"  cons={cons}, n_int={n_int}: CPU={cpu_val:.12e}, GPU={gpu_val:.12e}")
