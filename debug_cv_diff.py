#!/usr/bin/env python
"""Debug script to find exact source of CPU/GPU CV difference."""

import numpy as np
import torch
import mne
from mne.datasets import sample

# Suppress MNE logging
mne.set_log_level('ERROR')

from autoreject.autoreject import (
    _AutoReject, _run_local_reject_cv, _get_interp_chs,
    _interpolate_bad_epochs, _slicemean, _pbar, _GDKW
)
from autoreject.gpu_pipeline import (
    run_local_reject_cv_gpu_batch, _torch_median, GPUThresholdOptimizer
)
from autoreject.gpu_interpolation import gpu_batch_interpolate_all_n_interp
from sklearn.model_selection import KFold

np.random.seed(42)
torch.manual_seed(42)

# Create simple test data
n_epochs, n_channels, n_times = 30, 32, 200
sfreq = 200.0

# Create epochs
info = mne.create_info(
    ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
    sfreq=sfreq,
    ch_types=['eeg'] * n_channels
)

montage = mne.channels.make_standard_montage('standard_1020')
ch_names_32 = montage.ch_names[:n_channels]
info = mne.create_info(ch_names=ch_names_32, sfreq=sfreq, ch_types=['eeg'] * n_channels)

data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
epochs = mne.EpochsArray(data, info)
epochs.set_montage(montage)

picks = mne.pick_types(epochs.info, eeg=True)
ch_names = [epochs.ch_names[i] for i in picks]

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Simple thresh_func
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

# Parameters
n_interpolate = np.array([1, 4])
consensus = np.array([0.5, 0.9])
cv = KFold(n_splits=3, shuffle=True, random_state=42)

print("\n" + "=" * 70)
print("STEP-BY-STEP COMPARISON")
print("=" * 70)

# ============================================================================
# STEP 1: Fit _AutoReject (same for both)
# ============================================================================
print("\n--- STEP 1: Fit _AutoReject ---")

local_reject = _AutoReject(thresh_func=thresh_func, verbose=False, picks=picks, dots=None)
local_reject.fit(epochs)
ch_type = next(iter(local_reject.consensus_))

labels_original, bad_sensor_counts = local_reject._vote_bad_epochs(epochs, picks=picks)
print(f"  labels_original shape: {labels_original.shape}")
print(f"  bad_sensor_counts[:5]: {bad_sensor_counts[:5]}")

# ============================================================================
# STEP 2: Compute labels for each n_interp
# ============================================================================
print("\n--- STEP 2: Labels per n_interp ---")

labels_list = []
for n_interp in n_interpolate:
    local_reject.n_interpolate_[ch_type] = n_interp
    labels = local_reject._get_epochs_interpolation(
        epochs, labels=labels_original.copy(), picks=picks, 
        n_interpolate=n_interp, verbose=False
    )
    labels_list.append(labels)
    print(f"  n_interp={n_interp}: labels unique values: {np.unique(labels)}")

# ============================================================================
# STEP 3: Interpolate - CPU vs GPU
# ============================================================================
print("\n--- STEP 3: Interpolation CPU vs GPU ---")

X_original = epochs.get_data(picks, **_GDKW)
print(f"  X_original shape: {X_original.shape}")

# CPU interpolation for each n_interp
epochs_interp_cpu = []
for jdx, n_interp in enumerate(n_interpolate):
    labels = labels_list[jdx]
    interp_channels = _get_interp_chs(labels, epochs.ch_names, picks)
    epochs_copy = epochs.copy()
    _interpolate_bad_epochs(epochs_copy, interp_channels=interp_channels, 
                           picks=picks, dots=None, verbose=False)
    epochs_interp_cpu.append(epochs_copy.get_data(picks, **_GDKW))
    print(f"  CPU n_interp={n_interp}: shape {epochs_interp_cpu[-1].shape}")

# GPU interpolation
pos = epochs._get_channel_positions(picks)
optimizer = GPUThresholdOptimizer(device=device)

X_interp_gpu_list = gpu_batch_interpolate_all_n_interp(
    epochs, labels_list, picks, pos, device=optimizer.device, verbose=False
)

print(f"\n  Comparing CPU vs GPU interpolation:")
for jdx, n_interp in enumerate(n_interpolate):
    cpu_data = epochs_interp_cpu[jdx]
    gpu_data = X_interp_gpu_list[jdx].cpu().numpy().astype(np.float64)
    diff = np.abs(cpu_data - gpu_data)
    print(f"  n_interp={n_interp}: max_diff={diff.max():.2e}, mean_diff={diff.mean():.2e}")

# ============================================================================
# STEP 4: Compare CV fold calculations
# ============================================================================
print("\n--- STEP 4: CV Fold Calculations ---")

cv_splits = list(cv.split(np.zeros(len(epochs))))
X_gpu = optimizer._to_tensor(X_original)

n_interp = 4  # Focus on one value
jdx = 1  # Index in n_interpolate array
this_consensus = 0.5
idx_cons = 0  # Index in consensus array

print(f"\n  Testing n_interp={n_interp}, consensus={this_consensus}")

for fold, (train, test) in enumerate(cv_splits):
    print(f"\n  === Fold {fold} ===")
    print(f"    train indices: {train[:5]}... (len={len(train)})")
    print(f"    test indices: {test[:5]}... (len={len(test)})")
    
    # CPU calculation
    local_reject.n_interpolate_[ch_type] = n_interp
    local_reject.consensus_[ch_type] = this_consensus
    
    bad_epochs = local_reject._get_bad_epochs(
        bad_sensor_counts[train], picks=picks, ch_type=ch_type
    )
    good_epochs_idx = np.nonzero(np.invert(bad_epochs))[0]
    print(f"    bad_epochs in train: {np.sum(bad_epochs)}")
    print(f"    good_epochs_idx: {good_epochs_idx}")
    
    # CPU mean & score
    X_train_interp_cpu = epochs_interp_cpu[jdx][train]
    mean_cpu = _slicemean(X_train_interp_cpu, good_epochs_idx, axis=0)
    print(f"    mean_cpu shape: {mean_cpu.shape}")
    print(f"    mean_cpu[0,:5]: {mean_cpu[0,:5]}")
    
    X_test_cpu = X_original[test]
    median_cpu = np.median(X_test_cpu, axis=0)
    print(f"    median_cpu shape: {median_cpu.shape}")
    print(f"    median_cpu[0,:5]: {median_cpu[0,:5]}")
    
    score_cpu = -np.sqrt(np.mean((median_cpu - mean_cpu) ** 2))
    loss_cpu = -score_cpu
    print(f"    CPU score: {score_cpu:.10e}")
    print(f"    CPU loss: {loss_cpu:.10e}")
    
    # GPU calculation
    train_t = optimizer.torch.tensor(train, device=optimizer.device)
    test_t = optimizer.torch.tensor(test, device=optimizer.device)
    good_idx_t = optimizer.torch.tensor(good_epochs_idx, device=optimizer.device)
    
    X_interp_picks_gpu = X_interp_gpu_list[jdx]
    X_train_interp_gpu = X_interp_picks_gpu[train_t]
    X_good_gpu = X_train_interp_gpu[good_idx_t]
    mean_gpu = X_good_gpu.mean(dim=0)
    
    mean_gpu_np = mean_gpu.cpu().numpy().astype(np.float64)
    print(f"    mean_gpu[0,:5]: {mean_gpu_np[0,:5]}")
    print(f"    mean diff: {np.abs(mean_cpu - mean_gpu_np).max():.2e}")
    
    X_test_gpu = X_gpu[:, torch.tensor(picks, device=optimizer.device), :][test_t]
    median_gpu = _torch_median(X_test_gpu, dim=0)
    
    median_gpu_np = median_gpu.cpu().numpy().astype(np.float64)
    print(f"    median_gpu[0,:5]: {median_gpu_np[0,:5]}")
    print(f"    median diff: {np.abs(median_cpu - median_gpu_np).max():.2e}")
    
    sq_diff_gpu = (median_gpu - mean_gpu) ** 2
    score_gpu = -sq_diff_gpu.mean().sqrt()
    score_gpu_np = score_gpu.cpu().numpy()
    loss_gpu = -float(score_gpu_np)
    print(f"    GPU score: {float(score_gpu_np):.10e}")
    print(f"    GPU loss: {loss_gpu:.10e}")
    
    print(f"    LOSS DIFF: {abs(loss_cpu - loss_gpu):.2e}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
