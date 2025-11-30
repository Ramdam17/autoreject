#!/usr/bin/env python
"""
Phase 4 Validation Test: Pipeline complet AutoReject CPU vs GPU

Ce test valide que le pipeline AutoReject complet produit les mêmes résultats
entre CPU et GPU:
- Même consensus
- Même n_interpolate
- Même bad_epochs_idx
"""

import numpy as np
import mne

print("=" * 60)
print("PHASE 4 VALIDATION: Pipeline AutoReject complet")
print("=" * 60)

# Créer un montage EEG standard
montage = mne.channels.make_standard_montage('standard_1020')
n_channels = 32

ch_names = montage.ch_names[:n_channels]
info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
info.set_montage(montage)

# Créer des epochs synthétiques avec quelques artefacts
n_epochs = 30
n_times = 200
rng = np.random.RandomState(42)

# Données de base
data = rng.randn(n_epochs, n_channels, n_times) * 1e-6

# Ajouter quelques artefacts (epochs 5, 12, 25 ont des amplitudes élevées)
artifact_epochs = [5, 12, 25]
artifact_channels = [3, 7, 15]  # Quelques canaux avec artefacts
for ep in artifact_epochs:
    for ch in artifact_channels:
        data[ep, ch, :] += rng.randn(n_times) * 5e-5  # Signal 50x plus fort

epochs = mne.EpochsArray(data, info, verbose=False)
print(f"\nConfiguration:")
print(f"  Epochs: {n_epochs} x {n_channels} channels x {n_times} times")
print(f"  Epochs avec artefacts: {artifact_epochs}")
print(f"  Canaux avec artefacts: {[ch_names[i] for i in artifact_channels]}")

# Détecter le device
import torch
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"  Device GPU: {device}")

# ============================================================================
# TEST: AutoRejectLocal CPU vs GPU
# ============================================================================
print("\n" + "=" * 60)
print("TEST: AutoRejectLocal CPU vs GPU")
print("=" * 60)

from autoreject import AutoReject

# Paramètres communs
n_interpolate_values = [1, 4, 8]
consensus_values = [0.1, 0.5, 0.9]

print(f"\n  n_interpolate: {n_interpolate_values}")
print(f"  consensus: {consensus_values}")

# ---- CPU Version ----
print("\n  --- CPU Version ---")
try:
    ar_cpu = AutoReject(
        n_interpolate=n_interpolate_values,
        consensus=consensus_values,
        random_state=42,
        n_jobs=1,
        verbose=True
    )
    
    epochs_cpu = epochs.copy()
    ar_cpu.fit(epochs_cpu)
    
    print(f"  Optimal n_interpolate: {ar_cpu.n_interpolate_}")
    print(f"  Optimal consensus: {ar_cpu.consensus_}")
    
    # Transformer
    epochs_cpu_clean, reject_log_cpu = ar_cpu.transform(epochs_cpu, return_log=True)
    
    cpu_success = True
    
except Exception as e:
    print(f"  ❌ CPU AutoReject failed: {e}")
    import traceback
    traceback.print_exc()
    cpu_success = False

# ---- GPU Version ----
print("\n  --- GPU Version ---")
try:
    # AutoReject a un paramètre device pour forcer GPU
    ar_gpu = AutoReject(
        n_interpolate=n_interpolate_values,
        consensus=consensus_values,
        random_state=42,
        n_jobs=1,
        verbose=True,
        device='gpu'  # Force GPU
    )
    
    epochs_gpu = epochs.copy()
    ar_gpu.fit(epochs_gpu)
    
    print(f"  Optimal n_interpolate: {ar_gpu.n_interpolate_}")
    print(f"  Optimal consensus: {ar_gpu.consensus_}")
    
    # Transformer
    epochs_gpu_clean, reject_log_gpu = ar_gpu.transform(epochs_gpu, return_log=True)
    
    gpu_success = True
    
except Exception as e:
    print(f"  ❌ GPU AutoReject failed: {e}")
    import traceback
    traceback.print_exc()
    gpu_success = False

# ---- Comparaison ----
print("\n" + "=" * 60)
print("COMPARAISON CPU vs GPU")
print("=" * 60)

if cpu_success and gpu_success:
    # Comparer les hyperparamètres optimaux
    print(f"\n  Hyperparamètres optimaux:")
    print(f"  {'Paramètre':<20} {'CPU':>15} {'GPU':>15} {'Match':>10}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")
    
    # n_interpolate_ et consensus_ sont des dicts par ch_type
    cpu_n_interp = ar_cpu.n_interpolate_.get('eeg', ar_cpu.n_interpolate_)
    gpu_n_interp = ar_gpu.n_interpolate_.get('eeg', ar_gpu.n_interpolate_)
    cpu_consensus = ar_cpu.consensus_.get('eeg', ar_cpu.consensus_)
    gpu_consensus = ar_gpu.consensus_.get('eeg', ar_gpu.consensus_)
    
    n_interp_match = cpu_n_interp == gpu_n_interp
    consensus_match = cpu_consensus == gpu_consensus
    
    print(f"  {'n_interpolate':<20} {cpu_n_interp:>15} {gpu_n_interp:>15} {'✅' if n_interp_match else '❌':>10}")
    print(f"  {'consensus':<20} {cpu_consensus:>15.2f} {gpu_consensus:>15.2f} {'✅' if consensus_match else '❌':>10}")
    
    # Comparer les bad epochs
    bad_cpu = reject_log_cpu.bad_epochs
    bad_gpu = reject_log_gpu.bad_epochs
    bad_match = np.array_equal(bad_cpu, bad_gpu)
    
    print(f"\n  Bad epochs:")
    print(f"    CPU: {np.where(bad_cpu)[0].tolist()}")
    print(f"    GPU: {np.where(bad_gpu)[0].tolist()}")
    print(f"    Match: {'✅' if bad_match else '❌'}")
    
    # Comparer les données nettoyées
    data_cpu_clean = epochs_cpu_clean.get_data()
    data_gpu_clean = epochs_gpu_clean.get_data()
    
    if data_cpu_clean.shape == data_gpu_clean.shape:
        diff = np.abs(data_cpu_clean - data_gpu_clean)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n  Données nettoyées:")
        print(f"    Shape CPU: {data_cpu_clean.shape}")
        print(f"    Shape GPU: {data_gpu_clean.shape}")
        print(f"    Max diff: {max_diff:.2e}")
        print(f"    Mean diff: {mean_diff:.2e}")
        
        data_match = max_diff < 1e-6
        print(f"    Match: {'✅' if data_match else '❌'}")
    else:
        print(f"\n  ⚠️ Shapes différentes:")
        print(f"    CPU: {data_cpu_clean.shape}")
        print(f"    GPU: {data_gpu_clean.shape}")
        data_match = False
    
    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ PHASE 4")
    print("=" * 60)
    
    all_match = n_interp_match and consensus_match and bad_match and data_match
    
    if all_match:
        print("✅ PHASE 4 VALIDÉE: CPU et GPU produisent les mêmes résultats!")
        print("   - Mêmes hyperparamètres optimaux")
        print("   - Mêmes bad epochs")
        print("   - Mêmes données nettoyées")
    else:
        print("❌ PHASE 4 ÉCHOUÉE: Différences entre CPU et GPU")
        if not n_interp_match:
            print("   - n_interpolate différent")
        if not consensus_match:
            print("   - consensus différent")
        if not bad_match:
            print("   - bad_epochs différents")
        if not data_match:
            print("   - données nettoyées différentes")
            
else:
    print("\n❌ Impossible de comparer: une des versions a échoué")
    if not cpu_success:
        print("   - CPU a échoué")
    if not gpu_success:
        print("   - GPU a échoué")
