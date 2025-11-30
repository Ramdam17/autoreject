#!/usr/bin/env python
"""
Étape 6 Validation: Transform (application finale)

Ce test valide que ar.transform() produit les mêmes résultats
entre CPU et GPU (interpolation finale + suppression epochs).

Objectifs à valider:
1. GPU partout où possible
2. float64 par défaut, float32 si MPS
3. torch.linalg.pinv - déjà validé
4. CPU == GPU : mêmes epochs supprimés, données interpolées identiques
5. Tests passent
"""

import numpy as np
import torch
import mne
import os

mne.set_log_level('ERROR')

print("=" * 70)
print("ÉTAPE 6 VALIDATION: Transform (application finale)")
print("=" * 70)

# Seed pour reproductibilité
np.random.seed(42)
torch.manual_seed(42)

# Créer données
montage = mne.channels.make_standard_montage('standard_1020')
n_channels = 32
n_epochs = 40
n_times = 200

ch_names = montage.ch_names[:n_channels]
info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
info.set_montage(montage)

rng = np.random.RandomState(42)
data = rng.randn(n_epochs, n_channels, n_times) * 1e-6

# Artefacts
artifact_epochs = [2, 5, 8, 11, 15, 20, 25, 30, 35]
for i, ep in enumerate(artifact_epochs):
    n_bad = (i % 4) + 1
    bad_chs = rng.choice(n_channels, n_bad, replace=False)
    for ch in bad_chs:
        data[ep, ch, :] += rng.randn(n_times) * (3 + i) * 1e-5

epochs = mne.EpochsArray(data, info, verbose=False)

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nConfiguration:")
print(f"  Epochs: {n_epochs} x {n_channels} channels x {n_times} times")
print(f"  Epochs avec artefacts: {artifact_epochs}")
print(f"  Device: {device}")

# ============================================================================
# TEST 1: fit_transform() CPU vs GPU
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: fit_transform() CPU vs GPU")
print("=" * 70)

from autoreject import AutoReject
from autoreject.backends import clear_backend_cache

n_interpolate = np.array([1, 4, 8])
consensus = np.array([0.1, 0.5, 0.9])

# CPU
os.environ['AUTOREJECT_BACKEND'] = 'numpy'
clear_backend_cache()

print("  Fitting CPU...")
ar_cpu = AutoReject(
    n_interpolate=n_interpolate,
    consensus=consensus,
    cv=5,
    random_state=42,
    verbose=False
)
epochs_clean_cpu, reject_log_cpu = ar_cpu.fit_transform(epochs.copy(), return_log=True)

print(f"  CPU: {len(epochs_clean_cpu)} epochs restants")
print(f"  CPU bad_epochs: {np.where(reject_log_cpu.bad_epochs)[0].tolist()}")

# GPU
os.environ['AUTOREJECT_BACKEND'] = 'torch'
clear_backend_cache()

print("  Fitting GPU...")
ar_gpu = AutoReject(
    n_interpolate=n_interpolate,
    consensus=consensus,
    cv=5,
    random_state=42,
    verbose=False
)
epochs_clean_gpu, reject_log_gpu = ar_gpu.fit_transform(epochs.copy(), return_log=True)

print(f"  GPU: {len(epochs_clean_gpu)} epochs restants")
print(f"  GPU bad_epochs: {np.where(reject_log_gpu.bad_epochs)[0].tolist()}")

# Comparer
same_n_epochs = len(epochs_clean_cpu) == len(epochs_clean_gpu)
same_bad_epochs = np.array_equal(reject_log_cpu.bad_epochs, reject_log_gpu.bad_epochs)

print(f"\n  Même nombre d'epochs: {'✅' if same_n_epochs else '❌'}")
print(f"  Mêmes bad_epochs: {'✅' if same_bad_epochs else '❌'}")

if same_n_epochs and same_bad_epochs:
    # Comparer les données
    data_cpu = epochs_clean_cpu.get_data()
    data_gpu = epochs_clean_gpu.get_data()
    
    diff = np.abs(data_cpu - data_gpu)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  Data max diff: {max_diff:.2e}")
    print(f"  Data mean diff: {mean_diff:.2e}")
    
    if max_diff < 1e-10:
        print(f"  ✅ TEST 1 PASSED: Données bit-exact")
        test1_passed = True
    elif max_diff < 1e-5:
        print(f"  ✅ TEST 1 PASSED: Données très proches (acceptable MPS)")
        test1_passed = True
    else:
        print(f"  ❌ TEST 1 FAILED: Différences significatives")
        test1_passed = False
else:
    print(f"  ❌ TEST 1 FAILED: Epochs différents")
    test1_passed = False

# ============================================================================
# TEST 2: Reject log labels identiques
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Reject log labels identiques")
print("=" * 70)

labels_cpu = reject_log_cpu.labels
labels_gpu = reject_log_gpu.labels

print(f"  Labels shape: {labels_cpu.shape}")
print(f"  Labels unique CPU: {np.unique(labels_cpu)}")
print(f"  Labels unique GPU: {np.unique(labels_gpu)}")

if np.array_equal(labels_cpu, labels_gpu):
    print(f"  ✅ TEST 2 PASSED: Labels identiques")
    test2_passed = True
else:
    diff_count = np.sum(labels_cpu != labels_gpu)
    print(f"  ❌ TEST 2 FAILED: {diff_count} labels différents")
    test2_passed = False

# ============================================================================
# TEST 3: Transform sur nouvelles données
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: transform() sur nouvelles données")
print("=" * 70)

# Créer de nouvelles epochs
rng2 = np.random.RandomState(123)
data_new = rng2.randn(20, n_channels, n_times) * 1e-6

# Ajouter des artefacts similaires
for ep in [3, 7, 12]:
    bad_chs = rng2.choice(n_channels, 2, replace=False)
    for ch in bad_chs:
        data_new[ep, ch, :] += rng2.randn(n_times) * 5e-5

epochs_new = mne.EpochsArray(data_new, info, verbose=False)

# Appliquer les modèles déjà fittés
os.environ['AUTOREJECT_BACKEND'] = 'numpy'
clear_backend_cache()
epochs_new_cpu, log_new_cpu = ar_cpu.transform(epochs_new.copy(), return_log=True)

os.environ['AUTOREJECT_BACKEND'] = 'torch'
clear_backend_cache()
epochs_new_gpu, log_new_gpu = ar_gpu.transform(epochs_new.copy(), return_log=True)

print(f"  CPU: {len(epochs_new_cpu)} epochs restants")
print(f"  GPU: {len(epochs_new_gpu)} epochs restants")

same_n = len(epochs_new_cpu) == len(epochs_new_gpu)
same_bad = np.array_equal(log_new_cpu.bad_epochs, log_new_gpu.bad_epochs)

if same_n and same_bad:
    data_new_cpu = epochs_new_cpu.get_data()
    data_new_gpu = epochs_new_gpu.get_data()
    
    diff = np.abs(data_new_cpu - data_new_gpu)
    max_diff = np.max(diff)
    
    print(f"  Data max diff: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print(f"  ✅ TEST 3 PASSED: Transform identique")
        test3_passed = True
    else:
        print(f"  ❌ TEST 3 FAILED: Différences significatives")
        test3_passed = False
else:
    print(f"  ❌ TEST 3 FAILED: Epochs différents")
    test3_passed = False

# ============================================================================
# TEST 4: Hyperparamètres sélectionnés
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: Hyperparamètres sélectionnés identiques")
print("=" * 70)

print(f"  CPU n_interpolate_: {ar_cpu.n_interpolate_}")
print(f"  GPU n_interpolate_: {ar_gpu.n_interpolate_}")
print(f"  CPU consensus_: {ar_cpu.consensus_}")
print(f"  GPU consensus_: {ar_gpu.consensus_}")

same_params = (ar_cpu.n_interpolate_ == ar_gpu.n_interpolate_ and 
               ar_cpu.consensus_ == ar_gpu.consensus_)

if same_params:
    print(f"  ✅ TEST 4 PASSED: Mêmes hyperparamètres")
    test4_passed = True
else:
    print(f"  ❌ TEST 4 FAILED: Hyperparamètres différents")
    test4_passed = False

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ ÉTAPE 6 - Transform (application finale)")
print("=" * 70)

print(f"\n  TEST 1 (fit_transform): {'✅' if test1_passed else '❌'}")
print(f"  TEST 2 (reject log labels): {'✅' if test2_passed else '❌'}")
print(f"  TEST 3 (transform nouvelles données): {'✅' if test3_passed else '❌'}")
print(f"  TEST 4 (hyperparamètres): {'✅' if test4_passed else '❌'}")

all_passed = test1_passed and test2_passed and test3_passed and test4_passed

if all_passed:
    print(f"\n✅ ÉTAPE 6 VALIDÉE: Transform fonctionne identiquement CPU vs GPU")
else:
    print(f"\n❌ ÉTAPE 6 ÉCHOUÉE: Corriger les problèmes ci-dessus")
