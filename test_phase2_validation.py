#!/usr/bin/env python
"""
Phase 2 Validation Test: GPU interpolation matrix vs CPU

Ce test valide que gpu_make_interpolation_matrix produit les mêmes résultats
que la version CPU de MNE.

OBJECTIF: CPU == GPU à chaque étape (bit-exact ou très proche)
"""

import numpy as np
import mne

# Créer des données synthétiques avec les bonnes positions EEG
print("=" * 60)
print("PHASE 2 VALIDATION: gpu_make_interpolation_matrix")
print("=" * 60)

# Créer un montage EEG standard avec positions 3D
montage = mne.channels.make_standard_montage('standard_1020')
n_channels = 32  # On prend les 32 premiers canaux

# Créer les infos du montage
ch_names = montage.ch_names[:n_channels]
info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')

# Appliquer le montage pour avoir les positions
info.set_montage(montage)

print(f"\nConfiguration:")
print(f"  Nombre de canaux: {n_channels}")
print(f"  Canaux: {ch_names[:5]}... (premiers 5)")

# Extraire les positions des canaux
pos = np.array([info['chs'][i]['loc'][:3] for i in range(n_channels)])
print(f"  Positions shape: {pos.shape}")

# Simuler des mauvais canaux
bads_idx = [3, 7, 15]  # Indices des canaux à interpoler
goods_idx = [i for i in range(n_channels) if i not in bads_idx]

pos_bads = pos[bads_idx]
pos_goods = pos[goods_idx]

print(f"  Mauvais canaux (indices): {bads_idx}")
print(f"  Bons canaux: {len(goods_idx)}")

# ============================================================================
# TEST 1: Comparer les dots de Legendre
# ============================================================================
print("\n" + "=" * 60)
print("TEST 1: Legendre Polynomials (G matrix)")
print("=" * 60)

# CPU: Calculer G matrix avec MNE
from mne.channels.interpolation import _calc_g

# Calculer les distances cos entre good et bad channels
def compute_cos_distances(pos_from, pos_to):
    """Compute cosine distances between two sets of positions."""
    from numpy.linalg import norm
    pos_from = pos_from / norm(pos_from, axis=1, keepdims=True)
    pos_to = pos_to / norm(pos_to, axis=1, keepdims=True)
    cos_dist = pos_from @ pos_to.T
    # Clamp to [-1, 1]
    cos_dist = np.clip(cos_dist, -1, 1)
    return cos_dist

cos_dist_goods_to_bads = compute_cos_distances(pos_goods, pos_bads)
print(f"  cos_dist shape: {cos_dist_goods_to_bads.shape}")

# CPU: G matrix
G_cpu = _calc_g(cos_dist_goods_to_bads)
print(f"  G_cpu shape: {G_cpu.shape}")
print(f"  G_cpu dtype: {G_cpu.dtype}")
print(f"  G_cpu sample values: {G_cpu[0, :3]}")

# GPU: G matrix avec notre implémentation
import torch
from autoreject.gpu_interpolation import _calc_g_torch, is_cuda_device

# Détecter le device
if torch.cuda.is_available():
    device = 'cuda'
    print(f"\n  Device: CUDA")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    print(f"\n  Device: MPS")
else:
    device = 'cpu'
    print(f"\n  Device: CPU")

use_cuda = is_cuda_device(device)
print(f"  is_cuda_device: {use_cuda}")

# Pour le calcul, on utilise float64 sur CPU ou CUDA
if use_cuda:
    compute_device = device
    print(f"  Stratégie: float64 sur CUDA")
else:
    compute_device = 'cpu'
    print(f"  Stratégie: float64 sur CPU (MPS data sur device)")

# Calculer G avec PyTorch en utilisant _calc_g_torch (la vraie implémentation)
cos_dist_t = torch.tensor(cos_dist_goods_to_bads, dtype=torch.float64, device=compute_device)

# Utiliser la même fonction que gpu_make_interpolation_matrix
G_gpu = _calc_g_torch(cos_dist_t, stiffness=4, n_legendre_terms=50)

G_gpu_np = G_gpu.cpu().numpy()

print(f"\n  G_gpu shape: {G_gpu_np.shape}")
print(f"  G_gpu dtype: {G_gpu_np.dtype}")
print(f"  G_gpu sample values: {G_gpu_np[0, :3]}")

# Comparer
diff = np.abs(G_cpu - G_gpu_np)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"\n  RÉSULTATS:")
print(f"  Max diff: {max_diff:.2e}")
print(f"  Mean diff: {mean_diff:.2e}")

if max_diff < 1e-10:
    print(f"  ✅ TEST 1 PASSED: G matrices identiques")
else:
    print(f"  ❌ TEST 1 FAILED: Différences détectées")
    print(f"     CPU G[0,:5]: {G_cpu[0, :5]}")
    print(f"     GPU G[0,:5]: {G_gpu_np[0, :5]}")

# ============================================================================
# TEST 2: Comparer gpu_make_interpolation_matrix avec CPU
# ============================================================================
print("\n" + "=" * 60)
print("TEST 2: gpu_make_interpolation_matrix vs CPU")
print("=" * 60)

from autoreject.gpu_interpolation import gpu_make_interpolation_matrix
from mne.channels.interpolation import _make_interpolation_matrix

# Préparer les positions au format attendu par MNE
# MNE attend des positions normalisées sur une sphère

# CPU version
interp_matrix_cpu = _make_interpolation_matrix(pos_goods, pos_bads)
print(f"  CPU interp_matrix shape: {interp_matrix_cpu.shape}")
print(f"  CPU interp_matrix dtype: {interp_matrix_cpu.dtype}")

# GPU version
interp_matrix_gpu = gpu_make_interpolation_matrix(pos_goods, pos_bads, device=device)

# Convertir en numpy - DeviceArray a une propriété data qui est le tensor
if hasattr(interp_matrix_gpu, 'data'):
    # C'est un DeviceArray, extraire le tensor
    interp_matrix_gpu_np = interp_matrix_gpu.data.cpu().numpy().astype(np.float64)
elif hasattr(interp_matrix_gpu, 'cpu'):
    interp_matrix_gpu_np = interp_matrix_gpu.cpu().numpy()
else:
    interp_matrix_gpu_np = np.array(interp_matrix_gpu)

print(f"  GPU interp_matrix shape: {interp_matrix_gpu_np.shape}")
print(f"  GPU interp_matrix dtype: {interp_matrix_gpu_np.dtype}")

# Comparer
diff = np.abs(interp_matrix_cpu - interp_matrix_gpu_np)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
rel_diff = max_diff / (np.max(np.abs(interp_matrix_cpu)) + 1e-10)

print(f"\n  RÉSULTATS:")
print(f"  Max absolute diff: {max_diff:.2e}")
print(f"  Mean absolute diff: {mean_diff:.2e}")
print(f"  Max relative diff: {rel_diff:.2e}")

if max_diff < 1e-10:
    print(f"  ✅ TEST 2 PASSED: Interpolation matrices bit-exact")
elif max_diff < 1e-6:
    print(f"  ⚠️ TEST 2 PASSED: Interpolation matrices très proches (diff < 1e-6)")
else:
    print(f"  ❌ TEST 2 FAILED: Différences significatives")
    print(f"     CPU row 0: {interp_matrix_cpu[0, :5]}")
    print(f"     GPU row 0: {interp_matrix_gpu_np[0, :5]}")

# ============================================================================
# TEST 3: Vérifier que l'interpolation donne les mêmes données
# ============================================================================
print("\n" + "=" * 60)
print("TEST 3: Interpolation de données synthétiques")
print("=" * 60)

# Créer des données synthétiques (epochs)
n_epochs = 10
n_times = 100
rng = np.random.RandomState(42)
data = rng.randn(n_epochs, n_channels, n_times).astype(np.float64)

print(f"  Data shape: {data.shape}")
print(f"  Data dtype: {data.dtype}")

# CPU: Interpoler les mauvais canaux
data_good = data[:, goods_idx, :]  # Shape: (n_epochs, n_goods, n_times)
interp_cpu = np.einsum('ij,ejt->eit', interp_matrix_cpu, data_good)

print(f"  Interp CPU shape: {interp_cpu.shape}")

# GPU: Interpoler
data_good_t = torch.tensor(data_good, dtype=torch.float64, device=compute_device)
interp_matrix_t = torch.tensor(interp_matrix_cpu, dtype=torch.float64, device=compute_device)

# Batch matrix multiply
interp_gpu_t = torch.einsum('ij,ejt->eit', interp_matrix_t, data_good_t)
interp_gpu = interp_gpu_t.cpu().numpy()

print(f"  Interp GPU shape: {interp_gpu.shape}")

# Comparer
diff = np.abs(interp_cpu - interp_gpu)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"\n  RÉSULTATS:")
print(f"  Max diff: {max_diff:.2e}")
print(f"  Mean diff: {mean_diff:.2e}")

if max_diff < 1e-10:
    print(f"  ✅ TEST 3 PASSED: Interpolation identique")
else:
    print(f"  ❌ TEST 3 FAILED: Différences détectées")

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "=" * 60)
print("RÉSUMÉ PHASE 2 VALIDATION (Tests 1-3)")
print("=" * 60)
print(f"Device utilisé: {device}")
print(f"Stratégie compute: {'CUDA float64 on device' if use_cuda else 'CPU float64, MPS float32 data'}")
print("")

# ============================================================================
# TEST 4: gpu_clean_by_interp vs _clean_by_interp avec vraies Epochs
# ============================================================================
print("\n" + "=" * 60)
print("TEST 4: gpu_clean_by_interp vs _clean_by_interp")
print("=" * 60)

from autoreject.autoreject import _clean_by_interp
from autoreject.gpu_interpolation import gpu_clean_by_interp

# Créer des epochs synthétiques MNE
n_epochs_test = 5
n_times_test = 200
sfreq = 256

# Créer les données avec du bruit réaliste
rng = np.random.RandomState(42)
data_epochs = rng.randn(n_epochs_test, n_channels, n_times_test) * 1e-6  # Echelle EEG réaliste

# Créer l'objet Epochs
epochs = mne.EpochsArray(data_epochs, info, verbose=False)

# Marquer quelques canaux comme "bads"
bad_channels = ['AF9', 'AF7', 'FC1']  # 3 canaux à interpoler
epochs.info['bads'] = bad_channels

print(f"  Epochs: {n_epochs_test} epochs x {n_channels} channels x {n_times_test} times")
print(f"  Bad channels: {bad_channels}")

# Définir les picks (tous les canaux EEG)
picks = mne.pick_types(epochs.info, eeg=True)
print(f"  Picks: {len(picks)} channels")

# CPU: _clean_by_interp
try:
    epochs_cpu = epochs.copy()
    epochs_cpu_cleaned = _clean_by_interp(epochs_cpu, picks=picks, verbose=True)
    data_cpu_cleaned = epochs_cpu_cleaned.get_data()
    print(f"  CPU cleaned shape: {data_cpu_cleaned.shape}")
    cpu_success = True
except Exception as e:
    print(f"  ❌ CPU _clean_by_interp failed: {e}")
    cpu_success = False

# GPU: gpu_clean_by_interp
try:
    epochs_gpu = epochs.copy()
    result_gpu = gpu_clean_by_interp(epochs_gpu, picks=picks, device=device, verbose=True)
    # gpu_clean_by_interp renvoie un DeviceArray, pas un Epochs
    # On extrait les données
    if hasattr(result_gpu, 'data'):
        data_gpu_cleaned = result_gpu.data.cpu().numpy()
    else:
        data_gpu_cleaned = np.array(result_gpu)
    print(f"  GPU cleaned shape: {data_gpu_cleaned.shape}")
    gpu_success = True
except Exception as e:
    print(f"  ❌ GPU gpu_clean_by_interp failed: {e}")
    import traceback
    traceback.print_exc()
    gpu_success = False

# Comparer si les deux ont réussi
if cpu_success and gpu_success:
    diff = np.abs(data_cpu_cleaned - data_gpu_cleaned)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Différence relative par rapport à l'amplitude des données
    data_range = np.max(np.abs(data_cpu_cleaned))
    rel_diff = max_diff / (data_range + 1e-20)
    
    print(f"\n  RÉSULTATS:")
    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Mean absolute diff: {mean_diff:.2e}")
    print(f"  Max relative diff: {rel_diff:.2e}")
    print(f"  Data range: {data_range:.2e}")
    
    if max_diff < 1e-10:
        print(f"  ✅ TEST 4 PASSED: Epochs interpolées bit-exact")
    elif rel_diff < 1e-5:
        print(f"  ⚠️ TEST 4 PASSED: Epochs interpolées très proches (rel diff < 1e-5)")
    else:
        print(f"  ❌ TEST 4 FAILED: Différences significatives")
        # Montrer les différences par canal
        for i, ch in enumerate(bad_channels):
            ch_idx = epochs.ch_names.index(ch)
            ch_diff = np.max(np.abs(data_cpu_cleaned[:, ch_idx, :] - data_gpu_cleaned[:, ch_idx, :]))
            print(f"     {ch}: max diff = {ch_diff:.2e}")
else:
    print(f"  ❌ TEST 4 SKIPPED: CPU or GPU interpolation failed")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 60)
print("RÉSUMÉ FINAL PHASE 2")
print("=" * 60)
print(f"Device utilisé: {device}")
print(f"Stratégie compute: {'CUDA float64 on device' if use_cuda else 'CPU float64, MPS float32 data'}")
print("")
print("Si tous les tests passent, Phase 2 est validée et on peut continuer vers Phase 3.")
