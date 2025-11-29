#!/usr/bin/env python
"""
Benchmark réaliste basé sur les paramètres utilisateur réels.

Données utilisateur:
- AutoReject ICA: 2970 époques → 4h 35min (275 min)
- AutoReject Final: 5941 époques → 10h 30min (630 min)
- Canaux: ~128
- Paramètres: n_interpolate=[1,4,32], consensus=11 valeurs, cv=10
"""

import numpy as np
import mne
import time
import os
from importlib import reload


def create_test_epochs(n_epochs, n_channels, n_times, sfreq):
    """Créer des époques de test avec montage."""
    ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    phi = np.linspace(np.pi/4, 3*np.pi/4, n_channels)
    pos = np.column_stack([
        0.09 * np.sin(phi) * np.cos(theta),
        0.09 * np.sin(phi) * np.sin(theta),
        0.09 * np.cos(phi)
    ])
    montage = mne.channels.make_dig_montage(
        ch_pos={ch: p for ch, p in zip(ch_names, pos)}, 
        coord_frame='head'
    )
    info.set_montage(montage)
    
    np.random.seed(42)
    data = np.random.randn(n_epochs, n_channels, n_times) * 20e-6
    
    return mne.EpochsArray(data, info, verbose=False), ch_names


def benchmark_ptp_operation(data, n_repeats=10):
    """Benchmark peak-to-peak operation avec différents backends."""
    import autoreject.backends as backends_module
    
    results = {}
    
    for backend_name in ['numpy', 'torch']:
        os.environ['AUTOREJECT_BACKEND'] = backend_name
        reload(backends_module)
        backend = backends_module.get_backend()
        
        times = []
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(n_repeats):
                # Simuler ce que fait _vote_bad_epochs
                ptp = backend.ptp(data, axis=2)
                _ = backend.to_numpy(ptp)
            times.append(time.perf_counter() - start)
        
        results[backend_name] = np.median(times) * 1000  # ms
    
    return results


def benchmark_ransac_operations(n_epochs, n_times, n_channels=32):
    """Benchmark opérations RANSAC: correlation + median."""
    import autoreject.backends as backends_module
    
    # Simuler les données RANSAC
    np.random.seed(42)
    n_subset = 25
    pred_data = np.random.randn(n_epochs, n_subset, n_times).astype(np.float32)
    actual_data = np.random.randn(n_epochs, n_times).astype(np.float32)
    
    results = {}
    
    for backend_name in ['numpy', 'torch']:
        os.environ['AUTOREJECT_BACKEND'] = backend_name
        reload(backends_module)
        backend = backends_module.get_backend()
        
        times = []
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(n_channels):
                # Simuler ce que fait RANSAC pour chaque canal
                corrs = backend.correlation(pred_data, actual_data[:, np.newaxis, :])
                median_corr = backend.median(corrs, axis=0)
                _ = backend.to_numpy(median_corr)
            times.append(time.perf_counter() - start)
        
        results[backend_name] = np.median(times) * 1000  # ms
    
    return results


def benchmark_interpolation(epochs, n_bad_epochs=10, n_bad_channels=5):
    """Benchmark interpolation parallèle."""
    from autoreject.autoreject import _interpolate_bad_epochs
    
    n_epochs_total = len(epochs)
    n_channels = len(epochs.ch_names)
    ch_names = epochs.ch_names
    picks = np.arange(n_channels)
    
    # Créer interp_channels pour TOUTES les époques (liste vide si pas d'interpolation)
    bad_epochs_idx = np.linspace(0, n_epochs_total-1, n_bad_epochs, dtype=int)
    interp_channels = []
    for i in range(n_epochs_total):
        if i in bad_epochs_idx:
            bad_ch_idx = np.random.choice(n_channels, n_bad_channels, replace=False)
            interp_channels.append([ch_names[j] for j in bad_ch_idx])
        else:
            interp_channels.append([])
    
    results = {}
    
    for n_jobs in [1, -1]:
        times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = _interpolate_bad_epochs(
                epochs.copy(), interp_channels, picks, n_jobs=n_jobs
            )
            times.append(time.perf_counter() - start)
        
        key = f'n_jobs={n_jobs}'
        results[key] = np.median(times) * 1000  # ms
    
    return results


def main():
    # Paramètres basés sur vos données réelles
    n_epochs = 500  # Échantillon représentatif
    n_channels = 128
    n_times = 1000  # 2 sec @ 500 Hz
    sfreq = 500
    
    print('=' * 70)
    print('BENCHMARK RÉALISTE - Basé sur vos paramètres')
    print('=' * 70)
    print(f'Epochs: {n_epochs} (représentatif de vos 2970-5941)')
    print(f'Canaux: {n_channels}')
    print(f'Durée époque: {n_times/sfreq}s')
    print()
    
    # Créer les données
    epochs, ch_names = create_test_epochs(n_epochs, n_channels, n_times, sfreq)
    data = epochs.get_data()
    info = epochs.info
    
    # ========================================
    # Test 1: Peak-to-peak (ptp)
    # ========================================
    print('Test 1: Peak-to-peak (ptp sur données)')
    print('-' * 50)
    
    vote_results = benchmark_ptp_operation(data, n_repeats=10)
    print(f"  NumPy (10 appels): {vote_results['numpy']:.0f} ms")
    print(f"  Torch MPS (10 appels): {vote_results['torch']:.0f} ms")
    ptp_speedup = vote_results['numpy'] / vote_results['torch']
    print(f"  Speedup: {ptp_speedup:.1f}x")
    print()
    
    # ========================================
    # Test 2: RANSAC correlation
    # ========================================
    print('Test 2: RANSAC _compute_correlations')
    print('-' * 50)
    
    ransac_results = benchmark_ransac_operations(n_epochs, n_times, n_channels=32)
    print(f"  NumPy (32 canaux): {ransac_results['numpy']:.0f} ms")
    print(f"  Torch MPS (32 canaux): {ransac_results['torch']:.0f} ms")
    ransac_speedup = ransac_results['numpy'] / ransac_results['torch']
    print(f"  Speedup: {ransac_speedup:.1f}x")
    print()
    
    # ========================================
    # Test 3: Interpolation parallèle
    # ========================================
    print('Test 3: Interpolation (le goulot principal)')
    print('-' * 50)
    
    # Utiliser un sous-ensemble pour l'interpolation (plus rapide)
    epochs_small = mne.EpochsArray(data[:100], info, verbose=False)
    interp_results = benchmark_interpolation(epochs_small, n_bad_epochs=10, n_bad_channels=5)
    print(f"  n_jobs=1: {interp_results['n_jobs=1']:.0f} ms")
    print(f"  n_jobs=-1: {interp_results['n_jobs=-1']:.0f} ms")
    interp_speedup = interp_results['n_jobs=1'] / interp_results['n_jobs=-1']
    print(f"  Speedup: {interp_speedup:.1f}x")
    print()
    
    # ========================================
    # Estimation du gain total
    # ========================================
    print('=' * 70)
    print('ESTIMATION DU GAIN POUR VOS DONNÉES')
    print('=' * 70)
    
    # Votre temps réel
    your_time_ica = 275.3  # minutes
    your_time_final = 630.5  # minutes
    
    # Composition estimée du temps (basée sur profiling typique):
    # - 60% interpolation
    # - 25% _vote_bad_epochs (ptp)
    # - 15% RANSAC + autres
    
    print()
    print('Composition estimée du temps:')
    print('  - 60% Interpolation')
    print('  - 25% Vote bad epochs (ptp)')  
    print('  - 15% RANSAC + autres')
    print()
    
    # Gain pondéré
    weighted_speedup = 1 / (
        0.60 / interp_speedup + 
        0.25 / ptp_speedup + 
        0.15 / ransac_speedup
    )
    
    print('Gains par composant:')
    print(f'  - Interpolation (n_jobs=-1): {interp_speedup:.1f}x')
    print(f'  - Vote bad epochs (GPU): {ptp_speedup:.1f}x')
    print(f'  - RANSAC (GPU): {ransac_speedup:.1f}x')
    print()
    print(f'Gain total pondéré estimé: {weighted_speedup:.1f}x')
    print()
    
    # Temps estimés après optimisation
    new_time_ica = your_time_ica / weighted_speedup
    new_time_final = your_time_final / weighted_speedup
    
    print('TEMPS ESTIMÉS APRÈS OPTIMISATION:')
    print(f'  AutoReject ICA:   {your_time_ica:.0f} min → {new_time_ica:.0f} min ({new_time_ica/60:.1f}h)')
    print(f'  AutoReject Final: {your_time_final:.0f} min → {new_time_final:.0f} min ({new_time_final/60:.1f}h)')
    print()
    savings = (your_time_ica + your_time_final - new_time_ica - new_time_final)
    print(f'  Économie totale: {savings:.0f} min ({savings/60:.1f}h)')
    print()
    
    # ========================================
    # Comment activer les optimisations
    # ========================================
    print('=' * 70)
    print('COMMENT ACTIVER LES OPTIMISATIONS')
    print('=' * 70)
    print()
    print('1. Backend GPU (PyTorch MPS sur Mac):')
    print('   export AUTOREJECT_BACKEND=torch')
    print()
    print('2. Parallélisation CPU:')
    print('   ar = AutoReject(n_jobs=-1, ...)')
    print()
    print('3. Les deux ensemble pour un gain maximal!')
    print()


if __name__ == '__main__':
    main()
