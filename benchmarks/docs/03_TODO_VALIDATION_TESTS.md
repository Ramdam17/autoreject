# Document 3 : TODO - Tests de validation CPU vs GPU

## Objectif
Valider fonction par fonction les sorties et différences entre CPU (NumPy) et GPU (Torch).

## Jeu de données
- **Config** : highdensity_128ch
- **Paramètres** : 128 canaux, 300 epochs, 500Hz, 30% artifacts
- **CV** : 10 folds
- **n_interpolate** : [1, 2, 4, 8, 12, 16]
- **consensus** : [0.1, 0.2, 0.3, 0.4, 0.5]
- **random_state** : 42

---

## Tests à exécuter (dans l'ordre)

### Test 01 : _handle_picks
- **Fonctions** : utils._handle_picks
- **Attendu** : Sortie identique (pure Python, pas de backend)
- **Vérification** : picks_cpu == picks_gpu

### Test 02 : _check_data  
- **Fonctions** : utils._check_data
- **Attendu** : Aucune erreur levée pour les deux
- **Vérification** : Pas d'exception

### Test 03 : _should_use_gpu
- **Fonctions** : AutoReject._should_use_gpu
- **Attendu** : 
  - CPU: (False, 'cpu')
  - GPU: (True, 'mps')
- **Vérification** : Valeurs retournées correctes

### Test 04 : _get_picks_by_type
- **Fonctions** : utils._get_picks_by_type
- **Attendu** : Sortie identique
- **Vérification** : picks_by_type_cpu == picks_by_type_gpu

### Test 05 : get_backend
- **Fonctions** : backends.get_backend
- **Attendu** :
  - CPU: NumpyBackend
  - GPU: TorchBackend
- **Vérification** : Type du backend

### Test 06 : backend.ptp
- **Fonctions** : 
  - NumpyBackend.ptp
  - TorchBackend.ptp
- **Entrée** : epochs.get_data(picks) - shape (300, 128, n_times)
- **Attendu** : 
  - Valeurs proches (diff < 1e-6)
  - **ATTENTION** : dtype différent (float64 vs float32)
- **Vérification** : 
  - max(abs(ptp_cpu - ptp_gpu)) < 1e-6
  - ptp_cpu.dtype == float64
  - ptp_gpu.dtype == float32

### Test 07 : backend.median
- **Fonctions** :
  - NumpyBackend.median (np.median)
  - TorchBackend.median (_torch_median)
- **Entrée** : X_test shape (30, 128, n_times)
- **Attendu** : Valeurs identiques (même algorithme moyenne 2 valeurs centrales)
- **Vérification** : max(abs(median_cpu - median_gpu)) < 1e-9

### Test 08 : _clean_by_interp (Phase augmentation)
- **Fonctions** :
  - CPU: utils._clean_by_interp → _interpolate_bads_eeg
  - GPU: _clean_by_interp_gpu
- **Entrée** : epochs (300, 128, n_times)
- **Attendu** : Données augmentées similaires
- **Vérification** : max(abs(data_aug_cpu - data_aug_gpu)) < 1e-6

### Test 09 : Calcul PTP pour seuils
- **Fonctions** :
  - CPU: backend.ptp dans _compute_thresh
  - GPU: tensor.max()-min() dans compute_all_thresholds_gpu
- **Entrée** : data_augmented shape (600, 128, n_times)
- **Attendu** : PTP values similaires
- **Vérification** : 
  - max(abs(ptp_cpu - ptp_gpu)) < 1e-6
  - **ATTENTION** : float64 vs float32

### Test 10 : Seuils (thresholds) par canal
- **Fonctions** :
  - CPU: _compute_thresholds → _compute_thresh → bayes_opt
  - GPU: compute_thresholds_gpu → compute_all_thresholds_gpu → bayes_opt
- **Entrée** : epochs + random_state=42
- **Attendu** : Seuils similaires à ~1e-7 près
- **Vérification** :
  - Pour chaque canal: abs(thresh_cpu[ch] - thresh_gpu[ch]) < 1e-6
  - **ATTENTION** : dtype (float64 vs float32)
  - Compter nombre de canaux avec diff > 1e-8

### Test 11 : _vote_bad_epochs
- **Fonctions** : _AutoReject._vote_bad_epochs
- **Entrée** : epochs + thresholds (utiliser MÊMES thresholds pour isoler)
- **Attendu** : labels et bad_sensor_counts identiques
- **Vérification** :
  - labels_cpu == labels_gpu (exactement)
  - bad_sensor_counts_cpu == bad_sensor_counts_gpu (exactement)

### Test 12 : _vote_bad_epochs avec thresholds différents
- **Fonctions** : _AutoReject._vote_bad_epochs
- **Entrée** : epochs + thresholds CPU vs thresholds GPU
- **Attendu** : Différences dues à float32 vs float64
- **Vérification** :
  - Compter epochs avec bad_sensor_counts différents
  - Analyser impact sur loss grid

### Test 13 : _get_epochs_interpolation
- **Fonctions** : _AutoReject._get_epochs_interpolation
- **Entrée** : epochs + labels + n_interpolate=12
- **Attendu** : Labels identiques (pure Python logic)
- **Vérification** : labels_cpu == labels_gpu

### Test 14 : Interpolation (single epoch)
- **Fonctions** :
  - CPU: _interpolate_bads_eeg
  - GPU: gpu_interpolation (single epoch)
- **Entrée** : 1 epoch avec 5 canaux à interpoler
- **Attendu** : Données interpolées similaires
- **Vérification** : max(abs(data_interp_cpu - data_interp_gpu)) < 1e-9

### Test 15 : Interpolation batch (tous epochs)
- **Fonctions** :
  - CPU: _interpolate_bad_epochs (loop)
  - GPU: gpu_batch_interpolate_all_n_interp
- **Entrée** : epochs + labels pour n_interpolate=12
- **Attendu** : Données interpolées similaires
- **Vérification** : max(abs(X_interp_cpu - X_interp_gpu)) < 1e-9

### Test 16 : _get_bad_epochs
- **Fonctions** : _AutoReject._get_bad_epochs
- **Entrée** : bad_sensor_counts + consensus=0.3
- **Attendu** : Masque identique (pure Python)
- **Vérification** : bad_epochs_cpu == bad_epochs_gpu

### Test 17 : _slicemean
- **Fonctions** : autoreject.autoreject._slicemean
- **Entrée** : X_train_interp + good_epochs_idx
- **Attendu** : Mean identique
- **Vérification** : max(abs(mean_cpu - mean_gpu)) < 1e-12

### Test 18 : Score computation (single fold)
- **Fonctions** :
  - CPU: _AutoReject.score
  - GPU: Phase 3 score computation
- **Entrée** : X_test + mean_ (utiliser MÊMES mean_ et median)
- **Attendu** : Scores identiques
- **Vérification** : abs(score_cpu - score_gpu) < 1e-10

### Test 19 : CV fold (single fold, single consensus, single n_interp)
- **Fonctions** :
  - CPU: Boucle dans _run_local_reject_cv
  - GPU: Boucle dans run_local_reject_cv_gpu_batch
- **Entrée** : n_interp=12, consensus=0.3, fold=0
- **Attendu** : Loss identique si mêmes thresholds
- **Vérification** : abs(loss_cpu - loss_gpu) < 1e-8

### Test 20 : Full loss grid
- **Fonctions** :
  - CPU: _run_local_reject_cv
  - GPU: run_local_reject_cv_gpu_batch
- **Entrée** : Paramètres complets
- **Attendu** : Différences dues à float32 thresholds
- **Vérification** :
  - Pour chaque (consensus, n_interp, fold): calculer diff%
  - Identifier pattern de divergence
  - Analyser cause racine

### Test 21 : Optimal parameters
- **Fonctions** :
  - CPU: argmin sur loss grid
  - GPU: argmin sur loss grid
- **Entrée** : Loss grids CPU et GPU
- **Attendu** : 
  - CPU optimal: consensus=0.30, n_interpolate=16
  - GPU optimal: consensus=0.10, n_interpolate=12
- **Vérification** : Confirmer les valeurs optimales

---

## Format de sortie pour chaque test

```python
# Test XX : NOM_TEST
# ====================
# Entrées:
#   - param1: valeur
#   - param2: valeur
#
# CPU output:
#   - type: dtype
#   - shape: shape
#   - sample: premiers valeurs
#
# GPU output:
#   - type: dtype
#   - shape: shape
#   - sample: premiers valeurs
#
# Comparaison:
#   - max_diff: valeur
#   - mean_diff: valeur
#   - identical: True/False
#
# Verdict: ✅ PASS / ❌ FAIL
# Notes: observations importantes
```

---

## Script de test

Chaque test sera implémenté dans un fichier séparé :
- `benchmarks/validation/test_01_handle_picks.py`
- `benchmarks/validation/test_02_check_data.py`
- ...
- `benchmarks/validation/test_21_optimal_params.py`

Un script principal `run_all_validation_tests.py` exécutera tous les tests dans l'ordre.

---

## Statut des tests

| # | Test | Statut | Notes |
|---|------|--------|-------|
| 01 | _handle_picks | ⬜ TODO | |
| 02 | _check_data | ⬜ TODO | |
| 03 | _should_use_gpu | ⬜ TODO | |
| 04 | _get_picks_by_type | ⬜ TODO | |
| 05 | get_backend | ⬜ TODO | |
| 06 | backend.ptp | ⬜ TODO | |
| 07 | backend.median | ⬜ TODO | |
| 08 | _clean_by_interp | ⬜ TODO | |
| 09 | Calcul PTP seuils | ⬜ TODO | |
| 10 | Seuils (thresholds) | ⬜ TODO | |
| 11 | _vote_bad_epochs (mêmes thresh) | ⬜ TODO | |
| 12 | _vote_bad_epochs (thresh diff) | ⬜ TODO | |
| 13 | _get_epochs_interpolation | ⬜ TODO | |
| 14 | Interpolation single epoch | ⬜ TODO | |
| 15 | Interpolation batch | ⬜ TODO | |
| 16 | _get_bad_epochs | ⬜ TODO | |
| 17 | _slicemean | ⬜ TODO | |
| 18 | Score computation | ⬜ TODO | |
| 19 | CV single fold | ⬜ TODO | |
| 20 | Full loss grid | ⬜ TODO | |
| 21 | Optimal parameters | ⬜ TODO | |
