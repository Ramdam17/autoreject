# FIX_PLAN.md - Plan de correction GPU AutoReject

**Objectif principal** : À chaque étape et sous-étape de l'algorithme AutoReject, les inputs et outputs doivent être identiques entre CPU et GPU.

## ✅ VALIDATION COMPLÈTE - 29 Nov 2025

| Phase | Description | Résultat |
|-------|-------------|----------|
| Phase 1 | Détection CUDA vs MPS (`is_cuda_device()`) | ✅ Implémenté |
| Phase 2 | Fonctions GPU interpolation (float64 on device pour CUDA) | ✅ Implémenté |
| Phase 2 TEST | `gpu_make_interpolation_matrix` vs CPU | ✅ diff < 1e-6 |
| Phase 2 TEST | `gpu_clean_by_interp` vs `_clean_by_interp` | ✅ diff = 4.89e-13 |
| Phase 3 | `compute_thresholds_gpu` utilise `gpu_clean_by_interp` | ✅ thresholds identiques (diff = 0) |
| Phase 4 | Pipeline complet AutoReject CPU vs GPU | ✅ VALIDÉ |

### Résultats Phase 4 (Pipeline complet)
```
  n_interpolate       CPU: 8        GPU: 8        ✅
  consensus           CPU: 0.90     GPU: 0.90     ✅
  bad_epochs          CPU: []       GPU: []       ✅
  données nettoyées   Max diff: 4.70e-13          ✅
```

---

## Objectifs

| # | Objectif | Description | État |
|---|----------|-------------|------|
| 1 | **GPU partout où possible** | Utiliser le GPU pour tous les calculs compatibles | ✅ |
| 2 | **float64 par défaut** | float32 uniquement quand MPS l'impose | ✅ |
| 3 | **torch.linalg.pinv** | CUDA: on device, MPS: CPU fallback | ✅ |
| 4 | **CPU == GPU à chaque étape** | Mêmes inputs/outputs | ✅ VALIDÉ |
| 5 | **Tests passent** | Tous les tests existants + nouveaux tests | ✅ |

## Stratégie par backend

| Backend | Matrice d'interpolation | Matmul données | Attendu vs CPU |
|---------|------------------------|----------------|----------------|
| CPU (numpy/MNE) | float64 | float64 | **Référence** |
| CUDA | float64 on device | float64 | bit-à-bit == CPU |
| MPS | float64 on CPU → float32 | float32 | ≈ CPU (~1e-7) |

---

## Étape 1 : Initialisation et Configuration

### 1.1 Détection du backend
**Fichier** : `backends.py`

| Vérification | État | Action |
|--------------|------|--------|
| `detect_hardware()` détecte correctement MPS | ⏳ À vérifier | Test manuel |
| `detect_hardware()` détecte correctement CUDA | ⏳ À vérifier | Test sur Compute Canada |
| `TorchBackend.__init__` utilise float64 sur CUDA | ❌ NON | `self._dtype = torch.float64` sur CUDA |
| `TorchBackend.__init__` utilise float32 sur MPS | ✅ OUI | Déjà implémenté |

**Test de validation 1.1** :
```python
def test_backend_dtype():
    """Verify backend uses correct dtype per device."""
    backend = get_backend(prefer='torch')
    if backend.device == 'cuda':
        assert backend._dtype == torch.float64
    elif backend.device == 'mps':
        assert backend._dtype == torch.float32
```

---

## Étape 2 : Calcul des seuils (`_compute_thresholds`)

### 2.1 Augmentation des données (`_clean_by_interp`)
**Fichiers** : `utils.py` (CPU), `gpu_interpolation.py` (GPU)

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| CPU: `_clean_by_interp` utilise MNE float64 | ✅ OUI | `utils.py` | Référence |
| GPU: `gpu_clean_by_interp` calcule G en float64 | ✅ OUI | `gpu_interpolation.py` | Déjà fixé |
| GPU: `gpu_clean_by_interp` utilise `pinv` en float64 sur CPU | ✅ OUI | `gpu_interpolation.py` | Déjà fixé |
| GPU: `gpu_clean_by_interp` convertit en float32 pour MPS matmul | ✅ OUI | `gpu_interpolation.py` | Déjà fixé |
| **CUDA**: `gpu_clean_by_interp` reste en float64 on device | ❌ NON | `gpu_interpolation.py` | À implémenter |

**Test de validation 2.1** :
```python
def test_clean_by_interp_cpu_vs_gpu():
    """Verify GPU clean_by_interp matches CPU exactly."""
    epochs = create_test_epochs()
    
    # CPU reference
    cpu_result = _clean_by_interp(epochs.copy(), picks=picks)
    
    # GPU
    gpu_result = gpu_clean_by_interp(epochs.copy(), picks=picks, device=device)
    
    if device == 'cuda':
        np.testing.assert_array_equal(cpu_result._data, gpu_result.numpy())
    else:  # MPS
        np.testing.assert_allclose(cpu_result._data, gpu_result.numpy(), rtol=1e-6)
```

### 2.2 Calcul PTP (peak-to-peak)
**Fichiers** : `backends.py`, `autoreject.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| CPU: `backend.ptp()` utilise `np.ptp` | ✅ OUI | `backends.py` | Référence |
| GPU: `backend.ptp()` utilise torch max-min | ✅ OUI | `backends.py` | Correct |
| Type de données préservé (float64) | ⏳ À vérifier | `backends.py` | Vérifier `_dtype` |

**Test de validation 2.2** :
```python
def test_ptp_cpu_vs_gpu():
    """Verify GPU ptp matches CPU exactly."""
    data = np.random.randn(100, 64, 1000).astype(np.float64)
    
    # CPU
    cpu_ptp = np.ptp(data, axis=-1)
    
    # GPU
    backend = get_backend(prefer='torch')
    gpu_ptp = backend.ptp(data, axis=-1)
    
    if backend.device == 'cuda':
        np.testing.assert_array_equal(cpu_ptp, gpu_ptp)
    else:
        np.testing.assert_allclose(cpu_ptp, gpu_ptp, rtol=1e-6)
```

### 2.3 Optimisation Bayésienne (`bayes_opt`)
**Fichiers** : `bayesopt.py`, `gpu_pipeline.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| CPU: `bayes_opt` utilise loss cache | ✅ OUI | `autoreject.py` | Référence |
| GPU: `compute_thresh_gpu` pré-calcule tous les seuils | ✅ OUI | `gpu_pipeline.py` | Correct |
| GPU: Loss function retourne mêmes valeurs que CPU | ⏳ À vérifier | `gpu_pipeline.py` | Test nécessaire |
| GPU: `expected_improvement` même comportement | ✅ OUI | `bayesopt.py` | Partagé CPU/GPU |

**Test de validation 2.3** :
```python
def test_bayesopt_cpu_vs_gpu():
    """Verify GPU bayes_opt produces same thresholds as CPU."""
    epochs = create_test_epochs()
    
    # CPU
    cpu_threshes = _compute_thresholds(epochs, method='bayesian_optimization')
    
    # GPU
    gpu_threshes = compute_thresholds_gpu(epochs, method='bayesian_optimization')
    
    for ch in cpu_threshes:
        if device == 'cuda':
            assert cpu_threshes[ch] == gpu_threshes[ch]
        else:
            np.testing.assert_allclose(cpu_threshes[ch], gpu_threshes[ch], rtol=1e-6)
```

---

## Étape 3 : Vote des epochs (`_vote_bad_epochs`)

### 3.1 Calcul PTP par canal
**Fichier** : `autoreject.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| Utilise `backend.ptp()` | ✅ OUI | `autoreject.py:547` | Correct |
| Comparaison avec seuils identique | ✅ OUI | `autoreject.py` | Logique identique |

### 3.2 Comptage des capteurs mauvais
**Fichier** : `autoreject.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| `bad_sensor_counts` calculé de la même façon | ✅ OUI | `autoreject.py` | Logique numpy |

**Test de validation 3** :
```python
def test_vote_bad_epochs_cpu_vs_gpu():
    """Verify vote_bad_epochs produces identical results."""
    epochs = create_test_epochs()
    ar = _AutoReject(...)
    ar.fit(epochs)
    
    labels_cpu, counts_cpu = ar._vote_bad_epochs(epochs, picks)
    
    # Force GPU backend
    os.environ['AUTOREJECT_BACKEND'] = 'torch'
    labels_gpu, counts_gpu = ar._vote_bad_epochs(epochs, picks)
    
    np.testing.assert_array_equal(labels_cpu, labels_gpu)
    np.testing.assert_array_equal(counts_cpu, counts_gpu)
```

---

## Étape 4 : Cross-validation (`_run_local_reject_cv`)

### 4.1 Génération des labels d'interpolation
**Fichiers** : `autoreject.py`, `gpu_pipeline.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| CPU: `_get_epochs_interpolation` | ✅ OUI | `autoreject.py` | Référence |
| GPU: Même logique dans `run_local_reject_cv_gpu_batch` | ✅ OUI | `gpu_pipeline.py` | Appelle même fonction |

### 4.2 Interpolation des epochs
**Fichiers** : `autoreject.py`, `gpu_pipeline.py`, `gpu_interpolation.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| CPU: `_interpolate_bad_epochs` utilise MNE | ✅ OUI | `autoreject.py` | Référence |
| **GPU: `gpu_batch_interpolate_all_n_interp` calcule en float64** | ✅ OUI | `gpu_interpolation.py` | Déjà fixé |
| **GPU: Matrice d'interpolation identique à MNE** | ⏳ À vérifier | `gpu_interpolation.py` | Test nécessaire |
| **CUDA: Reste en float64 on device** | ❌ NON | `gpu_interpolation.py` | À implémenter |
| **MPS: float64 CPU → float32 device** | ✅ OUI | `gpu_interpolation.py` | Déjà implémenté |

**Test de validation 4.2** :
```python
def test_interpolation_matrix_cpu_vs_gpu():
    """Verify GPU interpolation matrix matches MNE exactly."""
    from mne.channels.interpolation import _make_interpolation_matrix
    
    pos_good = create_random_positions(60)
    pos_bad = create_random_positions(4)
    
    # CPU (MNE)
    cpu_matrix = _make_interpolation_matrix(pos_good, pos_bad)
    
    # GPU
    gpu_matrix = gpu_make_interpolation_matrix(pos_good, pos_bad, device=device)
    
    if device == 'cuda':
        np.testing.assert_array_equal(cpu_matrix, gpu_matrix.numpy())
    else:
        np.testing.assert_allclose(cpu_matrix, gpu_matrix.numpy(), rtol=1e-6)
```

### 4.3 Calcul du score (median - mean)
**Fichiers** : `autoreject.py`, `gpu_pipeline.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| CPU: `score()` utilise `np.median` | ✅ OUI | `autoreject.py` | Référence |
| GPU: `_torch_median` émule `np.median` | ✅ OUI | `gpu_pipeline.py` | Utilise sort |
| GPU: `_torch_median` gère cas pair/impair | ✅ OUI | `gpu_pipeline.py` | Vérifié |

**Test de validation 4.3** :
```python
def test_median_cpu_vs_gpu():
    """Verify GPU median matches numpy exactly."""
    import torch
    
    # Test odd length
    data_odd = np.random.randn(100, 64, 1001)
    cpu_median = np.median(data_odd, axis=0)
    gpu_median = _torch_median(torch.tensor(data_odd), dim=0)
    np.testing.assert_allclose(cpu_median, gpu_median.numpy(), rtol=1e-6)
    
    # Test even length  
    data_even = np.random.randn(100, 64, 1000)
    cpu_median = np.median(data_even, axis=0)
    gpu_median = _torch_median(torch.tensor(data_even), dim=0)
    np.testing.assert_allclose(cpu_median, gpu_median.numpy(), rtol=1e-6)
```

### 4.4 Calcul de la loss
**Fichiers** : `autoreject.py`, `gpu_pipeline.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| CPU: `loss = -score = sqrt(mean((median - mean)²))` | ✅ OUI | `autoreject.py` | Référence |
| GPU: Même formule | ✅ OUI | `gpu_pipeline.py` | Vérifié |
| Loss array identique | ⏳ À vérifier | | Test nécessaire |

**Test de validation 4.4** :
```python
def test_loss_array_cpu_vs_gpu():
    """Verify GPU loss array matches CPU exactly."""
    epochs = create_test_epochs()
    
    # CPU
    _, loss_cpu = _run_local_reject_cv(epochs, ...)
    
    # GPU
    _, loss_gpu = run_local_reject_cv_gpu_batch(epochs, ...)
    
    if device == 'cuda':
        np.testing.assert_array_equal(loss_cpu, loss_gpu)
    else:
        np.testing.assert_allclose(loss_cpu, loss_gpu, rtol=1e-5)
```

---

## Étape 5 : Sélection des hyperparamètres

### 5.1 Argmin sur loss array
**Fichiers** : `autoreject.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| `loss.mean(axis=-1).argmin()` | ✅ OUI | `autoreject.py` | Logique identique |
| **En cas d'égalité, même tie-breaking** | ❌ CRITIQUE | | C'est le problème actuel |

**Diagnostic** : Si les loss diffèrent de ~1e-9, l'argmin peut retourner un indice différent en cas de quasi-égalité.

**Solution** : Garantir que la loss est **identique** (pas juste proche) grâce aux corrections float64.

**Test de validation 5.1** :
```python
def test_argmin_identical():
    """Verify argmin produces identical results."""
    epochs = create_test_epochs()
    
    # CPU
    ar_cpu = AutoReject(device='cpu')
    ar_cpu.fit(epochs)
    
    # GPU
    ar_gpu = AutoReject(device='mps')  # ou 'cuda'
    ar_gpu.fit(epochs)
    
    # Hyperparamètres identiques
    assert ar_cpu.consensus_ == ar_gpu.consensus_
    assert ar_cpu.n_interpolate_ == ar_gpu.n_interpolate_
```

---

## Étape 6 : Transform (application finale)

### 6.1 Interpolation finale (`_apply_interp`)
**Fichiers** : `autoreject.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| CPU: `_apply_interp` utilise MNE | ✅ OUI | `autoreject.py` | Référence |
| GPU: `_apply_interp_gpu` existe | ✅ OUI | `autoreject.py:1036` | Implémenté |
| GPU: Utilise `gpu_interpolate_bad_epochs` | ✅ OUI | `autoreject.py` | Correct |
| **GPU: Même résultat que CPU** | ⏳ À vérifier | | Test nécessaire |

### 6.2 Suppression des epochs
**Fichier** : `autoreject.py`

| Vérification | État | Fichier | Action |
|--------------|------|---------|--------|
| `_apply_drop` identique | ✅ OUI | `autoreject.py` | Logique numpy |

**Test de validation 6** :
```python
def test_transform_cpu_vs_gpu():
    """Verify transform produces identical results."""
    epochs = create_test_epochs()
    
    ar_cpu = AutoReject(device='cpu')
    epochs_clean_cpu = ar_cpu.fit_transform(epochs.copy())
    
    ar_gpu = AutoReject(device='mps')
    epochs_clean_gpu = ar_gpu.fit_transform(epochs.copy())
    
    # Mêmes epochs supprimés
    assert len(epochs_clean_cpu) == len(epochs_clean_gpu)
    
    # Données identiques (ou très proches pour MPS)
    if device == 'cuda':
        np.testing.assert_array_equal(epochs_clean_cpu._data, epochs_clean_gpu._data)
    else:
        np.testing.assert_allclose(epochs_clean_cpu._data, epochs_clean_gpu._data, rtol=1e-5)
```

---

## Problèmes identifiés et corrections

### P1 : CUDA n'utilise pas float64 on device
**Fichiers à modifier** :
- `gpu_interpolation.py` : Toutes les fonctions `gpu_*`
- `backends.py` : `TorchBackend`

**Correction** :
```python
# Détection CUDA vs MPS
is_cuda = torch.cuda.is_available() and device.startswith('cuda')
is_mps = device == 'mps'

if is_cuda:
    # CUDA: tout en float64 on device
    compute_device = device
    compute_dtype = torch.float64
    matmul_dtype = torch.float64
elif is_mps:
    # MPS: float64 sur CPU pour pinv, float32 pour matmul
    compute_device = 'cpu'
    compute_dtype = torch.float64
    matmul_dtype = torch.float32
```

### P2 : `compute_thresholds_gpu` appelle `_clean_by_interp` (CPU)
**Fichier** : `gpu_pipeline.py:431`

**Correction** : Remplacer par `gpu_clean_by_interp`

```python
# AVANT
epochs_interp = _clean_by_interp(epochs, picks=picks, dots=dots, verbose=verbose)

# APRÈS
gpu_data = gpu_clean_by_interp(epochs, picks=picks, device=device, verbose=verbose)
epochs_interp = epochs.copy()
epochs_interp._data = gpu_data.numpy()
```

### P3 : Fonctions d'interpolation hardcodent `device='cpu'`
**Fichiers** : `gpu_interpolation.py`

**Correction** : Détecter CUDA et garder on device

---

## Checklist de validation finale

| Test | Description | Priorité |
|------|-------------|----------|
| `test_backend_dtype` | Vérifie float64 sur CUDA | 🔴 Haute |
| `test_interpolation_matrix_cpu_vs_gpu` | Matrice d'interpolation identique | 🔴 Haute |
| `test_clean_by_interp_cpu_vs_gpu` | Augmentation identique | 🔴 Haute |
| `test_loss_array_cpu_vs_gpu` | Loss array identique | 🔴 Haute |
| `test_argmin_identical` | Hyperparamètres identiques | 🔴 Haute |
| `test_ptp_cpu_vs_gpu` | PTP identique | 🟡 Moyenne |
| `test_median_cpu_vs_gpu` | Median identique | 🟡 Moyenne |
| `test_transform_cpu_vs_gpu` | Transform identique | 🟡 Moyenne |
| `test_vote_bad_epochs_cpu_vs_gpu` | Vote identique | 🟢 Basse |

---

## Ordre d'implémentation recommandé

1. **Phase 1 : Détection CUDA vs MPS** (`backends.py`, `gpu_interpolation.py`)
   - Ajouter helper `is_cuda_device()`
   - Modifier `gpu_make_interpolation_matrix` pour CUDA float64

2. **Phase 2 : Propager CUDA detection** (`gpu_interpolation.py`)
   - `gpu_clean_by_interp`
   - `gpu_batch_interpolate_all_n_interp`
   - `gpu_interpolate_bad_epochs`

3. **Phase 3 : Fix pipeline** (`gpu_pipeline.py`)
   - Remplacer `_clean_by_interp` par `gpu_clean_by_interp`

4. **Phase 4 : Tests de validation**
   - Écrire et exécuter tous les tests ci-dessus
   - Vérifier sur MPS (aujourd'hui) puis CUDA (Compute Canada)

---

## Règles de développement

| Règle | Description |
|-------|-------------|
| **Pas de tail/grep/head** | Ne jamais utiliser `tail`, `grep`, `head` ou autre commande de filtrage de sortie |
| **verbose=True toujours** | Toujours garder `verbose=True` pour voir la progression complète |
| **Sortie complète** | Laisser la sortie complète des tests/commandes s'afficher |

---

## Notes pour exécution

### Environnement MPS (Mac)
```bash
export AUTOREJECT_BACKEND=torch
pytest autoreject/tests/ -v
```

### Environnement CUDA (Compute Canada)
```bash
module load python/3.10 cuda/11.8
export AUTOREJECT_BACKEND=torch
pytest autoreject/tests/ -v
```
