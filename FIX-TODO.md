# FIX-TODO: Optimisations GPU - TERMINÉ ✅

## 🎯 Objectifs atteints

1. **CPU == GPU** : Résultats identiques entre CPU et GPU ✅
2. **Précision float64** : Maintenir float64 partout sauf MPS matmul ✅
3. **Speedup 4x minimum** : Accélération GPU vs CPU ✅ **17.09x atteint**
4. **Rétrocompatibilité** : Tests de référence passent ✅ (6/6 tests)
5. **Code maintenable** : Architecture propre avec backends ✅

---

## 📊 Historique des performances

| Version | Config | Temps GPU | Speedup | Fix appliqués |
|---------|--------|-----------|---------|---------------|
| Initial | 74ch×100ep | 7.97s | 4.16x | - |
| FIX 2 | 74ch×100ep | 6.80s | 4.74x | Batch all channels |
| FIX 1+2 | 74ch×100ep | 5.37s | 5.26x | + Éliminer MNE overhead |
| FIX 3 (BMM) | 74ch×100ep | 3.48s | 8.24x | + BMM + pre-median |
| **FINAL** | **128ch×300ep** | **10.4s** | **17.09x** | + Cache LOOCV + GPU interp |

---

## 🔧 Optimisations appliquées

### FIX 1: Éliminer l'overhead MNE epochs copy/getitem ✅

**Problème**: `epochs[...]` et `epochs.copy()` font des deep copies coûteuses.

**Solution**: Pré-extraction des données NumPy, passage via paramètre `data`.

**Fichiers modifiés**:
- `autoreject/gpu_pipeline.py` : pré-extraction dans `run_local_reject_cv_gpu_batch()`
- `autoreject/autoreject.py` : `_get_epochs_interpolation()` avec paramètre `data`

---

### FIX 2: Paralléliser tous les canaux ✅

**Problème**: `batched_channel_cv_loss()` traite un canal à la fois.

**Solution**: `batched_all_channels_cv_loss_parallel()` traite TOUS les canaux en parallèle.

**Fichiers modifiés**:
- `autoreject/gpu_pipeline.py` : nouvelle fonction batch

---

### FIX 3: BMM + Pre-computed medians ✅

**Problème**: Broadcast 4D crée 2.6GB de mémoire temporaire, médiane recalculée dans la boucle.

**Solution**: 
- BMM (batch matrix multiply) au lieu de broadcast 4D → **66x plus rapide**
- Pré-calcul des médianes avant la boucle de folds

**Code clé**:
```python
# BMM au lieu de 4D broadcast
data_perm = data_train.permute(1, 2, 0)  # (c, t, train)
good_perm = good_train.permute(1, 0, 2).float()  # (c, train, th)
masked_sum = torch.bmm(data_perm, good_perm)  # (c, t, th)
```

**Fichiers modifiés**:
- `autoreject/gpu_pipeline.py` : `batched_all_channels_cv_loss_parallel()`

---

### FIX 4: Cache LOOCV + GPU interpolation ✅

**Problème**: 
- `_interpolate_bad_epochs` utilise boucle MNE lente (copies, indexation)
- `gpu_clean_by_interp` calcule 128 matrices pinv séquentiellement

**Solution**:
- Cache global `_LOOCV_INTERP_CACHE` pour matrices d'interpolation LOOCV
- Pré-calcul de TOUTES les matrices en une fois
- Application batch via einsum: `torch.einsum('ij,ejt->eit', ...)`
- `_interpolate_bad_epochs_gpu()` pour path GPU complet

**Code clé**:
```python
# Cache des matrices LOOCV (une seule fois par géométrie)
interp_matrices = _get_loocv_interp_matrices(pos, picks, device, ...)

# Application batch
result_picks = torch.einsum('ij,ejt->eit', interp_matrices, data_picks)
```

**Fichiers modifiés**:
- `autoreject/gpu_interpolation.py` : `_LOOCV_INTERP_CACHE`, `_get_loocv_interp_matrices()`, `gpu_clean_by_interp()` optimisé
- `autoreject/autoreject.py` : `_interpolate_bad_epochs_gpu()`, `_AutoReject` avec paramètre `device`
- `autoreject/gpu_pipeline.py` : passage du `device` à `_AutoReject`

---

## 📈 Résultats finaux

### Configuration de test réaliste
- **128 canaux** (EEG haute densité)
- **300 epochs** (10 minutes @ 500Hz, epochs de 2s)
- **cv=10**, **n_interpolate=[1,2,4,8,12,16]**, **consensus=[0.1-0.5]**

### Performance
| Backend | Temps | Min |
|---------|-------|-----|
| CPU (numpy) | 178.0s | 3.0 min |
| GPU (torch/MPS) | **10.4s** | **0.2 min** |
| **Speedup** | | **17.09x** |

### Validation
- ✅ `consensus` identique CPU/GPU
- ✅ `n_interpolate` identique CPU/GPU  
- ✅ 6/6 tests unitaires passent

---

## 🔮 Pistes d'optimisation futures (non implémentées)

1. **Paralléliser n_interpolate dans batch interpolation** : Actuellement séquentiel, pourrait être parallélisé
2. **CUDA streams** : Pour overlap compute/transfer sur GPU NVIDIA
3. **Mixed precision (FP16)** : Pour GPU avec Tensor Cores
4. **Compilation JIT** : `torch.compile()` pour PyTorch 2.0+
