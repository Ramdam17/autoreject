# AutoReject GPU Pipeline - Plan de Diagnostic et Correction

> **Objectif** : Identifier et corriger les divergences entre l'implémentation CPU originale et l'implémentation GPU, en suivant l'ordre chronologique du pipeline.

---

## 📋 Vue d'ensemble du Pipeline

```
Données EEG brutes
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1: Calcul PTP (Peak-to-Peak)                               │
│   CPU: np.ptp(epoch, axis=-1)                                    │
│   GPU: torch.max - torch.min                                     │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2: Augmentation des données (_clean_by_interp)             │
│   CPU: mne.channels.interpolate_bads() + spherical splines       │
│   GPU: gpu_interpolation.py (réimplémentation PyTorch)           │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ ÉTAPE 3: Calcul des seuils (_compute_thresh)                     │
│   CPU: cross_val_score + GridSearchCV séquentiel                 │
│   GPU: batched_channel_cv_loss() matriciel                       │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ ÉTAPE 4: Vote des époques mauvaises                              │
│   CPU: _get_bad_epochs() avec np.median                          │
│   GPU: _torch_median() + vote tensoriel                          │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ ÉTAPE 5: Sélection des canaux à interpoler                       │
│   CPU: np.argsort + sélection séquentielle                       │
│   GPU: torch.argsort + sélection tensorielle                     │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ ÉTAPE 6: Interpolation des époques (_interpolate_bad_epochs)     │
│   CPU: mne.channels.interpolate_bads()                           │
│   GPU: interpolate_bads_gpu() + spherical splines PyTorch        │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ ÉTAPE 7: Calcul du score (métriques de qualité)                  │
│   CPU: np.median(np.log(data.var(axis=2)))                       │
│   GPU: torch.median + torch.log + torch.var                      │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ ÉTAPE 8: Grille de perte et sélection finale                     │
│   CPU: loss_grid[n_interp, consensus] avec argmin                │
│   GPU: loss_grid tensoriel + torch.argmin                        │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
   Résultats finaux (consensus, n_interpolate, thresholds)
```

---

## 🔬 Sprint 1 : Étape 1 - Calcul PTP (Peak-to-Peak)

### Description
Première opération sur les données : calcul de l'amplitude pic-à-pic pour chaque canal/époque.

### Fichiers concernés
| Fichier | Fonction | Ligne(s) |
|---------|----------|----------|
| `autoreject/autoreject.py` | `_compute_thresh()` | ~L250 |
| `autoreject/gpu_pipeline.py` | `batched_channel_cv_loss()` | ~L150 |

### Implémentation CPU originale
```python
# Dans _compute_thresh()
X = epochs.get_data(picks=picks)  # (n_epochs, n_channels, n_times)
ptp = np.ptp(X, axis=-1)  # (n_epochs, n_channels)
```

### Implémentation GPU actuelle
```python
# Dans batched_channel_cv_loss()
data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
ptp = data_tensor.max(dim=-1).values - data_tensor.min(dim=-1).values
```

### Tâches
- [x] **1.1** Créer script de diagnostic `benchmarks/diag_step1_ptp.py`
  - Charger un petit dataset (small_fast config)
  - Calculer PTP avec les deux méthodes
  - Comparer : `np.allclose(ptp_cpu, ptp_gpu.cpu().numpy(), rtol=1e-5)`
  - Afficher max_diff, mean_diff, positions des divergences
  - ✅ **RÉSULTAT** : PASS - Différence relative max: 1.1e-7 (acceptable)

- [x] **1.2** Vérifier l'impact de la précision float32 vs float64
  - MPS impose float32
  - Calculer la perte de précision théorique
  - Tester avec `torch.float64` sur CPU pour isoler le problème
  - ✅ **RÉSULTAT** : Perte de précision ~2e-11, négligeable

- [x] **1.3** Documenter les résultats
  - Si divergence > seuil acceptable : corriger
  - Si divergence négligeable : passer à l'étape 2
  - ✅ **RÉSULTAT** : Divergence négligeable → PASSER À L'ÉTAPE 2

### Critères de validation
- [x] Différence max PTP < 1e-5 (relatif) ✅
- [x] Pas de NaN/Inf ✅
- [x] Formes tensorielles identiques ✅

---

## 🔬 Sprint 2 : Étape 2 - Augmentation des données (_clean_by_interp)

### Description
**ÉTAPE CRITIQUE** : Création des données augmentées par interpolation. C'est ici que la réimplémentation GPU des splines sphériques peut diverger significativement.

### Résultat: ✅ VALIDÉ (après correction bug critique)

#### 🐛 Bug critique corrigé (1er Décembre 2025)

**Problème** : L'interpolation GPU donnait des résultats différents de MNE (~4500% d'écart !).

**Cause racine** : MNE centre les positions des capteurs autour de l'origine de la sphère ajustée AVANT de calculer la matrice d'interpolation :
```python
# MNE fait (dans _interpolate_bads_eeg):
radius, origin = _fit_sphere(pos_good)
pos_good = pos[goods_idx_pos] - origin  # ← CENTRAGE !
pos_bad = pos[bads_idx_pos] - origin    # ← CENTRAGE !
interpolation = _make_interpolation_matrix(pos_good, pos_bad)
```

Notre implémentation ne faisait PAS ce centrage, ce qui changeait significativement la matrice d'interpolation (diff = 0.6, soit 60% des poids !).

**Fichiers corrigés** :
1. `autoreject/utils.py` (ligne ~335) : ajout `pos_good -= center`, `pos_bad -= center`
2. `autoreject/gpu_interpolation.py` : 
   - `gpu_interpolate_bads_eeg()` : ajout `_fit_sphere` et centrage
   - `gpu_clean_by_interp()` : ajout `_fit_sphere` et centrage
   - Correction `DeviceArray(data, backend, device)` au lieu de `DeviceArray(data, backend='torch', device=...)`
3. `autoreject/gpu_pipeline.py` :
   - `run_local_reject_cv_gpu()` ligne ~830 : centrage avant normalisation
   - `run_local_reject_cv_gpu_v2()` ligne ~1058 : centrage des positions

#### Résultats après correction

Les tests montrent que l'implémentation GPU est **quasi-identique** à la version CPU :

| Test | Différence | Résultat |
|------|------------|----------|
| Polynômes de Legendre | 0 | ✅ IDENTIQUE |
| Fonction G (Green's function) | 0 | ✅ IDENTIQUE |
| Matrice d'interpolation | 2.85e-08 | ✅ PASS |
| `interpolate_bads` vs MNE | 1.44e-11 | ✅ IDENTIQUE |
| `_clean_by_interp` complet | 2.16e-07 (0.27% rel) | ✅ PASS |

La différence résiduelle est due à float32 (MPS) vs float64 (CPU).

### Fichiers concernés
| Fichier | Fonction | Ligne(s) |
|---------|----------|----------|
| `autoreject/utils.py` | `_clean_by_interp()` | ~L180 |
| `autoreject/utils.py` | `interpolate_bads()` | ~L220 |
| `autoreject/gpu_interpolation.py` | `interpolate_bads_gpu()` | ~L50 |
| `autoreject/gpu_interpolation.py` | `_compute_interpolation_matrix_gpu()` | ~L100 |
| `autoreject/gpu_interpolation.py` | `_legendre_table_gpu()` | ~L150 |
| `autoreject/gpu_interpolation.py` | `_calc_g_gpu()` | ~L200 |

### Implémentation CPU originale (MNE)
```python
# Dans interpolate_bads() → appelle MNE
epochs_interp = epochs.copy()
epochs_interp.info['bads'] = bad_chs
epochs_interp.interpolate_bads(reset_bads=True)
# Utilise : mne.channels.interpolation._make_interpolation_matrix()
# Basé sur : Green's function pour splines sphériques
# Précision : float64
```

### Implémentation GPU actuelle
```python
# Dans interpolate_bads_gpu()
interp_matrix = _compute_interpolation_matrix_gpu(pos_good, pos_bad, device)
# Utilise : Legendre polynomials table + calcul G matriciel
# Précision : float32 (contrainte MPS)
```

### Tâches
- [x] **2.1** Créer script de diagnostic `benchmarks/diag_step2_interp.py`
  - Prendre 1 époque avec 1 canal marqué mauvais
  - Interpoler avec CPU (MNE) et GPU
  - Comparer les données interpolées canal par canal
  - ✅ **RÉSULTAT** : PASS - Good channels identiques, bad channel diff < 1e-5

- [x] **2.2** Isoler la matrice d'interpolation
  - Extraire `interp_matrix` des deux implémentations
  - Comparer élément par élément
  - Identifier les sources de divergence (Legendre? G? inversion?)
  - ✅ **RÉSULTAT** : PASS - Diff max = 1.8e-8

- [x] **2.3** Vérifier le calcul des polynômes de Legendre
  ```python
  # CPU: scipy.special.lpmv ou table précalculée MNE
  # GPU: _legendre_table_gpu() → récurrence manuelle
  ```
  - Comparer les tables de Legendre pour n=1..7, m=0..n
  - ✅ **RÉSULTAT** : PASS - Diff = 0 (identique)

- [x] **2.4** Vérifier le calcul de la fonction G (Green's function)
  ```python
  # G(x) = 1/(4π) * Σ (2l+1)/(l(l+1)) * P_l(cos(θ))
  ```
  - Comparer les matrices G
  - ✅ **RÉSULTAT** : PASS - Diff = 0 (identique)

- [x] **2.5** Vérifier l'inversion de matrice
  ```python
  # CPU: np.linalg.lstsq ou solve
  # GPU: torch.linalg.lstsq
  ```
  - Comparer les solutions
  - ✅ **RÉSULTAT** : PASS - Intégré dans test matrice

- [x] **2.6** Quantifier l'erreur d'interpolation
  - MSE entre données interpolées CPU vs GPU
  - Visualiser les différences spatiales
  - ✅ **RÉSULTAT** : PASS - Max rel diff = 2.8e-5 (0.003%)

### Critères de validation
- [x] Matrice d'interpolation : diff max < 1e-4 ✅ (diff = 1.8e-8)
- [x] Données interpolées : MSE < 1e-6 ✅ (diff = 1e-12)
- [x] Pas de NaN/Inf dans les résultats ✅

---

## 🔬 Sprint 3 : Étape 3 - Calcul des seuils (_compute_thresh)

### Description
Cross-validation pour trouver le seuil optimal de rejet par canal. L'implémentation GPU utilise une approche matricielle batch au lieu de boucles séquentielles.

### Fichiers concernés
| Fichier | Fonction | Ligne(s) |
|---------|----------|----------|
| `autoreject/autoreject.py` | `_compute_thresh()` | ~L230-350 |
| `autoreject/autoreject.py` | `_ChannelAutoReject` | ~L150-200 |
| `autoreject/gpu_pipeline.py` | `GPUThresholdOptimizer` | ~L50-300 |
| `autoreject/gpu_pipeline.py` | `batched_channel_cv_loss()` | ~L150-250 |

### Implémentation CPU originale
```python
# Dans _compute_thresh()
for ch_idx in range(n_channels):
    X = epochs.get_data(picks=[ch_idx])
    cv = StratifiedKFold(n_splits=cv, shuffle=False)
    param_grid = {'thresh': threshes}
    gs = GridSearchCV(_ChannelAutoReject(), param_grid, cv=cv, scoring='neg_mean_squared_error')
    gs.fit(X, y)  # y = labels des époques (good/bad basé sur PTP)
    best_thresh[ch_idx] = gs.best_params_['thresh']
```

### Implémentation GPU actuelle
```python
# Dans batched_channel_cv_loss()
# 1. Calcul PTP pour tous les canaux en batch
ptp = data.max(dim=-1).values - data.min(dim=-1).values  # (n_epochs, n_channels)

# 2. Pour chaque seuil, calcul du loss en batch
for thresh in threshes:
    bad_mask = ptp > thresh  # (n_epochs, n_channels)
    # Calcul loss via CV folds matriciel
    losses[thresh_idx] = compute_cv_loss_batch(...)

# 3. Sélection du meilleur seuil par canal
best_thresh = threshes[losses.argmin(dim=0)]
```

### Tâches
- [ ] **3.1** Créer script de diagnostic `benchmarks/diag_step3_thresh.py`
  - Exécuter les deux implémentations sur 1 canal
  - Comparer les seuils trouvés
  - Comparer les courbes de loss

- [ ] **3.2** Vérifier la logique de cross-validation
  - Les folds sont-ils identiques ?
  - L'ordre des époques est-il préservé ?

- [ ] **3.3** Vérifier le calcul du loss
  ```python
  # CPU: neg_mean_squared_error sur données reconstruites
  # GPU: MSE tensoriel
  ```
  - Comparer les scores par fold

- [ ] **3.4** Vérifier le scoring
  - CPU utilise `scoring='neg_mean_squared_error'`
  - GPU utilise-t-il la même métrique ?

- [ ] **3.5** Vérifier la gestion des cas limites
  - Que se passe-t-il si tous les epochs sont bons/mauvais ?
  - Comportement avec seuil = min(ptp) ou max(ptp) ?

### Critères de validation
- [ ] Seuils identiques à ±5% (tolérance pour variations CV)
- [ ] Courbes de loss similaires (corrélation > 0.95)
- [ ] Meilleur seuil dans le même "voisinage"

---

## 🔬 Sprint 4 : Étape 4 - Vote des époques mauvaises

### Description
Détermination des époques à rejeter basée sur le consensus entre canaux.

### Fichiers concernés
| Fichier | Fonction | Ligne(s) |
|---------|----------|----------|
| `autoreject/autoreject.py` | `_get_bad_epochs()` | ~L400 |
| `autoreject/gpu_pipeline.py` | `_get_bad_epochs_gpu()` | ~L350 |

### Implémentation CPU originale
```python
def _get_bad_epochs(self, epochs, picks, threshes):
    X = epochs.get_data(picks=picks)
    ptp = np.ptp(X, axis=-1)  # (n_epochs, n_channels)
    bad_epoch_counts = np.zeros(len(epochs))
    for ch_idx, thresh in enumerate(threshes):
        bad_epoch_counts += (ptp[:, ch_idx] > thresh)
    n_bad_channels = bad_epoch_counts
    bad_epochs = n_bad_channels > (len(picks) * consensus)
    return bad_epochs
```

### Implémentation GPU actuelle
```python
def _get_bad_epochs_gpu(data, threshes, consensus, device):
    ptp = data.max(dim=-1).values - data.min(dim=-1).values
    bad_mask = ptp > threshes.unsqueeze(0)  # broadcast
    n_bad_channels = bad_mask.sum(dim=1)
    bad_epochs = n_bad_channels > (n_channels * consensus)
    return bad_epochs
```

### Tâches
- [ ] **4.1** Créer script de diagnostic `benchmarks/diag_step4_vote.py`
  - Comparer `bad_epochs` CPU vs GPU
  - Comparer `n_bad_channels` par époque

- [ ] **4.2** Vérifier le broadcasting des seuils
  - CPU: boucle sur canaux
  - GPU: broadcast (n_epochs, 1) vs (n_channels,)
  - Vérifier que les dimensions sont correctes

- [ ] **4.3** Vérifier le calcul du consensus
  - `consensus * n_channels` arrondi pareil ?
  - Comparaison `>` vs `>=` ?

### Critères de validation
- [ ] `bad_epochs` identiques (100% match)
- [ ] `n_bad_channels` identiques

---

## 🔬 Sprint 5 : Étape 5 - Sélection des canaux à interpoler

### Description
Pour chaque époque, sélection des K canaux les plus "mauvais" à interpoler.

### Fichiers concernés
| Fichier | Fonction | Ligne(s) |
|---------|----------|----------|
| `autoreject/autoreject.py` | `_run_local_reject_cv()` | ~L850 |
| `autoreject/gpu_pipeline.py` | `run_local_reject_cv_gpu()` | ~L600 |

### Implémentation CPU originale
```python
# Dans _run_local_reject_cv()
for epoch_idx in range(n_epochs):
    bad_chs_idx = np.argsort(ptp[epoch_idx])[-n_interpolate:]
    bad_chs = [ch_names[i] for i in bad_chs_idx]
    # Interpoler ces canaux
```

### Implémentation GPU actuelle
```python
# Dans run_local_reject_cv_gpu()
sorted_indices = torch.argsort(ptp, dim=1, descending=True)
bad_chs_indices = sorted_indices[:, :n_interpolate]
```

### Tâches
- [ ] **5.1** Créer script de diagnostic `benchmarks/diag_step5_select.py`
  - Comparer les canaux sélectionnés pour interpolation
  - Vérifier l'ordre de tri

- [ ] **5.2** Vérifier le comportement avec égalités
  - Si deux canaux ont le même PTP, l'ordre est-il déterministe ?
  - `np.argsort` vs `torch.argsort` : stable sort ?

- [ ] **5.3** Vérifier les indices vs noms de canaux
  - Mapping correct entre indices et noms ?

### Critères de validation
- [ ] Canaux sélectionnés identiques (ou équivalents si égalité PTP)
- [ ] Ordre de sélection cohérent

---

## 🔬 Sprint 6 : Étape 6 - Interpolation des époques

### Description
**ÉTAPE CRITIQUE #2** : Interpolation effective des canaux sélectionnés. Réutilise les fonctions de l'étape 2 mais appliquées dans le contexte de la CV.

### Fichiers concernés
| Fichier | Fonction | Ligne(s) |
|---------|----------|----------|
| `autoreject/autoreject.py` | `_interpolate_bad_epochs()` | ~L750 |
| `autoreject/gpu_pipeline.py` | `_interpolate_epochs_gpu()` | ~L500 |

### Implémentation CPU originale
```python
def _interpolate_bad_epochs(self, epochs, bad_epochs_idx, bad_chs_per_epoch):
    epochs_interp = epochs.copy()
    for epoch_idx in bad_epochs_idx:
        bad_chs = bad_chs_per_epoch[epoch_idx]
        # Créer une "mini-epoch" avec ce seul epoch
        epoch_data = epochs_interp[epoch_idx]
        epoch_data.info['bads'] = bad_chs
        epoch_data.interpolate_bads()
        epochs_interp._data[epoch_idx] = epoch_data.get_data()
    return epochs_interp
```

### Implémentation GPU actuelle
```python
def _interpolate_epochs_gpu(data, bad_chs_indices, interp_matrix, device):
    # Batch interpolation
    for epoch_idx in range(n_epochs):
        bad_idx = bad_chs_indices[epoch_idx]
        good_idx = ~bad_idx
        data[epoch_idx, bad_idx] = interp_matrix @ data[epoch_idx, good_idx]
    return data
```

### Tâches
- [ ] **6.1** Créer script de diagnostic `benchmarks/diag_step6_interp_epochs.py`
  - Appliquer les deux méthodes sur les mêmes époques/canaux
  - Comparer les données résultantes

- [ ] **6.2** Vérifier la matrice d'interpolation dynamique
  - La matrice change-t-elle selon les canaux mauvais ?
  - Est-elle recalculée correctement à chaque fois ?

- [ ] **6.3** Vérifier le batch processing
  - L'interpolation batch GPU est-elle équivalente aux appels séquentiels CPU ?

### Critères de validation
- [ ] Données interpolées : MSE < 1e-5
- [ ] Pas de "fuites" d'information entre époques

---

## 🔬 Sprint 7 : Étape 7 - Calcul du score

### Description
Calcul de la métrique de qualité pour évaluer chaque configuration (n_interpolate, consensus).

### Fichiers concernés
| Fichier | Fonction | Ligne(s) |
|---------|----------|----------|
| `autoreject/autoreject.py` | `BaseAutoReject.score()` | ~L120 |
| `autoreject/gpu_pipeline.py` | `_compute_score_gpu()` | ~L450 |

### Implémentation CPU originale
```python
def score(self, epochs):
    """Return the negative median log variance."""
    X = epochs.get_data()
    var = np.var(X, axis=2)  # variance par canal/époque
    log_var = np.log(var)
    return -np.median(log_var)
```

### Implémentation GPU actuelle
```python
def _compute_score_gpu(data, device):
    var = data.var(dim=-1)
    log_var = torch.log(var)
    return -_torch_median(log_var)
```

### Tâches
- [ ] **7.1** Créer script de diagnostic `benchmarks/diag_step7_score.py`
  - Calculer le score avec les deux méthodes
  - Comparer les valeurs

- [ ] **7.2** Vérifier `_torch_median()` vs `np.median()`
  ```python
  # np.median avec n pair : moyenne des deux valeurs centrales
  # torch.median : valeur centrale inférieure
  ```
  - Cette différence peut causer des divergences !

- [ ] **7.3** Vérifier le log de petites variances
  - `log(0)` = -inf : géré pareil ?
  - Variances très petites → instabilité numérique ?

### Critères de validation
- [ ] Scores identiques à ±1e-3
- [ ] Pas de NaN/Inf

---

## 🔬 Sprint 8 : Étape 8 - Grille de perte et sélection finale

### Description
Construction de la grille de perte pour toutes les combinaisons (n_interpolate, consensus) et sélection de la meilleure.

### Fichiers concernés
| Fichier | Fonction | Ligne(s) |
|---------|----------|----------|
| `autoreject/autoreject.py` | `_run_local_reject_cv()` | ~L900 |
| `autoreject/gpu_pipeline.py` | `run_local_reject_cv_gpu()` | ~L700 |

### Implémentation CPU originale
```python
# Dans _run_local_reject_cv()
loss_grid = np.zeros((len(n_interpolates), len(consensuses)))
for i, n_interp in enumerate(n_interpolates):
    for j, cons in enumerate(consensuses):
        # Appliquer n_interp et cons
        epochs_clean = self._apply_interp_and_reject(epochs, n_interp, cons)
        loss_grid[i, j] = -self.score(epochs_clean)

# Trouver le minimum
best_idx = np.unravel_index(loss_grid.argmin(), loss_grid.shape)
best_n_interp = n_interpolates[best_idx[0]]
best_consensus = consensuses[best_idx[1]]
```

### Implémentation GPU actuelle
```python
# Dans run_local_reject_cv_gpu()
loss_grid = torch.zeros((len(n_interpolates), len(consensuses)), device=device)
# ... calcul parallélisé ...
best_idx = loss_grid.argmin()
best_i, best_j = best_idx // len(consensuses), best_idx % len(consensuses)
```

### Tâches
- [ ] **8.1** Créer script de diagnostic `benchmarks/diag_step8_grid.py`
  - Comparer les grilles de perte complètes
  - Visualiser les différences

- [ ] **8.2** Vérifier le calcul de l'argmin
  - `np.unravel_index` vs division/modulo PyTorch
  - Comportement si plusieurs minima égaux ?

- [ ] **8.3** Vérifier l'accumulation des erreurs
  - Les erreurs des étapes précédentes s'accumulent-elles ?
  - Quelle est la contribution de chaque étape à l'erreur finale ?

### Critères de validation
- [ ] Grilles de perte : corrélation > 0.99
- [ ] Même minimum sélectionné (ou équivalent)

---

## 📊 Scripts de diagnostic à créer

```
benchmarks/
├── diag_step1_ptp.py
├── diag_step2_interp.py
├── diag_step3_thresh.py
├── diag_step4_vote.py
├── diag_step5_select.py
├── diag_step6_interp_epochs.py
├── diag_step7_score.py
├── diag_step8_grid.py
└── diag_full_comparison.py  # Exécute tout et génère un rapport
```

---

## 🎯 Métriques de succès globales

| Métrique | Objectif | Priorité |
|----------|----------|----------|
| Consensus match | 100% | P0 |
| n_interpolate match | ±1 | P0 |
| Thresholds correlation | >0.99 | P1 |
| Temps GPU < CPU | ≥2x speedup | P2 |
| Mémoire GPU | <4GB pour 128ch | P2 |

---

## 📝 Notes

### Sur le dossier legacy/
> **Question** : Est-ce utile de mettre le code original dans un dossier `legacy/` ?

**Oui, très utile pour** :
- Avoir une référence "ground truth" facilement accessible
- Pouvoir importer et comparer directement dans les scripts de diagnostic
- Éviter de chercher dans l'historique git

**Structure suggérée** :
```
legacy/
├── __init__.py
├── autoreject_original.py   # Copie de autoreject.py avant modifications
├── utils_original.py        # Copie de utils.py avant modifications
└── README.md                # Explication de la version
```

---

## 🚀 Prochaine action

**Sprint 1 - Tâche 1.1** : Créer `benchmarks/diag_step1_ptp.py` et valider que le calcul PTP est identique.
