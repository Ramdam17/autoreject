# Document 2 : GPU Call Stack (AutoReject.fit - Backend Torch)

## Configuration analysée
- **Jeu de données** : highdensity_128ch (128 canaux, 300 epochs, 500Hz)
- **Paramètres** : n_interpolate=[1,2,4,8,12,16], consensus=[0.1,0.2,0.3,0.4,0.5], cv=10
- **Backend** : torch (os.environ['AUTOREJECT_BACKEND'] = 'torch')
- **Device** : mps (Mac) ou cuda (Linux/Windows)

---

## 1. Point d'entrée : AutoReject.fit(epochs)

```
AutoReject.fit(epochs)
│
├── 1.1  utils._handle_picks(info, picks)
│        └── Appelant: AutoReject.fit
│        └── Paramètres: info=epochs.info, picks=None (default)
│        └── Retourne: array d'indices de canaux [0, 1, ..., 127]
│
├── 1.2  utils._check_data(epochs, picks, verbose)
│        └── Appelant: AutoReject.fit
│        └── Vérifie: preloaded, positions canaux, types supportés
│
├── 1.3  AutoReject._should_use_gpu(epochs)   ← DIFFÉRENCE AVEC CPU
│        └── Appelant: AutoReject.fit
│        └── Appelle: _is_gpu_available() → True
│        └── Condition: n_epochs >= 50
│        └── Retourne: (True, 'mps')
│
├── 1.4  utils._get_picks_by_type(info, picks_)
│        └── Retourne: [('eeg', [0, 1, ..., 127])]
│
├── 1.5  functools.partial(compute_thresholds_gpu, ...)   ← DIFFÉRENCE
│        └── Crée: thresh_func avec device='mps', method, random_state
│
└── 1.6  BOUCLE sur picks_by_type (1 itération pour EEG)
         │
         └── 1.6.1  run_local_reject_cv_gpu_batch(...)   ← DIFFÉRENCE
                    │
                    └── Voir section 2 ci-dessous
```

---

## 2. run_local_reject_cv_gpu_batch - Validation croisée GPU batch

```
run_local_reject_cv_gpu_batch(epochs, thresh_func, picks_, n_interpolate, cv, consensus, dots, verbose, n_jobs, device)
│
├── 2.1  Initialisation loss array
│        └── loss = np.zeros((5, 6, 10))  # (n_consensus, n_interpolate, n_folds)
│
├── 2.2  _AutoReject(..., device=device).fit(epochs)
│        │
│        └── 2.2.1  _AutoReject.fit(epochs)
│                   │
│                   ├── 2.2.1.1  utils._handle_picks(info, picks)
│                   │
│                   ├── 2.2.1.2  thresh_func(epochs.copy(), ...)
│                   │            │
│                   │            └── compute_thresholds_gpu(...)   ← DIFFÉRENCE
│                   │                 │
│                   │                 └── Voir section 3 ci-dessous
│                   │
│                   └── (reste comme CPU: get_reject_log, _interpolate, _slicemean)
│
├── 2.3  _AutoReject._vote_bad_epochs(epochs, picks_)
│        │
│        └── Voir section 4 ci-dessous (utilise TorchBackend)
│
├── 2.4  PHASE 1: Pré-calcul labels pour tous les n_interpolate
│        │
│        └── BOUCLE sur n_interpolate (6 itérations)
│            │
│            └── _AutoReject._get_epochs_interpolation(epochs, labels, picks_, n_interp, data=X_full)
│                 │
│                 └── Optimisation: data pré-extrait pour éviter epochs[idx].get_data()
│
├── 2.5  PHASE 2: GPU batch interpolation   ← DIFFÉRENCE MAJEURE
│        │
│        └── gpu_interpolation.gpu_batch_interpolate_all_n_interp(...)
│             │
│             └── Voir section 5 ci-dessous
│
└── 2.6  PHASE 3: Batch CV evaluation   ← DIFFÉRENCE MAJEURE
         │
         └── Voir section 6 ci-dessous
```

---

## 3. compute_thresholds_gpu - Calcul des seuils GPU

```
compute_thresholds_gpu(epochs, picks, device, method='bayesian_optimization', random_state, verbose)
│
├── 3.1  _get_or_compute_loocv_matrices(epochs, picks, device)
│        │
│        ├── 3.1.1  [Si cache miss] compute_loocv_interp_matrices_gpu(...)
│        │          │
│        │          ├── compute_spherical_spline_G_matrix_gpu(...)
│        │          │   │
│        │          │   ├── _compute_legendre_polynomials_gpu(...)
│        │          │   │   └── Évaluation Clenshaw des polynômes de Legendre
│        │          │   │
│        │          │   └── Retourne: G_matrix tensor GPU
│        │          │
│        │          └── Retourne: interp_matrices shape (n_picks, n_picks)
│        │
│        └── Retourne: matrices d'interpolation LOOCV (depuis cache ou calcul)
│
├── 3.2  _clean_by_interp_gpu(epochs, picks, interp_matrices)
│        │
│        └── Augmentation des données via LOOCV interpolation (GPU)
│             └── Retourne: epochs_interp (600 epochs = 300 original + 300 augmenté)
│
├── 3.3  GPUThresholdOptimizer(device='mps')
│        └── Initialisation de l'optimiseur GPU
│
├── 3.4  data_gpu = optimizer._to_tensor(X_full)   ← CONVERSION FLOAT32
│        └── Tensor GPU shape (600, 128, n_times), dtype=float32
│
├── 3.5  cv_splits = list(StratifiedShuffleSplit(...).split(X, y))
│
└── 3.6  optimizer.compute_all_thresholds_gpu(data_gpu, picks, cv_splits, y, method, random_state)
         │
         └── 3.6.1  Calcul PTP pour tous les canaux (GPU)
         │          │
         │          └── ptp_all = data.max(dim=-1).values - data.min(dim=-1).values
         │               └── Shape: (600, 128), dtype=float32
         │
         ├── 3.6.2  Construction tensor de seuils
         │          │
         │          └── threshes_all = _to_tensor(threshes_all_np)
         │               └── Shape: (128, 600), dtype=float32
         │
         ├── 3.6.3  batched_all_channels_cv_loss_parallel(...)   ← CLÉ GPU
         │          │
         │          └── Voir section 3.7 ci-dessous
         │
         └── 3.6.4  BOUCLE sur canaux: bayes_opt avec cache de losses
                    │
                    └── bayesopt.bayes_opt(cached_loss_func, initial_x, all_threshes, ...)
                         └── Retourne: best_thresh pour ce canal
```

### 3.7 batched_all_channels_cv_loss_parallel - Calcul CV batch GPU

```
batched_all_channels_cv_loss_parallel(data_all_channels, ptp_all, threshes_all, cv_splits)
│
├── BOUCLE sur folds (10 folds)
│   │
│   ├── data_train = data_all_channels[train_idx]   # (540, 128, n_times)
│   ├── data_test = data_all_channels[test_idx]     # (60, 128, n_times)
│   ├── ptp_train = ptp_all[train_idx]              # (540, 128)
│   │
│   ├── # Pour CHAQUE canal et CHAQUE seuil: calcul masque good epochs
│   ├── ptp_train_exp = ptp_train.T.unsqueeze(-1)   # (128, 540, 1)
│   ├── threshes_exp = threshes_all.unsqueeze(1)    # (128, 1, 600)
│   ├── good_train = ptp_train_exp <= threshes_exp  # (128, 540, 600) bool
│   │
│   ├── # Calcul mean_ pour tous canaux/seuils en parallèle (BMM)
│   ├── data_perm = data_train.permute(1, 2, 0)     # (128, n_times, 540)
│   ├── good_perm = good_train.permute(1, 0, 2).float()  # (128, 540, 600)
│   ├── masked_sum = torch.bmm(data_perm, good_perm)     # (128, n_times, 600)
│   ├── counts = good_train.sum(dim=1)              # (128, 600)
│   ├── mean_all = masked_sum / counts.unsqueeze(1) # (128, n_times, 600)
│   │
│   ├── # Médiane test (partagée pour tous seuils)
│   ├── median_test = _torch_median(data_test, dim=0)  # (128, n_times)
│   │
│   └── # Score pour tous canaux/seuils
│       ├── sq_diff = (median_test.unsqueeze(-1) - mean_all) ** 2
│       └── fold_losses = sq_diff.mean(dim=1).sqrt()  # (128, 600)
│
└── Retourne: all_losses.mean(dim=0)  # (128, 600) - moyenne sur folds
```

---

## 4. _vote_bad_epochs avec TorchBackend

```
_AutoReject._vote_bad_epochs(epochs, picks)
│
├── backends.get_backend() → TorchBackend   ← DIFFÉRENCE
│
├── data = epochs.get_data(picks)  # shape: (300, 128, n_times), dtype=float64
│
├── backend.ptp(data, axis=-1)   ← DIFFÉRENCE: retourne float32!
│   │
│   └── TorchBackend.ptp implementation:
│        ├── tensor = torch.tensor(data, dtype=torch.float32, device='mps')
│        ├── ptp = tensor.max(dim=-1).values - tensor.min(dim=-1).values
│        └── return ptp.cpu().numpy()  # dtype=float32
│
├── deltas = ptp.T  # shape: (128, 300), dtype=float32
│
└── BOUCLE sur canaux: delta > thresh (comparaison float32 vs float64!)
    └── Retourne: (labels, bad_sensor_counts)
```

---

## 5. gpu_batch_interpolate_all_n_interp - Interpolation batch GPU

```
gpu_batch_interpolate_all_n_interp(epochs, labels_list, picks, pos, device, verbose)
│
├── 5.1  X_gpu = torch.tensor(epochs.get_data(), dtype=torch.float32, device=device)
│        └── Shape: (300, n_channels_total, n_times)
│
├── 5.2  interp_cache = {}  # Cache des matrices d'interpolation
│
└── 5.3  BOUCLE sur labels_list (6 n_interp values)
         │
         ├── _get_interp_chs(labels, ch_names, picks)
         │   └── Liste des canaux à interpoler par epoch
         │
         ├── all_interp_ch_indices = [[picks.index(ch) for ch in epoch_chs] for epoch_chs in interp_channels]
         │
         ├── data_gpu = X_gpu[:, picks, :].clone()  # (300, 128, n_times)
         │
         └── BOUCLE sur epochs (300)
             │
             ├── Si len(bad_ch_indices) == 0: continue
             │
             ├── cache_key = tuple(sorted(bad_ch_indices))
             │
             ├── [Si cache miss] Calcul matrice d'interpolation
             │   │
             │   ├── good_idx = [i for i in range(n_picks) if i not in bad_ch_indices]
             │   ├── pos_good = pos[good_idx]
             │   ├── pos_bad = pos[bad_ch_indices]
             │   │
             │   ├── # Calcul matrice spline sphérique GPU
             │   ├── G_good_good = compute_G_matrix_gpu(pos_good, pos_good)
             │   ├── G_bad_good = compute_G_matrix_gpu(pos_bad, pos_good)
             │   │
             │   ├── # Résolution système linéaire pour coefficients
             │   ├── coefficients = torch.linalg.solve(G_good_good, ...)
             │   │
             │   └── interp_matrix = G_bad_good @ coefficients
             │
             └── # Application interpolation
                 ├── good_data = data_gpu[epoch_idx, good_idx, :]  # (n_good, n_times)
                 ├── interpolated = interp_matrix @ good_data      # (n_bad, n_times)
                 └── data_gpu[epoch_idx, bad_idx, :] = interpolated
         
         └── X_interp_all_gpu.append(data_gpu)

└── Retourne: X_interp_all_gpu  # Liste de 6 tensors (300, 128, n_times)
```

---

## 6. Phase 3 CV evaluation batch GPU

```
PHASE 3: Batch CV evaluation
│
├── X_gpu = optimizer._to_tensor(X_full)  # (300, 128, n_times), float32
├── X_picks_gpu = X_gpu[:, picks_t, :]    # (300, 128, n_times)
│
└── BOUCLE sur n_interpolate (6 valeurs)
    │
    ├── X_interp_picks_gpu = X_interp_all_gpu[jdx]  # (300, 128, n_times)
    │
    └── BOUCLE sur cv.split (10 folds)
        │
        ├── train_t = torch.tensor(train, device=device)  # (270,)
        ├── test_t = torch.tensor(test, device=device)    # (30,)
        │
        ├── # Médiane test pré-calculée (partagée entre consensus)
        ├── X_test = X_picks_gpu[test_t]           # (30, 128, n_times)
        ├── median_X = _torch_median(X_test, dim=0)  # (128, n_times)
        │
        ├── scores_gpu = torch.zeros(n_consensus, device=device)
        │
        └── BOUCLE sur consensus (5 valeurs)
            │
            ├── # Vérification validité
            ├── if this_consensus * n_channels <= n_interp:
            │   └── scores_gpu[idx] = float('-inf')
            │
            ├── # Calcul bad_epochs (CPU)
            ├── local_reject.consensus_[ch_type] = this_consensus
            ├── bad_epochs = local_reject._get_bad_epochs(bad_sensor_counts[train], ...)
            ├── good_epochs_idx = np.nonzero(np.invert(bad_epochs))[0]
            │
            ├── # Calcul mean sur GPU
            ├── good_idx_t = torch.tensor(good_epochs_idx, device=device)
            ├── X_train_interp = X_interp_picks_gpu[train_t]  # (270, 128, n_times)
            ├── X_good = X_train_interp[good_idx_t]           # (n_good, 128, n_times)
            ├── mean_gpu = X_good.mean(dim=0)                 # (128, n_times)
            │
            └── # Score GPU
                ├── sq_diff = (median_X - mean_gpu) ** 2
                └── scores_gpu[idx] = -sq_diff.mean().sqrt()
        
        ├── # SINGLE sync par fold (optimisation)
        └── scores_np = scores_gpu.cpu().numpy()
            └── loss[idx, jdx, fold] = -scores_np[idx]
```

---

## 7. Différences clés CPU vs GPU

| Aspect | CPU (NumPy) | GPU (Torch) |
|--------|-------------|-------------|
| **Backend** | NumpyBackend | TorchBackend |
| **dtype données** | float64 | float32 |
| **compute_thresholds** | _compute_thresholds (séquentiel) | compute_thresholds_gpu (batch) |
| **CV thresholds** | Parallel(_compute_thresh) ×128 canaux | batched_all_channels_cv_loss_parallel |
| **Interpolation** | _interpolate_bad_epochs (×300 epochs) | gpu_batch_interpolate_all_n_interp |
| **CV evaluation** | _run_local_reject_cv | run_local_reject_cv_gpu_batch |
| **ptp** | np.ptp() → float64 | tensor.max()-min() → float32 |
| **median** | np.median() → float64 | _torch_median() → float32 |

---

## 8. Points critiques identifiés

### 8.1 Conversion float32
```python
# gpu_pipeline.py ligne 131
def _to_tensor(self, data, dtype=None):
    if dtype is None:
        dtype = self.torch.float32  # ← TOUJOURS float32 par défaut
    return self.torch.tensor(data, dtype=dtype, device=self.device)
```

### 8.2 Thresholds stockés en float32
```python
# Après compute_all_thresholds_gpu
best_thresholds = np.zeros(n_channels)  # float64
for ch_idx in range(n_channels):
    best_thresholds[ch_idx] = all_threshes[best_idx]  # all_threshes est float32!
```

### 8.3 PTP retourne float32
```python
# backends.py TorchBackend.ptp
tensor = self.torch.tensor(data, dtype=self.torch.float32, device=self.device)
ptp = tensor.max(dim=axis).values - tensor.min(dim=axis).values
return ptp.cpu().numpy()  # ← float32 même si input était float64
```

---

## 9. Récapitulatif chronologique GPU

| # | Fonction | Module | Différence avec CPU |
|---|----------|--------|---------------------|
| 1 | AutoReject.fit | autoreject.py | - |
| 2 | _handle_picks | utils.py | - |
| 3 | _check_data | utils.py | - |
| 4 | **_should_use_gpu** | autoreject.py | Retourne (True, 'mps') |
| 5 | _get_picks_by_type | utils.py | - |
| 6 | **partial(compute_thresholds_gpu)** | autoreject.py | Nouveau thresh_func |
| 7 | **run_local_reject_cv_gpu_batch** | gpu_pipeline.py | Remplace _run_local_reject_cv |
| 8 | _AutoReject.fit | autoreject.py | Utilise thresh_func GPU |
| 9 | **compute_thresholds_gpu** | gpu_pipeline.py | Remplace _compute_thresholds |
| 10 | **_get_or_compute_loocv_matrices** | gpu_pipeline.py | Nouveau |
| 11 | **GPUThresholdOptimizer** | gpu_pipeline.py | Nouveau |
| 12 | **_to_tensor (float32!)** | gpu_pipeline.py | Conversion critique |
| 13 | **batched_all_channels_cv_loss_parallel** | gpu_pipeline.py | Nouveau |
| 14 | **_torch_median** | gpu_pipeline.py | Remplace np.median |
| 15 | bayes_opt | bayesopt.py | Identique mais avec cache |
| 16 | _vote_bad_epochs | autoreject.py | TorchBackend.ptp (float32) |
| 17 | _get_epochs_interpolation | autoreject.py | Avec data pré-extrait |
| 18 | **gpu_batch_interpolate_all_n_interp** | gpu_interpolation.py | Nouveau |
| 19 | **Phase 3 batch CV** | gpu_pipeline.py | Nouveau |
| 20 | **_torch_median** | gpu_pipeline.py | Pour score |
