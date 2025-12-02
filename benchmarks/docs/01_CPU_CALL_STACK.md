# Document 1 : CPU Call Stack (AutoReject.fit - Backend NumPy)

## Configuration analysée
- **Jeu de données** : highdensity_128ch (128 canaux, 300 epochs, 500Hz)
- **Paramètres** : n_interpolate=[1,2,4,8,12,16], consensus=[0.1,0.2,0.3,0.4,0.5], cv=10
- **Backend** : numpy (os.environ['AUTOREJECT_BACKEND'] = 'numpy')
- **Device** : cpu

---

## 1. Point d'entrée : AutoReject.fit(epochs)

```
AutoReject.fit(epochs)
│
├── 1.1  utils._handle_picks(info, picks)
│        └── Appelant: AutoReject.fit
│        └── Paramètres: info=epochs.info, picks=None (default)
│        └── Retourne: array d'indices de canaux [0, 1, ..., 127] (128 canaux EEG)
│
├── 1.2  utils._check_data(epochs, picks, verbose)
│        └── Appelant: AutoReject.fit
│        └── Paramètres: epochs, picks_=picks_, verbose=True
│        └── Vérifie: preloaded, positions canaux valides, types supportés
│        └── Retourne: None (lève ValueError si invalide)
│
├── 1.3  AutoReject._should_use_gpu(epochs)
│        └── Appelant: AutoReject.fit
│        └── Paramètres: epochs
│        └── Retourne: (False, 'cpu') car device='cpu' ou backend='numpy'
│
├── 1.4  utils._get_picks_by_type(info, picks_)
│        └── Appelant: AutoReject.fit
│        └── Paramètres: info=epochs.info, picks=picks_
│        └── Retourne: [('eeg', [0, 1, ..., 127])] pour EEG 128 canaux
│
├── 1.5  utils._compute_dots(info, templates=None) [si MEG présent, sinon None]
│        └── Appelant: AutoReject.fit
│        └── Paramètres: info (MEG picks)
│        └── Retourne: (self_dots, cross_dots) ou None pour EEG seul
│
├── 1.6  functools.partial(_compute_thresholds, ...)
│        └── Appelant: AutoReject.fit
│        └── Crée: thresh_func avec n_jobs, method, random_state, dots pré-remplis
│
└── 1.7  BOUCLE sur picks_by_type (1 itération pour EEG)
         │
         └── Pour ch_type='eeg', this_picks=[0..127]:
              │
              └── 1.7.1  _run_local_reject_cv(epochs, thresh_func, this_picks, ...)
                         │
                         └── Voir section 2 ci-dessous
```

---

## 2. _run_local_reject_cv - Validation croisée locale

```
_run_local_reject_cv(epochs, thresh_func, picks_, n_interpolate, cv, consensus, dots, verbose, n_jobs)
│
├── 2.1  Initialisation loss array
│        └── loss = np.zeros((5, 6, 10))  # (n_consensus, n_interpolate, n_folds)
│
├── 2.2  _AutoReject(...).fit(epochs)
│        │
│        └── 2.2.1  _AutoReject.fit(epochs)
│                   │
│                   ├── 2.2.1.1  utils._handle_picks(info, picks)
│                   │
│                   ├── 2.2.1.2  utils._check_data(epochs, picks_, verbose)
│                   │
│                   ├── 2.2.1.3  utils._get_picks_by_type(picks_, info)
│                   │
│                   ├── 2.2.1.4  thresh_func(epochs.copy(), ...)
│                   │            │
│                   │            └── _compute_thresholds(epochs, method, random_state, picks, ...)
│                   │                 │
│                   │                 └── Voir section 3 ci-dessous
│                   │
│                   ├── 2.2.1.5  _AutoReject.get_reject_log(epochs, picks)
│                   │            │
│                   │            └── Voir section 4 ci-dessous
│                   │
│                   ├── 2.2.1.6  _get_interp_chs(labels, ch_names, picks)
│                   │            └── Retourne: liste de listes de noms de canaux à interpoler
│                   │
│                   ├── 2.2.1.7  _interpolate_bad_epochs(epochs_copy, interp_channels, ...)
│                   │            │
│                   │            └── Voir section 5 ci-dessous
│                   │
│                   └── 2.2.1.8  _slicemean(epochs_copy.get_data(), good_epochs_idx, axis=0)
│                                └── Retourne: mean_ (template moyen des bonnes epochs)
│
├── 2.3  _AutoReject._vote_bad_epochs(epochs, picks_)
│        │
│        └── Voir section 4.1 ci-dessous
│
└── 2.4  BOUCLE sur n_interpolate (6 itérations: [1,2,4,8,12,16])
         │
         ├── 2.4.1  _AutoReject._get_epochs_interpolation(epochs, labels, picks_, n_interp)
         │          │
         │          └── Voir section 4.2 ci-dessous
         │
         ├── 2.4.2  _get_interp_chs(labels, ch_names, picks_)
         │
         ├── 2.4.3  _interpolate_bad_epochs(epochs_interp, interp_channels, picks_, dots, verbose, n_jobs)
         │          │
         │          └── Voir section 5 ci-dessous
         │
         └── 2.4.4  BOUCLE sur cv.split(X) (10 folds)
                    │
                    └── 2.4.4.1  BOUCLE sur consensus (5 valeurs: [0.1, 0.2, 0.3, 0.4, 0.5])
                                 │
                                 ├── _AutoReject._get_bad_epochs(bad_sensor_counts[train], ch_type, picks_)
                                 │
                                 ├── _slicemean(epochs_interp[train].get_data(), good_epochs_idx, axis=0)
                                 │
                                 └── _AutoReject.score(X[test])
                                      │
                                      ├── backends.get_backend() → NumpyBackend
                                      ├── backend.median(X, axis=0) → np.median()
                                      └── Retourne: -RMSE(median - mean_)
```

---

## 3. _compute_thresholds - Calcul des seuils par canal

```
_compute_thresholds(epochs, method='bayesian_optimization', random_state, picks, augment=True, dots, verbose, n_jobs)
│
├── 3.1  utils._handle_picks(info, picks)
│
├── 3.2  utils._check_data(epochs, picks, verbose, check_loc=augment)
│
├── 3.3  utils._get_picks_by_type(picks, info)
│        └── Si multiple types: appel récursif pour chaque type
│
├── 3.4  [Si augment=True] utils._clean_by_interp(epochs, picks, dots, verbose)
│        │
│        └── 3.4.1  BOUCLE sur chaque canal (128 itérations)
│                   │
│                   ├── epochs.info['bads'] = [ch_name]
│                   │
│                   ├── utils.interpolate_bads(epochs, picks, dots, reset_bads=True, mode='fast')
│                   │   │
│                   │   ├── [EEG] _interpolate_bads_eeg(inst, picks)
│                   │   │        │
│                   │   │        ├── mne.bem._fit_sphere(pos_good)
│                   │   │        ├── mne.channels.interpolation._make_interpolation_matrix(pos_good, pos_bad)
│                   │   │        └── mne.channels.interpolation._do_interp_dots(inst, interpolation, goods_idx, bads_idx)
│                   │   │
│                   │   └── [MEG] _interpolate_bads_meg_fast(inst, picks, dots, mode)
│                   │
│                   └── epochs_interp._data[:, pick] = epochs._data[:, pick_interp]
│
├── 3.5  sklearn.model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state)
│
└── 3.6  joblib.Parallel(n_jobs)(delayed(_compute_thresh)(...) for pick in picks)
         │
         └── 3.6.1  _compute_thresh(this_data, method='bayesian_optimization', cv, y, random_state)
                    │                              [128 appels parallèles si n_jobs>1]
                    │
                    ├── 3.6.1.1  backends.get_backend() → NumpyBackend
                    │
                    ├── 3.6.1.2  backend.ptp(this_data, axis=1) → np.ptp()
                    │            └── all_threshes = np.sort(deltas)
                    │
                    ├── 3.6.1.3  Définition de func(thresh) [closure]
                    │            │
                    │            └── sklearn.model_selection.cross_val_score(est, this_data, y, cv)
                    │                 │
                    │                 └── _ChannelAutoReject.fit(X) + score(X)
                    │                      │
                    │                      ├── backend.ptp(X, axis=1)
                    │                      └── _slicemean(X, keep, axis=0)
                    │
                    ├── 3.6.1.4  Initialisation: initial_x = all_threshes[indices linspace]
                    │
                    └── 3.6.1.5  bayesopt.bayes_opt(func, initial_x, all_threshes, expected_improvement, max_iter=10)
                                 │
                                 └── Voir section 6 ci-dessous
```

---

## 4. _vote_bad_epochs et helpers

### 4.1 _vote_bad_epochs
```
_AutoReject._vote_bad_epochs(epochs, picks)
│
├── labels = np.zeros((n_epochs, n_channels))  # (300, 128)
│
├── backends.get_backend() → NumpyBackend
│
├── data = epochs.get_data(picks)  # shape: (300, 128, n_times)
│
├── backend.ptp(data, axis=-1) → np.ptp()  # shape: (300, 128)
│   └── deltas = ptp.T  # shape: (128, 300)
│
└── BOUCLE sur canaux: marquer labels[bad_epochs_idx, ch_idx] = 1
    └── Retourne: (labels, bad_sensor_counts)
```

### 4.2 _get_epochs_interpolation
```
_get_epochs_interpolation(epochs, labels, picks, n_interpolate, verbose, data)
│
├── Pre-compute: data = epochs.get_data() si nécessaire
│
└── BOUCLE sur epochs (300 itérations)
    │
    ├── Si n_bads == 0: continue
    │
    ├── Si n_bads <= n_interpolate:
    │   └── interp_chs_mask = (labels[epoch_idx] == 1)
    │
    └── Si n_bads > n_interpolate:
        │
        ├── backends.get_backend() → NumpyBackend
        │
        ├── backend.ptp(epoch_data, axis=-1) → np.ptp()
        │
        ├── Trier canaux par ptp décroissant
        │
        └── Sélectionner les n_interpolate pires canaux
        
    └── labels[epoch_idx][interp_chs_mask] = 2
    
└── Retourne: labels (avec 0=good, 1=bad non interpolé, 2=bad interpolé)
```

### 4.3 _get_bad_epochs
```
_get_bad_epochs(bad_sensor_counts, ch_type, picks)
│
├── sorted_epoch_idx = np.argsort(bad_sensor_counts)[::-1]
│
├── n_consensus = self.consensus_[ch_type] * n_channels  # ex: 0.3 * 128 = 38.4
│
└── bad_epochs = (bad_sensor_counts >= n_consensus)
    └── Retourne: masque booléen des epochs à rejeter
```

---

## 5. _interpolate_bad_epochs - Interpolation des epochs

```
_interpolate_bad_epochs(epochs, interp_channels, picks, dots, verbose, n_jobs, use_gpu=False, device=None)
│
├── [Si use_gpu=True]: _interpolate_bad_epochs_gpu(...) → Non utilisé ici
│
└── [Mode CPU séquentiel n_jobs=1]:
    │
    └── BOUCLE sur epochs avec progress bar (300 itérations)
        │
        ├── epoch = epochs[epoch_idx]  # Extraction d'une epoch
        │
        ├── epoch.info['bads'] = interp_chs  # Liste des canaux à interpoler
        │
        ├── utils.interpolate_bads(epoch, dots, picks, reset_bads=True)
        │   │
        │   ├── [EEG] _interpolate_bads_eeg(epoch, picks=eeg_picks_interp)
        │   │        │
        │   │        ├── mne.bem._fit_sphere(pos_good)
        │   │        │   └── Calcule le centre et rayon de la sphère
        │   │        │
        │   │        ├── mne.channels.interpolation._make_interpolation_matrix(pos_good, pos_bad)
        │   │        │   └── Matrice d'interpolation spherical splines
        │   │        │
        │   │        └── mne.channels.interpolation._do_interp_dots(epoch, interpolation, goods_idx, bads_idx)
        │   │            └── epoch._data[bads_idx] = interpolation @ epoch._data[goods_idx]
        │   │
        │   └── [MEG] _interpolate_bads_meg_fast(epoch, picks, dots, mode)
        │
        └── epochs._data[epoch_idx] = epoch._data  # Update in-place
```

---

## 6. bayes_opt - Optimisation bayésienne

```
bayesopt.bayes_opt(f, initial_x, all_x, acquisition=expected_improvement, max_iter=10, debug=False, random_state)
│
├── 6.1  Initialisation
│        │
│        └── BOUCLE sur initial_x (~40 points):
│            │
│            └── y.append(f(x))  # Évalue la fonction objectif
│                 │
│                 └── func(thresh) [closure depuis _compute_thresh]
│                      │
│                      ├── sklearn.model_selection.cross_val_score(_ChannelAutoReject(thresh=thresh), this_data, y, cv)
│                      │   │
│                      │   └── 10 folds de validation croisée
│                      │        │
│                      │        ├── _ChannelAutoReject.fit(X_train)
│                      │        │   │
│                      │        │   ├── backend.ptp(X, axis=1)
│                      │        │   └── _slicemean(X, keep, axis=0)
│                      │        │
│                      │        └── _ChannelAutoReject.score(X_test)
│                      │            │
│                      │            ├── backend.median(X, axis=0)
│                      │            └── -sqrt(mean((median - mean_)^2))
│                      │
│                      └── Retourne: -mean(cv_scores)  # Objectif à minimiser
│
├── 6.2  sklearn.gaussian_process.GaussianProcessRegressor(random_state).fit(X, y)
│
└── 6.3  BOUCLE d'optimisation (max_iter=10 itérations):
         │
         ├── gp.fit(X, y)  # Mise à jour du modèle GP
         │
         ├── acquisition = expected_improvement(gp, best_f, all_x)
         │   │
         │   ├── gp.predict(x[:, None], return_std=True)  # Prédiction GP
         │   │
         │   ├── Z = (y - best_y) / (y_std + 1e-12)
         │   │
         │   └── EI = (y - best_y) * norm.cdf(Z) + y_std * norm.pdf(Z)
         │
         ├── new_x = all_x[acquisition.argmin()]  # Point suivant à évaluer
         │
         └── new_f = f(new_x)  # Évaluation de la fonction
         
└── Retourne: (best_x, best_f)  # Meilleur seuil trouvé
```

---

## 7. Récapitulatif chronologique complet

| # | Fonction | Appelant | Paramètres clés | Retour |
|---|----------|----------|-----------------|--------|
| 1 | AutoReject.fit | User | epochs (300×128×n_times) | self |
| 2 | _handle_picks | AutoReject.fit | picks=None | [0..127] |
| 3 | _check_data | AutoReject.fit | check_loc=True | None |
| 4 | _should_use_gpu | AutoReject.fit | - | (False, 'cpu') |
| 5 | _get_picks_by_type | AutoReject.fit | - | [('eeg', [0..127])] |
| 6 | partial(_compute_thresholds) | AutoReject.fit | method, n_jobs, dots | thresh_func |
| 7 | _run_local_reject_cv | AutoReject.fit (loop) | n_interpolate, consensus, cv | (local_reject, loss) |
| 8 | _AutoReject.fit | _run_local_reject_cv | thresh_func, picks_, dots | self |
| 9 | _compute_thresholds | _AutoReject.fit | method='bayesian_optimization' | threshes_ dict |
| 10 | _clean_by_interp | _compute_thresholds | augment=True | epochs_interp |
| 11 | interpolate_bads (×128) | _clean_by_interp | mode='fast' | epochs |
| 12 | _interpolate_bads_eeg | interpolate_bads | - | None (in-place) |
| 13 | _make_interpolation_matrix | _interpolate_bads_eeg | - | matrix |
| 14 | _do_interp_dots | _interpolate_bads_eeg | - | None (in-place) |
| 15 | StratifiedShuffleSplit | _compute_thresholds | test_size=0.2 | cv object |
| 16 | Parallel(_compute_thresh) | _compute_thresholds | ×128 canaux | threshes list |
| 17 | _compute_thresh | Parallel | method='bayesian_optimization' | best_thresh |
| 18 | get_backend | _compute_thresh | - | NumpyBackend |
| 19 | backend.ptp | _compute_thresh | - | deltas (300,) |
| 20 | bayes_opt | _compute_thresh | max_iter=10 | (best_thresh, best_f) |
| 21 | func(thresh) [closure] | bayes_opt (×~50) | thresh | CV loss |
| 22 | cross_val_score | func | cv=10 folds | scores |
| 23 | _ChannelAutoReject.fit | cross_val_score (×10) | thresh | self |
| 24 | _ChannelAutoReject.score | cross_val_score (×10) | - | -RMSE |
| 25 | GaussianProcessRegressor.fit | bayes_opt (×10) | - | gp model |
| 26 | expected_improvement | bayes_opt (×10) | - | EI values |
| 27 | get_reject_log | _AutoReject.fit | - | RejectLog |
| 28 | _vote_bad_epochs | get_reject_log | - | (labels, bad_sensor_counts) |
| 29 | backend.ptp | _vote_bad_epochs | - | deltas |
| 30 | _get_epochs_interpolation | get_reject_log | n_interpolate | labels updated |
| 31 | _get_bad_epochs | get_reject_log | consensus | bad_epochs mask |
| 32 | _get_interp_chs | _AutoReject.fit | - | interp_channels list |
| 33 | _interpolate_bad_epochs | _AutoReject.fit | use_gpu=False | None (in-place) |
| 34 | interpolate_bads (×300) | _interpolate_bad_epochs | - | None (in-place) |
| 35 | _slicemean | _AutoReject.fit | - | mean_ template |
| 36+ | **CV LOOP** | _run_local_reject_cv | Répétition pour chaque (n_interp, fold, consensus) | loss grid |

---

## 8. Fonctions clés utilisant le backend

| Fonction | Appel Backend | NumPy Implementation |
|----------|---------------|---------------------|
| _vote_bad_epochs | backend.ptp(data, axis=-1) | np.ptp() |
| _get_epochs_interpolation | backend.ptp(epoch_data, axis=-1) | np.ptp() |
| _compute_thresh | backend.ptp(this_data, axis=1) | np.ptp() |
| _ChannelAutoReject.fit | backend.ptp(X, axis=1) | np.ptp() |
| _ChannelAutoReject.score | backend.median(X, axis=0) | np.median() |
| _AutoReject.score | backend.median(X, axis=0) | np.median() |
