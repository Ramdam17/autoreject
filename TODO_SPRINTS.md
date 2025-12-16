# AutoReject GPU - Roadmap vers PR Officielle

## Objectif Final
Préparer une Pull Request propre pour le repo officiel `autoreject` avec :
1. Implémentation GPU qui accélère significativement le traitement
2. Fallback CPU 100% compatible avec l'implémentation legacy
3. Documentation et tests complets

---

## Sprint 01 - Parité Legacy CPU ✅ (En cours)

**Branche**: `feature/sprint-01-legacy-parity`

**Objectif**: Garantir que le code CPU (avec `AUTOREJECT_BACKEND='numpy'`) produit des résultats **bit-identical** au code legacy officiel.

### Analyse des divergences (Terminé)

| Fichier | Statut | Risque de divergence |
|---------|--------|---------------------|
| `utils.py` | ✅ 100% identique | Aucun |
| `bayesopt.py` | ✅ 100% identique | Aucun |
| `ransac.py` | ⚠️ 1 fonction divergente | Faible (backend abstraction) |
| `autoreject.py` | ⚠️ Appels via backend | Faible avec NumPy backend |

### Tâches

- [ ] **1.1** Créer `autoreject/tests/test_legacy_parity.py`
  - Fixture pour forcer `AUTOREJECT_BACKEND='numpy'`
  - Import du module legacy via `importlib`

- [ ] **1.2** Test `test_autoreject_fit_matches_legacy()`
  - Données synthétiques (32 canaux, 50 epochs, `random_state=42`)
  - Comparer `threshes_`, `bad_epochs_idx`, `reject_log`
  - Assertion `np.testing.assert_array_equal()` pour booléens
  - Assertion `np.testing.assert_allclose(rtol=1e-14)` pour numériques

- [ ] **1.3** Test `test_ransac_correlations_match_legacy()`
  - Comparer `Ransac._compute_correlations()` legacy vs current

- [ ] **1.4** Test `test_compute_thresholds_matches_legacy()`
  - Comparer `compute_thresholds()` legacy vs current

- [ ] **1.5** Test `test_numba_matches_numpy()`
  - Valider que NumbaBackend produit les mêmes résultats que NumpyBackend

- [ ] **1.6** Test fonctions individuelles
  - `test_ptp_backends_match()`
  - `test_median_backends_match()`
  - `test_correlation_backends_match()`

- [ ] **1.7** Exécuter tous les tests et valider

### Critères de succès
- Tous les tests passent avec `pytest autoreject/tests/test_legacy_parity.py`
- Aucune divergence numérique détectée entre legacy et current (NumPy backend)

---

## Sprint 02 - Optimisations CPU (À venir)

**Branche**: `feature/sprint-02-cpu-optimizations`

**Objectif**: Explorer les améliorations de performance CPU sans casser la parité.

### Tâches planifiées

- [ ] **2.1** Pré-calculs de matrices
  - Identifier les matrices recalculées inutilement
  - Cacher les matrices d'interpolation

- [ ] **2.2** Parallélisation Numba
  - Valider que NumbaBackend ne diverge pas
  - Optimiser les boucles critiques avec `@jit(parallel=True)`

- [ ] **2.3** Autres pistes
  - Évaluer `scipy.linalg` vs `numpy.linalg`
  - Évaluer chunking pour gros datasets

- [ ] **2.4** Benchmarks CPU
  - Comparer NumPy vs Numba sur différentes tailles

### Critères de succès
- Tests de parité Sprint 01 toujours verts
- Gain de performance mesurable sur CPU

---

## Sprint 03 - Nettoyage et Documentation (À venir)

**Branche**: `feature/sprint-03-cleanup-docs`

**Objectif**: Préparer le code pour la PR officielle.

### Tâches planifiées

- [ ] **3.1** Nettoyage du code
  - Supprimer code mort
  - Uniformiser le style (PEP8, docstrings)
  - Simplifier les abstractions inutiles

- [ ] **3.2** Documentation
  - Docstrings numpy-style complètes
  - README mis à jour avec instructions GPU
  - Exemples d'utilisation

- [ ] **3.3** Configuration
  - `AUTOREJECT_BACKEND` documenté
  - Fallback behavior documenté

- [ ] **3.4** Tests finaux
  - Couverture de tests > 90%
  - Tests sur CI (Linux, macOS, Windows)

- [ ] **3.5** Préparation PR
  - CHANGELOG
  - Commit history propre
  - Description PR

### Critères de succès
- Code prêt pour review externe
- Documentation complète
- CI verte

---

## Notes Techniques

### Variable d'environnement `AUTOREJECT_BACKEND`

| Valeur | Effet |
|--------|-------|
| `numpy` | Force le backend NumPy (référence legacy) |
| `numba` | Backend Numba avec parallélisation CPU |
| `torch` | Backend PyTorch (CUDA/MPS si disponible) |
| `jax` | Backend JAX (CUDA si disponible) |
| Non défini | Auto-détection (priorité: GPU > Numba > NumPy) |

### Commandes utiles

```bash
# Forcer NumPy backend pour tests de parité
AUTOREJECT_BACKEND=numpy pytest autoreject/tests/test_legacy_parity.py

# Lancer les benchmarks
python benchmarks/run_single.py --config standard_64ch

# Comparer GPU vs CPU
python benchmarks/run_single.py --config standard_64ch --compare
```

---

## Historique

- **2025-12-16**: Création du plan, début Sprint 01
