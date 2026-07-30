# Legacy AutoReject - Original CPU Implementation

Ce dossier contient l'implémentation **originale** d'AutoReject, avant les modifications pour le support GPU.

## Fichiers

| Fichier | Description |
|---------|-------------|
| `autoreject_original.py` | Implémentation principale : `AutoReject`, `GlobalAutoReject`, `_compute_thresh`, etc. |
| `utils_original.py` | Utilitaires : `_clean_by_interp`, `interpolate_bads` (via MNE) |
| `bayesopt_original.py` | Optimisation bayésienne pour la recherche d'hyperparamètres |
| `ransac_original.py` | Implémentation RANSAC pour la détection de mauvais canaux |

## Utilisation

Ces fichiers servent de **référence "ground truth"** pour comparer les résultats de l'implémentation GPU.

```python
# Import depuis legacy pour comparaison
from legacy import AutoReject as AutoRejectOriginal
from legacy import _clean_by_interp as clean_by_interp_original

# Import de l'implémentation actuelle (GPU)
from autoreject import AutoReject as AutoRejectGPU

# Comparaison
ar_orig = AutoRejectOriginal()
ar_gpu = AutoRejectGPU()

# Exécuter et comparer les résultats
```

## Important

- **Ne pas modifier** ces fichiers
- Ils représentent l'état du code qui produit les résultats `CPU_Original` dans les benchmarks
- Toute correction doit être faite dans `/autoreject/`, pas ici
