# Réconciliation des branches du fork — 2026-07-29

Task 1 du plan `ok-il-est-temps-shimmying-petal`. Read-only : aucune branche n'a été
modifiée. Se termine sur une recommandation qui attend validation.

## 1. Topologie réelle

Quatre branches portent du travail GPU. Elles ne sont pas dans une relation linéaire.

| Branche | HEAD | Rôle | Statut |
|---|---|---|---|
| `origin/main` | `b369f88` | **Ce que PPSP installe** (`uv.lock:65`) | 48 commits d'historique GPU |
| `main` (local) | `8ed59f5` | Branche PR-ready, historique squashé | 9 devant / 48 derrière `origin/main` |
| `origin/fix/gpu-pipeline-divergence` | `dd9b2e7` | **Infra de validation, jamais mergée** | +9760 / −2156 vs `origin/main` |
| `origin/feat/gpu-kernel-exploration` | `51677fc` | Kernels CUDA + benchmark Narval A100 | exploration |
| `upstream/main` | `b4e218e` | Amont | 6 commits en attente |

`main` local et `origin/main` sont deux **reconstructions parallèles du même travail**,
pas deux points d'une même ligne : local = `upstream@16be71c` + 5 commits squashés
(backends → interpolation → seuils/CV → API → docs) ; `origin/main` = les 48 commits
de développement. Diff sur les 4 fichiers GPU : +422 / −514, soit 106 hunks
(`backends.py` 33, `autoreject.py` 53, `gpu_pipeline.py` 10, `gpu_interpolation.py` 10).

**Conséquence sur l'issue #1 :** elle a été déposée en lisant le checkout local.
J'ai vérifié que les constats tiennent aussi dans `b369f88` (même
`StratifiedShuffleSplit`, même code mort `compute_thresh_gpu`, même `_to_tensor`
float32) — mais tout correctif doit viser `origin/main`, seule branche dont les
résultats d'étude dépendent.

## 2. Correction : F1 et F2 ne sont pas le même genre de problème

Le plan les présentait tous deux comme une incohérence de dtype. L'historique dit
autre chose, et la distinction change quoi faire.

### F2 — décision assumée, prémisse falsifiée depuis

Deux commits successifs, en décembre 2025 :

- `3c66149 fix(mps): use CPU float64 for interpolation and score computation`
  ajoute un aller-retour CPU pour calculer l'interpolation et le score en float64 sur
  MPS. Message : *« exact numerical conformity with CPU legacy at ~1% performance cost.
  **CUDA path unchanged (float64 natively supported)** »*.
- `69fd2b2 perf: remove unnecessary CPU float64 transfers for MPS` **retire** cette
  conformité, un mois plus tard. Message : *« Variance testing showed that GPU float32
  results are within the natural variance of the algorithm, so CPU float64 conformity
  transfers are unnecessary. »* Gain : 10.14 s → 4.86 s (−52 %), speedup 14× → 29×.

Le raisonnement de `69fd2b2` **est exactement l'argument de Rémy du 29/07 à 16h31** :
si la perturbation float32 reste sous la variance naturelle de l'algorithme, elle
n'est pas un problème. Et il était **juste à l'échelle où il a été mesuré** —
autoreject seul.

Ce que les données de Guillaume ajoutent, c'est que cette échelle n'est pas la bonne.
En aval d'autoreject, ICA et ICLabel sont des étages **discrets** : ICLabel tranche à
`min_probability_for_exclusion: 0.5`, et AR2 sélectionne un point de grille. Une
perturbation sous la variance de l'étage 1 peut franchir un seuil de décision à
l'étage 3. La prémisse n'est pas fausse, sa **portée** l'est.

Le test qui a servi de justification, `test_variance.py` (91 lignes, ajouté par
`69fd2b2`), **n'existe dans aucun arbre final** — supprimé avant `b369f88`. La
décision est documentée, sa preuve ne l'est plus.

→ Corriger F2 est une **décision méthodologique de Rémy**, pas une correction de bug :
on rend 52 % de performance MPS pour de la conformité. Ce plan la mesure, ne la tranche pas.

### F1 — régression CUDA non documentée, elle

`GPUThresholdOptimizer._to_tensor` (`gpu_pipeline.py:137-138`, **identique dans
`b369f88`**) force `torch.float32` quand `dtype=None`, et `compute_thresholds_gpu`
l'appelle sans `dtype`. Ça date des tout premiers commits GPU (`ff87e75`, `e5c4839`),
donc **avant** les deux commits ci-dessus.

D'où une contradiction nette : `3c66149` affirme *« CUDA path unchanged (float64
natively supported) »*, et `backends.py:784-792` sélectionne bien float64 pour CUDA —
mais la recherche de seuils, elle, n'a **jamais** été en float64 sur CUDA. Le travail
de conformité de l'auteur reposait sur une hypothèse que `_to_tensor` invalidait
silencieusement.

C'est le point le plus exploitable du diagnostic, et il est spécifique à
l'environnement où la divergence a été observée : **PPSP tourne sur CUDA**
(`torch v2.11.0+cu130` dans les logs de Guillaume), où float64 est gratuit et déjà
sélectionné partout ailleurs dans le même pipeline. Sur ce device, le float32 de la
recherche de seuils est une perte sèche sans contrepartie de performance justifiée
ni documentée.

→ Corriger F1 sur CUDA n'a pas d'arbitrage : ni le coût de MPS, ni la justification
de `69fd2b2` ne s'y appliquent.

## 3. Correction : l'infra de validation n'est pas perdue

F9 du plan disait les sources du framework perdues. C'est vrai pour **un** framework
et faux pour l'autre — il y en a deux, de noms quasi identiques :

| | `Benchmark/` (majuscule, local) | `benchmarks/` (minuscule, sur `fix/gpu-pipeline-divergence`) |
|---|---|---|
| suivi par git | **non** (0 fichier) | oui |
| sources | **perdues** (`__pycache__` seul) | présentes |
| contenu | 200+ `results/*.json`, sweep 15 seeds | `run_all.py`, `run_single.py`, `config.yaml`, `generate_report.py` |

Et cette branche porte, en plus, exactement ce que `numerical-testing` prescrit et
que le plan proposait d'écrire :

- **`legacy/`** — implémentation CPU d'origine **figée** (`autoreject_original.py`,
  `utils_original.py`, `bayesopt_original.py`, `ransac_original.py`), documentée comme
  référence ground-truth. C'est la « reference implementation » de la Task 10, déjà écrite.
- **`autoreject/tests/references/*.npz`** — fixtures versionnées `v1` pour
  `_vote_bad_epochs`, `_compute_thresholds`, `Ransac.fit`, `_interpolate_bad_epochs`,
  `local_reject_cv`, avec `tools/generate_references.py` pour les régénérer et un
  README qui interdit de les modifier sans changement algorithmique intentionnel.
- **`autoreject/tests/test_retrocompat.py`** — 545 lignes de tests de non-régression.
- **`POTENTIAL_BUG_SPHERE_CENTERING.md`** — l'enquête sphere-centering déjà écrite
  (conclusion cohérente avec F7 : le revert `dd9b2e7` était correct).

→ Les Tasks 2, 3 et 10 deviennent **récupérer et compléter**, plus **reconstruire**.
Économie substantielle, et surtout : cette infra est meilleure que ce que j'aurais
écrit, parce qu'elle gèle une vraie référence CPU au lieu d'en réimplémenter une.

## 4. Nouveau constat — F10 : la suite retrocompat ne peut pas détecter F1/F2

En lisant `test_retrocompat.py`, deux angles morts qui expliquent pourquoi le problème
a survécu à une suite de tests conçue exprès pour l'attraper :

1. **Toutes les tolérances continues sont `rtol=1e-5`** (lignes 269, 294, 338, 450,
   509, 542). C'est la tolérance float32 de `numerical-testing`. La suite a été
   calibrée *pour accepter* la divergence float32 — elle passe avec F1 et F2 en place,
   par construction. Elle a été écrite dans la même période que la décision `69fd2b2`,
   ce qui est cohérent, mais la rend structurellement aveugle à ce qu'on cherche.
2. **Les sorties discrètes ne sont pas asservies.** Seul `vote_bad_epochs` utilise
   `assert_array_equal` (sur `labels`). `consensus_`, `n_interpolate_` et les ensembles
   d'epochs rejetées passent uniquement par `assert_allclose` sur des grandeurs
   continues, ou pas du tout. Or c'est précisément là que Guillaume mesure la
   divergence : point de grille AR2 différent dans 11/14 tâches.

C'est la même leçon que ses tables : `assert_allclose` valide « les nombres sont
proches », jamais « les décisions sont identiques ». Pour une méthode dont la sortie
est un ensemble d'epochs et deux hyperparamètres discrets, le second est le seul
critère qui compte.

## 5. Recommandation

**Base canonique : `origin/main` (`b369f88`)** — c'est ce que `uv.lock` installe ;
corriger ailleurs ne change rien aux résultats de l'étude.

Séquence proposée, dans cet ordre :

1. **Merger `origin/fix/gpu-pipeline-divergence` dans `origin/main`** (ou en cherry-pick
   sélectif : `legacy/`, `tests/references/`, `test_retrocompat.py`, `benchmarks/`).
   Récupère l'infra sans toucher au code numérique. Faible risque, gros gain.
2. **Corriger F10 d'abord** — durcir la suite récupérée : égalité exacte sur les
   sorties discrètes, tolérances par device au lieu d'un `rtol=1e-5` uniforme. Sans
   ça, aucun correctif suivant n'est vérifiable.
3. **Corriger F1** (float64 sur CUDA/CPU dans `_to_tensor`). Pas d'arbitrage.
4. **Mesurer F2** (Tasks 5-6) et te laisser trancher MPS float32 vs conformité.
5. **Cherry-pick `29bec2a`** (F4, `optimizer=None`).
6. **Re-dériver `main` local** depuis la base corrigée pour la PR upstream.

**Aucun rebase ni force-push effectué** : réécrire l'historique d'une branche que
`uv.lock` épingle casserait la résolution de dépendance côté PPSP. L'étape 1 demande
ton go.

## 6. Points ouverts

- `origin/feat/gpu-kernel-exploration` (kernels CUDA + benchmark A100) n'est pas
  audité ici. À faire si cette voie doit être reprise.
- Les 106 hunks `origin/main` ↔ `main` local ne sont pas classés un par un
  cosmetic/semantic/feature. Devenu secondaire : la recommandation est de prendre
  `origin/main` comme base et de re-dériver le local, ce qui rend le classement
  inutile. À reprendre seulement si tu préfères l'inverse.
