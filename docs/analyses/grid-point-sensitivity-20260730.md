# Sensibilité du point de grille — 2026-07-30

Mesure destinée à tester **mon propre mécanisme** : j'ai affirmé qu'un arrondi
float32 (~1e-7 relatif) pouvait déplacer le point de grille sélectionné par
`bayes_opt`, et que c'était par là que la divergence de backend entrait dans la
cascade. **La mesure ne soutient pas cette affirmation.** Elle est consignée ici
parce qu'une hypothèse falsifiée oriente la suite autant qu'une confirmée.

## Ce qui a été mesuré

Sur des surfaces de loss réelles produites par
`batched_all_channels_cv_loss_parallel` (donc les vraies losses CV, pas un
modèle jouet), deux quantités :

1. **La marge** : écart relatif entre le meilleur et le deuxième meilleur point
   de grille. C'est elle qui décide si une perturbation peut faire basculer une
   sélection.
2. **Le taux de basculement** effectif, en perturbant les losses de façon
   multiplicative et en relançant `bayes_opt`.

## Marge best-vs-2e, par échelle

| config | pts de grille | médiane | p05 | min | canaux < 1e-7 |
|---|---|---|---|---|---|
| synthétique 32 ch / 30 ep | 30 | 2.44e-03 | 4.20e-04 | 2.83e-04 | 0/32 |
| MNE sample EEG 60 ch / 120 ep | 120 | 2.23e-03 | 6.24e-05 | 1.86e-05 | 0/60 |
| synthétique 64 ch / 120 ep | 120 | 6.85e-04 | 9.18e-05 | 4.19e-05 | 0/64 |
| synthétique **128 ch / 182 ep** | 182 | 4.73e-04 | 2.07e-05 | **2.48e-06** | 0/128 |
| synthétique 128 ch / 400 ep | 400 | 1.98e-04 | 1.32e-05 | **2.20e-06** | 0/128 |

La marge se resserre bien avec l'échelle — la grille s'affine quand il y a plus
d'epochs — mais elle reste **au minimum 2.2e-06**, soit environ 20× au-dessus de
l'arrondi float32. **Aucun canal, dans aucune configuration, n'a de marge
inférieure à 1e-7.** La ligne 128 ch / 182 ep correspond au régime de l'étude
(EGI 128, 182 epochs sur `01restEyesOpen`).

## Taux de basculement, par amplitude de perturbation

| amplitude | tel quel (le GP refit son noyau) | `optimizer=None` (29bec2a) | `argmin` simple |
|---|---|---|---|
| **1e-07** | **0/60** | **0/60** | **0/60** |
| 1e-05 | 0/60 | 0/60 | 0/60 |
| 1e-03 | 8/60 | 6/60 | 5/60 |
| 1e-02 | 33/60 | 36/60 | 35/60 |

(MNE sample EEG, 60 canaux ; le jeu synthétique 32 ch donne la même forme.)

## Deux conclusions

**1. F4 n'est pas un amplificateur — mais c'est bien un bug, pour une autre
raison.** Je présentais le fix upstream `29bec2a`
(`GaussianProcessRegressor(optimizer=None)`) comme *« un amplificateur numérique
placé exactement dans l'étage que Guillaume identifie comme première divergence
dans 6/14 tâches »*. C'était un raisonnement, pas une mesure, et il est faux :
`optimizer=None` ne réduit pas le taux de basculement (6 vs 8 à 1e-3, 36 vs 33 à
1e-2 — du bruit, aucune amélioration systématique).

Il s'est trouvé qu'il corrige autre chose, découvert par accident en lançant
`test_retrocompat` et `test_backend_parity` dans cet ordre : **7 tests sur 20
échouaient** sur

```
sklearn.exceptions.ConvergenceWarning: The optimal value found for dimension 0
of parameter k1__constant_value is close to the specified lower bound 1e-05.
```

Le `conftest` du projet transforme les warnings en erreurs (`error::`), et
`expected_improvement` n'entoure d'un `simplefilter("ignore")` que `gp.predict`,
pas `gp.fit`. L'optimisation d'hyperparamètres du noyau émet donc un warning qui
remonte, de façon dépendante de l'ordre d'exécution. `optimizer=None` supprime
l'optimisation, donc le warning : **20/20 passent** après cherry-pick.

C'est une fragilité réelle de la suite, et le fix upstream est le bon. Mais la
raison n'a rien à voir avec celle que j'avais avancée.

**Réserve à garder en tête :** `29bec2a` est un changement algorithmique
intentionnel en amont. Vérifié ici que le chemin numpy reste **bit-exact** contre
les références figées après cherry-pick — donc la sélection ne change pas sur ce
jeu de 32 canaux. Ce n'est pas une garantie générale : les `.npz` ont été générés
avec l'ancien GP. À revérifier sur les données de l'étude avant de considérer que
le changement est neutre là aussi.

**2. Le mécanisme que j'avais avancé pour F1/F2 ne tient pas — sous cette forme.**
Perturber les *losses* de 1e-7 ne déplace pas un point de grille : la marge
protège la sélection d'un facteur ≥ 20. **Mais c'était la mauvaise perturbation à
tester.** Voir la section suivante, qui corrige cette conclusion.

---

## Correction — le mécanisme existe, par un autre chemin (mesuré le 30/07)

La conclusion ci-dessus comparait deux quantités mal appariées. J'ai perturbé les
losses de 1e-7 en gardant les données fixes. Or dans la vraie divergence de
backend, ce ne sont pas les losses qui bougent de 1e-7 : ce sont **les données**,
et elles déplacent *toute la surface* de loss. Une marge best-vs-2e de 2.2e-06
n'est pas une protection contre ça — c'est au contraire ce qui rend le
basculement possible, puisqu'un déplacement de surface du même ordre suffit.

Découvert en écrivant un test qui exerce réellement `AutoReject.fit()` sur le
chemin GPU (60 epochs, device forcé, espion sur
`run_local_reject_cv_gpu_batch`) : **il échoue**, et de façon reproductible.

### Ce que le test trouve

Épochque 55, sur 1920 éléments de `reject_log.labels`, exactement 2 diffèrent :

| | numpy (float64) | MPS (float32) |
|---|---|---|
| `labels[55, 2]` (EEG003) | **2** = interpolé | **0** = bon |
| `labels[55, 24]` (EEG025) | **1** = mauvais, non interpolé | **2** = interpolé |

`consensus_` identique, `n_interpolate_` identique, **ensemble des epochs
rejetées identique**. Seules les décisions par canal diffèrent : le même epoch
survit dans les deux cas, avec un canal différent interpolé. **Les données
retenues ne sont pas les mêmes.** C'est exactement le `03firstName` de Guillaume
(29 = 29 epochs mais `n_interpolate_` 4 vs 16), reproduit en test unitaire.

### La cause, vérifiée

Deux hypothèses testées et **écartées** avant d'arriver à la bonne :

- *L'argsort des PTP par canal dans `_get_epochs_interpolation` basculerait.* Non :
  le classement est identique en float32 et float64 (EEG003 rang 0, EEG025 rang
  14 dans les deux cas), et l'écart relatif minimum entre canaux adjacents sur
  les 60 epochs est 4.3e-07, avec **0/1860 sous 1e-7**.
- *La comparaison `delta > thresh` basculerait sur un epoch à la frontière.* Non
  pour EEG025, dont les deux seuils ne diffèrent que de 9.5e-09 et qui est classé
  mauvais des deux côtés.

La vraie cause est sur EEG003, et elle est d'une autre nature :

```
EEG003, seuil sélectionné    numpy = 2.030141815181105e-04
                               MPS = 1.414362923242152e-04
                    écart relatif = 3.03e-01     <-- 30 %, pas 1e-8
delta (PTP epoch 55)              = 1.771477136274092e-04   -> ENTRE les deux
```

Les deux backends ont sélectionné des **points de grille substantiellement
différents**, et le PTP de l'epoch 55 tombe entre les deux — d'où le
basculement, puis le changement de canal interpolé par effet de bord dans le même
epoch.

### Fréquence, et pourquoi `augment` est indispensable

| n_epochs | `augment=False` | `augment=True` |
|---|---|---|
| 30 | 0/32 canaux, max 9.9e-08 | 0/32, max 8.5e-08 |
| 60 | 0/32, max 8.5e-08 | **1/32, max 3.03e-01** |
| 120 | 0/32, max 6.9e-08 | 0/32, max 1.4e-07 |

Le saut n'apparaît **jamais** sans augmentation, à aucune taille. Il exige
`augment=True`, qui est le mode du vrai pipeline : la moitié interpolée des
données est construite en float32 sur MPS, ce qui perturbe les données d'environ
5e-07 relatif, ce qui déplace la surface de loss du même ordre que la marge
inter-points (2.2e-06). Rarement — 1 canal sur 32 dans une configuration sur
trois — cet écart suffit à envoyer l'argmin sur un point de grille lointain.

Rare par canal, mais sur 128 canaux × 14 tâches ce n'est pas négligeable, et
chaque occurrence se propage en aval.

### Le garde-fou existant ne peut pas y remédier

`_vote_bad_epochs` contient déjà une mitigation (`autoreject.py:607-620`) :

```python
if use_f32_thresh:   # torch + mps
    thresh = np.nextafter(np.float32(thresh), np.float32(np.inf))
```

Son commentaire identifie le bon phénomène — *« MPS only supports float32, which
can cause edge-case flips when comparing to float64 thresholds »*. Mais elle
arrondit la **représentation** du seuil, alors que le problème est que la
**valeur** du seuil a été calculée à partir de données différentes. Arrondir
1.41e-04 au float32 suivant ne le rapproche pas de 2.03e-04.

### Conséquence sur la valeur du correctif de dtype

Elle est **plus grande** que ce que je concluais, et pour CUDA précisément.
Sur CUDA, `gpu_interpolation` calcule déjà en float64 (`gpu_interpolation.py:797-801`)
et, avec le correctif, la recherche de seuils aussi. Les deux sources de
perturbation disparaissent donc, et ce mécanisme ne peut plus se déclencher —
alors que sur MPS le float32 de l'interpolation est une contrainte matérielle et
le mécanisme subsiste. C'est consigné en `xfail` documenté sur MPS et en
assertion stricte sur CUDA dans `test_fit_gpu_path_matches_per_channel_labels`.

**Ceci reste une inférence pour CUDA**, faute de matériel ici. C'est l'assertion
que le test fera tomber, ou pas, sur Narval.

## Bilan — ce qui est établi

1. **Le correctif de dtype est mesuré et utile.** Sur un device float64,
   `compute_thresholds_gpu` passe de 0/32 à **32/32 bit-exact** contre la
   référence figée de `legacy/` (médiane 3.73e-08 → 0).
2. **Le chemin non augmenté est robuste.** 0/32 canaux divergent au-delà de
   l'arrondi, à toutes les tailles testées. Perturber les losses de 1e-7 ne
   déplace aucun point de grille (marges ≥ 2.2e-06).
3. **Le chemin augmenté — celui du vrai pipeline — ne l'est pas.** L'interpolation
   float32 déplace la surface de loss du même ordre que la marge inter-points, et
   envoie rarement l'argmin sur un point lointain (1/32 canaux à 60 epochs, saut
   de 30 %). Conséquence : mêmes epochs retenues, **canaux interpolés
   différents**, données retenues différentes.
4. **Le mécanisme de Guillaume est reproduit en test unitaire.** Son
   `03firstName` (29 = 29 epochs, `n_interpolate_` 4 vs 16) a le même profil que
   l'épochque 55 mesurée ici.
5. **Sur CUDA le mécanisme devrait disparaître** — interpolation et recherche de
   seuils toutes deux en float64. Inféré, pas mesuré : c'est ce que le test
   tranchera sur Narval.

## Ce qui reste ouvert

Ce document établit un mécanisme **suffisant** pour produire des données retenues
différentes. Il n'établit pas qu'il est **le seul**, ni qu'il suffit à expliquer
des écarts de 17–29 % d'epochs après AR2 : ici l'ensemble des epochs rejetées
reste identique, seul le contenu change. Le passage de « contenu différent » à
« 20 epochs de plus ou de moins » demande la traversée ICA → ICLabel, où une
décomposition différente change les composantes exclues autour du seuil 0.5.

Ce qui le testerait : instrumenter le pipeline étage par étage sur les données de
l'étude et mesurer l'amplitude de la divergence **à l'entrée** d'ICA, pas
seulement à sa sortie. Côté PPSP.

## Deux erreurs de méthode à retenir

- **J'ai perturbé la mauvaise quantité.** Tester la sensibilité en bruitant les
  losses, données fixes, ne dit rien sur une divergence qui vient des données.
- **Un test qui passe ne prouve rien s'il n'exerce pas le chemin.** La première
  version du test de décisions discrètes passait sans jamais toucher le GPU :
  30 epochs < le seuil de 50 de `should_use_gpu`, et `AUTOREJECT_BACKEND=numpy`
  fuité par l'import de `test_retrocompat`. Il faut espionner l'appel et
  asserter que la porte s'est ouverte.

## Reproduction

`/scratchpad/bayesopt_sensitivity.py` et `sensitivity_sweep.py`. À porter dans
`Benchmark/` versionné (Task 2-3) — ces mesures doivent être rejouables, c'est
précisément le reproche fait à `test_variance.py`.
