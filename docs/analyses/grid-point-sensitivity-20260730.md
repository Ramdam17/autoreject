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

**2. Le mécanisme que j'avais avancé pour F1/F2 ne tient pas non plus.**
Un arrondi de 1e-7 ne peut pas déplacer un point de grille AutoReject, à aucune
échelle testable, y compris celle de l'étude. La marge protège la sélection d'un
facteur ≥ 20.

## Ce qui reste établi, et ce qui ne l'est plus

**Établi (mesuré).** Le correctif de dtype est réel et utile : sur un device
float64, `compute_thresholds_gpu` passe de 0/32 à **32/32 bit-exact** contre la
référence figée de `legacy/`, médiane 3.73e-08 → 0. Ça supprime une divergence
inutile et rend le chemin GPU reproductible contre la référence. C'est une
justification de **reproductibilité**, pas de changement de décision.

**Plus établi.** Que ce correctif explique les écarts de 17–29 % d'epochs
retenues mesurés par Guillaume. Je l'ai laissé entendre ; la mesure l'exclut par
ce mécanisme-là.

## Où chercher ensuite

L'hypothèse restante que ces mesures rendent la plus plausible — et qui est,
elle aussi, à mesurer avant d'être affirmée — est que la perturbation n'entre pas
par une sélection discrète mais par un chemin **sans marge** :

AR1 interpole des canaux même quand il ne rejette rien (Guillaume l'observe
lui-même sur `04memory1` : 0/81 rejeté, `n_interpolate_=32`). L'erreur
d'interpolation est *directe*, pas médiée par un écart best-vs-2e. Ce signal
interpolé alimente ICA, qui est une optimisation itérative : rien ne garantit
qu'un changement d'entrée de 1e-7 mène au même optimum local. Une décomposition
différente donne des composantes différentes, donc des exclusions ICLabel
différentes autour du seuil 0.5, et AR2 voit alors des données *réellement*
différentes — une grosse perturbation, plus 1e-7. Ce qui est cohérent avec le
constat de Guillaume lui-même : AR2 *« is the amplifier, never the source »*.

Cette chaîne n'est pas démontrée ici. Ce qui la testerait : instrumenter le
pipeline étage par étage sur les données de l'étude et mesurer l'amplitude de la
divergence à l'entrée d'ICA, pas seulement à sa sortie. Côté PPSP.

## Reproduction

`/scratchpad/bayesopt_sensitivity.py` et `sensitivity_sweep.py`. À porter dans
`Benchmark/` versionné (Task 2-3) — ces mesures doivent être rejouables, c'est
précisément le reproche fait à `test_variance.py`.
