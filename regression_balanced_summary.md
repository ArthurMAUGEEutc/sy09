# Regression supervisee sur variables quantitatives equilibrees

## Objectif

La demande etait d'ajouter un modele de regression en evitant les variables/cibles trop desequilibrees. Si une classe contient presque tous les eleves, un modele peut etre "precis" en predisant toujours la classe majoritaire, sans apprendre de structure interessante.

Ici, on evite donc la cible classique `G3 >= 10` :

| Dataset | Echec (`G3 < 10`) | Reussite (`G3 >= 10`) | Part majoritaire |
|---|---:|---:|---:|
| Maths | 130 | 265 | 67.1% |
| Portugais | 100 | 549 | 84.6% |

On utilise plutot une cible logistique equilibree : `G3 >= mediane` dans chaque matiere.

| Dataset | Seuil | Classe basse | Classe haute | Part classe haute |
|---|---:|---:|---:|---:|
| Maths | 11 | 186 | 209 | 52.9% |
| Portugais | 12 | 301 | 348 | 53.6% |

## Variables retenues

Selection quantitative retenue : `age`, `Medu`, `Fedu`, `studytime`, `famrel`, `freetime`, `goout`, `Walc`, `health`, `absences`.

Variables exclues car trop concentrees dans une seule modalite :

| Variable | Maths | Portugais | Raison |
|---|---:|---:|---|
| `traveltime` | 65.1% en modalite 1 | 56.4% en modalite 1 | trop de trajets courts |
| `failures` | 79.0% en modalite 0 | 84.6% en modalite 0 | presque tout le monde sans echec |
| `Dalc` | 69.9% en modalite 1 | 69.5% en modalite 1 | consommation semaine tres faible pour la majorite |

## Modeles entraines

Deux versions sont produites :

- `context_only` : seulement les variables quantitatives equilibrees ci-dessus. C'est le modele interpretable.
- `with_G1_G2` : ajoute `G1` et `G2`. C'est le benchmark predictif, car les notes precedentes sont naturellement les meilleurs predicteurs de `G3`.

### Regression lineaire sur `G3`

| Dataset | Modele | MAE baseline | MAE modele | R2 |
|---|---|---:|---:|---:|
| Maths | context_only | 3.69 | 3.47 | 0.122 |
| Maths | with_G1_G2 | 3.69 | 1.35 | 0.806 |
| Portugais | context_only | 2.45 | 2.22 | 0.171 |
| Portugais | with_G1_G2 | 2.45 | 0.76 | 0.876 |

### Regression logistique sur `G3 >= mediane`

| Dataset | Modele | Baseline majoritaire | Accuracy test | Balanced accuracy | ROC-AUC | CV accuracy |
|---|---|---:|---:|---:|---:|---:|
| Maths | context_only | 0.529 | 0.555 | 0.555 | 0.560 | 0.582 |
| Maths | with_G1_G2 | 0.529 | 0.933 | 0.933 | 0.981 | 0.914 |
| Portugais | context_only | 0.538 | 0.667 | 0.665 | 0.738 | 0.666 |
| Portugais | with_G1_G2 | 0.538 | 0.928 | 0.929 | 0.982 | 0.921 |

## Interpretation courte

Le modele `context_only` reste modeste, surtout en mathematiques : les variables sociales et comportementales seules expliquent peu la note finale. En portugais, le signal est plus clair : education parentale (`Fedu`, `Medu`) et `studytime` augmentent la probabilite d'etre au-dessus de la mediane, tandis que `Walc` et `absences` la reduisent.

Avec `G1` et `G2`, les deux regressions deviennent tres performantes. `G2` est le predicteur dominant, puis `G1`, ce qui est coherent : la note finale suit fortement la trajectoire scolaire deja observee.

Conclusion pour la soutenance : on peut presenter `context_only` pour l'interpretation des facteurs socio-comportementaux, puis `with_G1_G2` comme modele predictif principal. La cible est volontairement equilibree, donc les scores ne viennent pas d'une simple prediction de la classe majoritaire.

## Fichiers generes

- Code : `D04_regression_balanced.py`
- Metriques : `regression_outputs/D04_regression_metrics.csv`
- Selection des variables : `regression_outputs/D04_selected_predictors.csv`
- Variables exclues : `regression_outputs/D04_excluded_predictors.csv`
- Equilibre cible : `regression_outputs/D04_target_balance.csv`
- Coefficients lineaires : `regression_outputs/D04_linear_coefficients.csv`
- Odds ratios logistiques : `regression_outputs/D04_logistic_odds_ratios.csv`
- Figures : `figure/D04_target_balance.png`, `figure/D04_linear_coefficients_context_only.png`, `figure/D04_linear_coefficients_with_G1_G2.png`, `figure/D04_logistic_coefficients_context_only.png`, `figure/D04_logistic_coefficients_with_G1_G2.png`
