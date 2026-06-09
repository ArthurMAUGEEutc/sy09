# Plan d'attaque — `apprentissage_supervisé_oracle.ipynb`

## Objectif

Jusqu'ici, on a prédit les **clusters K-means** — des classes "artificielles", construites par un algorithme non supervisé pour maximiser la séparation dans l'espace ACP. Ici on change de paradigme : on prédit de **vraies variables du dataset** (vérité terrain, "oracle"), pas une étiquette qu'on a nous-même fabriquée.

Intérêt : voir si nos méthodes (KNN, LDA/QDA, régression logistique) se comportent différemment sur de vraies classes — et ce que ça nous apprend en creux sur la nature des clusters K-means (est-ce qu'ils étaient "trop faciles" à prédire parce qu'ils étaient géométriquement optimisés pour ça ?).

---

## 1. Choix des variables cibles

On veut des booléennes **ni triviales ni trop déséquilibrées** (sinon le modèle naïf gagne déjà à ~90%, comme pour `higher`/`internet`/`nursery`/`schoolsup`).

Vérification des proportions de `True` (recalculées) :

| Variable | mat | por | Retenue ? |
|---|---|---|---|
| `famsup` | 61.3% | 61.3% | ✅ — équilibre correct, dispo sur les deux datasets |
| `paid` | 45.8% | 6.0% | ⚠️ — équilibré sur mat seulement ; **mat uniquement** |
| `activities` | 50.9% | 48.5% | ✅ — quasi parfaitement équilibré, sur les deux |
| `romantic` | 33.4% | 36.8% | ✅ — déséquilibre modéré, sur les deux |

Écartées : `schoolsup` (~11%), `higher` (~90-95%), `internet` (~77-83%), `nursery` (~80%) — trop proches d'une classe majoritaire écrasante.

- [ ] Confirmer ces proportions dans le notebook (cellule de rappel/vérif, pas de surprise après nettoyage bool)
- [ ] Pour chaque cible : afficher la répartition des classes avant de se lancer (un déséquilibre 33/67 n'a pas le même impact qu'un 50/50 sur l'interprétation de l'accuracy)

---

## 2. Réutiliser ce qu'on a déjà appris (ne pas repartir de 0)

On reprend telle quelle la méthodologie rodée sur `apprentissage_mat`/`apprentissage_por` :

- [ ] **Sélection de variables** par ANOVA / η² (réutiliser `anova_select()` écrite pour `apprentissage_por`) — entre la variable cible (binaire) et les variables explicatives quantitatives/catégorielles encodées
- [ ] **Vérification des hypothèses** avant chaque modèle :
  - homoscédasticité (heatmaps de covariance + ratio σ_max/σ_min) pour LDA/QDA
  - linéarité des log-odds pour la régression logistique
- [ ] **Mêmes modèles**, dans le même ordre : naïf (classe majoritaire) → KNN → LDA (+ `shrinkage='auto'` si besoin) → QDA (+ `reg_param`) → régression logistique
- [ ] **Mêmes tableaux de bilan** (comparaison accuracy / commentaire sur pourquoi tel modèle marche moins bien)

---

## 3. Point de vigilance spécifique : la fuite de données (data leakage)

Avec les clusters K-means, la fuite n'était pas un risque (le cluster est une fonction de *toutes* les variables). Ici, la cible est **une vraie colonne du dataset** — il faut donc se demander, pour chaque variable cible, s'il existe une feature qui serait un *proxy* trop direct :

- [ ] `paid` (cours payants) : vérifier qu'on n'utilise pas une variable qui encoderait quasi-directement "la famille a les moyens de payer des cours" de façon tautologique (ex. croiser avec `Medu`/`Fedu`/`Mjob`/`Fjob` est OK — c'est le but — mais il faut rester vigilant si une variable s'avère être un alias)
- [ ] `famsup` / `activities` / `romantic` : a priori pas de proxy évident, mais à vérifier via les η² (un η² anormalement élevé, comme `guardian_other` pour les clusters, est un signal d'alerte à creuser avant de l'utiliser)

---

## 4. Attentes à poser avant de lancer les modèles

- [ ] **L'accuracy sera probablement plus basse que sur les clusters** (on était à 85-93% sur les clusters K-means). C'est normal et *informatif* : les clusters K-means sont par construction les groupes les plus séparables dans l'espace ACP, alors que `romantic`/`activities`/`famsup` sont des comportements réels, bruités, pas optimisés pour être linéairement séparables par nos variables.
- [ ] Si l'accuracy est proche du modèle naïf (classe majoritaire), ce n'est pas un échec de méthode : ça veut dire que **la variable cible n'est probablement pas "lisible"** dans les variables dont on dispose (ce qui est en soi une conclusion intéressante pour le rapport).
- [ ] Pour les cibles déséquilibrées (`romantic`, `paid`), ne pas se fier qu'à l'accuracy globale : regarder aussi la matrice de confusion / le rappel par classe (un modèle qui prédit toujours "False" peut avoir 67% d'accuracy sur `romantic` sans rien apprendre).

---

## 5. Idée de croisement de variables (`paid` × `famsup`, etc.)

Proposition de l'utilisateur : au lieu de prédire une seule variable booléenne, croiser deux variables logiquement liées (ex. *soutien scolaire* = `famsup` et/ou `paid`) pour créer une cible plus riche.

- [ ] **D'abord vérifier la cohérence logique du croisement** avant de coder quoi que ce soit : `famsup` (aide aux devoirs par la famille) et `paid` (cours particuliers payants) sont deux formes de soutien scolaire *a priori indépendantes* (une famille peut faire les deux, l'un, l'autre, ou ni l'un ni l'autre) — le croisement a du sens parce qu'il répond à une vraie question : *"la famille soutient-elle l'élève, et sous quelle forme ?"*
- [ ] Construire la cible croisée à 4 modalités : `(famsup=T, paid=T)` / `(famsup=T, paid=F)` / `(famsup=F, paid=T)` / `(famsup=F, paid=F)` — et vérifier l'effectif de chaque modalité avant de se lancer (un croisement peut très vite donner des classes minuscules et inexploitables)
- [ ] Si une modalité est trop petite (< ~30 individus), envisager un repli sur une cible binaire dérivée plus grossière, par exemple : *"au moins une forme de soutien"* (`famsup OR paid`) vs *"aucune"* — à tester seulement sur `mat` puisque `paid` y est équilibré
- [ ] Comparer : est-ce que prédire le croisement est *plus* ou *moins* facile que prédire chaque variable séparément ? (Hypothèse à tester, pas à présupposer — ça peut aller dans les deux sens : plus de modalités = problème plus dur, mais parfois la combinaison capture mieux un "profil familial" cohérent que chaque variable isolée)

---

## 6. Structure proposée du notebook

1. **Intro / rappel** : objectif, différence avec l'approche clusters, méthodo réutilisée (lien vers `apprentissage_mat`/`apprentissage_por`)
2. **Préparation** : chargement des données nettoyées, vérification des proportions des 4 cibles candidates
3. **Section par variable simple** (`famsup`, `activities`, `romantic` sur mat+por ; `paid` sur mat seul) :
   - répartition des classes
   - sélection ANOVA / η²
   - vérif hypothèses
   - KNN / LDA / QDA / LogReg
   - bilan (accuracy + matrice de confusion si déséquilibré)
4. **Section croisement** (`famsup` × `paid`, sur mat) :
   - justification logique + effectifs des modalités
   - modélisation (mêmes étapes)
   - comparaison avec les variables isolées
5. **Bilan global** : tableau récapitulatif de toutes les cibles testées (accuracy, déséquilibre, meilleur modèle)
6. **Conclusion** : qu'est-ce que ça change par rapport aux clusters K-means ? Est-ce que les variables qui structuraient les clusters (G3, Medu/Fedu, failures, Mjob...) permettent aussi de prédire des comportements réels, ou est-ce que les clusters "sur-apprenaient" une structure géométrique sans signification comportementale ?

---

## Ordre d'exécution suggéré

1. `famsup` (équilibré, dispo sur les deux datasets — bon point de départ "facile")
2. `activities` (quasi 50/50 — cas le plus propre statistiquement)
3. `romantic` (déséquilibre modéré — premier vrai test de robustesse)
4. `paid` (mat uniquement — cas particulier à isoler)
5. Croisement `famsup` × `paid`
6. Bilan global + conclusion
