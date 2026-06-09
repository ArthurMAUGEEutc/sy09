# Modèles d'apprentissage supervisé — checklist

Notebook cible : `apprentissage.ipynb`

---

## Idées à explorer

- [ ] **1. Prédire le cluster K-means (k=4, 42 variables)**
  - Cible : labels K-means (4 archétypes)
  - Features : variables interprétables (goout, studytime, Medu, sex, higher, internet, romantic, absences, failures...)
  - Modèles : KNN + LDA
  - Objectif : trouver une règle de décision lisible pour chaque archétype

- [ ] **2. Prédire le décrochage scolaire (binaire)**
  - Cible : `à_risque = 1` si G3 < 10 OU failures ≥ 2
  - Features : comportementales + familiales (sans G1, G2)
  - Modèles : KNN + LDA
  - Objectif : identifier les signaux d'alerte précoce du décrochage

- [ ] **3. Généralisation mat → por**
  - Entraîner sur student-mat, tester sur student-por
  - Mesurer la chute de performance (accuracy train vs test cross-dataset)
  - Cible : classe_alcool ou G3 discrétisé
  - Objectif : tester si les patterns appris en maths se généralisent au portugais

- [ ] **4. Prédire G3 — features socio-familiales uniquement**
  - Cible : G3 (régression ou classification 3 classes)
  - Features : Medu, Fedu, Mjob, Fjob, studytime, failures, absences, higher, internet (sans G1, G2, sans alcool)
  - Modèles : régression linéaire + LDA
  - Objectif : isoler la contribution du capital familial hors trajectoire scolaire

- [ ] **5. Prédire classe_alcool sans data leakage**
  - Cible : classe_alcool (3 classes)
  - Features : goout, studytime, freetime, absences, failures, sex, age, address, internet, romantic — **sans Dalc ni Walc**
  - Modèles : KNN + LDA
  - Objectif : version propre de la prédiction d'alcool, sans fuite de données
