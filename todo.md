[] Bien regarder les types
[] voir les corrélation pour l'ACP



Question :
[] L'éducation/métier des parents impacte-t-il les notes ?
[] Mère ou père : qui a le plus d'influence ?
[] Cet effet diffère-t-il selon le sexe de l'élève ?
[] Ces tendances sont-elles les mêmes en maths et en portugais ?

Voici la problématique formalisée :

---

## Problématique générale
**Quels facteurs socio-familiaux et comportementaux influencent la réussite académique des élèves ?**

---

## Axes d'analyse

### Axe 1 — Profil parental & réussite
- L'éducation (`Medu`, `Fedu`) et le métier (`Mjob`, `Fjob`) des parents influencent-ils les notes ?
- Mère ou père : qui a le plus d'influence ?
- Quand les niveaux d'éducation des parents divergent, le tuteur légal (`guardian`) a-t-il plus de poids ?
- Cet effet varie-t-il selon le sexe de l'élève ?

### Axe 2 — Contexte socio-démographique
- L'école (`school`), l'âge (`age`), l'adresse (`address` urbain/rural), la taille de la famille (`famsize`) et le statut des parents (`Pstatus`) ont-ils un impact ?
- Le temps de trajet (`traveltime`) pénalise-t-il la réussite ?

### Axe 3 — Comportement & mode de vie
- La consommation d'alcool (`Dalc`, `Walc`) impacte-t-elle les notes ? où même est-ce que si on habite loin, est-ce que ça impacte la consomation d'alcool ?
- Cet effet est-il différent selon le sexe ?

### Axe 4 — Comparaison Maths vs Portugais
- Les tendances observées sont-elles les mêmes dans les deux matières ?
- Les deux populations sont-elles comparables ?


### Axe 5 - Apprentissage Supervisé
- apprendre sur un échantillon avec les étudiants alcoolique puis essayer sur un autre échantillon

#### Suite du modèle LDA (classe_alcool, 3 classes, avec Dalc + sex) :
- [X] Interpréter les axes LDA — coefficients discriminants pour comprendre le poids réel de chaque feature -> PSMR
- [ ] Visualiser la projection LDA — projeter les individus sur les 2 axes discriminants (comme ACP mais optimisée pour séparer les classes)
- [ ] Analyser les erreurs — qui sont les élèves mal classés ? Est-ce systématique (ex : filles Régulières confondues avec Modéré) ?

#### Comparaison avec student_por :
- [ ] Comparer la distribution de Walc/Dalc entre mat et por — est-ce que les classes alcool sont les mêmes dans les deux populations ?
- [ ] Entraîner sur student_mat, tester sur student_por — test de généralisation du modèle
- [ ] Comparer les modèles indépendants — entraîner un LDA sur chaque dataset et comparer les accuracies et coefficients

---

**Variable cible principale :** `G3` (note finale), avec `G1` et `G2` pour observer la progression.


Faire l'ACP sur G3 et G1,G2 pour les analyses en rapport avec la note 


on va essayer freetime goout traveltime absences failures et studytime pour commencer. A terme j'aimerais bien recuperer ces paramètre et pourquoi pas d'autre (qu'on a pas mis dans le tableau de corrélation) pour faire un apprentissage supervisé pour apprendre sur le style "non-alcoolique, modéré, alcoolique". On va faire ça après




04/06

[ ] Regarder comment les cluster se sont redistribué entre K=3 et K=4,5
[X] Regarder la théorie de corrélation entre romantic et absence. 
[ ] Regarder en fonction du métier des parents
[ ] Faire d'autre visualisation pour mieux voir les nuages de points
[ ] Rajouter l'indice de rand entre quanti et toute les variables pour mat et por pour montrer l'impact 