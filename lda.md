# Interprétation des axes LDA — classe_alcool (3 classes)

**Modèle :** LDA sur `classe_alcool` (Non-consommateur / Modéré / Régulier)  
**Features :** `goout`, `studytime`, `freetime`, `failures`, `absences`, `age`, `Dalc`, `sex_M`  
**Accuracy CV 10-fold :** 68.2%

---

## Variance expliquée par axe

| Axe | Variance expliquée |
|-----|--------------------|
| LD1 | **98.2%** |
| LD2 | 1.8% |

Presque toute la séparation entre classes est capturée par **un seul axe (LD1)**. LD2 n'apporte quasiment rien. Cela signifie que le problème est essentiellement **unidimensionnel** : il existe une direction principale dans l'espace des features qui discrimine les 3 classes, et les deux autres dimensions ne séparent presque pas.

---

## Axe LD1 — Le gradient de consommation (98.2% de la variance)

| Feature | Coefficient | Interprétation |
|---------|-------------|----------------|
| `Dalc` | **+1.595** | Poids dominant — consommation en semaine |
| `goout` | +0.387 | Sorties avec amis |
| `sex_M` | +0.192 | Être un homme |
| `studytime` | −0.136 | Temps d'étude (frein à la consommation) |
| `failures` | +0.076 | Échecs scolaires |
| `freetime` | −0.073 | (effet ambigu, légèrement négatif) |
| `absences` | +0.062 | Absences |
| `age` | −0.027 | (effet quasi nul) |

**LD1 est un axe "profil alcool" :**  
Un score LD1 élevé = fort `Dalc` + beaucoup de sorties + être un homme + peu d'étude → prédit **Régulier**.  
Un score LD1 faible = peu de `Dalc` + moins de sorties + être une femme + plus d'étude → prédit **Non-consommateur**.  
Les Modérés se retrouvent entre les deux.

**`Dalc` est de loin la feature la plus discriminante** (coefficient 4× supérieur à `goout`). Le modèle s'appuie principalement sur la consommation en semaine pour séparer les classes, ce qui est logique puisque `Dalc` entre dans le calcul du `score_alcool` qui définit les classes.

---

## Axe LD2 — Contraste étude/âge vs loisirs (1.8% de la variance)

| Feature | Coefficient | Interprétation |
|---------|-------------|----------------|
| `studytime` | **−0.778** | Temps d'étude (fort poids négatif) |
| `age` | +0.563 | Âge |
| `goout` | +0.423 | Sorties |
| `freetime` | −0.406 | Temps libre |
| `Dalc` | −0.316 | Consommation semaine |
| `failures` | −0.217 | Échecs |

LD2 oppose les élèves **jeunes qui étudient beaucoup et ont peu de temps libre** aux **élèves plus âgés qui sortent plus**. Mais comme il n'explique que 1.8% de la variance discriminante, il ne joue pratiquement aucun rôle dans la classification finale.

---

## Moyennes par classe (features normalisées)

| Classe | goout | studytime | freetime | failures | absences | age | Dalc | sex_M |
|--------|-------|-----------|----------|----------|----------|-----|------|-------|
| Non-consommateur | −0.33 | +0.31 | −0.08 | −0.13 | −0.11 | −0.20 | −0.54 | −0.21 |
| Modéré | −0.03 | −0.07 | −0.11 | −0.04 | −0.01 | +0.08 | −0.24 | −0.09 |
| Régulier | +0.77 | −0.47 | +0.45 | +0.37 | +0.24 | +0.20 | +1.72 | +0.66 |

Le profil **Régulier** est très marqué : score `Dalc` normalisé à +1.72 (très au-dessus de la moyenne), `goout` à +0.77, `sex_M` à +0.66 (majorité d'hommes), et `studytime` à −0.47.  
Le profil **Non-consommateur** est l'exact opposé : peu de `Dalc`, moins de sorties, plus de temps d'étude, légèrement plus de femmes.  
Le profil **Modéré** est proche de zéro sur toutes les dimensions — c'est la classe "moyenne", ce qui explique qu'elle soit la plus difficile à séparer des deux autres.

---

## Conclusion

Le LDA confirme ce qu'on observait dans les analyses exploratoires :
- **`Dalc` est le prédicteur numéro 1**, loin devant tous les autres
- **`goout` est le 2ème signal comportemental pertinent**
- **Le sexe a un rôle significatif** : être un homme pousse vers les classes de consommation élevée
- **`studytime` joue en sens inverse** : étudier davantage est associé à moins de consommation
- La séparation Non-consommateur / Régulier est nette, mais **Modéré reste flou** par construction (classe intermédiaire)
