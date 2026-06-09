# Description des variables

| Variable | Description | Explication |
|----------|-------------|-------------|
| school | École de l'élève | Binaire: 'GP' : Gabriel Pereira ; 'MS' : Mousinho da Silveira|
| sex | Sexe de l'élève | Binaire: 'F' - Female ; 'M' Male |
| age | Âge de l'élève ||
| address | Type d'adresse du domicile | Binaire: 'U' : urban or 'R' : rural|
| famsize | Taille de la famille | Binaire : LE3 : <= 3 ; GT3 > 3 |
| Pstatus | Statut de cohabitation des parents | Binaire: 'T' : Living Together or 'A' : Appart|
| Medu | Niveau d'éducation de la mère | [0:4] |
| Fedu | Niveau d'éducation du père | [0:4] |
| Mjob | Emploi de la mère | teacher, other, services, health, at_home|
| Fjob | Emploi du père | teacher, other, services, health, at_home|
| reason | Raison du choix de l'école | course, other, home, reputation |
| guardian | Tuteur légal de l'élève | mother, father, other |
| traveltime | Temps de trajet domicile → école |
| studytime | Temps de travail hebdomadaire |
| failures | Nombre d'échecs scolaires passés |
| schoolsup | Soutien éducatif supplémentaire | (yes/no) converted in bool|
| famsup | Soutien éducatif familial | (yes/no) converted in bool|
| paid | Cours particuliers payants | (yes/no) converted in bool|
| activities | Activités extra-scolaires | (yes/no) converted in bool|
| nursery | A fréquenté une école maternelle | (yes/no) converted in bool|
| higher | Souhaite poursuivre des études supérieures | (yes/no) converted in bool|
| internet | Accès à Internet à domicile | (yes/no) converted in bool|
| romantic | En relation amoureuse | (yes/no) converted in bool|
| famrel | Qualité des relations familiales |
| freetime | Temps libre après l'école |
| goout | Fréquence de sorties avec des amis |
| Dalc | Consommation d'alcool en semaine |
| Walc | Consommation d'alcool le week-end |
| health | État de santé actuel |
| absences | Nombre d'absences scolaires |
| G1 | Note du premier trimestre |
| G2 | Note du deuxième trimestre |
| G3 | Note finale |



# Détails des colonnes pour student-mat

## Colonnes numériques (`int64`)

| Colonne | Nulls | Min | Max | Moyenne |
|---------|-------|-----|-----|---------|
| age | 0 | 15.0 | 22.0 | 16.7 | -> Age
| Medu | 0 | 0.0 | 4.0 | 2.75 | -> Mother -> Education
| Fedu | 0 | 0.0 | 4.0 | 2.52 | -> Father Education
| traveltime | 0 | 1.0 | 4.0 | 1.45 | -> Home To School Travel time
| studytime | 0 | 1.0 | 4.0 | 2.04 |
| failures | 0 | 0.0 | 3.0 | 0.33 |
| famrel | 0 | 1.0 | 5.0 | 3.94 |
| freetime | 0 | 1.0 | 5.0 | 3.24 |
| goout | 0 | 1.0 | 5.0 | 3.11 |
| Dalc | 0 | 1.0 | 5.0 | 1.48 |
| Walc | 0 | 1.0 | 5.0 | 2.29 |
| health | 0 | 1.0 | 5.0 | 3.55 |
| absences | 0 | 0.0 | 75.0 | 5.71 |
| G1 | 0 | 3.0 | 19.0 | 10.91 |
| G2 | 0 | 0.0 | 19.0 | 10.71 |
| G3 | 0 | 0.0 | 20.0 | 10.42 |

## Colonnes texte (`str`)

| Colonne | Nulls | Nulls_% |
|---------|-------|---------|
| school | 0 | 0.0% |
| sex | 0 | 0.0% |
| address | 0 | 0.0% |
| famsize | 0 | 0.0% |
| Pstatus | 0 | 0.0% |
| Mjob | 0 | 0.0% |
| Fjob | 0 | 0.0% |
| reason | 0 | 0.0% |
| guardian | 0 | 0.0% |
| schoolsup | 0 | 0.0% |
| famsup | 0 | 0.0% |
| paid | 0 | 0.0% |
| activities | 0 | 0.0% |
| nursery | 0 | 0.0% |
| higher | 0 | 0.0% |
| internet | 0 | 0.0% |
| romantic | 0 | 0.0% |






# 📊 Détails des colonnes — student-por

## Colonnes numériques (`int64`)

| Colonne | Nulls | Min | Max | Moyenne |
|---------|-------|-----|-----|---------|
| age | 0 | 15.0 | 22.0 | 16.74 |
| Medu | 0 | 0.0 | 4.0 | 2.51 |
| Fedu | 0 | 0.0 | 4.0 | 2.31 |
| traveltime | 0 | 1.0 | 4.0 | 1.57 |
| studytime | 0 | 1.0 | 4.0 | 1.93 |
| failures | 0 | 0.0 | 3.0 | 0.22 |
| famrel | 0 | 1.0 | 5.0 | 3.93 |
| freetime | 0 | 1.0 | 5.0 | 3.18 |
| goout | 0 | 1.0 | 5.0 | 3.18 |
| Dalc | 0 | 1.0 | 5.0 | 1.5 |
| Walc | 0 | 1.0 | 5.0 | 2.28 |
| health | 0 | 1.0 | 5.0 | 3.54 |
| absences | 0 | 0.0 | 32.0 | 3.66 |
| G1 | 0 | 0.0 | 19.0 | 11.4 |
| G2 | 0 | 0.0 | 19.0 | 11.57 |
| G3 | 0 | 0.0 | 19.0 | 11.91 |

## Colonnes texte (`str`)

| Colonne | Nulls | Nulls_% |
|---------|-------|---------|
| school | 0 | 0.0% |
| sex | 0 | 0.0% |
| address | 0 | 0.0% |
| famsize | 0 | 0.0% |
| Pstatus | 0 | 0.0% |
| Mjob | 0 | 0.0% |
| Fjob | 0 | 0.0% |
| reason | 0 | 0.0% |
| guardian | 0 | 0.0% |
| schoolsup | 0 | 0.0% |
| famsup | 0 | 0.0% |
| paid | 0 | 0.0% |
| activities | 0 | 0.0% |
| nursery | 0 | 0.0% |
| higher | 0 | 0.0% |
| internet | 0 | 0.0% |
| romantic | 0 | 0.0% |