
Le rapport : 5 pages (sans compter les annexes)

La soutenance : 3 minutes de présentation + 12 minutes de questions/réponses. 

1. Introduction -> montrer des graphiques sur l'importance de l'éducation des parents sur les résultat scolaire (parler du groupe Excellence par education qui est un groupe redondant de tout les clusters)

2. Ensuite parler des Cluster K-means sur mat, por, mat x por

3. parler des modèle de préduction sur l'éducation des parents 

4. Limites -> modèle d'apprentissage supervisée limité, volonté de ne pas prédire G3 car trop facile avec G1,G2 et sans intérêt sans 






Idée : Pour la Section 1 — Introduction (montrer l'importance de l'éducation parentale)

Le modèle apporte un argument fort et contre-intuitif : on peut prédire l'éducation de la mère depuis les résultats et comportements de l'enfant à 76.7% (CV-5). Ce renversement de la causalité est rhétoriquement puissant — si l'empreinte de l'éducation maternelle est lisible dans le profil de l'élève, c'est qu'elle est réelle et profonde.

Pour la Section 3 — Modèles de prédiction

Trois arguments précis à construire :

1. L'empreinte comportementale pure
Sans Fedu (aucune info sur le père), le modèle prédit encore Medu à 66.4% depuis uniquement les notes, échecs, comportements et ambitions. Les variables les plus prédictives — failures, higher, G1/G2/G3, Dalc — sont des médiateurs : l'éducation maternelle se transmet via des valeurs (ambition d'études supérieures), des habitudes (temps d'étude, alcool) et se reflète dans la réussite scolaire. Ce n'est pas une corrélation statistique abstraite : c'est une transmission mesurable.

2. La reproduction sociale inter-parentale
L'apport de Fedu au modèle est de +10.3 points. Cela quantifie l'homogamie éducative — les parents ont tendance à avoir des niveaux d'éducation similaires, ce qui concentre l'avantage ou le désavantage éducatif au niveau du foyer entier.

3. Lien avec le cluster "Excellence"
Le modèle confirme ce qu'on observe dans les clusters K-means : le groupe à haute réussite (cluster "Excellence") est aussi celui où Medu/Fedu sont les plus élevés. Ce n'est pas une coïncidence — c'est la même structure, vue sous un angle supervisé. Les deux approches (non-supervisée et supervisée) convergent vers la même conclusion.

En résumé pour le rapport : "L'éducation parentale n'influence pas directement les notes — elle façonne un profil comportemental complet (ambition, discipline, rapport à l'alcool) qui lui-même détermine la réussite. Notre modèle de régression logistique, en prédisant le niveau d'éducation de la mère depuis le seul profil de l'élève à 66% (sans information sur le père), rend ce mécanisme statistiquement visible."