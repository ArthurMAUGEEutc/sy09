# =========================================================
# ACP / PCA sur le dataset Student Performance
# =========================================================

# Installer si nécessaire :
# pip install pandas scikit-learn matplotlib seaborn

# =========================================================
# 1. IMPORTS
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================================================
# 2. CHARGEMENT DU CSV
# =========================================================

# Remplacer par ton fichier CSV
df = pd.read_csv("dataset/student-mat.csv")

# Aperçu
print("===== APERÇU DES DONNÉES =====")
print(df.head())

print("\n===== INFORMATIONS =====")
print(df.info())

# =========================================================
# 3. SUPPRESSION DES VALEURS MANQUANTES
# =========================================================

df = df.dropna()

# =========================================================
# 4. OPTION : RETIRER LES NOTES
# =========================================================
# Décommente si tu veux une ACP socio-comportementale

# df = df.drop(columns=["G1", "G2", "G3"])

# =========================================================
# 5. ENCODAGE DES VARIABLES CATÉGORIELLES
# =========================================================

df_encoded = pd.get_dummies(df, drop_first=True)

print("\n===== DONNÉES ENCODÉES =====")
print(df_encoded.head())

# =========================================================
# 6. STANDARDISATION
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df_encoded)

# =========================================================
# 7. ACP
# =========================================================

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

# =========================================================
# 8. VARIANCE EXPLIQUÉE
# =========================================================

explained_variance = pca.explained_variance_ratio_

print("\n===== VARIANCE EXPLIQUÉE =====")

for i, var in enumerate(explained_variance):
    print(f"Composante {i+1}: {var:.4f}")

# =========================================================
# 9. SCREE PLOT
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance.cumsum(),
    marker='o'
)

plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.title("Scree Plot - ACP")

plt.grid(True)

plt.show()

# =========================================================
# 10. ACP À 2 COMPOSANTES
# =========================================================

pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_scaled)

# =========================================================
# 11. DATAFRAME DES COMPOSANTES
# =========================================================

pca_df = pd.DataFrame(
    data=X_pca_2d,
    columns=["PC1", "PC2"]
)

print("\n===== COORDONNÉES PCA =====")
print(pca_df.head())

# =========================================================
# 12. VISUALISATION DES INDIVIDUS
# =========================================================

plt.figure(figsize=(10, 8))

sns.scatterplot(
    x="PC1",
    y="PC2",
    data=pca_df
)

plt.title("Projection des étudiants sur les 2 premières composantes")

plt.xlabel(
    f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}% variance)"
)

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

plt.grid(True)

plt.show()

# =========================================================
# 13. CONTRIBUTION DES VARIABLES
# =========================================================

loadings = pd.DataFrame(
    pca_2d.components_.T,
    columns=["PC1", "PC2"],
    index=df_encoded.columns
)

print("\n===== CONTRIBUTIONS DES VARIABLES =====")

print(loadings)

# =========================================================
# 14. VARIABLES LES PLUS IMPORTANTES
# =========================================================

print("\n===== VARIABLES IMPORTANTES POUR PC1 =====")

print(
    loadings["PC1"]
    .sort_values(ascending=False)
    .head(10)
)

print("\n===== VARIABLES IMPORTANTES NÉGATIVES POUR PC1 =====")

print(
    loadings["PC1"]
    .sort_values()
    .head(10)
)

# =========================================================
# 15. CERCLE DES CORRÉLATIONS
# =========================================================

plt.figure(figsize=(12, 12))

for i, var in enumerate(df_encoded.columns):

    plt.arrow(
        0,
        0,
        loadings.iloc[i, 0],
        loadings.iloc[i, 1],
        head_width=0.02,
        alpha=0.5
    )

    plt.text(
        loadings.iloc[i, 0] * 1.1,
        loadings.iloc[i, 1] * 1.1,
        var,
        fontsize=8
    )

plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Cercle des corrélations")

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

plt.grid(True)

plt.show()

# =========================================================
# 16. EXPORT DES RÉSULTATS
# =========================================================

plt.figure(figsize=(18,12))

corr = df_encoded.corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Matrice de corrélation")

plt.show()

plt.figure(figsize=(10,6))

sns.histplot(df["G3"], bins=20, kde=True)

plt.title("Distribution des notes finales G3")

plt.xlabel("Note finale")

plt.show()

plt.figure(figsize=(10,6))

sns.boxplot(
    x="Walc",
    y="G3",
    data=df
)

plt.title("Impact de l'alcool du week-end sur les notes")

plt.show()

plt.figure(figsize=(10,6))

sns.scatterplot(
    x="studytime",
    y="G3",
    data=df
)

plt.title("Temps d'étude vs note finale")

plt.show()

cols = ["G1", "G2", "G3", "studytime", "absences", "failures"]

sns.pairplot(df[cols])

plt.show()

plt.figure(figsize=(10,6))

plt.plot(
    np.cumsum(pca.explained_variance_ratio_),
    marker='o'
)

plt.xlabel("Nombre de composantes")
plt.ylabel("Variance cumulée")

plt.title("Variance expliquée cumulée")

plt.grid()

plt.show()

plt.figure(figsize=(10,8))

scatter = plt.scatter(
    X_pca_2d[:,0],
    X_pca_2d[:,1],
    c=df["G3"],
    cmap="viridis"
)

plt.colorbar(scatter, label="G3")

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("ACP colorée par la note finale")

plt.show()

plt.figure(figsize=(10,8))

sns.scatterplot(
    x=X_pca_2d[:,0],
    y=X_pca_2d[:,1],
    hue=df["sex"]
)

plt.title("ACP selon le sexe")

plt.show()

contrib = pd.DataFrame(
    np.abs(pca.components_),
    columns=df_encoded.columns
)

pc1 = contrib.iloc[0].sort_values(ascending=False)[:15]

plt.figure(figsize=(12,6))

sns.barplot(
    x=pc1.values,
    y=pc1.index
)

plt.title("Variables importantes pour PC1")

plt.show()

pca_df.to_csv("resultats_pca.csv", index=False)

loadings.to_csv("contributions_variables.csv")

print("\n===== EXPORT TERMINÉ =====")
print("Fichier créé : resultats_pca.csv")
print("Fichier créé : contributions_variables.csv")

# =========================================================
# FIN
# =========================================================