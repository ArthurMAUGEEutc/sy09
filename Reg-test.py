# =========================================================
# PROJET SY09 — ANALYSE COMPLETE DES REGRESSIONS
# Student Performance Dataset (Math)
#
# Objectif :
# Construire une analyse explicative complète :
#
# 1. Régressions linéaires :
#    - réussite scolaire
#    - absences
#    - alcool
#    - comportement social
#
# 2. Régressions logistiques :
#    - ambition scolaire (higher)
#    - internet
#    - paid
#    - romantic
#
# 3. Analyses par clusters
#
# 4. Importance des variables

# 5. Analyse sociologique des résultats
#
# =========================================================

# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.metrics import (
    adjusted_rand_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    precision_score,
    recall_score,
)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =========================================================
# CHARGEMENT DATASET
# =========================================================

# df = pd.read_csv("dataset/student-mat.csv")
df = pd.read_csv("dataset/student-por.csv")

df_mat = pd.read_csv("dataset/student-mat.csv")
df_por = pd.read_csv("dataset/student-por.csv")

merge_keys = [
    "school", "sex", "age", "address", "famsize",
    "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
    "reason", "nursery", "internet"
]

df_merge = pd.merge(df_mat, df_por, on=merge_keys, suffixes=("_mat", "_por"))


# =========================================================
# FUSION SUR VARIABLES COMMUNES
# =========================================================

merge_keys = [
    "school", "sex", "age", "address", "famsize",
    "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
    "reason", "nursery", "internet"
]

df_merge = pd.merge(df_mat, df_por, on=merge_keys, suffixes=("_mat", "_por"))


print("="*80)
print("DATASET STUDENT PERFORMANCE — ANALYSE COMPLETE")
print("="*80)

print("\nDimensions :", df.shape)

print("\nColonnes :")
print(df.columns.tolist())

# =========================================================
# VARIABLES
# =========================================================

quantitative_vars = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3"
]

binary_vars = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
    "internet",
    "romantic"
]

# =========================================================
# ENCODAGE
# =========================================================

binary_map = {
    "yes": 1,
    "no": 0,
    "F": 0,
    "M": 1
}

df_encoded = df.copy()

for col in binary_vars:
    df_encoded[col] = df_encoded[col].map(binary_map)

df_encoded["sex"] = df_encoded["sex"].map(binary_map)

# =========================================================
# STATISTIQUES GENERALES
# =========================================================

print("\n" + "="*80)
print("STATISTIQUES GENERALES")
print("="*80)

print(df_encoded.describe())

# =========================================================
# DISTRIBUTIONS
# =========================================================

print("\n" + "="*80)
print("DISTRIBUTIONS DES VARIABLES")
print("="*80)

fig, axes = plt.subplots(4, 4, figsize=(18, 14))

axes = axes.flatten()

for i, col in enumerate(quantitative_vars):

    sns.histplot(
        df_encoded[col],
        kde=True,
        ax=axes[i]
    )

    axes[i].set_title(col)

plt.tight_layout()
plt.show()

# =========================================================
# VARIABLES EXPLICATIVES PRINCIPALES
# =========================================================

features = [
    "Medu",
    "Fedu",
    "failures",
    "sex",
    "internet",
    "famsup",
    "studytime",
    "Dalc",
    "Walc",
    "goout",
    "absences"
]

# =========================================================
# STANDARDISATION
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df_encoded[features])

# =========================================================
# MATRICE DE CORRELATION
# =========================================================

print("\n" + "="*80)
print("MATRICE DE CORRELATION")
print("="*80)

corr = df_encoded[
    quantitative_vars +
    ["internet", "famsup", "higher", "paid"]
].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Corrélations")
plt.show()


# =========================================================
# CREATION CLASSES DE PERFORMANCE
# =========================================================

def grade_class(x):
    if x < 10:
        return 0   # échec
    elif x <= 14:
        return 1   # moyen
    else:
        return 2   # bon

df_merge["G3_mat_class"] = df_merge["G3_mat"].apply(grade_class)
df_merge["G3_por_class"] = df_merge["G3_por"].apply(grade_class)


# =========================================================
# ALIGNEMENT LOGIQUE ARI
# =========================================================

ari = adjusted_rand_score(
    df_merge["G3_mat_class"],
    df_merge["G3_por_class"]
)

df_mat["G3_bin"] = pd.qcut(df_mat["G3"], q=3, labels=False)
df_por["G3_bin"] = pd.qcut(df_por["G3"], q=3, labels=False)

df_merge["G3_mat_bin"] = pd.qcut(df_merge["G3_mat"], q=3, labels=False)
df_merge["G3_por_bin"] = pd.qcut(df_merge["G3_por"], q=3, labels=False)

ari_bins = adjusted_rand_score(
    df_merge["G3_mat_bin"],
    df_merge["G3_por_bin"]
)


# =========================================================
# =========================================================
# REGRESSIONS LINEAIRES
# =========================================================
# =========================================================

def analyse_regression_lineaire(
    target,
    features,
    remove_G1G2=False
):

    print("\n" + "#"*80)
    print(f"REGRESSION LINEAIRE — {target}")
    print("#"*80)

    local_features = [
        f for f in features
        if f != target
    ]

    if remove_G1G2:
        local_features = [
            x for x in local_features
            if x not in ["G1", "G2"]
        ]

    X = df_encoded[local_features]
    y = df_encoded[target]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    acc= accuracy_score(y_test, y_pred.round())


    print("\nPerformances :")
    print(f"R2    : {r2:.3f}")
    print(f"RMSE  : {rmse:.3f}")
    print(f"MAE   : {mae:.3f}")
    print(f"Accuracy : {acc:.3f}")


    coeffs = pd.DataFrame({
        "Variable": local_features,
        "Coefficient": model.coef_
    })

    coeffs["AbsCoeff"] = np.abs(coeffs["Coefficient"])

    coeffs = coeffs.sort_values(
        by="AbsCoeff",
        ascending=False
    )

    print("\nVariables influentes :")
    print(coeffs)

    # =====================================================
    # VISUALISATION COEFFICIENTS
    # =====================================================

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=coeffs,
        x="Coefficient",
        y="Variable"
    )

    plt.title(f"Importance des variables — {target}")
    plt.show()

    # =====================================================
    # PREDICTION VS REALITE
    # =====================================================

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.7
    )

    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")

    plt.title(f"Prédictions — {target}")

    plt.show()

    # =====================================================
    # INTERPRETATION
    # =====================================================

    print("\nINTERPRETATION :")

    top_positive = coeffs.sort_values(
        by="Coefficient",
        ascending=False
    ).head(3)

    top_negative = coeffs.sort_values(
        by="Coefficient",
        ascending=True
    ).head(3)

    print("\nVariables positives :")
    print(top_positive[["Variable", "Coefficient"]])

    print("\nVariables negatives :")
    print(top_negative[["Variable", "Coefficient"]])

    return model, coeffs

# =========================================================
# REGRESSION G3 SANS G1/G2
# =========================================================

analyse_regression_lineaire(
    target="G3",
    features=features,
    remove_G1G2=True
)

# =========================================================
# REGRESSION ABSENCES
# =========================================================

analyse_regression_lineaire(
    target="absences",
    features=features
)

# =========================================================
# REGRESSION DALC
# =========================================================

analyse_regression_lineaire(
    target="Dalc",
    features=features
)

# =========================================================
# REGRESSION WALC
# =========================================================

analyse_regression_lineaire(
    target="Walc",
    features=features
)

# =========================================================
# =========================================================
# REGRESSIONS LOGISTIQUES
# =========================================================
# =========================================================

def analyse_regression_logistique(
    target,
    features,
    threshold_balance=0.15
):

    print("\n" + "#"*80)
    print(f"REGRESSION LOGISTIQUE — {target}")
    print("#"*80)

    local_features = [
        f for f in features
        if f != target
    ]

    X = df_encoded[local_features]
    y = df_encoded[target]

    print("\nDistribution des classes :")
    print(y.value_counts(normalize=True))

    # =====================================================
    # TEST EQUILIBRE
    # =====================================================

    if y.value_counts(normalize=True).min() < threshold_balance:

        print("\n⚠️ Variable trop déséquilibrée")
        print("Analyse ignorée")

        return None

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=500)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # =====================================================
    # METRIQUES
    # =====================================================

    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    evs = explained_variance_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)


    print("\nPerformances :")
    print(f"Accuracy           : {acc:.3f}")
    print(f"Balanced Accuracy  : {balanced_acc:.3f}")
    print(f"F1-score           : {f1:.3f}")
    print(f"ROC AUC            : {roc:.3f}")
    print(f"Explained Variance : {evs:.3f}")
    print(f"Precision          : {precision:.3f}")
    print(f"Recall             : {recall:.3f}")

    print("\nClassification report :")
    print(classification_report(y_test, y_pred))

    # =====================================================
    # MATRICE CONFUSION
    # =====================================================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Matrice confusion — {target}")

    plt.xlabel("Prediction")
    plt.ylabel("Reel")

    plt.show()

    # =====================================================
    # COEFFICIENTS
    # =====================================================

    coeffs = pd.DataFrame({
        "Variable": local_features,
        "Coefficient": model.coef_[0]
    })

    coeffs["AbsCoeff"] = np.abs(coeffs["Coefficient"])

    coeffs = coeffs.sort_values(
        by="AbsCoeff",
        ascending=False
    )

    print("\nVariables importantes :")
    print(coeffs)

    # =====================================================
    # VISUALISATION
    # =====================================================

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=coeffs,
        x="Coefficient",
        y="Variable"
    )

    plt.title(f"Importance variables — {target}")

    plt.show()

    # =====================================================
    # INTERPRETATION
    # =====================================================

    print("\nINTERPRETATION :")

    print("\nFacteurs augmentant la probabilite :")

    print(
        coeffs.sort_values(
            by="Coefficient",
            ascending=False
        ).head(3)
    )

    print("\nFacteurs diminuant la probabilite :")

    print(
        coeffs.sort_values(
            by="Coefficient",
            ascending=True
        ).head(3)
    )

    return model, coeffs

# =========================================================
# HIGHER
# =========================================================

analyse_regression_logistique(
    target="higher",
    features=features
)

# =========================================================
# INTERNET
# =========================================================

analyse_regression_logistique(
    target="internet",
    features=features
)

# =========================================================
# PAID
# =========================================================

analyse_regression_logistique(
    target="paid",
    features=features
)

# =========================================================
# ROMANTIC
# =========================================================

analyse_regression_logistique(
    target="romantic",
    features=features
)

# =========================================================
# =========================================================
# ACP + KMEANS
# =========================================================
# =========================================================

print("\n" + "="*80)
print("ACP + KMEANS")
print("="*80)

X_cluster = pd.get_dummies(
    df.drop(columns=["G3"]),
    drop_first=True
)

scaler_cluster = StandardScaler()

X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# =========================================================
# ACP
# =========================================================

pca = PCA()

X_pca_full = pca.fit_transform(X_cluster_scaled)

explained = pca.explained_variance_ratio_

cumulated = np.cumsum(explained)

print("\nVariance expliquee :")

for i, val in enumerate(explained):

    print(
        f"CP{i+1:2d} : "
        f"{val*100:5.1f}%   "
        f"cumule : {cumulated[i]*100:5.1f}%"
    )

# =========================================================
# CHOIX 80%
# =========================================================

n_components = np.argmax(cumulated >= 0.80) + 1

print(f"\nNombre composantes 80% : {n_components}")

pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_cluster_scaled)



# =========================================================
# KMEANS
# =========================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

"""
def plot_kmeans_full(X_pca_full, X_pca3, k, df_full, num_cols):
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X_pca_full)

    palette = ['steelblue', 'orange', 'tomato', 'mediumseagreen', 'purple']
    var3 = pca3_full.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # Scatter CP1/CP2
    for c in range(k):
        mask = labels == c
        axes[0].scatter(X_pca3[mask, 0], X_pca3[mask, 1],
                        c=palette[c], label=f'Cluster {c} (n={mask.sum()})',
                        alpha=0.6, s=35, edgecolors='none')
    axes[0].set_xlabel(f'CP1 ({var3[0]*100:.1f}%)')
    axes[0].set_ylabel(f'CP2 ({var3[1]*100:.1f}%)')
    axes[0].set_title(f'K-means k={k} — CP1/CP2 (42 vars)')
    axes[0].legend(fontsize=9)

    # Profil heatmap sur variables quantitatives + higher + sex
    df_tmp = df_full.copy()
    df_tmp['cluster'] = labels
    profile = df_tmp.groupby('cluster')[num_cols].mean()
    pn = (profile - profile.mean()) / profile.std()

    im = axes[1].imshow(pn.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    axes[1].set_xticks(range(len(num_cols)))
    axes[1].set_xticklabels(num_cols, rotation=35, ha='right')
    axes[1].set_yticks(range(k))
    axes[1].set_yticklabels([f'Cluster {i}' for i in range(k)])
    axes[1].set_title(f'Profil moyen — variables quantitatives (k={k})')
    plt.colorbar(im, ax=axes[1], label='Écarts à la moyenne')
    for i in range(k):
        for j in range(len(num_cols)):
            axes[1].text(j, i, f'{profile.values[i,j]:.1f}',
                         ha='center', va='center', fontsize=7)

    plt.suptitle(f'K-means étendu — k={k} (42 variables)', fontsize=13)
    plt.tight_layout()
    plt.show()

    # Profil des variables non-quantitatives
    cols_bool = [c for c in bool_cols if c in df_tmp.columns]
    if cols_bool:
        profil_bool = df_tmp.groupby('cluster')[cols_bool].mean().round(2)
        print(f"Profil booléens (proportion par cluster) :")
        print(profil_bool.to_string())
    print()
    return labels
"""


clusters = kmeans.fit_predict(X_pca)
# labels4 = plot_kmeans_clusters(X_pca, clusters)
df_encoded["cluster"] = clusters

print("\nTailles clusters :")
print(df_encoded["cluster"].value_counts())
# =========================================================
# VISUALISATION
# =========================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=clusters,
    palette="Set1"
)

plt.title("ACP + KMeans")
plt.show()


# =========================================================
# =========================================================
# ANALYSES PAR CLUSTER
# =========================================================
# =========================================================

targets_cluster = [
    "higher",
    "internet",
    "paid",
    "romantic"
]

for cluster_id in sorted(df_encoded["cluster"].unique()):

    print("\n" + "#"*80)
    print(f"ANALYSE CLUSTER {cluster_id}")
    print("#"*80)

    df_cluster = df_encoded[
        df_encoded["cluster"] == cluster_id
    ].copy()

    print("\nTaille :", len(df_cluster))

    print("\nMoyennes quantitatives :")

    print(
        df_cluster[
            quantitative_vars
        ].mean().sort_values(
            ascending=False
        )
    )

    print("\nProportions binaires :")

    print(
        df_cluster[
            binary_vars
        ].mean().sort_values(
            ascending=False
        )
    )

    # =====================================================
    # REGRESSIONS PAR CLUSTER
    # =====================================================

    for target in targets_cluster:

        print("\n" + "-"*60)
        print(f"Target : {target}")
        print("-"*60)

        y = df_cluster[target]

        if y.value_counts(normalize=True).min() < 0.15:

            print("\n⚠️ Variable desequilibree")

            continue

        # Vérification nombre de classes

        if len(np.unique(y)) < 2:

            print("\n⚠️ Une seule classe présente")
            print("Régression impossible")

            continue

        local_features = [
            f for f in features
            if f != target
        ]

        X = df_cluster[local_features]

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )

        model = LogisticRegression(max_iter=500)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        print("\nPerformances :")

        print(
            "Balanced Accuracy :",
            round(
                balanced_accuracy_score(
                    y_test,
                    y_pred
                ),
                3
            )
        )

        print(
            "F1-score :",
            round(
                f1_score(
                    y_test,
                    y_pred
                ),
                3
            )
        )

        print(
            "ROC AUC :",
            round(
                roc_auc_score(
                    y_test,
                    y_prob
                ),
                3
            )
        )

        coeffs = pd.DataFrame({
            "Variable": local_features,
            "Coefficient": model.coef_[0]
        })

        coeffs["AbsCoeff"] = np.abs(
            coeffs["Coefficient"]
        )

        coeffs = coeffs.sort_values(
            by="AbsCoeff",
            ascending=False
        )

        print("\nVariables importantes :")
        print(coeffs)

# =========================================================
# =========================================================
# VISUALISATIONS FINALES
# =========================================================
# =========================================================

# =========================================================
# G3 PAR CLUSTER
# =========================================================

print(
    df_encoded.groupby("cluster")["G3"]
    .describe()[["mean","std","min","max"]]
)

plt.figure(figsize=(10, 6))


sns.boxplot(
    x="cluster",
    y="G3",
    data=df_encoded
)

plt.title("Distribution G3 par cluster")

plt.show()

# =========================================================
# ABSENCES PAR CLUSTER
# =========================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    x="cluster",
    y="absences",
    data=df_encoded
)

plt.title("Distribution absences par cluster")

plt.show()

# =========================================================
# DALC PAR CLUSTER
# =========================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    x="cluster",
    y="Dalc",
    data=df_encoded
)

plt.title("Dalc par cluster")

plt.show()

# =========================================================
# WALC PAR CLUSTER
# =========================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    x="cluster",
    y="Walc",
    data=df_encoded
)

plt.title("Walc par cluster")

plt.show()

# =========================================================
# FIN
# =========================================================

print("\n" + "="*80)
print("FIN DES ANALYSES")
print("="*80)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7,6))

sns.scatterplot(
    x=df_merge["G3_mat"],
    y=df_merge["G3_por"]
)

plt.xlabel("G3 Math")
plt.ylabel("G3 Portugais")
plt.title("Relation entre performances math et portugais")
plt.show()

df_merge["gap"] = df_merge["G3_mat"] - df_merge["G3_por"]

sns.histplot(df_merge["gap"], kde=True)
plt.title("Différence performance Math - Portugais")
plt.show()

print("\n==============================")
print("ARI Math vs Portugais (G3 classes)")
print("==============================")
print("ARI =", round(ari, 4))

print("\nARI (bins quantiles) =", round(ari_bins, 4))

print("\nCorrélation G3 mat vs por :")

print(df_merge[["G3_mat", "G3_por"]].corr())
