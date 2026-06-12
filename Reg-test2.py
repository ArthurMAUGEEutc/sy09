# ==============================================================================
# PROJET SY09 — PIPELINE D'ANALYSE STATISTIQUE ET DE MODÉLISATION AVANCÉE
# Dataset : Student Performance (Focus: Mathématiques & Analyse Cross-Discipline)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)

# Configuration esthétique globale des graphiques
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11, 
    'axes.labelsize': 12, 
    'axes.titlesize': 14,
    'figure.titlesize': 16
})

print("="*80)
print("DÉBUT DE L'EXÉCUTION DE LA PIPELINE STATISTIQUE SY09")
print("="*80)

# ==============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================
print("\n" + "="*80)
print("1. CHARGEMENT ET ENCODAGE DES DATASETS")
print("="*80)

# Chargement des fichiers sources
df_mat = pd.read_csv("dataset/student-mat.csv")
df_por = pd.read_csv("dataset/student-por.csv")

# Copie de travail sur les mathématiques pour l'analyse comportementale principale
df = df_mat.copy()
print(f"[INFO] Dataset Mathématiques (Principal) chargé. Dimensions : {df.shape}")
print(f"[INFO] Dataset Portugais (Comparatif) chargé.     Dimensions : {df_por.shape}")

# Définition thématique des variables du jeu de données
quantitative_vars = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", 
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", 
    "G1", "G2", "G3"
]

binary_vars = [
    "schoolsup", "famsup", "paid", "activities", "higher", "internet", "romantic"
]

# Variables explicatives sélectionnées (Facteurs extra-scolaires fondamentaux)
features_model = [
    "Medu", "Fedu", "failures", "sex", "internet", 
    "famsup", "studytime", "Dalc", "Walc", "goout", "absences"
]

# Encodage binaire uniforme (yes/no -> 1/0 et F/M -> 0/1)
binary_map = {"yes": 1, "no": 0, "F": 0, "M": 1}
df_encoded = df.copy()

for col in binary_vars:
    df_encoded[col] = df_encoded[col].map(binary_map)
df_encoded["sex"] = df_encoded["sex"].map(binary_map)

print("\n[INFO] Aperçu des données encodées (5 premières lignes) :")
print(df_encoded[features_model + ["G3"]].head())

# ==============================================================================
# 2. ANALYSE EXPLORATOIRE DES DISTRIBUTIONS
# ==============================================================================
print("\n" + "="*80)
print("2. ANALYSE EXPLORATOIRE ET STATISTIQUES DESCRIPTIVES")
print("="*80)

print("\n[STAT] Statistiques descriptives de la variable cible 'G3' (Maths) :")
print(df_encoded["G3"].describe())

print("\n[STAT] Statistiques descriptives de la variable asymétrique 'absences' :")
print(df_encoded["absences"].describe())

# Graphique 1 : Zoom sur la distribution non-linéaire des absences
plt.figure(figsize=(9, 5))
sns.histplot(df_encoded["absences"], kde=True, color="crimson", bins=30)
plt.axvline(df_encoded["absences"].median(), color="black", linestyle="--", 
            label=f"Médiane = {df_encoded['absences'].median()}")
plt.title("Histogramme de distribution de la variable 'absences' (Asymétrie positive)")
plt.xlabel("Nombre d'absences de l'étudiant")
plt.ylabel("Fréquence (Nombre d'élèves)")
plt.legend()
plt.tight_layout()
plt.show()

# Graphique 2 : Matrice de corrélation linéaire
corr_vars = quantitative_vars + ["internet", "famsup", "higher", "paid"]
corr_matrix = df_encoded[corr_vars].corr()

plt.figure(figsize=(13, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
            cbar_kws={'label': 'Coefficient de corrélation de Pearson'})
plt.title("Matrice de corrélation linéaire des descripteurs quantitatifs et binaires")
plt.tight_layout()
plt.show()

# ==============================================================================
# 3. MODÉLISATION PAR RÉGRESSION LINÉAIRE GLOBALE
# ==============================================================================
print("\n" + "="*80)
print("3. MODÈLES DE RÉGRESSION LINÉAIRE GLOBALE")
print("="*80)

saved_split_data = {}

def pipeline_regression_lineaire(target, features_list, remove_G1G2=False):
    print(f"\n👉 EXÉCUTION RÉGRESSION LINÉAIRE — CIBLE : {target}")
    if remove_G1G2:
        print("   (Note : Exclusion volontaire de G1 et G2 pour éviter la multicolinéarité)")
    
    local_features = [f for f in features_list if f != target]
    if remove_G1G2:
        local_features = [f for f in local_features if f not in ["G1", "G2"]]
        
    X = df_encoded[local_features]
    y = df_encoded[target]
    
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    if target == "G3" and remove_G1G2:
        saved_split_data['X_train'] = X_train
        saved_split_data['y_train'] = y_train
        saved_split_data['features'] = local_features
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   [MÉTRIQUES] R2       = {r2:.4f}")
    print(f"   [MÉTRIQUES] RMSE     = {rmse:.4f}")
    print(f"   [MÉTRIQUES] MAE      = {mae:.4f}")
    
    coeffs = pd.DataFrame({"Variable": local_features, "Coefficient": model.coef_})
    coeffs["AbsCoeff"] = coeffs["Coefficient"].abs()
    coeffs = coeffs.sort_values(by="AbsCoeff", ascending=False)
    
    print("   [COEFFICIENTS] Liste ordonnée des variables influentes (Beta standardisés) :")
    print(coeffs.to_string(index=False))
    
    # Graphique 3 : Poids des coefficients linéaires
    plt.figure(figsize=(10, 5))
    sns.barplot(data=coeffs, x="Coefficient", y="Variable", palette="vlag")
    plt.axvline(0, color="black", linestyle="-", linewidth=1)
    plt.title(f"Coefficients standardisés (Beta) — Modèle linéaire '{target}'")
    plt.xlabel("Valeur du coefficient standardisé")
    plt.ylabel("Variables explicatives")
    plt.tight_layout()
    plt.show()
    
    return model, coeffs

# Exécution des régressions
_, _ = pipeline_regression_lineaire("G3", features_model, remove_G1G2=True)
_, _ = pipeline_regression_lineaire("absences", features_model)
_, _ = pipeline_regression_lineaire("Dalc", features_model)
_, _ = pipeline_regression_lineaire("Walc", features_model)

# ==============================================================================
# 4. COMPARAISON AVEC UNE APPROCHE NON-LINÉAIRE (RANDOM FOREST)
# ==============================================================================
print("\n" + "="*80)
print("4. COMPARAISON MÉTHODOLOGIQUE : IMPORTANCE NON-LINÉAIRE (RANDOM FOREST)")
print("="*80)

if 'X_train' in saved_split_data:
    print("👉 ENTRAÎNEMENT DU MODÈLE RANDOM FOREST (Cible: G3, sans G1/G2)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(saved_split_data['X_train'], saved_split_data['y_train'])
    
    importances = pd.DataFrame({
        'Variable': saved_split_data['features'],
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("   [RF IMPORTANCE] Part de variance/pureté expliquée par chaque descripteur :")
    print(importances.to_string(index=False))
    
    # Graphique 4 : Importance non-linéaire des variables
    plt.figure(figsize=(10, 5))
    sns.barplot(data=importances, x='Importance', y='Variable', palette='viridis')
    plt.title("Importance des facteurs extra-scolaires sur G3 (Modèle Random Forest)")
    plt.xlabel("Pureté de la baisse d'impureté moyenne (Feature Importance)")
    plt.ylabel("Variables explicatives")
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 5. MODÉLISATION PAR RÉGRESSION LOGISTIQUE GLOBALE
# ==============================================================================
print("\n" + "="*80)
print("5. MODÈLES DE RÉGRESSION LOGISTIQUE GLOBALE")
print("="*80)

def pipeline_regression_logistique(target, features_list, threshold_balance=0.15):
    print(f"\n👉 EXÉCUTION RÉGRESSION LOGISTIQUE — CIBLE : {target}")
    
    local_features = [f for f in features_list if f != target]
    X = df_encoded[local_features]
    y = df_encoded[target]
    
    class_dist = y.value_counts(normalize=True)
    print(f"   [DISTRIBUTION] Proportions réelles des classes pour '{target}' :")
    for val, prop in class_dist.items():
        print(f"      Classe {val} : {prop*100:.2f}%")
        
    if class_dist.min() < threshold_balance:
        print(f"   ⚠️ [ALERTE] Variable '{target}' trop déséquilibrée (< {threshold_balance*100}%). Analyse globale ignorée.")
        return None
        
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"   [MÉTRIQUES] Accuracy          = {accuracy_score(y_test, y_pred):.4f}")
    print(f"   [MÉTRIQUES] Balanced Accuracy = {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"   [MÉTRIQUES] ROC AUC Score     = {roc_auc_score(y_test, y_prob):.4f}")
    
    print("   [RAPPORT DE CLASSIFICATION] :")
    print(classification_report(y_test, y_pred))
    
    coeffs = pd.DataFrame({"Variable": local_features, "Coefficient": model.coef_[0]})
    coeffs["AbsCoeff"] = coeffs["Coefficient"].abs()
    coeffs = coeffs.sort_values(by="AbsCoeff", ascending=False)
    print("   [COEFFICIENTS LOGISTIQUES] Poids des variables (Log-Odds) :")
    print(coeffs.to_string(index=False))

    # Graphique 5 : Matrice de confusion
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matrice de confusion — Classification '{target}'")
    plt.xlabel("Classe prédite (0=Non, 1=Oui)")
    plt.ylabel("Classe réelle")
    plt.tight_layout()
    plt.show()
    
    # Graphique 6 (Spécifique pour 'paid') : Courbe ROC
    if target == "paid":
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc_score(y_test, y_prob):.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de Faux Positifs (FPR)')
        plt.ylabel('Taux de Vrais Positifs (TPR)')
        plt.title("Courbe ROC — Modèle de recours aux cours payants ('paid')")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    return model

# Exécution des régressions logistiques
for target_bin in ["higher", "internet", "paid", "romantic"]:
    pipeline_regression_logistique(target_bin, features_model)

# ==============================================================================
# 6. RÉDUCTION DIMENSIONNELLE ET COMPORTEMENT LOCAL (ACP + K-MEANS)
# ==============================================================================
print("\n" + "="*80)
print("6. SEGMENTATION LOCALE : ACP ET PARTITIONNEMENT K-MEANS")
print("="*80)

X_cluster = pd.get_dummies(df.drop(columns=["G3"]), drop_first=True)
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

pca_full = PCA()
pca_full.fit(X_cluster_scaled)
cum_variance = np.cumsum(pca_full.explained_variance_ratio_)

print("\n[ACP] Variances expliquées individuelles et cumulées par axe :")
for i, (val, cum) in enumerate(zip(pca_full.explained_variance_ratio_, cum_variance)):
    print(f"   Axe {i+1:2d} : {val*100:5.2f}% (Cumulé : {cum*100:5.2f}%)")
    if i >= 14: # Tronquer l'affichage pour la console
        print("   [...] axes suivants masqués par souci de lisibilité")
        break

n_components = np.argmax(cum_variance >= 0.80) + 1
print(f"\n[ACP] Nombre optimal de composantes pour capturer 80% de la variance : {n_components}")

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_cluster_scaled)

print("\n👉 APPLICATION DU CLUSTERING K-MEANS (K=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
clusters_labels = kmeans.fit_predict(X_pca)
df_encoded["cluster"] = clusters_labels

print("\n[EFFECTIFS] Répartition finale des effectifs au sein des 4 clusters d'élèves :")
print(df_encoded["cluster"].value_counts())

# Graphique 7 : Projection spatiale des clusters (Plan Factoriel CP1/CP2)
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters_labels, palette="Set1", alpha=0.8, edgecolor='none')
plt.title("Visualisation locale des 4 clusters d'élèves (Premier Plan Factoriel)")
plt.xlabel(f"Composante Principale 1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Composante Principale 2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(title="Cluster assigné")
plt.tight_layout()
plt.show()

# ==============================================================================
# 7. ENRICHISSEMENT : CERCLE DES CORRÉLATIONS DE L'ACP
# ==============================================================================
print("\n" + "="*80)
print("7. INTERPRÉTATION DU SENS DES AXES DE L'ACP (LOADINGS)")
print("="*80)

loadings = pd.DataFrame(pca.components_[:2].T, columns=['CP1', 'CP2'], index=X_cluster.columns)
top_loadings = loadings.loc[(loadings['CP1'].abs() > 0.18) | (loadings['CP2'].abs() > 0.18)]

print("\n[ACP LOADINGS] Coordonnées des variables les plus significatives sur les axes 1 et 2 :")
print(top_loadings.sort_values(by="CP1", ascending=False))

# Graphique 8 : Cercle des corrélations approché
plt.figure(figsize=(8, 8))
for var in top_loadings.index:
    plt.arrow(0, 0, top_loadings.loc[var, 'CP1'], top_loadings.loc[var, 'CP2'], 
              head_width=0.025, head_length=0.025, color='darkviolet', alpha=0.7)
    plt.text(top_loadings.loc[var, 'CP1']*1.12, top_loadings.loc[var, 'CP2']*1.12, 
             var, fontsize=9, ha='center', va='center')

circle = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--', linewidth=0.8)
plt.gca().add_patch(circle)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("Cercle des corrélations des variables structurantes (Seuil contribution > 0.18)")
plt.xlabel(f"CP1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"CP2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)")
plt.tight_layout()
plt.show()

# ==============================================================================
# 8. ENRICHISSEMENT : CARTE THERMIQUE DES PROFILS DE CLUSTERS (HEATMAP)
# ==============================================================================
print("\n" + "="*80)
print("8. DESCRIPTEURS MOYENS ET CARTE THERMIQUE DES CLUSTERS")
print("="*80)

features_profil = ["G3", "absences", "failures", "studytime", "goout", "Dalc", "Walc", "Medu", "Fedu"]
profile_means = df_encoded.groupby('cluster')[features_profil].mean()

print("\n[CLUSTER PROFILES] Valeurs moyennes brutes pour chaque groupe d'étudiants :")
print(profile_means.to_string())

# Centrage-réduction des profils pour l'expression en nombre d'écarts-types
profile_scaled = (profile_means - df_encoded[features_profil].mean()) / df_encoded[features_profil].std()

# Graphique 9 : Carte thermique comparative des profils de clusters
plt.figure(figsize=(11, 5))
sns.heatmap(profile_scaled, annot=profile_means.values, fmt=".2f", cmap="RdBu_r", center=0, 
            cbar_kws={'label': "Écart à la moyenne de la population (en Écarts-types)"})
plt.title("Profils moyens réels et centrés-réduits des 4 clusters identifiés")
plt.xlabel("Variables quantitatives clés")
plt.ylabel("Sous-groupes d'étudiants (Clusters)")
plt.tight_layout()
plt.show()

# ==============================================================================
# 9. RÉGRESSIONS LOGISTIQUES LOCALES PAR CLUSTER
# ==============================================================================
print("\n" + "="*80)
print("9. ENTRAÎNEMENT DES LOGIQUES LOCALES (AU SEIN DE CHAQUE SOUS-ESPACE)")
print("="*80)

targets_cluster = ["higher", "internet", "paid", "romantic"]

for cluster_id in sorted(df_encoded["cluster"].unique()):
    df_cluster = df_encoded[df_encoded["cluster"] == cluster_id].copy()
    print(f"\n⚡ ANALYSE LOCALE DU CLUSTER {cluster_id} (Effectif local n = {len(df_cluster)}) :")
    
    for target in targets_cluster:
        y_c = df_cluster[target]
        if y_c.value_counts(normalize=True).min() < 0.15 or len(np.unique(y_c)) < 2:
            print(f"   [-] Target '{target}' ignorée (Trop déséquilibrée ou classe unique dans ce cluster).")
            continue
            
        local_features = [f for f in features_model if f != target]
        X_c_scaled = StandardScaler().fit_transform(df_cluster[local_features])
        X_t, X_v, y_t, y_v = train_test_split(X_c_scaled, y_c, test_size=0.3, random_state=42, stratify=y_c)
        
        log_loc = LogisticRegression(max_iter=500).fit(X_t, y_t)
        score_loc = balanced_accuracy_score(y_v, log_loc.predict(X_v))
        print(f"   [+] Target '{target:8s}' -> Balanced Accuracy Évaluée localement = {score_loc:.4f}")

# ==============================================================================
# 10. BOÎTES À MOUSTACHES DE SYNTHÈSE COMPORTEMENTALE INTER-CLUSTERS
# ==============================================================================
# Les graphiques 10 et 11 s'affichent ici en arrière-plan sans nécessiter de prints consoles.
plt.figure(figsize=(9, 5))
sns.boxplot(x="cluster", y="G3", data=df_encoded, palette="Set2")
plt.title("Distribution des notes finales (G3) selon le cluster d'appartenance")
plt.xlabel("Identifiant du Cluster")
plt.ylabel("Note finale G3 (Mathématiques)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.boxplot(x="cluster", y="absences", data=df_encoded, palette="Set2")
plt.title("Volume d'absentéisme constaté selon le profil des clusters")
plt.xlabel("Identifiant du Cluster")
plt.ylabel("Nombre d'absences relevées")
plt.tight_layout()
plt.show()

# ==============================================================================
# 11. RECHERCHE CROISÉE ET ANALYSE MULTI-DISCIPLINAIRE (MATHS VS PORTUGAIS)
# ==============================================================================
print("\n" + "="*80)
print("11. COMPARAISON DES DEUX MATIÈRES (MATHÉMATIQUES VS PORTUGAIS)")
print("="*80)

merge_keys = [
    "school", "sex", "age", "address", "famsize", "Pstatus", 
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]
df_merge = pd.merge(df_mat, df_por, on=merge_keys, suffixes=("_mat", "_por"))
print(f"[INFO] Nombre d'élèves identifiés et appariés sur les deux matières : {len(df_merge)}")

print("\n[CORR] Matrice de corrélation linéaire entre les notes G3 de Math et Portugais :")
print(df_merge[["G3_mat", "G3_por"]].corr())

# Classification en 3 tranches discrètes (Échec, Moyen, Bon)
def grade_class(x):
    return 0 if x < 10 else (1 if x <= 14 else 2)

df_merge["G3_mat_class"] = df_merge["G3_mat"].apply(grade_class)
df_merge["G3_por_class"] = df_merge["G3_por"].apply(grade_class)

ari_score = adjusted_rand_score(df_merge["G3_mat_class"], df_merge["G3_por_class"])
print(f"\n[RAND INDEX] Indice de Rand Ajusté (ARI) calculé inter-matières : {ari_score:.4f}")

# Graphique 12 : Nuage de points Mathématiques vs Portugais
plt.figure(figsize=(7, 6))
sns.scatterplot(x=df_merge["G3_mat"], y=df_merge["G3_por"], color="seagreen", alpha=0.6)
plt.xlabel("Note finale G3 (Mathématiques)")
plt.ylabel("Note finale G3 (Portugais)")
plt.title("Nuage de points des performances finales inter-matières (G3)")
plt.tight_layout()
plt.show()

# ==============================================================================
# 12. ENRICHISSEMENT : MATRICE DE CONTINGENCE CROISÉE (PROFIL-LIGNE MATH VS POR)
# ==============================================================================
print("\n" + "="*80)
print("12. STRUCTURE DE LA TABLE DE CONTINGENCE CROISÉE (PROFIL-LIGNE)")
print("="*80)

contingency_matrix = pd.crosstab(
    df_merge["G3_mat_class"], 
    df_merge["G3_por_class"], 
    normalize='index'
)

print("\n[CONTINGENCE %] Répartition relative des profils de Mathématiques (lignes) en Portugais (colonnes) :")
print("               (Colonnes indexées : 0 = Échec, 1 = Moyen, 2 = Bon)")
print(contingency_matrix)

# Graphique 13 : Visualisation de la table de contingence
plt.figure(figsize=(7, 5))
sns.heatmap(contingency_matrix, annot=True, fmt=".1%", cmap="YlGnBu", 
            xticklabels=['Échec (<10)', 'Moyen (10-14)', 'Bon (>14)'], 
            yticklabels=['Échec (<10)', 'Moyen (10-14)', 'Bon (>14)'])
plt.title("Distribution relative des élèves de Mathématiques au sein du cours de Portugais")
plt.xlabel("Niveau de performance finale en Portugais")
plt.ylabel("Niveau de performance finale en Mathématiques")
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("FIN DU TRAITEMENT")
print("="*80)