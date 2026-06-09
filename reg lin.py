# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =========================================================
# DATASET
# =========================================================

df = pd.read_csv("dataset/student-mat.csv")

# =========================================================
# VARIABLES EXPLICATIVES
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
    "goout"
]

# =========================================================
# ENCODAGE
# =========================================================

df_encoded = df.copy()

binary_map = {
    "yes": 1,
    "no": 0,
    "F": 0,
    "M": 1
}

for col in ["sex", "internet", "famsup"]:
    df_encoded[col] = df_encoded[col].map(binary_map)

# =========================================================
# STANDARDISATION
# =========================================================

X_base = df_encoded[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_base)

# =========================================================
# FONCTION EVALUATION CLASSIFICATION
# =========================================================

def evaluate_classifier(model, X_train, X_test, y_train, y_test, name="Model"):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    print("\n" + "="*60)
    print(name)
    print("="*60)

    print(f"Accuracy           : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy  : {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-score           : {f1_score(y_test, y_pred):.3f}")

    if y_prob is not None:
        print(f"ROC AUC            : {roc_auc_score(y_test, y_prob):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    return model

# =========================================================
# =========================================================
# 1. REGRESSION LOGISTIQUE SUR HIGHER
# =========================================================
# =========================================================

print("\n" + "#"*70)
print("REGRESSION LOGISTIQUE — HIGHER")
print("#"*70)

y = df_encoded["higher"].map(binary_map)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

logreg_higher = LogisticRegression(max_iter=500)

evaluate_classifier(
    logreg_higher,
    X_train,
    X_test,
    y_train,
    y_test,
    "Prediction de HIGHER"
)

# =========================================================
# COEFFICIENTS
# =========================================================

coeffs = pd.DataFrame({
    "Variable": features,
    "Coefficient": logreg_higher.coef_[0]
})

coeffs["AbsCoeff"] = np.abs(coeffs["Coefficient"])

coeffs = coeffs.sort_values(
    by="AbsCoeff",
    ascending=False
)

print("\nVariables importantes pour HIGHER:")
print(coeffs)

# =========================================================
# VISUALISATION
# =========================================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=coeffs,
    x="Coefficient",
    y="Variable"
)

plt.title("Importance des variables — HIGHER")
plt.show()

# =========================================================
# =========================================================
# 2. REGRESSION LINEAIRE SUR G3 SANS G1/G2
# =========================================================
# =========================================================

print("\n" + "#"*70)
print("REGRESSION LINEAIRE — G3 SANS G1/G2")
print("#"*70)

y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42
)

linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

print(f"R2    : {r2_score(y_test, y_pred):.3f}")
print(f"RMSE  : {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

coeffs_g3 = pd.DataFrame({
    "Variable": features,
    "Coefficient": linreg.coef_
})

coeffs_g3["AbsCoeff"] = np.abs(coeffs_g3["Coefficient"])

coeffs_g3 = coeffs_g3.sort_values(
    by="AbsCoeff",
    ascending=False
)

print("\nVariables influentes sur G3:")
print(coeffs_g3)

# =========================================================
# =========================================================
# 3. REGRESSION SUR ABSENCES
# =========================================================
# =========================================================

print("\n" + "#"*70)
print("REGRESSION LINEAIRE — ABSENCES")
print("#"*70)

y = df["absences"]

linreg_abs = LinearRegression()

linreg_abs.fit(X_train, y_train)

y_pred_abs = linreg_abs.predict(X_test)

print(f"R2    : {r2_score(y_test, y_pred_abs):.3f}")
print(f"RMSE  : {np.sqrt(mean_squared_error(y_test, y_pred_abs)):.3f}")

coeffs_abs = pd.DataFrame({
    "Variable": features,
    "Coefficient": linreg_abs.coef_
})

coeffs_abs["AbsCoeff"] = np.abs(coeffs_abs["Coefficient"])

coeffs_abs = coeffs_abs.sort_values(
    by="AbsCoeff",
    ascending=False
)

print("\nVariables influentes sur absences:")
print(coeffs_abs)

# =========================================================
# =========================================================
# 4. ACP + KMEANS
# =========================================================
# =========================================================

print("\n" + "#"*70)
print("ACP + KMEANS")
print("#"*70)

X_cluster = pd.get_dummies(
    df.drop(columns=["G3"]),
    drop_first=True
)

scaler_cluster = StandardScaler()

X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

pca = PCA(n_components=0.8)

X_pca = pca.fit_transform(X_cluster_scaled)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

clusters = kmeans.fit_predict(X_pca)

df["cluster"] = clusters

print("\nTaille des clusters:")
print(df["cluster"].value_counts())

# =========================================================
# =========================================================
# 5. REGRESSION PAR CLUSTER
# =========================================================
# =========================================================

print("\n" + "#"*70)
print("REGRESSION PAR CLUSTER")
print("#"*70)

targets = [
    "higher",
    "internet",
    "romantic",
    "paid"
]

for cluster_id in sorted(df["cluster"].unique()):

    print("\n" + "="*70)
    print(f"CLUSTER {cluster_id}")
    print("="*70)

    df_cluster = df[df["cluster"] == cluster_id].copy()

    # Encodage
    for col in ["sex", "internet", "famsup", "higher", "paid", "romantic"]:
        df_cluster[col] = df_cluster[col].map(binary_map)

    X = df_cluster[features]

    scaler_local = StandardScaler()

    X_scaled_local = scaler_local.fit_transform(X)

    for target in targets:

        print("\n" + "-"*60)
        print(f"Target : {target}")
        print("-"*60)

        y = df_cluster[target]

        # Eviter les classes trop déséquilibrées
        counts = y.value_counts(normalize=True)

        print("\nDistribution:")
        print(counts)

        if counts.min() < 0.15:
            print("⚠️ Variable trop déséquilibrée — analyse ignorée")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_local,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )

        model = LogisticRegression(max_iter=500)

        try:

            evaluate_classifier(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                f"{target} — Cluster {cluster_id}"
            )

            coeffs_cluster = pd.DataFrame({
                "Variable": features,
                "Coefficient": model.coef_[0]
            })

            coeffs_cluster["AbsCoeff"] = np.abs(
                coeffs_cluster["Coefficient"]
            )

            coeffs_cluster = coeffs_cluster.sort_values(
                by="AbsCoeff",
                ascending=False
            )

            print("\nVariables importantes:")
            print(coeffs_cluster)

        except Exception as e:
            print(f"Erreur : {e}")

# =========================================================
# =========================================================
# 6. MATRICE DE CORRELATION
# =========================================================
# =========================================================

plt.figure(figsize=(12,10))

corr_vars = features + ["G3", "absences"]

corr_df = df_encoded[corr_vars]

sns.heatmap(
    corr_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matrice de corrélation")
plt.show()

# =========================================================
# =========================================================
# 7. VISUALISATION G3 PAR CLUSTER
# =========================================================
# =========================================================

plt.figure(figsize=(10,6))

sns.boxplot(
    x="cluster",
    y="G3",
    data=df
)

plt.title("Distribution de G3 par cluster")
plt.show()

# =========================================================
# =========================================================
# 8. VISUALISATION ABSENCES PAR CLUSTER
# =========================================================
# =========================================================

plt.figure(figsize=(10,6))

sns.boxplot(
    x="cluster",
    y="absences",
    data=df
)

plt.title("Distribution des absences par cluster")
plt.show()