"""
D04 - Regression on well-balanced quantitative variables.

Goal:
- avoid a target where one class dominates;
- keep quantitative predictors whose distribution is not concentrated in one value;
- compare linear regression on G3 and logistic regression on a balanced G3 class.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "donnees"
OUT_DIR = ROOT / "regression_outputs"
FIG_DIR = ROOT / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = [
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
    "G3",
]

BASE_PREDICTORS = [
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
]


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "mat": pd.read_csv(DATA_DIR / "student-mat.csv"),
        "por": pd.read_csv(DATA_DIR / "student-por.csv"),
    }


def concentration_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in NUMERIC_COLS:
        counts = df[col].value_counts().sort_index()
        max_share = counts.max() / n
        rows.append(
            {
                "dataset": dataset,
                "variable": col,
                "n_unique": int(counts.size),
                "modal_value": counts.idxmax(),
                "modal_share": max_share,
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "std": df[col].std(ddof=0),
            }
        )
    return pd.DataFrame(rows)


def select_balanced_predictors(screening: pd.DataFrame) -> list[str]:
    # We keep variables that are not dominated by one value in either subject.
    keep = []
    for col in BASE_PREDICTORS:
        subset = screening[screening["variable"] == col]
        if (subset["modal_share"] <= 0.55).all():
            keep.append(col)
    return keep


def make_balanced_target(df: pd.DataFrame) -> tuple[pd.Series, float]:
    # Median split gives a meaningful and balanced academic target.
    threshold = float(df["G3"].median())
    y = (df["G3"] >= threshold).astype(int)
    return y, threshold


def linear_model(
    dataset: str, df: pd.DataFrame, predictors: list[str], model_name: str
) -> tuple[dict, pd.DataFrame]:
    X = df[predictors]
    y = df["G3"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)

    metrics = {
        "dataset": dataset,
        "model": model_name,
        "target": "G3",
        "threshold": np.nan,
        "n_features": len(predictors),
        "baseline_metric": mean_absolute_error(y_test, baseline),
        "primary_metric": mean_absolute_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
    }

    reg = model.named_steps["reg"]
    coefs = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model_name,
            "feature": predictors,
            "coef_standardized": reg.coef_,
        }
    ).sort_values("coef_standardized", key=lambda s: s.abs(), ascending=False)
    return metrics, coefs


def logistic_model(
    dataset: str, df: pd.DataFrame, predictors: list[str], model_name: str
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    X = df[predictors]
    y, threshold = make_balanced_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    majority = max(y_test.mean(), 1 - y_test.mean())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy", "f1", "roc_auc"],
    )

    metrics = {
        "dataset": dataset,
        "model": model_name,
        "target": f"G3 >= median ({threshold:.0f})",
        "threshold": threshold,
        "n_features": len(predictors),
        "baseline_metric": majority,
        "primary_metric": accuracy_score(y_test, pred),
        "rmse": np.nan,
        "r2": np.nan,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
        "cv_accuracy_std": cv_scores["test_accuracy"].std(),
        "cv_balanced_accuracy_mean": cv_scores["test_balanced_accuracy"].mean(),
        "cv_f1_mean": cv_scores["test_f1"].mean(),
        "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
    }

    logit = model.named_steps["logit"]
    odds = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model_name,
            "feature": predictors,
            "coef_standardized": logit.coef_[0],
            "odds_ratio": np.exp(logit.coef_[0]),
        }
    ).sort_values("coef_standardized", key=lambda s: s.abs(), ascending=False)
    return metrics, odds, confusion_matrix(y_test, pred)


def plot_target_balance(target_rows: list[dict]) -> None:
    table = pd.DataFrame(target_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table))
    ax.bar(x - 0.18, table["low_or_equal_pct"], width=0.36, label="G3 < median")
    ax.bar(x + 0.18, table["high_pct"], width=0.36, label="G3 >= median")
    ax.axhline(50, color="gray", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(table["dataset"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share (%)")
    ax.set_title("Balanced logistic target by subject")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "D04_target_balance.png", dpi=180)
    plt.close(fig)


def plot_coefficients(df: pd.DataFrame, filename: str, value_col: str, title: str) -> None:
    top = (
        df.assign(abs_value=lambda d: d[value_col].abs())
        .sort_values(["dataset", "abs_value"], ascending=[True, False])
        .groupby("dataset")
        .head(8)
        .copy()
    )
    top["label"] = top["dataset"] + " - " + top["feature"]
    top = top.sort_values(value_col)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(top[value_col] >= 0, "tomato", "steelblue")
    ax.barh(top["label"], top[value_col], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    data = load_data()

    screening = pd.concat(
        [concentration_table(df, name) for name, df in data.items()],
        ignore_index=True,
    )
    predictors = select_balanced_predictors(screening)
    excluded = [col for col in BASE_PREDICTORS if col not in predictors]
    feature_sets = {
        "context_only": predictors,
        "with_G1_G2": predictors + ["G1", "G2"],
    }

    metrics_rows = []
    coef_rows = []
    odds_rows = []
    target_rows = []
    confusions = []

    for dataset, df in data.items():
        y, threshold = make_balanced_target(df)
        target_rows.append(
            {
                "dataset": dataset,
                "threshold": threshold,
                "low_or_equal_n": int((y == 0).sum()),
                "high_n": int((y == 1).sum()),
                "low_or_equal_pct": 100 * (y == 0).mean(),
                "high_pct": 100 * (y == 1).mean(),
            }
        )

        for suffix, feature_list in feature_sets.items():
            lin_metrics, lin_coefs = linear_model(
                dataset, df, feature_list, f"linear_{suffix}"
            )
            log_metrics, odds, cm = logistic_model(
                dataset, df, feature_list, f"logistic_{suffix}"
            )

            metrics_rows.extend([lin_metrics, log_metrics])
            coef_rows.append(lin_coefs)
            odds_rows.append(odds)
            confusions.append(
                {
                    "dataset": dataset,
                    "model": f"logistic_{suffix}",
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    linear_coefs = pd.concat(coef_rows, ignore_index=True)
    odds_ratios = pd.concat(odds_rows, ignore_index=True)
    targets = pd.DataFrame(target_rows)
    confusions_df = pd.DataFrame(confusions)

    screening.to_csv(OUT_DIR / "D04_distribution_screening.csv", index=False)
    pd.Series(predictors, name="selected_predictors").to_csv(
        OUT_DIR / "D04_selected_predictors.csv", index=False
    )
    pd.Series(excluded, name="excluded_predictors").to_csv(
        OUT_DIR / "D04_excluded_predictors.csv", index=False
    )
    targets.to_csv(OUT_DIR / "D04_target_balance.csv", index=False)
    metrics.to_csv(OUT_DIR / "D04_regression_metrics.csv", index=False)
    linear_coefs.to_csv(OUT_DIR / "D04_linear_coefficients.csv", index=False)
    odds_ratios.to_csv(OUT_DIR / "D04_logistic_odds_ratios.csv", index=False)
    confusions_df.to_csv(OUT_DIR / "D04_logistic_confusion_matrices.csv", index=False)

    plot_target_balance(target_rows)
    for suffix in feature_sets:
        plot_coefficients(
            linear_coefs[linear_coefs["model"] == f"linear_{suffix}"],
            f"D04_linear_coefficients_{suffix}.png",
            "coef_standardized",
            f"Linear regression coefficients on G3 - {suffix}",
        )
        plot_coefficients(
            odds_ratios[odds_ratios["model"] == f"logistic_{suffix}"],
            f"D04_logistic_coefficients_{suffix}.png",
            "coef_standardized",
            f"Logistic regression coefficients for high G3 - {suffix}",
        )

    print("Selected quantitative predictors:")
    print(", ".join(predictors))
    print("\nExcluded because one value dominates in at least one subject:")
    print(", ".join(excluded) if excluded else "None")
    print("\nTarget balance:")
    print(targets.round(3).to_string(index=False))
    print("\nRegression metrics:")
    cols = [
        "dataset",
        "model",
        "target",
        "baseline_metric",
        "primary_metric",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "r2",
        "cv_accuracy_mean",
    ]
    print(metrics.reindex(columns=cols).round(3).to_string(index=False))
    print("\nTop logistic odds ratios:")
    print(
        odds_ratios.groupby(["dataset", "model"])
        .head(6)[["dataset", "model", "feature", "coef_standardized", "odds_ratio"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
