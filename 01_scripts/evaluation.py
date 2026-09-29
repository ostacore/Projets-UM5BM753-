"""Score an already-decided classifier or clustering: cross-validated screening,
single-model test-set metrics, and clustering metrics vs. the true diagnosis.
Choosing a hyperparameter first (number of clusters, DBSCAN's eps) is a
different concern, handled in cluster_selection.py instead.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
                              recall_score, f1_score, fbeta_score, roc_auc_score,
                              make_scorer, homogeneity_score, completeness_score,
                              v_measure_score, adjusted_rand_score)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# Supervised models

def cross_validate_classifiers(classifiers: dict, X, y, cv_folds: int = 10,
                                random_state: int = 42) -> pd.DataFrame:
    """Run stratified k-fold cross-validation for each classifier and summarize results.

    F2 (beta=2) is included alongside F1: it weighs recall twice as heavily as
    precision, matching this project's clinical priority of not missing a
    malignant case better than F1's equal weighting.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f2": make_scorer(fbeta_score, beta=2),
        "roc_auc": "roc_auc",
    }

    rows = []
    for name, clf in classifiers.items():
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def cross_validate_classifiers_folds(classifiers: dict, X, y, metric: str = "recall",
                                      cv_folds: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Same cross-validation as cross_validate_classifiers, but return the raw
    per-fold scores for one metric (long format: one row per model per fold),
    for a boxplot of score spread across folds instead of just the mean.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorer = make_scorer(fbeta_score, beta=2) if metric == "f2" else metric

    rows = []
    for name, clf in classifiers.items():
        scores = cross_validate(clf, X, y, cv=cv, scoring=scorer)
        for fold_score in scores["test_score"]:
            rows.append({"model": name, metric: fold_score})
    return pd.DataFrame(rows)


def evaluate_classifier(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit one model and report the same 6 metrics as the screening stage, on the
    held-out test set (accuracy, precision, recall, f1, f2, roc_auc). For a single
    already-tuned model, not a substitute for cross-validation during model selection.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "f2": fbeta_score(y_test, y_pred, beta=2),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def bootstrap_logreg_coefficients(model, X_train, y_train, feature_names, n_bootstrap=1000, random_state=42):
    """Estimate 95% confidence intervals for Logistic Regression coefficients via bootstrapping.

    Resamples the training data with replacement and refits the model over multiple
    iterations to compute the empirical 2.5th and 97.5th percentiles for each feature's
    coefficient, assessing the stability of the learned weights.
    """
    coefs = pd.Series(model.coef_[0], index=feature_names)
    rng = np.random.default_rng(random_state)
    boot_coefs = np.empty((n_bootstrap, len(feature_names)))

    for b in range(n_bootstrap):
        idx = rng.integers(0, len(X_train), len(X_train))
        boot_model = LogisticRegression(C=model.C, random_state=random_state, max_iter=5000)
        boot_model.fit(X_train[idx], y_train[idx])
        boot_coefs[b] = boot_model.coef_[0]

    ci_low, ci_high = np.percentile(boot_coefs, [2.5, 97.5], axis=0)

    return pd.DataFrame({
        "coefficient": coefs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "odds_ratio_per_SD": np.exp(coefs),
    }).sort_values("coefficient", key=np.abs, ascending=False)


def compute_logodds_contributions(model, selected_features, train_df_selected, X_wpbc_scaled_df, wpbc_status,
                                  missed_index):
    """Calculate the mean contribution of each feature to the Logistic Regression log-odds.

    Extracts the model's coefficients and applies them to the standardized feature values
    to compute the average log-odds impact for specific patient subgroups (Original Malignant,
    WPBC Correct, WPBC Missed).
    """
    coefs = pd.Series(model.coef_[0], index=selected_features)
    intercept = model.intercept_[0]

    def mean_contribution(df_scaled):
        return (df_scaled[selected_features] * coefs).mean()

    contrib_original_malignant = mean_contribution(
        train_df_selected[train_df_selected["diagnosis"] == "Original - Malignant"])
    contrib_wpbc_correct = mean_contribution(
        X_wpbc_scaled_df.loc[wpbc_status[wpbc_status == "Prognostic - Predicted malignant"].index])
    contrib_wpbc_missed = mean_contribution(X_wpbc_scaled_df.loc[missed_index])

    contrib_table = pd.DataFrame({
        "Original - Malignant (train avg.)": contrib_original_malignant,
        "Prognostic - Predicted malignant (avg.)": contrib_wpbc_correct,
        "Prognostic - Missed (avg.)": contrib_wpbc_missed,
    })
    contrib_table.loc["intercept"] = intercept
    contrib_table.loc["log-odds (sum)"] = contrib_table.drop(index="intercept").sum() + intercept

    return contrib_table

# Unsupervised models

def evaluate_clusterers(clusterers: dict, X, y_true) -> pd.DataFrame:
    """Evaluate clustering algorithms against the true diagnosis labels.

    Silhouette score is label-free (internal); homogeneity, completeness,
    V-measure, and ARI use the diagnosis only for external validation, never for
    fitting the clustering algorithm.
    """
    rows = []
    for name, algo in clusterers.items():
        labels = algo.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil = silhouette_score(X, labels) if n_clusters > 1 else np.nan
        homogeneity = homogeneity_score(y_true, labels)
        completeness = completeness_score(y_true, labels)
        v_measure = v_measure_score(y_true, labels)
        ari = adjusted_rand_score(y_true, labels)
        rows.append({"model": name, "n_clusters_found": n_clusters,
                      "silhouette": sil, "homogeneity": homogeneity,
                      "completeness": completeness, "v_measure": v_measure,
                      "ARI_vs_diagnosis": ari})
    return pd.DataFrame(rows)


