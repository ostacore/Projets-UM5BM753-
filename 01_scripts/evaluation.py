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
