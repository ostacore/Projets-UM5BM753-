"""Reusable model-evaluation functions, shared across notebooks.

`cross_validate_classifiers` and `run_kmeans_range` are extracted as-is from
01_Milestone1.ipynb (screening stage: many models, cross-validated).
`evaluate_classifier` is new: a single-model, train/test evaluation used by the
"zoom" analyses (one tuned model, evaluated once on the held-out test set) rather
than by the screening stage.
"""

import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)


def cross_validate_classifiers(classifiers: dict, X, y, cv_folds: int = 10,
                                random_state: int = 42) -> pd.DataFrame:
    """Run stratified k-fold cross-validation for each classifier and summarize results."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    rows = []
    for name, clf in classifiers.items():
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def evaluate_classifier(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit one model and report the same 5 metrics as the screening stage, on the
    held-out test set (accuracy, precision, recall, f1, roc_auc). For a single
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
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def run_kmeans_range(X, k_range=range(2, 8), random_state: int = 42) -> pd.DataFrame:
    """Run K-Means for a range of k values and report inertia and silhouette score."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)
