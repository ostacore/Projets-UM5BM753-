"""The set of baseline classifiers used for the initial supervised screening.

Kept separate from evaluation.py: this file only says *which* models and
hyperparameters to compare, not *how* to evaluate them.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


def get_baseline_classifiers(random_state: int = 42) -> dict:
    """Return a dictionary of standard baseline classifiers.

    Covers linear, kernel, generative and ensemble families -- including a plain
    Decision Tree alongside Random Forest/AdaBoost to measure the benefit of
    ensembling directly.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVM (linear)": SVC(kernel="linear", probability=True, random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "SVM (poly)": SVC(kernel="poly", degree=3, probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),  # unregularized QDA fails:
        # covariance matrix of the benign class is not full rank (several feature
        # pairs have |r| > 0.95, see the correlation heatmap)
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=random_state),
    }


def tune_logreg_with_1se(X, y, param_grid, cv_folds=10, random_state=42):
    """Perform a cross-validated grid search for Logistic Regression and apply the 1-SE rule.

    Tunes the regularization parameter C by maximizing the chosen scoring metric (recall),
    then selects the simplest model (smallest C) whose mean score is within one
    standard error of the absolute best score, ensuring stronger regularization.
    """
    grid_logreg = GridSearchCV(
        LogisticRegression(random_state=random_state, max_iter=5000),
        param_grid, cv=cv_folds, scoring='recall', n_jobs=-1
    )
    grid_logreg.fit(X, y)

    res = pd.DataFrame(grid_logreg.cv_results_)
    best_row = res.loc[res["mean_test_score"].idxmax()]
    one_se_floor = best_row["mean_test_score"] - (best_row["std_test_score"] / np.sqrt(cv_folds))
    one_se_C = res.loc[res["mean_test_score"] >= one_se_floor, "param_C"].astype(float).min()

    return grid_logreg, res, one_se_C


def benchmark_feature_selectors(selectors_dict, X_train, y_train, cv_folds, scoring_dict, random_state=42):
    """Evaluate multiple feature selection methods using a baseline Logistic Regression.

    Fits each selector to the training data, applies the resulting feature mask,
    and cross-validates a baseline model on the reduced feature space. Returns a
    DataFrame ranking the methods by a combined score of clinical performance
    (Recall/F1/F2) and sparsity (fewer features preferred).
    """
    benchmark_rows = []
    for name, selector in selectors_dict.items():
        mask = selector.get_support()
        cv_results = cross_validate(
            LogisticRegression(random_state=random_state), X_train[:, mask], y_train,
            cv=cv_folds, scoring=scoring_dict
        )
        benchmark_rows.append({
            "model": name,
            "n_features": int(mask.sum()),
            "accuracy_mean": cv_results["test_accuracy"].mean(),
            "precision_mean": cv_results["test_precision"].mean(),
            "recall_mean": cv_results["test_recall"].mean(),
            "f1_mean": cv_results["test_f1"].mean(),
            "f2_mean": cv_results["test_f2"].mean(),
            "roc_auc_mean": cv_results["test_roc_auc"].mean(),
        })

    df_benchmark = pd.DataFrame(benchmark_rows)
    _ranking_metrics = ["recall_mean", "f1_mean", "f2_mean"]
    ranks_perf = df_benchmark.set_index("model")[_ranking_metrics].rank(ascending=False)
    ranks_feat = df_benchmark.set_index("model")["n_features"].rank(ascending=True)

    df_benchmark["avg_rank_score"] = pd.concat([ranks_perf, ranks_feat], axis=1).mean(axis=1).values
    return df_benchmark.sort_values("avg_rank_score")


