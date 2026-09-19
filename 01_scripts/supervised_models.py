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
