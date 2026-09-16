"""The exact set of baseline classifiers screened in 01_Milestone1.ipynb.

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

    Three SVM kernels are compared (linear, RBF, polynomial) since the PCA projection
    in Section 3 suggested a boundary close to linear, worth checking directly against
    the more flexible RBF and polynomial kernels rather than assuming RBF is best.
    LDA/QDA and AdaBoost are added to cover families not otherwise represented: a
    generative linear model (LDA, shared covariance) paired with its quadratic
    counterpart (QDA, per-class covariance), and sequential boosting. A single Decision
    Tree is added alongside Random Forest and AdaBoost so the benefit of ensembling
    (bagging/boosting) over one interpretable tree can be measured directly.
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
        # covariance matrix of the benign class is not full rank (Section 3's
        # correlation heatmap shows several |r| > 0.95 feature pairs)
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=random_state),
    }
