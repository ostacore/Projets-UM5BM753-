"""Plotting functions for this project's figures: data exploration, supervised-model
results (ROC, confusion matrix, decision tree), and clustering results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Exploratory data analysis
def plot_class_balance(dframe: pd.DataFrame, out_path: str):
    """Bar plot of the diagnosis class distribution."""
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = dframe["diagnosis"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette=["#4C72B0", "#C44E52"], legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, f"{v} ({100*v/len(dframe):.1f}%)", ha="center")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Number of samples")
    ax.set_title("Class distribution: Benign (B) vs Malignant (M)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_correlation_heatmap(dframe: pd.DataFrame, feature_list: list, out_path: str):
    """Heatmap of the Pearson correlation matrix across all features."""
    corr = dframe[feature_list].corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Pearson correlation matrix - 30 morphological features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def get_top_correlated_pairs(dframe: pd.DataFrame, feature_list: list,
                              threshold: float = 0.95) -> list:
    """Return feature pairs with |Pearson correlation| above the given threshold,
    sorted by decreasing correlation."""
    corr = dframe[feature_list].corr().abs()
    pairs = []
    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):
            if corr.iloc[i, j] > threshold:
                pairs.append((feature_list[i], feature_list[j], corr.iloc[i, j]))
    return sorted(pairs, key=lambda x: -x[2])


def plot_boxplots_by_diagnosis(dframe: pd.DataFrame, feature_list: list, out_path: str,
                                n_cols: int = 5):
    """Grid of boxplots comparing each feature's distribution across diagnosis groups."""
    n_feat = len(feature_list)
    n_rows = int(np.ceil(n_feat / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i, f in enumerate(feature_list):
        sns.boxplot(data=dframe, x="diagnosis", y=f, hue="diagnosis",
                    palette=["#4C72B0", "#C44E52"], legend=False, ax=axes[i])
        axes[i].set_title(f, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature distributions by diagnosis (mean features)", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pca_projection(X_scaled: np.ndarray, y: np.ndarray, out_path: str,
                         random_state: int = 42):
    """2D PCA projection of the standardized data, colored by diagnosis."""
    pca_local = PCA(n_components=2, random_state=random_state)
    X_pca_local = pca_local.fit_transform(X_scaled)
    var_explained = pca_local.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = np.where(y == 1, "Malignant", "Benign")
    sns.scatterplot(x=X_pca_local[:, 0], y=X_pca_local[:, 1], hue=labels,
                     palette={"Benign": "#4C72B0", "Malignant": "#C44E52"},
                     alpha=0.8, ax=ax)
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)")
    ax.set_title("PCA projection of the 30 standardized features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()
    return pca_local, X_pca_local


def plot_pca_scree(X_scaled: np.ndarray, out_path: str, n_components: int = 15,
                    random_state: int = 42):
    """Scree plot: individual and cumulative explained variance per component."""
    pca_local = PCA(n_components=n_components, random_state=random_state)
    pca_local.fit(X_scaled)
    cum_var = np.cumsum(pca_local.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, n_components + 1), pca_local.explained_variance_ratio_,
           alpha=0.6, label="Individual")
    ax.plot(range(1, n_components + 1), cum_var, marker="o", color="#C44E52",
            label="Cumulative")
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA scree plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()
    return cum_var

# Supervised models 

def plot_roc_curves(classifiers: dict, X_tr, y_tr, X_te, y_te, out_path: str,
                     title: str = "ROC curves (held-out test set)"):
    """Fit each classifier on the train set and plot ROC curves on the held-out test set."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        y_score = clf.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_cv_score_boxplot(cv_scores: pd.DataFrame, metric: str, out_path: str,
                           title: str = None):
    """Boxplot of per-fold cross-validation scores, one box per model. Shows the
    spread across folds, not just the mean shown in the summary table."""
    fig, ax = plt.subplots(figsize=(9, 5))
    order = cv_scores.groupby("model")[metric].median().sort_values(ascending=False).index
    sns.boxplot(data=cv_scores, x="model", y=metric, order=order, hue="model",
                palette="Blues_r", legend=False, ax=ax)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_title(title or f"{metric} across CV folds, by model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_confusion_matrices(classifiers: dict, X_tr, y_tr, X_te, y_te, out_path: str):
    """Plot a grid of confusion matrices, one per fitted classifier, on the test set."""
    n = len(classifiers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, clf) in zip(axes, classifiers.items()):
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        cm = confusion_matrix(y_te, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=["Benign", "Malignant"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()

# Unsupervised models 

def plot_elbow_and_silhouette(kmeans_results: pd.DataFrame, out_path: str):
    """Side-by-side elbow (inertia) and silhouette score plots across k."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(kmeans_results["k"], kmeans_results["inertia"], marker="o")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method")

    axes[1].plot(kmeans_results["k"], kmeans_results["silhouette"], marker="o", color="#C44E52")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette analysis")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_silhouette_by_k(results: pd.DataFrame, out_path: str,
                          title: str = "Silhouette analysis"):
    """Silhouette score across a range of k, single panel (for methods without
    centroids: Agglomerative, Spectral -- no inertia to show alongside it)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results["k"], results["silhouette"], marker="o", color="#C44E52")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_cluster_vs_diagnosis(X_pca_2d: np.ndarray, y_true: np.ndarray, cluster_labels: np.ndarray,
                               out_path: str, title: str = "Predicted clusters vs true diagnosis"):
    """Side-by-side PCA scatter comparing true diagnosis and predicted clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    diag_labels = np.where(y_true == 1, "Malignant", "Benign")
    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=diag_labels,
                     palette={"Benign": "#4C72B0", "Malignant": "#C44E52"},
                     alpha=0.8, ax=axes[0])
    axes[0].set_title("True diagnosis")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=cluster_labels,
                     palette="Set2", alpha=0.8, ax=axes[1])
    axes[1].set_title("Predicted clusters")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_all_clusters_vs_diagnosis(X_pca_2d: np.ndarray, y_true: np.ndarray,
                                    cluster_labels_dict: dict, out_path: str,
                                    n_cols: int = 4):
    """Grid comparing the true diagnosis to every clustering method's partition."""
    diag_labels = np.where(y_true == 1, "Malignant", "Benign")
    n_panels = 1 + len(cluster_labels_dict)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.8 * n_rows))
    axes = axes.flatten()

    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=diag_labels,
                     palette={"Benign": "#4C72B0", "Malignant": "#C44E52"},
                     alpha=0.8, ax=axes[0])
    axes[0].set_title("True diagnosis")
    axes[0].legend(fontsize=8, loc="best", title=None)

    # Each panel gets its own legend: cluster numbers are arbitrary per algorithm,
    # not comparable across panels.
    for i, (name, labels) in enumerate(cluster_labels_dict.items(), start=1):
        sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=labels,
                         palette="Set2", alpha=0.8, ax=axes[i])
        axes[i].set_title(name, fontsize=10)
        axes[i].legend(fontsize=8, loc="best", title=None)

    for ax in axes[:n_panels]:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle("True diagnosis vs. predicted clusters, all methods")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_ari_barplot(clustering_eval: pd.DataFrame, out_path: str,
                      title: str = "ARI vs. true diagnosis, by clustering method"):
    """Bar chart of ARI vs. the true diagnosis, one bar per clustering method.
    ARI (not silhouette) is what actually drives the method choice here: a high
    silhouette can still mean a poor match to the true diagnosis (e.g. one giant
    cluster and a tiny one), so silhouette alone is not shown for comparison."""
    ordered = clustering_eval.sort_values("ARI_vs_diagnosis", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=ordered, x="model", y="ARI_vs_diagnosis", hue="model",
                palette="Blues_r", legend=False, ax=ax)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["model"], rotation=45, ha="right")
    ax.set_ylabel("ARI vs. true diagnosis")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_decision_tree_structure(X_tr, y_tr, feature_list: list, out_path: str,
                                  max_depth: int = 3, random_state: int = 42):
    """Fit a depth-limited Decision Tree and plot its structure (max_depth is
    capped for readability only, not used for the model's real performance)."""
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree_clf.fit(X_tr, y_tr)
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(tree_clf, feature_names=feature_list, class_names=["Benign", "Malignant"],
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"Decision Tree structure (max_depth={max_depth}, for readability only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()
