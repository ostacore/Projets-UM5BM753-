"""The set of baseline clusterers used for the initial unsupervised screening.

Kept separate from evaluation.py: this file only says *which* algorithms and
hyperparameters to compare, not *how* to evaluate them.
"""

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN


def get_baseline_clusterers(random_state: int = 42) -> dict:
    """Return standard clustering algorithms configured to look for 2 clusters.

    Covers centroid-based (K-Means), hierarchical (3 linkages), graph-based
    (Spectral) and density-based (DBSCAN) approaches.
    """
    return {
        "K-Means": KMeans(n_clusters=2, n_init=10, random_state=random_state),
        "Agglomerative (ward)": AgglomerativeClustering(n_clusters=2, linkage="ward"),
        "Agglomerative (average)": AgglomerativeClustering(n_clusters=2, linkage="average"),
        "Agglomerative (complete)": AgglomerativeClustering(n_clusters=2, linkage="complete"),
        "Spectral": SpectralClustering(n_clusters=2, random_state=random_state,
                                        affinity="nearest_neighbors"),
        "DBSCAN": DBSCAN(eps=2.5, min_samples=10),
    }
