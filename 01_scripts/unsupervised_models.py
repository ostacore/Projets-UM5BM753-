"""The exact set of baseline clusterers screened in 01_Milestone1.ipynb.

Kept separate from evaluation.py: this file only says *which* algorithms and
hyperparameters to compare, not *how* to evaluate them.
"""

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN


def get_baseline_clusterers(random_state: int = 42) -> dict:
    """Return standard clustering algorithms configured to look for 2 clusters.

    All three linkage criteria are compared for Agglomerative clustering (Ward,
    average, complete), since the linkage rule changes what "distance between
    clusters" means and can noticeably change the resulting partition. Spectral
    clustering is added as a graph-based alternative that can capture non-convex
    cluster shapes that K-Means/Ward cannot.
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
