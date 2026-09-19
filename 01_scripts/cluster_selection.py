"""Choosing the number of clusters for Spectral Clustering (Section 4.5's zoom):
this function searches for a hyperparameter, it does not evaluate an
already-decided clustering.
"""

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


def run_spectral_range(X, k_range=range(2, 8), random_state: int = 42) -> pd.DataFrame:
    """Run Spectral Clustering for a range of n_clusters and report the silhouette
    score (no inertia column: Spectral Clustering has no centroids)."""
    rows = []
    for k in k_range:
        sc = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", random_state=random_state)
        labels = sc.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": sil})
    return pd.DataFrame(rows)
