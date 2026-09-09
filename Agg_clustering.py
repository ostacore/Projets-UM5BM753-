import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Load data
breast_data = pd.read_csv('/Users/apple/Desktop/Project UE1_AIDA/10_Breast_Cancer_Wisconsin_Diagnostic.csv')

#Diagnosis
breast_data["diagnosis"] = breast_data["diagnosis"].map({"M": 0, "B": 1})

#Features
data_features = breast_data.drop(["id", "diagnosis"], axis=1)

#Standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_features)

#Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=2)
agg_clusters = agg.fit_predict(features_scaled)

breast_data["agg_cluster"] = agg_clusters

#Comparison with the diagnosis
table_agg = pd.crosstab(breast_data["diagnosis"],breast_data["agg_cluster"])
print(table_agg)

#silhouette
silhouette_agg = silhouette_score(features_scaled,agg_clusters)
print("Silhouette Score :", silhouette_agg)

#Accuracy
accuracy_agg = accuracy_score(breast_data["diagnosis"],breast_data["agg_cluster"])
print("Accuracy :", accuracy_agg)

#Dendogram
linkage_matrix = linkage(features_scaled, method="ward")

plt.figure(figsize=(12, 6))

dendrogram(linkage_matrix,truncate_mode="lastp",p=30)

plt.xlabel("Patients / groupes")
plt.ylabel("Distance")
plt.title("Dendrogramme - Agglomerative Clustering")

plt.show()

#PCA 
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)
plt.figure(figsize=(10, 7))

plt.scatter(features_pca[:, 0],features_pca[:, 1],c=agg_clusters,alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Agglomerative Clustering")

plt.colorbar(label="Cluster")
plt.show()
