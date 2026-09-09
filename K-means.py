import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#give the program access to the data
breast_data = pd.read_csv('/Users/apple/Desktop/Project UE1_AIDA/10_Breast_Cancer_Wisconsin_Diagnostic.csv')

#transform the diagnosis into M = 1 and B = 0
breast_data["diagnosis"] = breast_data["diagnosis"].map({"M": 0, "B": 1})

#extract only the features (we remove the id and the diagnosis)
data_features = breast_data.drop(["id", "diagnosis"], axis=1)

#Standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_features)

#K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_clusters = kmeans.fit_predict(features_scaled)
breast_data["km_cluster"] = kmeans_clusters


table_km = pd.crosstab(breast_data["diagnosis"], breast_data["km_cluster"])

accuracy = accuracy_score(breast_data["diagnosis"], breast_data["km_cluster"])

print(table_km)
print(accuracy)

inertias = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)

    score = silhouette_score(features_scaled, clusters)
    silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.show()


pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

#PCA with K-means cluster
plt.figure(figsize=(8,6))
plt.scatter(features_pca[:, 0],features_pca[:, 1],c=breast_data["km_cluster"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - K-means clusters")
plt.show()

#PCA with real diagnosis
plt.figure(figsize=(8,6))
plt.scatter(features_pca[:, 0],features_pca[:, 1],c=breast_data["diagnosis"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Diagnostic réel")
plt.show()

#Importance of each component
pca_components = pd.DataFrame(pca.components_,columns=data_features.columns,index=["PC1", "PC2"])
print(pca_components.loc["PC1"].abs().sort_values(ascending=False))

#how much PC1 and PC2 are important
print(pca.explained_variance_ratio_.sum())
print(pca.explained_variance_ratio_)