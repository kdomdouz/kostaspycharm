# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Creating a dataset for clustering
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Visualizing the dataset
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Generated Dataset")
plt.show()

# Applying KMeans Clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Getting the centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')

# Plotting the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.title("K-Means Clusters with Centroids")
plt.show()
