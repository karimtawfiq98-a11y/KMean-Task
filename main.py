import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create hypothetical points
points = np.array([
    [2, 3],
    [3, 4],
    [2, 4],
    [8, 8],
    [9, 8],
    [10, 10],
    [25, 30],
    [24, 29],
    [26, 31]
])

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(points)

# Get cluster centers
centers = kmeans.cluster_centers_

# Plot the results
plt.scatter(points[:, 0], points[:, 1],
            c=kmeans.labels_, s=100)

plt.scatter(centers[:, 0], centers[:, 1],
            color='red', marker='X', s=200)

plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()