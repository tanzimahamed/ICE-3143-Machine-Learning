# Experiment 9: K-Means Clustering

**Course:** Machine Learning Lab
**Experiment:** 09 – Lab Performance
**Topic:** Unsupervised Learning – K-Means Clustering
**Language / Tools:** Python, NumPy, Matplotlib, Scikit-learn

---

## Objective

To understand and implement the **K-Means clustering algorithm** using Scikit-learn, group unlabeled data points into clusters, and visualize the resulting clusters along with their centers.

---

## Theory in Brief

**K-Means** is an unsupervised learning algorithm that partitions a dataset into **K** clusters. Each data point is assigned to the cluster whose center (centroid) is nearest.

**Algorithm steps:**

1. Choose the number of clusters **K**.
2. Initialize **K** centroids.
3. **Assign** each data point to its nearest centroid (usually by Euclidean distance).
4. **Update** each centroid to the mean of the points assigned to it.
5. Repeat steps 3-4 until the centroids stop changing (convergence) or the maximum number of iterations is reached.

**Objective function (inertia):** K-Means minimizes the sum of squared distances between each point and its assigned centroid.

---

## Implementation Overview

| Step | Description |
|------|-------------|
| 1 | Import libraries (`numpy`, `matplotlib`, `sklearn`) |
| 2 | Generate synthetic data with `make_blobs` (300 samples, 4 centers, `cluster_std=0.60`, `random_state=0`) |
| 3 | Create and fit `KMeans(n_clusters=4, random_state=0)` |
| 4 | Extract `cluster_centers_` and predicted `labels_` |
| 5 | Visualize clusters (colored points) and centroids (red markers) with a scatter plot |

### Core Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Create sample data points
data, labels = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.60)

# Initialize and fit the K-means model
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(data)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
predicted_labels = kmeans.labels_

# Visualize
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, label='Cluster Centers')
plt.legend()
plt.title('K-means Clustering')
plt.show()
```

---

## Result

The model groups the 300 data points into **4 distinct clusters**. The scatter plot shows each cluster in a different color, with the **red markers** indicating the learned cluster centers.

> Add your output image here after uploading it to the repo:
<p align="center">
  <img src="images/exp09.png" width="500" alt="K-means Clustering Result">
</p>

---

## How to Run

**Option 1: Google Colab**
Upload `Ml_09_K_means_Clustering.ipynb` to [Google Colab](https://colab.research.google.com/) and run all cells (all libraries are pre-installed).

**Option 2: Local machine**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install numpy matplotlib scikit-learn jupyter
jupyter notebook Ml_09_K_means_Clustering.ipynb
```

---

## Repository Structure

```
.
├── Ml_09_K_means_Clustering.ipynb   # Lab notebook
├── images/
│   └── kmeans_result.png            # Output plot (optional)
└── README.md
```

---

## Key Takeaways

- K-Means is a simple and fast clustering method for **unlabeled** data.
- The number of clusters **K must be chosen in advance**.
- `random_state` makes the result reproducible.
- It works best when clusters are roughly spherical and similar in size.
- Methods like the **Elbow method** or **Silhouette score** can help choose a good K.

---

## Possible Improvements

- Use the Elbow method / Silhouette score to select the best K.
- Try different values of K and compare results.
- Apply K-Means to a real-world dataset (e.g., Iris, customer segmentation).

---

## Author

**Tanzim Ahamed**

