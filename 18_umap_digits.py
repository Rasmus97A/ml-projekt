import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score

digits = load_digits()
X = digits.data / 16.0       # pixels already share one unit (0-16), so no StandardScaler needed
y = digits.target

X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

clusters = KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(X)
print("Adjusted Rand Index:", round(adjusted_rand_score(y, clusters), 3))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=8)
axes[0].set_title("UMAP, coloured by TRUE digit")
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap="tab10", s=8)
axes[1].set_title("UMAP, coloured by KMeans cluster (labels never used)")
plt.show()