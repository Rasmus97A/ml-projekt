import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from prep import load_clean_data

df = load_clean_data()
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print("Variance kept by each component:", pca.explained_variance_ratio_.round(3))

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Passengers squeezed into 2D (colour = survived, only for display)")
plt.show()