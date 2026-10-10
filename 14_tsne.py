import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from prep import load_clean_data

df = load_clean_data()
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_scaled = StandardScaler().fit_transform(X)

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm", alpha=0.6)
plt.title("t-SNE (colour = survived, only for display)")
plt.show()