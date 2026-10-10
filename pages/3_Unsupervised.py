import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from prep import load_clean_data

st.title("Unsupervised learning on Titanic")

drop_ports = st.checkbox("Remove the port columns (Embarked_C/Q/S)", value=True)
st.caption("Unticked, KMeans splits passengers almost purely by port. Ticked, it finds passenger types.")

@st.cache_data
def prepare(drop_ports):
    df = load_clean_data()
    X = df.drop(columns=["Survived"])
    if drop_ports:
        X = X.drop(columns=["Embarked_C", "Embarked_Q", "Embarked_S"])
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
    return df, X_scaled, X_pca, pca.explained_variance_ratio_, X_tsne

@st.cache_data
def silhouette_curve(drop_ports):
    X_scaled = prepare(drop_ports)[1]
    return [silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled))
            for k in range(2, 9)]

df, X_scaled, X_pca, variance, X_tsne = prepare(drop_ports)

# --- Choosing k ---
st.header("Choosing the number of clusters")
fig, ax = plt.subplots()
ax.plot(list(range(2, 9)), silhouette_curve(drop_ports), "o-", color="orange")
ax.set_xlabel("k")
ax.set_ylabel("Silhouette score")
st.pyplot(fig)

k = st.slider("Number of clusters (k)", 2, 8, 3)
labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)

# --- 2D pictures ---
st.header("Passengers in 2D")
colour_by = st.radio("Colour the dots by", ["Survived", "KMeans cluster"], horizontal=True)
colours = df["Survived"] if colour_by == "Survived" else labels
cmap = "coolwarm" if colour_by == "Survived" else "tab10"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=colours, cmap=cmap, alpha=0.6, s=15)
axes[0].set_title(f"PCA (keeps {variance.sum():.0%} of the variation)")
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colours, cmap=cmap, alpha=0.6, s=15)
axes[1].set_title("t-SNE (axes have no meaning)")
st.pyplot(fig)

# --- What the clusters contain ---
st.header("What each cluster looks like")
profile = df.assign(Cluster=labels).groupby("Cluster").mean().round(2)
profile["Passengers"] = df.assign(Cluster=labels)["Cluster"].value_counts().sort_index()
st.dataframe(profile)
st.caption("The Survived column is each cluster's survival rate. KMeans never saw it.")