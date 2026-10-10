import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from prep import load_clean_data
pd.set_option("display.max_columns", None)

df = load_clean_data()
X = df.drop(columns=["Survived", "Embarked_C", "Embarked_Q", "Embarked_S"])
X_scaled = StandardScaler().fit_transform(X)

# Try several values of k and score each
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    score = silhouette_score(X_scaled, km.labels_)
    print(f"k={k}  silhouette={score:.3f}")

# Fit with k=3 and look at what the groups mean
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df["Cluster"] = km.labels_

print(df.groupby("Cluster").mean().round(2))
print(pd.crosstab(df["Cluster"], df["Survived"], normalize="index").round(2))