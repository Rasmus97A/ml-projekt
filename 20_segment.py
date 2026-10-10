import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)

customers = pd.read_csv("customers.csv")
X_scaled = StandardScaler().fit_transform(customers)

ks = range(2, 9)
inertias, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k}  inertia={km.inertia_:.0f}  silhouette={silhouettes[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(ks), inertias, "o-")
axes[0].set_title("Elbow: inertia per k")
axes[0].set_xlabel("k")
axes[1].plot(list(ks), silhouettes, "o-", color="orange")
axes[1].set_title("Silhouette per k")
axes[1].set_xlabel("k")
plt.show()

# Final model: change best_k below after you have looked at the plots
best_k = 4
km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_scaled)
customers["segment"] = km.labels_

print(customers.groupby("segment").mean().round(1))
print(customers["segment"].value_counts().sort_index())
customers.to_csv("customers_segmented.csv", index=False)