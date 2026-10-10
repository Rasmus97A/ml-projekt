import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from prep import load_clean_data

os.makedirs("images", exist_ok=True)

def save(fig, name):
    fig.savefig(f"images/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", name)

df = load_clean_data()
X = df.drop(columns=["Survived"])
y = df["Survived"]

# 1. EDA: survival by sex and class
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.barplot(x="Sex_encoded", y="Survived", data=df, ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Male", "Female"])
axes[0].set_xlabel("")
axes[0].set_title("Survival rate by sex")
sns.barplot(x="Pclass", y="Survived", data=df, ax=axes[1])
axes[1].set_title("Survival rate by class")
save(fig, "eda.png")

# 2. ROC and precision-recall for three models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Neural Network (MLP)": make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    ),
}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, m in models.items():
    proba = m.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
    precision, recall, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(recall, precision, label=name)
axes[0].plot([0, 1], [0, 1], "k--", label="Coin flip")
axes[0].set_title("ROC curve")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].legend()
axes[1].set_title("Precision-recall curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
save(fig, "roc_pr.png")

# 3. t-SNE of the passengers
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(
    StandardScaler().fit_transform(X)
)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm", alpha=0.6, s=15)
ax.set_title("t-SNE of passengers (red = survived, blue = died)")
ax.set_xticks([])
ax.set_yticks([])
save(fig, "tsne.png")

# 4. Digits: UMAP, true labels vs KMeans
digits = load_digits()
X_digits = digits.data / 16.0
X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X_digits)
clusters = KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(X_digits)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=digits.target, cmap="tab10", s=8)
axes[0].set_title("UMAP coloured by true digit")
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap="tab10", s=8)
axes[1].set_title("UMAP coloured by KMeans cluster (no labels used)")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
save(fig, "digits_umap.png")

# 5. Customer segments
customers = pd.read_csv("customers_segmented.csv")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(customers["income_k"], customers["spending_score"], c=customers["segment"], cmap="tab10", alpha=0.7)
ax.set_xlabel("Income (thousand per year)")
ax.set_ylabel("Spending score")
ax.set_title("Customer segments found by KMeans")
save(fig, "customers.png")