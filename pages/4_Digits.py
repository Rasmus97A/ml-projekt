import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import umap
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, adjusted_rand_score,
                             classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split

st.title("Handwritten digits")

@st.cache_resource
def train_models():
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(42)

    digits = load_digits()
    X = digits.images / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train.reshape(len(X_train), -1), y_train)

    cnn = keras.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = cnn.fit(X_train[..., np.newaxis], y_train, validation_split=0.15,
                      epochs=30, batch_size=32, verbose=0)
    return rf, cnn, history.history, X_test, y_test

@st.cache_data
def umap_and_kmeans():
    digits = load_digits()
    X = digits.data / 16.0
    X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    clusters = KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(X)
    return X_2d, clusters, digits.target

with st.spinner("Training the models (only the first time)..."):
    rf, cnn, history, X_test, y_test = train_models()

rf_pred = rf.predict(X_test.reshape(len(X_test), -1))
cnn_proba = cnn.predict(X_test[..., np.newaxis], verbose=0)
cnn_pred = np.argmax(cnn_proba, axis=1)

# --- Headline numbers ---
col1, col2, col3 = st.columns(3)
col1.metric("Random Forest accuracy", f"{accuracy_score(y_test, rf_pred):.2%}")
col2.metric("CNN accuracy", f"{accuracy_score(y_test, cnn_pred):.2%}")
col3.metric("CNN macro ROC-AUC", f"{roc_auc_score(y_test, cnn_proba, multi_class='ovr'):.3f}")

# --- Example images ---
st.header("What the model sees: 8×8 pixels")
fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i], cmap="gray")
    ax.set_title(str(y_test[i]))
    ax.axis("off")
st.pyplot(fig)

# --- Confusion matrix (interactive) ---
st.header("Confusion matrix")
choice = st.radio("Model", ["CNN", "Random Forest"], horizontal=True)
pred = cnn_pred if choice == "CNN" else rf_pred
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Purples", ax=ax)
st.pyplot(fig)

st.subheader(f"Classification report ({choice})")
report = classification_report(y_test, pred, output_dict=True)
st.dataframe(pd.DataFrame(report).T.round(2))

# --- Learning curve ---
st.header("CNN learning curve")
fig, ax = plt.subplots()
ax.plot(history["accuracy"], label="training")
ax.plot(history["val_accuracy"], label="validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)

# --- The CNN's mistakes ---
st.header("The CNN's mistakes")
wrong = np.where(cnn_pred != y_test)[0]
if len(wrong) > 0:
    fig, axes = plt.subplots(1, len(wrong), figsize=(1.5 * len(wrong), 2))
    for ax, i in zip(np.atleast_1d(axes), wrong):
        ax.imshow(X_test[i], cmap="gray")
        ax.set_title(f"true {y_test[i]}\nguess {cnn_pred[i]}", fontsize=9)
        ax.axis("off")
    st.pyplot(fig)
else:
    st.write("No mistakes on the test set.")

# --- Unsupervised: UMAP + KMeans ---
st.header("Without labels: UMAP and KMeans")
X_2d, clusters, target = umap_and_kmeans()
st.metric("Adjusted Rand Index (KMeans vs true digits)", f"{adjusted_rand_score(target, clusters):.3f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=target, cmap="tab10", s=8)
axes[0].set_title("Coloured by true digit")
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap="tab10", s=8)
axes[1].set_title("Coloured by KMeans cluster (labels never used)")
st.pyplot(fig)