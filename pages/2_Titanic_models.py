import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from prep import load_clean_data

st.title("Titanic models")

@st.cache_resource
def train_models():
    df = load_clean_data()
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
        ),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models, X, y, X_test, y_test

@st.cache_data
def cv_table():
    models, X, y, _, _ = train_models()
    rows = []
    for name, m in models.items():
        scores = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
        rows.append({"Model": name, "Mean CV AUC": round(scores.mean(), 3),
                     "Lowest fold": round(scores.min(), 3), "Highest fold": round(scores.max(), 3)})
    return pd.DataFrame(rows)

models, X, y, X_test, y_test = train_models()

# --- One model in detail (interactive) ---
st.header("Confusion matrix and classification report")
name = st.selectbox("Choose a model", list(models))
pred = models[name].predict(X_test)

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Died", "Survived"], cmap="Blues", ax=ax)
st.pyplot(fig)

report = classification_report(y_test, pred, target_names=["Died", "Survived"], output_dict=True)
st.dataframe(pd.DataFrame(report).T.round(2))

# --- All models compared ---
st.header("ROC and precision-recall curves")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for model_name, m in models.items():
    proba = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, label=f"{model_name} (AUC {roc_auc_score(y_test, proba):.3f})")
    precision, recall, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(recall, precision, label=model_name)
axes[0].plot([0, 1], [0, 1], "k--", label="Coin flip")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].legend()
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
st.pyplot(fig)

# --- Feature importance ---
st.header("Feature importance (Random Forest)")
importances = pd.Series(models["Random Forest"].feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots()
importances.plot(kind="barh", ax=ax)
st.pyplot(fig)

# --- Cross-validation ---
st.header("Cross-validation (5 folds, all 891 passengers)")
st.dataframe(cv_table(), hide_index=True)
st.caption("A single test split can mislead, so these averages are the fairer comparison.")