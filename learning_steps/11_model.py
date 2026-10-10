import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Load and clean data ---
df = pd.read_csv("titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})
embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
df = pd.concat([df, embarked_dummies], axis=1)

df = df.drop(columns=["Sex", "Embarked", "Name", "Ticket", "Cabin", "PassengerId"])

# --- Train/test split ---
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions[:10])
print(y_test[:10].values)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")

cm = confusion_matrix(y_test, predictions)
print("Confusion matrix:")
print(cm)

print(classification_report(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
print("Random Forest accuracy:", f"{accuracy_score(y_test, rf_predictions):.2%}")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

# ROC-AUC: needs probabilities, not just 0/1 guesses
lr_proba = model.predict_proba(X_test)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]
print("Logistic Regression ROC-AUC:", round(roc_auc_score(y_test, lr_proba), 3))
print("Random Forest ROC-AUC:", round(roc_auc_score(y_test, rf_proba), 3))

importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print(importances)

importances.sort_values().plot(kind="barh")
plt.title("Feature importance (Random Forest)")
plt.tight_layout()
plt.show()

from sklearn.model_selection import cross_val_score

lr_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring="accuracy")
rf_scores = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), X, y, cv=5, scoring="accuracy")

print("Logistic Regression:", lr_scores.round(3), "-> mean", lr_scores.mean().round(3))
print("Random Forest:      ", rf_scores.round(3), "-> mean", rf_scores.mean().round(3))

from sklearn.metrics import roc_curve, precision_recall_curve

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC curve
for name, proba in [("Logistic Regression", lr_proba), ("Random Forest", rf_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, label=name)
axes[0].plot([0, 1], [0, 1], "k--", label="Coin flip")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (recall)")
axes[0].set_title("ROC curve")
axes[0].legend()

# Precision-Recall curve
for name, proba in [("Logistic Regression", lr_proba), ("Random Forest", rf_proba)]:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(recall, precision, label=name)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall curve")
axes[1].legend()

plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn the scaling from training data only
X_test_scaled = scaler.transform(X_test)         # apply the same scaling to test data

mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
mlp.fit(X_train_scaled, y_train)

mlp_pred = mlp.predict(X_test_scaled)
mlp_proba = mlp.predict_proba(X_test_scaled)[:, 1]
print("MLP accuracy:", f"{accuracy_score(y_test, mlp_pred):.2%}")
print("MLP ROC-AUC:", round(roc_auc_score(y_test, mlp_proba), 3))

from sklearn.pipeline import make_pipeline

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Neural Network (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)),
}

for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
    print(f"{name:22s} {scores.round(3)} -> mean {scores.mean():.3f}")