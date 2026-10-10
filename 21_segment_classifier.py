import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("customers_segmented.csv")
X = df.drop(columns=["segment"])
y = df["segment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print(classification_report(y_test, pred))
auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class="ovr")
print("Macro ROC-AUC (one-vs-rest):", round(auc, 3))

ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Greens")
plt.title("Predicting the customer segment")
plt.show()

# A brand-new customer the model has never seen
new_customer = pd.DataFrame([{
    "income_k": 85, "spending_score": 90, "age": 35, "visits_per_month": 11
}])
print("New customer -> segment", rf.predict(new_customer)[0])
print(rf.predict_proba(new_customer).round(2))