import joblib
from sklearn.ensemble import RandomForestClassifier
from prep import load_clean_data

df = load_clean_data()
X = df.drop(columns=["Survived"])
y = df["Survived"]

model = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=5, random_state=42
)
model.fit(X, y)

joblib.dump({"model": model, "columns": list(X.columns)}, "titanic_model.joblib")
print("Saved. Columns:", list(X.columns))