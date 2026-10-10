from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from xgboost import XGBClassifier
from prep import load_clean_data

df = load_clean_data()
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Same split as Step 11, so the numbers are comparable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline: default settings, scored with cross-validation on the TRAINING data only
rf_default = RandomForestClassifier(n_estimators=200, random_state=42)
print("Default RF, CV AUC:", cross_val_score(rf_default, X_train, y_train, cv=5, scoring="roc_auc").mean().round(3))

searches = {
    "Random Forest": GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 300], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 3, 5]},
        cv=5, scoring="roc_auc",
    ),
    "XGBoost": GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        {"n_estimators": [100, 300], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1]},
        cv=5, scoring="roc_auc",
    ),
}

for name, search in searches.items():
    search.fit(X_train, y_train)
    best = search.best_estimator_
    proba = best.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print("  Best settings:", search.best_params_)
    print("  CV AUC (training data):", round(search.best_score_, 3))
    print("  Test AUC:", round(roc_auc_score(y_test, proba), 3))
    print("  Test accuracy:", f"{accuracy_score(y_test, best.predict(X_test)):.2%}")