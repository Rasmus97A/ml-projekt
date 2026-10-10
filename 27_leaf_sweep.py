from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from prep import load_clean_data

df = load_clean_data()
X = df.drop(columns=["Survived"])
y = df["Survived"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("leaf    AUC    Brier (lower = better)")
for leaf in [1, 3, 5, 10, 15, 25, 40]:
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, random_state=42)
    auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc").mean()
    brier = -cross_val_score(rf, X, y, cv=cv, scoring="neg_brier_score").mean()
    print(f"{leaf:<6}  {auc:.3f}  {brier:.3f}")