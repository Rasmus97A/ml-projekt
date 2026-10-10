import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from prep import load_clean_data

df = load_clean_data()
X = df.drop(columns=["Survived"])
y = df["Survived"]

models = {
    "leaf=3": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42).fit(X, y),
    "leaf=15": RandomForestClassifier(n_estimators=300, min_samples_leaf=15, random_state=42).fit(X, y),
}

def chance(model, age, fare):
    row = {"Pclass": 3, "Age": age, "SibSp": 0, "Parch": 0, "Fare": fare,
           "Sex_encoded": 0, "Embarked_C": False, "Embarked_Q": False, "Embarked_S": True}
    return model.predict_proba(pd.DataFrame([row])[X.columns])[0, 1]

for fare in [7.0, 7.5, 7.8, 8.0, 8.5, 9.0]:
    line = "   ".join(f"{name}: {chance(m, 30, fare):.0%}" for name, m in models.items())
    print(f"fare {fare}:  {line}")

# Which exact fares do 3rd-class men actually pay?
men3 = df[(df["Pclass"] == 3) & (df["Sex_encoded"] == 0)]
print(men3["Fare"].value_counts().head(8))