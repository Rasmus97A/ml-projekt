import joblib
import pandas as pd

saved = joblib.load("titanic_model.joblib")
model, columns = saved["model"], saved["columns"]

def chance(age, fare, pclass=3, sex=0, sibsp=0, parch=0, port="S"):
    row = {
        "Pclass": pclass, "Age": age, "SibSp": sibsp, "Parch": parch, "Fare": fare,
        "Sex_encoded": sex,
        "Embarked_C": port == "C", "Embarked_Q": port == "Q", "Embarked_S": port == "S",
    }
    return model.predict_proba(pd.DataFrame([row])[columns])[0, 1]

for fare in [7.0, 7.5, 7.8, 8.0, 8.5, 9.0]:
    print(f"fare {fare}:  age 30 -> {chance(30, fare):.0%}   age 28 -> {chance(28, fare):.0%}")