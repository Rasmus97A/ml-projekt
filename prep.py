import pandas as pd

def load_clean_data():
    df = pd.read_csv("titanic.csv")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})
    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
    df = pd.concat([df, embarked_dummies], axis=1)

    df = df.drop(columns=["Sex", "Embarked", "Name", "Ticket", "Cabin", "PassengerId"])
    return df