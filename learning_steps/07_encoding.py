import pandas as pd

df = pd.read_csv("titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Option A: Label encoding for Sex (only 2 categories - safe to do this way)
df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})
print(df[["Sex", "Sex_encoded"]].head())

# Option B: One-hot encoding for Embarked (3 categories - S, C, Q)
embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
print(embarked_dummies.head())

# Attach the one-hot columns onto the main dataframe
df = pd.concat([df, embarked_dummies], axis=1)

# Now drop the original text columns we've replaced, plus columns that are
# just identifiers/free text with no predictive pattern a model could learn
df = df.drop(columns=["Sex", "Embarked", "Name", "Ticket", "Cabin"])
df = df.drop(columns=["PassengerId"])

print(df.head())
print(df.columns.tolist())