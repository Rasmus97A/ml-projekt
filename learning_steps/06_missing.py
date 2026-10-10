import pandas as pd

df = pd.read_csv("titanic.csv")

median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)

most_common_port = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(most_common_port)

print(df.isnull().sum())