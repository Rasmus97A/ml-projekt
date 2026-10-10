import pandas as pd

df = pd.read_csv("titanic.csv")

print(df.head())        # first 5 rows
print(df.shape)         # (rows, columns)
print(df.info())        # column names, data types, and non-null counts
print(df.describe())    # mean, min, max, etc. for numeric columns
print(df["Survived"].value_counts())   # counts of each unique values
print(df.isnull().sum())