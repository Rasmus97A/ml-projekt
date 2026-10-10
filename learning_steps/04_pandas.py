import pandas as pd

data = {
    "name": ["Owen Harris", "Florence Briggs", "Laina Heikkinen", "Lily Futrelle", "William Allen"],
    "age": [22, 38, 26, 35, 35],
    "class": [3, 1, 3, 1, 3],
    "survived": [False, True, True, True, False],
}

df = pd.DataFrame(data)

print(df)
print(df["age"])          
print(df["age"].mean())
print(df["survived"].sum())