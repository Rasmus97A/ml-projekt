from prep import load_clean_data

df = load_clean_data()
print(df.groupby(["Sex_encoded", "Pclass"])["Survived"].agg(["mean", "count"]).round(2))
narrow = df[
    (df["Sex_encoded"] == 0) & (df["Pclass"] == 3)
    & (df["Fare"].between(6, 10)) & (df["Age"].between(25, 35))
]
print("Similar passengers:", len(narrow), "-> survival rate:", round(narrow["Survived"].mean(), 2))
print("Rows with Age == 28:", (df["Age"] == 28).sum())