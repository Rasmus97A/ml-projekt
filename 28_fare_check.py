from prep import load_clean_data

df = load_clean_data()
print(df.groupby("Pclass")["Fare"].describe().round(1))