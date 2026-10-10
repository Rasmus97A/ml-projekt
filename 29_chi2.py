import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv("titanic.csv")

for col in ["Sex", "Pclass", "Embarked"]:
    table = pd.crosstab(df[col], df["Survived"])    # counts for each combination
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"{col:9s} chi2 = {chi2:6.1f}   p = {p:.2e}")

    print(pd.crosstab(df["Embarked"], df["Pclass"], normalize="index").round(2))