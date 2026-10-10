import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load and clean data (everything from Steps 5-8) ---
df = pd.read_csv("titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})
embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
df = pd.concat([df, embarked_dummies], axis=1)

df = df.drop(columns=["Sex", "Embarked", "Name", "Ticket", "Cabin", "PassengerId"])

# --- Visualization ---
sns.barplot(x="Sex_encoded", y="Survived", data=df)
plt.title("Survival rate by sex (0 = male, 1 = female)")
plt.show()

# Survival rate by passenger class
sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival rate by class")
plt.show()

# Age distribution, split by survival
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30)
plt.title("Age distribution: survived vs not")
plt.show()

# Correlation heatmap - how do all numeric columns relate to each other?
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.show()