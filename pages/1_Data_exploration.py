import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency
from prep import load_clean_data

st.title("Exploring the Titanic data")

@st.cache_data
def get_data():
    return load_clean_data(), pd.read_csv("titanic.csv")

df, raw = get_data()

# --- Interactive bar chart ---
st.header("Survival rate by group")
column = st.selectbox("Group passengers by", ["Sex_encoded", "Pclass", "SibSp", "Parch"])
fig, ax = plt.subplots()
sns.barplot(x=column, y="Survived", data=df, ax=ax)
ax.set_ylabel("Share who survived")
st.pyplot(fig)
if column == "Sex_encoded":
    st.caption("0 = male, 1 = female")

# --- Age histogram ---
st.header("Age distribution")
fig, ax = plt.subplots()
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30, ax=ax)
st.pyplot(fig)
st.caption("The spike at 28 is the 177 missing ages filled with the median.")

# --- Correlation heatmap ---
st.header("Correlation matrix")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)

# --- Chi-square tests ---
st.header("Chi-square tests")
rows = []
for col in ["Sex", "Pclass", "Embarked"]:
    chi2, p, dof, expected = chi2_contingency(pd.crosstab(raw[col], raw["Survived"]))
    rows.append({"Column": col, "chi2": round(chi2, 1), "p-value": f"{p:.2e}"})
st.dataframe(pd.DataFrame(rows), hide_index=True)