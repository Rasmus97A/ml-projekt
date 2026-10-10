import joblib
import pandas as pd
import streamlit as st
from prep import load_clean_data
st.title("Titanic survival predictor")

saved = joblib.load("titanic_model.joblib")
model = saved["model"]
columns = saved["columns"]

pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
sibsp = st.number_input("Siblings/spouses aboard", 0, 8, 0)
parch = st.number_input("Parents/children aboard", 0, 6, 0)
fare = st.slider("Ticket price", 0.0, 512.0, 30.0)
embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"])
@st.cache_data
def fare_ranges():
    data = load_clean_data()
    return data.groupby("Pclass")["Fare"].quantile([0.05, 0.95]).unstack()

ranges = fare_ranges()
low, high = ranges.loc[pclass, 0.05], ranges.loc[pclass, 0.95]
if fare < low or fare > high:
    st.warning(
        f"A ticket price of {fare:.1f} is unusual for class {pclass} "
        f"(90% of passengers paid between {low:.1f} and {high:.1f}). "
        "The estimate below is less reliable."
    )
row = {
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Sex_encoded": 1 if sex == "female" else 0,
    "Embarked_C": embarked == "C",
    "Embarked_Q": embarked == "Q",
    "Embarked_S": embarked == "S",
}
input_df = pd.DataFrame([row])[columns]

probability = model.predict_proba(input_df)[0, 1]
rounded = round(probability * 20) / 20      # nearest 5%
st.metric("Estimated chance of survival", f"about {rounded:.0%}")
st.caption("Based on a model trained on 891 passengers. Treat this as a rough estimate, not an exact figure.")
st.progress(float(probability))