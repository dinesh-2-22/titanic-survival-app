import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details below:")

# User inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", 1, 100, 25)
sibsp = st.number_input("Number of siblings/spouses aboard", 0, 10, 0)
parch = st.number_input("Number of parents/children aboard", 0, 10, 0)
fare = st.number_input("Ticket Fare", min_value=0, max_value=600, value=50, step=10)

# Convert inputs (sex → number)
sex_val = 0 if sex == "male" else 1

# Features must match the way you trained your model
features = np.array([[pclass, sex_val, age, sibsp, parch, fare]])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("🎉 This passenger would have SURVIVED!")
    else:
        st.error("💀 This passenger would NOT have survived")