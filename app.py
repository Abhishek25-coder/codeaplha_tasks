import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's medical details below.")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Female", "Male"])

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    80,
    250,
    120
)

chol = st.number_input(
    "Cholesterol",
    100,
    600,
    200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["No", "Yes"]
)

restecg = st.selectbox(
    "Rest ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    60,
    250,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

oldpeak = st.number_input(
    "Old Peak",
    0.0,
    10.0,
    1.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Major Vessels",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# Convert text values to numbers
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

if st.button("Predict"):

    patient = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    patient = scaler.transform(patient)

    prediction = model.predict(patient)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")