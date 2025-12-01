import streamlit as st
import numpy as np
import pickle

# Load Model
model = pickle.load(open("heart-disease-prediction-knn-model.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction System", layout="wide")

st.title("❤️ Heart Disease Prediction System")
st.write("Provide the patient's health details to predict the risk of heart disease.")

# Sidebar Inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", (1, 0))
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", 100, 600, 250)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", (1, 0))
    restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", (0, 1, 2))
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", (1, 0))
    oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox("Slope (0-2)", (0, 1, 2))
    ca = st.sidebar.selectbox("Major Vessels (0-3)", (0, 1, 2, 3))
    thal = st.sidebar.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", (1, 2, 3))

    data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])

    return data

# Get input
input_data = user_input_features()

st.subheader("🎯 Model Input Data")
st.write(input_data)

# Predict Button
if st.button("🔍 Predict Heart Disease"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🔴 **High Risk of Heart Disease Detected**")
    else:
        st.success("🟢 **No Significant Risk of Heart Disease Detected**")
