import streamlit as st
import pickle
import os
import numpy as np

# ===============================
# LOAD MODELS SAFELY
# ===============================
BASE_DIR = os.path.dirname(__file__)

def load_model(model_name):
    path = os.path.join(BASE_DIR, "models", model_name)
    return pickle.load(open(path, "rb"))

diabetes_model = load_model("diabetes_model.pkl")
heart_model = load_model("heart_model.pkl")
liver_model = load_model("liver_model.pkl")

# ===============================
# UI
# ===============================
st.set_page_config(page_title="Multi Disease Prediction", layout="wide")

st.title("🩺 Multi Disease Prediction System")

menu = ["Diabetes", "Heart Disease", "Liver Disease"]
choice = st.sidebar.selectbox("Select Disease", menu)

# ===============================
# DIABETES
# ===============================
if choice == "Diabetes":
    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose Level")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        result = diabetes_model.predict(input_data)

        if result[0] == 1:
            st.error("Diabetic")
        else:
            st.success("Not Diabetic")

# ===============================
# HEART
# ===============================
elif choice == "Heart Disease":
    st.header("Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.number_input("Sex (1=Male, 0=Female)")
    cp = st.number_input("Chest Pain Type")
    trestbps = st.number_input("Resting BP")
    chol = st.number_input("Cholesterol")
    fbs = st.number_input("Fasting Blood Sugar")
    restecg = st.number_input("Rest ECG")
    thalach = st.number_input("Max Heart Rate")
    exang = st.number_input("Exercise Induced Angina")

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]])
        result = heart_model.predict(input_data)

        if result[0] == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("Healthy Heart")

# ===============================
# LIVER
# ===============================
elif choice == "Liver Disease":
    st.header("Liver Disease Prediction")

    age = st.number_input("Age")
    total_bilirubin = st.number_input("Total Bilirubin")
    direct_bilirubin = st.number_input("Direct Bilirubin")
    alk_phos = st.number_input("Alkaline Phosphotase")
    sgpt = st.number_input("SGPT")
    sgot = st.number_input("SGOT")

    if st.button("Predict Liver Disease"):
        input_data = np.array([[age, total_bilirubin, direct_bilirubin, alk_phos, sgpt, sgot]])
        result = liver_model.predict(input_data)

        if result[0] == 1:
            st.error("Liver Disease Detected")
        else:
            st.success("Healthy Liver")
