import streamlit as st
import pickle
import numpy as np
import os

# ===============================
# LOAD MODELS
# ===============================
BASE_DIR = os.path.dirname(__file__)

def load_model(name):
    return pickle.load(open(os.path.join(BASE_DIR, "models", name), "rb"))

models = {
    "Diabetes": load_model("diabetes_model.pkl"),
    "Heart": load_model("heart_model.pkl"),
    "Breast Cancer": load_model("breast_cancer_model.pkl"),
    "Lung Cancer": load_model("lung_cancer_model.pkl"),
}

# ===============================
# UI
# ===============================
st.set_page_config(page_title="Multi Disease AI System", layout="wide")
st.title("🧠 AI Multi Disease Prediction System")

disease = st.sidebar.selectbox("Select Disease", list(models.keys()))

# ===============================
# INPUT SYSTEM (DYNAMIC)
# ===============================
st.subheader(f"{disease} Prediction")

num_inputs = st.number_input("Enter number of features", 5)

inputs = []
for i in range(num_inputs):
    val = st.number_input(f"Feature {i+1}", key=i)
    inputs.append(val)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict"):
    try:
        data = np.array(inputs).reshape(1, -1)
        result = models[disease].predict(data)

        if result[0] == 1:
            st.error(f"{disease} Detected ❌")
        else:
            st.success(f"No {disease} ✅")

    except Exception as e:
        st.error(f"Error: {e}")
