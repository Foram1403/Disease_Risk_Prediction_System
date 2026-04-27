import streamlit as st
import pandas as pd
import os
from utils.model_loader import load_model
from utils.predict import predict
from analytics import show_analytics

st.set_page_config(page_title="AI Healthcare", layout="wide")

# LOAD MODELS
models = {}

try:
    models["Diabetes"] = load_model("diabetes_model.pkl")
    models["Heart"] = load_model("heart_model.pkl")
    models["Breast Cancer"] = load_model("breast_cancer_model.pkl")
    models["Lung Cancer"] = load_model("lung_cancer_model.pkl")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")

if not models:
    st.stop()
# UI
st.title("🧠 AI Healthcare Dashboard")

disease = st.sidebar.selectbox("Select Disease", list(models.keys()))


# LOAD DATA
data_files = {
    "Diabetes": "data_sets/diabetes dataset.csv",
    "Heart": "data_sets/heart.csv",
    "Breast Cancer": "data_sets/breast_cancer.csv",
    "Lung Cancer": "data_sets/lung_cancer.csv",
}

df = pd.read_csv(data_files[disease])

# ANALYTICS
show_analytics(df, df.columns[-1])

# INPUTS
st.write("Models folder content:", os.listdir("models"))
st.subheader("Enter Patient Data")

inputs = []
for i in range(len(df.columns) - 1):
    val = st.number_input(f"Feature {i+1}", key=i)
    inputs.append(val)

# PREDICT
if st.button("Predict"):
    result = predict(models[disease], inputs)

    if result[0] == 1:
        st.error("Disease Detected ❌")
    else:
        st.success("No Disease ✅")
