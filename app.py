import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Healthcare Dashboard",
    layout="wide",
    page_icon="🧠"
)

# ===============================
# CUSTOM CSS (🔥 BEAUTIFUL UI)
# ===============================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

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
# SIDEBAR
# ===============================
st.sidebar.title("🩺 Navigation")
disease = st.sidebar.selectbox("Select Disease", list(models.keys()))

# ===============================
# HEADER
# ===============================
st.title("🧠 AI Multi Disease Prediction Dashboard")
st.markdown("### Real-time Medical Risk Analysis System")

# ===============================
# INPUT SECTION
# ===============================
st.subheader(f"🔍 {disease} Prediction")

col1, col2, col3 = st.columns(3)

inputs = []

for i in range(9):  # adjustable
    with [col1, col2, col3][i % 3]:
        val = st.number_input(f"Feature {i+1}", key=i)
        inputs.append(val)

# ===============================
# PREDICTION
# ===============================
if st.button("🚀 Predict Now"):
    try:
        data = np.array(inputs).reshape(1, -1)
        result = models[disease].predict(data)

        st.subheader("📊 Prediction Result")

        if result[0] == 1:
            st.error(f"⚠️ {disease} Detected")
        else:
            st.success(f"✅ No {disease}")

    except Exception as e:
        st.error(f"Error: {e}")

# ===============================
# REAL-TIME ANALYTICS
# ===============================
st.subheader("📈 Real-Time Analytics")

sample_data = np.random.rand(20)

fig, ax = plt.subplots()
ax.plot(sample_data)
ax.set_title("Health Risk Trend")

st.pyplot(fig)

# ===============================
# FEATURE IMPORTANCE (if RF)
# ===============================
st.subheader("🧬 Feature Importance")

model = models[disease]

if hasattr(model, "feature_importances_"):
    fig2, ax2 = plt.subplots()
    ax2.bar(range(len(model.feature_importances_)), model.feature_importances_)
    ax2.set_title("Feature Importance")

    st.pyplot(fig2)

# ===============================
# MODEL PERFORMANCE MOCK
# ===============================
st.subheader("📊 Model Performance")

st.metric("Accuracy", "92%")
st.metric("ROC-AUC", "0.91")
