import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="CardioCare | Heart Disease Analytics",
    page_icon="❤️",
    layout="wide"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; border: none; transition: 0.3s; }
    .stButton>button:hover { background-color: #d43f3f; border: none; }
    .prediction-card { 
        padding: 30px; 
        border-radius: 15px; 
        background-color: white; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
        border: 1px solid #eee;
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# --- Resource Loading ---
@st.cache_data
def load_data():
    # Loading the heart_cleveland_upload.csv dataset
    df = pd.read_csv("heart_cleveland_upload.csv")
    if 'condition' in df.columns:
        df = df.rename(columns={'condition': 'target'})
    return df

@st.cache_resource
def load_model():
    # Loading the heart-disease-prediction-RF-model.pkl
    model_path = "heart-disease-prediction-RF-model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

# --- App Setup ---
df = load_data()
model = load_model()
# Features used for prediction
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
st.sidebar.title("CardioCare")
page = st.sidebar.radio("Navigation", ["Risk Predictor", "Clinical Analytics"])

# --- Page 1: Risk Predictor ---
if page == "Risk Predictor":
    st.title("🔬 Cardiovascular Risk Assessment")
    st.markdown("Complete the clinical profile below to generate a diagnostic risk report.")

    if model is None:
        st.error("Error: Prediction model assets not found.")
    else:
        with st.container():
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.subheader("👤 Patient Profile")
                age = st.number_input("Age", 1, 100, 45)
                sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
                cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0:Typical, 1:Atypical, 2:Non-anginal, 3:Asymptomatic")
                trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
                chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 240)

            with col2:
                st.subheader("🧬 Clinical Data")
                thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
                exang = st.radio("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
                oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
                ca = st.selectbox("Vessels Colored by Flourosopy (0-3)", [0, 1, 2, 3])
                thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])

            with st.expander("Additional Laboratory Metrics"):
                fbs = st.checkbox("Fasting Blood Sugar > 120 mg/dl")
                restecg = st.select_slider("Resting ECG Result", options=[0, 1, 2])
                slope = st.select_slider("Slope of Peak Exercise ST", options=[0, 1, 2])

            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("RUN DIAGNOSTIC ASSESSMENT"):
                # Building array in the order expected by the model
                input_data = np.array([[age, sex, cp, trestbps, chol, int(fbs), restecg, 
                                       thalach, exang, oldpeak, slope, ca, thal]])
                
                prediction = model.predict(input_data)
                
                st.markdown("---")
                if prediction[0] == 1:
                    st.error("### 🚩 Result: High Risk Detected")
                    st.info("Assessment: Clinical indicators suggest a high probability of heart disease. Immediate specialist consultation is recommended.")
                else:
                    st.success("### ✅ Result: No Significant Risk Detected")
                    st.info("Assessment: Current clinical metrics are within the low-risk threshold for heart disease based on the predictive model.")

# --- Page 2: Clinical Analytics (EDA) ---
elif page == "Clinical Analytics":
    st.title("📊 Clinical Trends Analysis")
    st.markdown("Visual analysis of the heart disease dataset used to train the prediction model.")
    
    tab1, tab2 = st.tabs(["Feature Distribution", "Clinical Correlations"])
    
    with tab1:
        selected_feature = st.selectbox("Select metric to compare with patient outcome:", features)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df, x=selected_feature, hue='target', kde=True, palette="vlag", multiple="stack")
        plt.title(f"Impact of {selected_feature} on Heart Disease Risk")
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Global Correlation Map")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
        st.pyplot(fig)
