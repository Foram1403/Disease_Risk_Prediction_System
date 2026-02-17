
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- Page Config ---
# st.set_page_config(page_title="CardioSense AI", layout="wide", page_icon="❤️")

# # --- Custom Styling (The "No One Uses This" UI) ---
# st.markdown("""
#     <style>
#     /* Main Background and Fonts */
#     .main { background-color: #f0f2f6; }
#     h1, h2, h3 { color: #1e3a8a; font-family: 'Helvetica Neue', sans-serif; }
    
#     /* Neumorphic Card Styling */
#     .st-emotion-cache-12w0qpk { border-radius: 20px; border: none; box-shadow: 8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff; padding: 25px; }
    
#     /* Custom Button */
#     .stButton>button {
#         width: 100%;
#         border-radius: 12px;
#         height: 3em;
#         background: linear-gradient(145deg, #2563eb, #1d4ed8);
#         color: white;
#         font-weight: bold;
#         border: none;
#         box-shadow: 5px 5px 10px #bec8d9;
#         transition: 0.3s;
#     }
#     .stButton>button:hover { transform: translateY(-2px); box-shadow: 7px 7px 14px #bec8d9; }
    
#     /* Navigation Bar */
#     .nav-container { display: flex; justify-content: center; gap: 20px; padding: 10px; background: white; border-radius: 50px; margin-bottom: 30px; box-shadow: inset 2px 2px 5px #bec8d9, inset -2px -2px 5px #ffffff; }
#     </style>
#     """, unsafe_allow_html=True)

# # --- Resource Loading ---
# @st.cache_resource
# def load_prediction_engine():
#     model_path = "heart-disease-prediction-RF-model.pkl" #
#     if os.path.exists(model_path):
#         with open(model_path, "rb") as f:
#             return pickle.load(f)
#     return None

# @st.cache_data
# def load_core_data():
#     df = pd.read_csv("heart_cleveland_upload.csv") #
#     if 'condition' in df.columns:
#         df = df.rename(columns={'condition': 'target'})
#     return df

# # Initialize Data/Model
# df = load_core_data()
# model = load_prediction_engine()

# # --- Top Navigation ---
# st.markdown("<h1 style='text-align: center;'>🫀 CardioSense AI Console</h1>", unsafe_allow_html=True)
# selected_tab = st.tabs(["🚀 Live Diagnostic", "📈 Population Trends", "ℹ️ System Info"])

# # --- TAB 1: LIVE DIAGNOSTIC ---
# with selected_tab[0]:
#     st.write("### Patient Assessment")
    
#     with st.container():
#         # Split inputs into logical groupings
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.markdown("#### 👤 Profile")
#             age = st.number_input("Age", 1, 110, 45)
#             sex = st.segmented_control("Biological Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
#             cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="0:Typical, 1:Atypical, 2:Non-anginal, 3:Asymptomatic")

#         with col2:
#             st.markdown("#### 🩺 Vitals")
#             trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
#             thalach = st.slider("Max Heart Rate", 60, 220, 150)
#             chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)

#         with col3:
#             st.markdown("#### 🧪 Lab Data")
#             ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
#             thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])
#             oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)

#         with st.expander("Advanced Clinical Parameters"):
#             ac1, ac2, ac3 = st.columns(3)
#             exang = ac1.checkbox("Exercise-Induced Angina")
#             fbs = ac2.checkbox("Fasting Blood Sugar > 120mg/dl")
#             slope = ac3.select_slider("ST Slope", options=[0, 1, 2])
#             restecg = 1 # Static default to match features

#     st.markdown("---")
    
#     if st.button("RUN AI DIAGNOSIS"):
#         if model:
#             # Prepare data in exact feature order used in training
#             # age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
#             features = np.array([[age, sex, cp, trestbps, chol, int(fbs), restecg, 
#                                  thalach, int(exang), oldpeak, slope, ca, thal]])
            
#             prediction = model.predict(features)[0]
            
#             # Custom Result Layout
#             res_col1, res_col2 = st.columns([1, 2])
            
#             with res_col1:
#                 # Simple Visual Gauge using Matplotlib
#                 fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
#                 val = 0.8 if prediction == 1 else 0.2
#                 ax.barh(0, val * np.pi, color='#e63946' if prediction == 1 else '#2a9d8f', height=0.5)
#                 ax.set_xlim(0, np.pi)
#                 ax.set_axis_off()
#                 st.pyplot(fig)

#             with res_col2:
#                 if prediction == 1:
#                     st.error("## ⚠️ High Risk Detected")
#                     st.write("Patient data shows significant correlation with cardiovascular conditions. Clinical intervention recommended.")
#                 else:
#                     st.success("## ✅ Low Risk Detected")
#                     st.write("Biometric markers are within expected ranges for this profile.")
#         else:
#             st.error("Model file not found. Please upload 'heart-disease-prediction-RF-model.pkl'.")

# # --- TAB 2: POPULATION TRENDS ---
# with selected_tab[1]:
#     st.write("### Data Insights")
#     if df is not None:
#         t_col1, t_col2 = st.columns(2)
        
#         with t_col1:
#             st.write("#### Age vs. Risk")
#             fig, ax = plt.subplots()
#             sns.kdeplot(data=df, x="age", hue="target", fill=True, palette="crest", ax=ax)
#             st.pyplot(fig)
            
#         with t_col2:
#             st.write("#### Correlation Heatmap")
#             fig, ax = plt.subplots(figsize=(10, 7))
#             sns.heatmap(df.corr(), annot=False, cmap="RdBu_r", ax=ax)
#             st.pyplot(fig)
#     else:
#         st.info("Upload 'heart_cleveland_upload.csv' to see analytics.")

# # --- TAB 3: SYSTEM INFO ---
# # --- TAB 3: SYSTEM INFO (Improved About Content) ---
# with selected_tab[2]:
#     st.markdown("""
#         <div style='background: white; padding: 30px; border-radius: 20px; box-shadow: 5px 5px 15px #d1d9e6;'>
#             <h2 style='color: #1e3a8a; margin-top: 0;'>About CardioSense AI</h2>
#             <p style='color: #475569; font-size: 1.1rem; line-height: 1.6;'>
#                 CardioSense AI is a state-of-the-art diagnostic support tool designed to assist healthcare professionals 
#                 in the early detection of cardiovascular risks. By leveraging advanced <b>Machine Learning</b> 
#                 algorithms and historical clinical data, the system identifies subtle patterns in patient biomarkers 
#                 that may indicate the presence of heart disease.
#             </p>
#             <hr style='border: 0.5px solid #e2e8f0;'>
#             <h4 style='color: #1e3a8a;'>How It Works</h4>
#             <p style='color: #475569;'>
#                 The engine is powered by a <b>Random Forest Classifier</b> trained on the gold-standard 
#                 <i>Cleveland Heart Disease Dataset</i>. It analyzes 13 critical health metrics, 
#                 ranging from basic biometrics like age and sex to complex indicators such as ST-segment 
#                 depression and major vessel fluoroscopy.
#             </p>
#             <div style='background: #f8fafc; padding: 20px; border-left: 5px solid #2563eb; border-radius: 10px;'>
#                 <h5 style='margin-top: 0; color: #1e3a8a;'>⚠️ Medical Disclaimer</h5>
#                 <p style='margin-bottom: 0; font-size: 0.95rem; color: #64748b;'>
#                     This tool is intended for educational and research purposes only. It is designed to act as a 
#                     supplemental screening aid and <b>must not</b> be used as a replacement for professional 
#                     medical advice, diagnosis, or treatment. Always seek the advice of a qualified physician 
#                     regarding any medical condition.
#                 </p>
#             </div>
#             <br>
#             <p style='font-size: 0.85rem; color: #94a3b8; text-align: center;'>
#                 <b>Version 2.0</b> | Optimized for Streamlit Cloud Deployment | Developed with ❤️ for better health.
#             </p>
#         </div>
#     """, unsafe_allow_html=True)

import streamlit as st
import numpy as np
import pickle
from sklearn.inspection import permutation_importance
import pandas as pd

st.set_page_config(page_title="AI Multi-Disease Prediction System", layout="wide")

st.title("🧠 AI-Based Multi-Disease Risk Prediction System")
st.write("Select a disease and answer symptom-based questions.")

# ==========================================
# Disease Selection
# ==========================================

disease_option = st.sidebar.selectbox(
    "Select Disease",
    ["Heart Disease", "Diabetes", "Lung Disease"]
)

# ==========================================
# Load Corresponding Model
# ==========================================

if disease_option == "Heart Disease":
    model = pickle.load(open("models/heart_model.pkl", "rb"))
    scaler = pickle.load(open("models/heart_scaler.pkl", "rb"))

elif disease_option == "Diabetes":
    model = pickle.load(open("models/diabetes_model.pkl", "rb"))
    scaler = pickle.load(open("models/diabetes_scaler.pkl", "rb"))

elif disease_option == "Lung Disease":
    model = pickle.load(open("models/lung_model.pkl", "rb"))
    scaler = pickle.load(open("models/lung_scaler.pkl", "rb"))

# ==========================================
# Symptom-Based Inputs
# (Keep structure same as dataset columns!)
# ==========================================

st.sidebar.header("Symptom Based Questions")

age = st.sidebar.slider("Age", 1, 100, 40)
bp = st.sidebar.number_input("Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol / Glucose Level", 50, 400, 200)
smoking = st.sidebar.selectbox("Do you smoke?", [0, 1])
symptom1 = st.sidebar.selectbox("Do you have chest pain / cough?", [0, 1])
symptom2 = st.sidebar.selectbox("Do you feel fatigue / breathlessness?", [0, 1])

input_data = np.array([[age, bp, chol, smoking, symptom1, symptom2]])

# ==========================================
# Prediction
# ==========================================

if st.button("Predict Risk"):

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🔴 High Risk Detected")
    else:
        st.success("🟢 Low Risk")

    st.write(f"### Risk Probability: {round(probability*100,2)}%")

    # ==========================================
    # Permutation Importance (Explainability)
    # ==========================================

    st.subheader("Feature Importance (Model Explanation)")

    dummy_X = scaled_input
    dummy_y = prediction

    result = permutation_importance(model, dummy_X, dummy_y, n_repeats=5)

    importance = result.importances_mean

    for i, val in enumerate(importance):
        st.write(f"Feature {i+1} Importance: {round(val,4)}")

    # ==========================================
    # Disease Knowledge Section
    # ==========================================

    st.subheader("Disease Information & Guidance")

    if disease_option == "Heart Disease":
        st.write("""
        ✔ Causes: High BP, cholesterol, smoking  
        ✔ Symptoms: Chest pain, fatigue  
        ✔ Prevention: Exercise, healthy diet  
        ✔ Consult doctor for ECG / lipid profile
        """)

    elif disease_option == "Diabetes":
        st.write("""
        ✔ Causes: Insulin resistance  
        ✔ Symptoms: Frequent urination, fatigue  
        ✔ Prevention: Control sugar intake  
        ✔ Monitor blood glucose regularly
        """)

    elif disease_option == "Lung Disease":
        st.write("""
        ✔ Causes: Smoking, pollution  
        ✔ Symptoms: Cough, breathlessness  
        ✔ Prevention: Avoid smoking  
        ✔ Seek medical advice if symptoms worsen
        """)


