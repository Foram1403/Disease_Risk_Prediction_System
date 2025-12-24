# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="CardioCare | Heart Disease Analytics",
#     page_icon="❤️",
#     layout="wide"
# )

# # --- Custom CSS for Modern UI ---
# st.markdown("""
#     <style>
#     .main { background-color: #f8f9fa; }
#     .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; border: none; transition: 0.3s; }
#     .stButton>button:hover { background-color: #d43f3f; border: none; }
#     .prediction-card { 
#         padding: 30px; 
#         border-radius: 15px; 
#         background-color: white; 
#         box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
#         border: 1px solid #eee;
#         margin-bottom: 20px;
#     }
#     h1, h2, h3 { color: #1e293b; }
#     </style>
#     """, unsafe_allow_html=True)

# # --- Resource Loading ---
# @st.cache_data
# def load_data():
#     # Loading the heart_cleveland_upload.csv dataset
#     df = pd.read_csv("heart_cleveland_upload.csv")
#     if 'condition' in df.columns:
#         df = df.rename(columns={'condition': 'target'})
#     return df

# @st.cache_resource
# def load_model():
#     # Loading the heart-disease-prediction-RF-model.pkl
#     model_path = "heart-disease-prediction-RF-model.pkl"
#     if os.path.exists(model_path):
#         with open(model_path, "rb") as f:
#             return pickle.load(f)
#     return None

# # --- App Setup ---
# df = load_data()
# model = load_model()
# # Features used for prediction
# features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
#             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# # --- Sidebar Navigation ---
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
# st.sidebar.title("CardioCare")
# page = st.sidebar.radio("Navigation", ["Risk Predictor", "Clinical Analytics"])

# # --- Page 1: Risk Predictor ---
# if page == "Risk Predictor":
#     st.title("🔬 Cardiovascular Risk Assessment")
#     st.markdown("Complete the clinical profile below to generate a diagnostic risk report.")

#     if model is None:
#         st.error("Error: Prediction model assets not found.")
#     else:
#         with st.container():
#             st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
#             col1, col2 = st.columns(2, gap="large")
            
#             with col1:
#                 st.subheader("👤 Patient Profile")
#                 age = st.number_input("Age", 1, 100, 45)
#                 sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
#                 cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0:Typical, 1:Atypical, 2:Non-anginal, 3:Asymptomatic")
#                 trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
#                 chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 240)

#             with col2:
#                 st.subheader("🧬 Clinical Data")
#                 thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
#                 exang = st.radio("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
#                 oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
#                 ca = st.selectbox("Vessels Colored by Flourosopy (0-3)", [0, 1, 2, 3])
#                 thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])

#             with st.expander("Additional Laboratory Metrics"):
#                 fbs = st.checkbox("Fasting Blood Sugar > 120 mg/dl")
#                 restecg = st.select_slider("Resting ECG Result", options=[0, 1, 2])
#                 slope = st.select_slider("Slope of Peak Exercise ST", options=[0, 1, 2])

#             st.markdown('</div>', unsafe_allow_html=True)
            
#             if st.button("RUN DIAGNOSTIC ASSESSMENT"):
#                 # Building array in the order expected by the model
#                 input_data = np.array([[age, sex, cp, trestbps, chol, int(fbs), restecg, 
#                                        thalach, exang, oldpeak, slope, ca, thal]])
                
#                 prediction = model.predict(input_data)
                
#                 st.markdown("---")
#                 if prediction[0] == 1:
#                     st.error("### 🚩 Result: High Risk Detected")
#                     st.info("Assessment: Clinical indicators suggest a high probability of heart disease. Immediate specialist consultation is recommended.")
#                 else:
#                     st.success("### ✅ Result: No Significant Risk Detected")
#                     st.info("Assessment: Current clinical metrics are within the low-risk threshold for heart disease based on the predictive model.")

# # --- Page 2: Clinical Analytics (EDA) ---
# elif page == "Clinical Analytics":
#     st.title("📊 Clinical Trends Analysis")
#     st.markdown("Visual analysis of the heart disease dataset used to train the prediction model.")
    
#     tab1, tab2 = st.tabs(["Feature Distribution", "Clinical Correlations"])
    
#     with tab1:
#         selected_feature = st.selectbox("Select metric to compare with patient outcome:", features)
#         fig, ax = plt.subplots(figsize=(10, 5))
#         sns.histplot(data=df, x=selected_feature, hue='target', kde=True, palette="vlag", multiple="stack")
#         plt.title(f"Impact of {selected_feature} on Heart Disease Risk")
#         st.pyplot(fig)
        
#     with tab2:
#         st.subheader("Global Correlation Map")
#         fig, ax = plt.subplots(figsize=(12, 8))
#         sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
#         st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# --- Config & Unique Theme ---
st.set_page_config(page_title="NEURAL-HEART DX", layout="wide")

# Unique CSS: Glassmorphism + Futuristic Medical Glow
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e);
        font-family: 'Inter', sans-serif;
        color: #e9ecef;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-bottom: 20px;
    }
    .step-header {
        font-family: 'Orbitron', sans-serif;
        color: #4cc9f0;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4361ee, #4cc9f0);
        color: white; border: none; border-radius: 30px;
        padding: 0.5rem 2rem; font-family: 'Orbitron';
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(76, 201, 240, 0.4);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Asset Loading ---
@st.cache_data
def load_assets():
    df = pd.read_csv("heart_cleveland_upload.csv") #
    if 'condition' in df.columns:
        df = df.rename(columns={'condition': 'target'}) #
    
    model = None
    if os.path.exists("heart-disease-prediction-RF-model.pkl"): #
        with open("heart-disease-prediction-RF-model.pkl", "rb") as f:
            model = pickle.load(f) #
    return df, model

df, model = load_assets()

# --- Session State for Flow ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

# --- Top Navigation Visualizer ---
cols = st.columns(4)
steps = ["Demographics", "Vitals", "Lab Tests", "Diagnosis"]
for i, name in enumerate(steps):
    color = "#4cc9f0" if st.session_state.step >= i+1 else "rgba(255,255,255,0.2)"
    cols[i].markdown(f"<div style='border-bottom: 4px solid {color}; text-align:center; padding:10px; font-family:Orbitron; font-size:10px; color:{color}'>{name}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Main Walkthrough Flow ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown('<p class="step-header">Phase 01: Demographics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    st.session_state.patient_data['age'] = c1.number_input("Patient Age", 1, 110, 50) #
    st.session_state.patient_data['sex'] = c2.selectbox("Biological Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female") #
    st.session_state.patient_data['cp'] = st.select_slider("Chest Pain Complexity (CP)", options=[0, 1, 2, 3]) #
    st.button("Continue to Vitals →", on_click=next_step)

elif st.session_state.step == 2:
    st.markdown('<p class="step-header">Phase 02: Hemodynamics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    st.session_state.patient_data['trestbps'] = c1.slider("Resting BP (mm/Hg)", 80, 200, 120) #
    st.session_state.patient_data['thalach'] = c2.slider("Peak Heart Rate (MHR)", 60, 220, 150) #
    st.session_state.patient_data['exang'] = st.checkbox("Exercise-Induced Angina Presence") #
    
    col_nav = st.columns([1, 4, 1])
    col_nav[0].button("← Back", on_click=prev_step)
    col_nav[2].button("Next →", on_click=next_step)

elif st.session_state.step == 3:
    st.markdown('<p class="step-header">Phase 03: Biochemical Markers</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    st.session_state.patient_data['chol'] = c1.number_input("Serum Cholestoral", 100, 600, 240) #
    st.session_state.patient_data['oldpeak'] = c2.number_input("ST Depression", 0.0, 6.0, 1.0) #
    st.session_state.patient_data['ca'] = c3.selectbox("Flourosopy Vessels", [0, 1, 2, 3]) #
    
    # Hidden/Default advanced metrics to simplify UI
    st.session_state.patient_data['fbs'] = 0 
    st.session_state.patient_data['restecg'] = 1
    st.session_state.patient_data['slope'] = 1
    st.session_state.patient_data['thal'] = 2

    col_nav = st.columns([1, 4, 1])
    col_nav[0].button("← Back", on_click=prev_step)
    col_nav[2].button("Analyze →", on_click=next_step)

elif st.session_state.step == 4:
    st.markdown('<p class="step-header">Phase 04: Diagnostic Output</p>', unsafe_allow_html=True)
    
    d = st.session_state.patient_data
    # Formatting input for model prediction
    input_arr = np.array([[d['age'], d['sex'], d['cp'], d['trestbps'], d['chol'], 
                           d['fbs'], d['restecg'], d['thalach'], int(d['exang']), 
                           d['oldpeak'], d['slope'], d['ca'], d['thal']]])
    
    if model:
        prediction = model.predict(input_arr)[0] #
        prob = model.predict_proba(input_arr)[0][prediction] #
        
        # Unique Visual: Diagnostic Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Confidence Level %", 'font': {'family': "Orbitron", 'color': "#4cc9f0"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': "#4cc9f0"},
                'bar': {'color': "#f72585" if prediction == 1 else "#4cc9f0"},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255,255,255,0.05)'},
                    {'range': [50, 100], 'color': 'rgba(255,255,255,0.1)'}
                ],
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.markdown("<h2 style='text-align:center; color:#f72585;'>⚠️ HIGH RISK DETECTED</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align:center; color:#4cc9f0;'>✅ LOW RISK DETECTED</h2>", unsafe_allow_html=True)
            
    st.button("New Assessment ↺", on_click=lambda: st.session_state.update({"step": 1}))

st.markdown('</div>', unsafe_allow_html=True)

# --- Bottom Analytics (Conditional) ---
if st.session_state.step == 4:
    with st.expander("Compare with Global Clinical Trends"):
        feat = st.selectbox("Select metric to compare:", ['age', 'chol', 'thalach', 'trestbps'])
        fig_trend = px.violin(df, y=feat, x="target", color="target", box=True, 
                             points="all", template="plotly_dark", 
                             color_discrete_sequence=["#4cc9f0", "#f72585"])
        st.plotly_chart(fig_trend, use_container_width=True)
