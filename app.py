# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import os

# # 1. Cache the model loading for better performance
# @st.cache_resource
# def load_prediction_assets():
#     model_path = "heart-disease-prediction-RF-model.pkl"
#     if os.path.exists(model_path):
#         with open(model_path, "rb") as f:
#             return pickle.load(f)
#     return None

# def main():
#     st.set_page_config(page_title="Heart Disease Prediction System", layout="wide")
    
#     st.title("❤️ Heart Disease Prediction System")
#     st.markdown("""
#     Provide the patient's health details below. This system uses a Random Forest model 
#     trained on clinical data to assess heart disease risk.
#     """)

#     model = load_prediction_assets()
    
#     if model is None:
#         st.error("Model file not found. Please ensure the .pkl file is in the application directory.")
#         return

#     # 2. Use columns for a better layout instead of just the sidebar
#     st.subheader("📋 Patient Health Metrics")
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
#         cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
#         trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

#     with col2:
#         chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 240)
#         fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
#         restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
#         thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)

#     with col3:
#         exang = st.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
#         oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
#         slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
#         ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
#         thal = st.selectbox("Thal", options=[1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversable Defect"][x-1])

#     # 3. Organize input for prediction
#     input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

#     st.divider()

#     # 4. Centered Predict Button
#     _, btn_col, _ = st.columns([2, 1, 2])
#     with btn_col:
#         predict_btn = st.button("🔍 Run Risk Assessment", use_container_width=True)

#     if predict_btn:
#         with st.spinner("Analyzing health metrics..."):
#             prediction = model.predict(input_features)
            
#             # 5. Display results with visual indicators
#             if prediction[0] == 1:
#                 st.error("### 🔴 High Risk Detected")
#                 st.write("The model suggests a high probability of heart disease. Please consult a medical professional.")
#             else:
#                 st.success("### 🟢 Low Risk Detected")
#                 st.write("The model predicts no significant risk based on the provided parameters.")
            
#             # Optional: Display confidence if the model supports it
#             if hasattr(model, "predict_proba"):
#                 prob = model.predict_proba(input_features)[0][1]
#                 st.info(f"Model Confidence Score: {prob:.2%}")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --- Configuration ---
st.set_page_config(page_title="Heart Disease Analytics & Prediction", layout="wide")

# --- Utility Functions ---
@st.cache_data
def load_data():
    # Loading the dataset provided in the repository
    df = pd.read_csv("heart_cleveland_upload.csv")
    # Rename 'condition' to 'target' for clarity if needed
    if 'condition' in df.columns:
        df = df.rename(columns={'condition': 'target'})
    return df

@st.cache_resource
def load_model():
    # Load the Random Forest model from the local directory
    model_path = "heart-disease-prediction-RF-model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

# --- App Setup ---
df = load_data()
model = load_model()
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Model Insights (SHAP)"])

# --- Page: Home ---
if page == "Home":
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("""
    Welcome to the Heart Disease Diagnostic Tool. This application allows you to:
    1. **Explore** clinical data trends.
    2. **Predict** heart disease risk for individual patients.
    3. **Understand** the "why" behind model predictions using SHAP values.
    """)
    st.image("https://images.unsplash.com/photo-1505751172876-fa1923c5c528?auto=format&fit=crop&w=800", use_column_width=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# --- Page: EDA ---
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='target', data=df, palette='viridis', ax=ax)
        ax.set_xticklabels(['No Disease', 'Disease'])
        st.pyplot(fig)
        
    with col2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select a feature to visualize:", feature_cols)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x=selected_feature, hue='target', kde=True, element="step", ax=ax)
    st.pyplot(fig)

# --- Page: Prediction ---
elif page == "Prediction":
    st.title("🔬 Patient Risk Assessment")
    
    if model is None:
        st.error("Model file 'heart-disease-prediction-RF-model.pkl' not found.")
    else:
        with st.form("prediction_form"):
            st.write("Enter patient details:")
            c1, c2, c3 = st.columns(3)
            
            # Form Inputs
            age = c1.number_input("Age", 1, 120, 50)
            sex = c1.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            cp = c1.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
            
            trestbps = c2.number_input("Resting BP", 80, 200, 120)
            chol = c2.number_input("Cholesterol", 100, 600, 200)
            fbs = c2.selectbox("Fasting Blood Sugar > 120", [0, 1])
            
            restecg = c3.selectbox("Resting ECG (0-2)", [0, 1, 2])
            thalach = c3.number_input("Max Heart Rate", 60, 220, 150)
            exang = c3.selectbox("Exercise Induced Angina", [0, 1])
            
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
            slope = st.selectbox("Slope", [0, 1, 2])
            ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
            thal = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3])
            
            submit = st.form_submit_button("Predict Risk")
            
        if submit:
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction = model.predict(features)
            
            if prediction[0] == 1:
                st.error("### 🔴 Result: High Risk of Heart Disease")
            else:
                st.success("### 🟢 Result: No Significant Risk Detected")

# --- Page: Model Insights (SHAP) ---
elif page == "Model Insights (SHAP)":
    st.title("🧠 Model Explainability")
    st.write("This section shows how much each feature contributed to the model's decision across the entire dataset.")
    
    if model is not None:
        # Prepare background data for SHAP
        X = df[feature_cols]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        fig, ax = plt.subplots()
        # Summary plot for class 1 (Disease)
        shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
        st.pyplot(fig)
        st.info("The features at the top (e.g., 'ca' or 'cp') are the most important predictors for this model.")
    else:
        st.error("Model required for SHAP analysis.")

# if __name__ == "__main__":
#     main()

