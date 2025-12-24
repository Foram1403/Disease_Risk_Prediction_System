# import streamlit as st
# import numpy as np
# import pickle

# # Load Model
# model = pickle.load(open("heart-disease-prediction-knn-model.pkl", "rb"))

# st.set_page_config(page_title="Heart Disease Prediction System", layout="wide")

# st.title("❤️ Heart Disease Prediction System")
# st.write("Provide the patient's health details to predict the risk of heart disease.")

# # Sidebar Inputs
# st.sidebar.header("User Input Parameters")

# def user_input_features():
#     age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
#     sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", (1, 0))
#     cp = st.sidebar.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
#     trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
#     chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", 100, 600, 250)
#     fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", (1, 0))
#     restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", (0, 1, 2))
#     thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
#     exang = st.sidebar.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", (1, 0))
#     oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)
#     slope = st.sidebar.selectbox("Slope (0-2)", (0, 1, 2))
#     ca = st.sidebar.selectbox("Major Vessels (0-3)", (0, 1, 2, 3))
#     thal = st.sidebar.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", (1, 2, 3))

#     data = np.array([[
#         age, sex, cp, trestbps, chol, fbs, restecg,
#         thalach, exang, oldpeak, slope, ca, thal
#     ]])

#     return data

# # Get input
# input_data = user_input_features()

# st.subheader("🎯 Model Input Data")
# st.write(input_data)

# # Predict Button
# if st.button("🔍 Predict Heart Disease"):
#     prediction = model.predict(input_data)

#     if prediction[0] == 1:
#         st.error("🔴 **High Risk of Heart Disease Detected**")
#     else:
#         st.success("🟢 **No Significant Risk of Heart Disease Detected**")



# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from io import BytesIO

# Modeling & preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP
import shap

st.set_page_config(page_title="Heart Disease — MultiPage App", layout="wide")

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_data
def load_dataset(path=None):
    """Load dataset from provided path or default location."""
    default_paths = [
        "heart_cleveland_upload.csv",
        "/mnt/data/heart_cleveland_upload.csv",
        "./heart_cleveland_upload.csv"
    ]
    if path is not None:
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            pass
    for p in default_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                continue
    return None

@st.cache_resource
def load_model(pickle_path="heart-disease-prediction-RF-model.pkl"):
    """Load the trained model pickle. Returns model or None."""
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.warning(f"Failed to load model from {pickle_path}: {e}")
            return None
    else:
        return None

@st.cache_resource
def build_scaler_from_df(df, feature_cols):
    """Fit a StandardScaler on dataframe features (used to scale user input)."""
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)
    return scaler

def predict_from_input(model, scaler, X_input):
    """Scale X_input and return model prediction and probability (if available)."""
    if scaler is not None:
        Xs = scaler.transform(X_input)
    else:
        Xs = X_input
    pred = model.predict(Xs)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(Xs)
    return pred, prob

# ---------------------------
# Sidebar / Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Live Prediction", "Exploratory Data Analysis", "Model Explainability (SHAP)", "About"])

# ---------------------------
# Load dataset & model
# ---------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Loaded uploaded dataset")
        dataset_source = "uploaded"
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        df = None
        dataset_source = None
else:
    df = load_dataset()
    dataset_source = "default" if df is not None else None

model = load_model()
if model is None:
    st.sidebar.warning("Pretrained model not found at 'heart-disease-prediction-RF-model.pkl'. Prediction and SHAP pages will be disabled until model is available.")

# Determine columns to use for features
# We'll attempt to detect common columns used in your project.
default_feature_cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
if df is not None:
    # if those columns exist, use them; otherwise infer numeric columns (excluding target)
    cols_lower = [c.lower() for c in df.columns]
    present = [c for c in default_feature_cols if c in cols_lower]
    if present:
        # map to actual case-sensitive col names in df
        feature_cols = [df.columns[cols_lower.index(c)] for c in present]
    else:
        # fallback: choose numeric columns excluding 'target' if present
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in [c.lower() for c in numeric_cols]:
            numeric_cols = [c for c in numeric_cols if c.lower() != 'target']
        feature_cols = numeric_cols[:13]  # first 13 numeric features
else:
    feature_cols = default_feature_cols  # will be used for UI only

# If df exists and model exists, build scaler
scaler = None
if df is not None and model is not None:
    # ensure feature_cols are in df
    if all([c in df.columns for c in feature_cols]):
        try:
            # Fit scaler on dataset features (safe for transforming user inputs)
            scaler = build_scaler_from_df(df, feature_cols)
        except Exception:
            scaler = None

# ---------------------------
# Pages
# ---------------------------
if page == "Home":
    st.title("❤️ Heart Disease Prediction — Dashboard")
    st.markdown("""
    This multi-page app includes:
    - **Live Prediction** — predict for a single patient using the pre-trained model.  
    - **Exploratory Data Analysis (EDA)** — interactive charts, correlations, distributions.  
    - **Model Explainability (SHAP)** — global summary and per-sample explanations.  
    """)
    st.write("App status:")
    st.write(f"- Dataset loaded: {'Yes' if df is not None else 'No'} ({dataset_source})")
    st.write(f"- Model loaded: {'Yes' if model is not None else 'No'}")
    st.write("Use the sidebar to navigate pages.")

    if df is not None:
        with st.expander("Preview dataset (first 10 rows)"):
            st.dataframe(df.head(10))

    st.info("If the model isn't loading or the dataset doesn't match expected columns, upload a CSV via the sidebar or place files under your project root.")

elif page == "Live Prediction":
    st.title("🔬 Live Prediction")
    st.write("Enter patient data in the form below. The app scales inputs (using dataset-derived scaler) and uses the pretrained model for prediction.")

    if model is None:
        st.error("Model not loaded. Prediction not available.")
    else:
        st.subheader("Input patient data")

        # Create form-like inputs based on feature_cols
        user_input = {}
        cols = st.columns(2)
        for i, feat in enumerate(feature_cols):
            col = cols[i % 2]
            # make reasonable input widgets depending on feature name
            lname = feat.lower()
            if 'age' in lname:
                user_input[feat] = col.number_input(feat, min_value=1, max_value=120, value=int(df[feat].median()) if df is not None and feat in df else 45)
            elif 'sex' in lname:
                # allow 1/0 or M/F
                if df is not None and df[feat].nunique() <= 2:
                    options = sorted(df[feat].unique().tolist())
                    user_input[feat] = col.selectbox(feat, options, index=0)
                else:
                    user_input[feat] = col.selectbox(feat, [1, 0], index=0)
            elif lname in ['cp','restecg','slope','ca','thal','exang','fbs']:
                # small integer categories
                if df is not None and feat in df:
                    vals = sorted(df[feat].dropna().unique().astype(int).tolist())
                    if len(vals) > 0:
                        user_input[feat] = col.selectbox(feat, vals, index=0)
                    else:
                        user_input[feat] = col.number_input(feat, value=0)
                else:
                    user_input[feat] = col.number_input(feat, value=0)
            elif 'chol' in lname or 'trestbps' in lname or 'thalach' in lname:
                if df is not None and feat in df:
                    med = int(df[feat].median())
                    user_input[feat] = col.number_input(feat, value=med)
                else:
                    user_input[feat] = col.number_input(feat, value=120)
            elif 'oldpeak' in lname:
                user_input[feat] = col.number_input(feat, value=0.0, format="%.2f")
            else:
                # generic numeric
                user_input[feat] = col.number_input(feat, value=0)

        # Build input array in the same order as feature_cols
        X_input = np.array([[user_input[feat] for feat in feature_cols]], dtype=float)

        if st.button("Predict"):
            try:
                pred, prob = predict_from_input(model, scaler, X_input)
                label = int(pred[0])
                if label == 1:
                    st.error("🔴 Model Prediction: **Heart Disease Present (1)**")
                else:
                    st.success("🟢 Model Prediction: **No Heart Disease Detected (0)**")
                if prob is not None:
                    st.write("Prediction probabilities:")
                    st.write(prob.round(3))
            except Exception as e:
                st.error(f"Prediction error: {e}")

elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    if df is None:
        st.error("No dataset loaded. Upload a CSV file via the sidebar or place `heart_cleveland_upload.csv` in project root.")
    else:
        st.subheader("Dataset Overview")
        st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
        if 'target' in [c.lower() for c in df.columns]:
            # show value counts for target
            target_col_name = [c for c in df.columns if c.lower() == 'target'][0]
            st.write("Target distribution:")
            vc = df[target_col_name].value_counts().rename_axis('class').reset_index(name='count')
            st.bar_chart(data=vc.set_index('class'))

        with st.expander("Show dataset (first 50 rows)"):
            st.dataframe(df.head(50))

        # Feature distributions (select up to 4 features)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            st.info("No numeric columns available for plotting.")
        else:
            st.subheader("Feature distributions")
            sel_feats = st.multiselect("Select numeric features to plot histograms", num_cols, default=num_cols[:4])
            for feat in sel_feats:
                fig, ax = plt.subplots(figsize=(6,3))
                sns.histplot(df[feat].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {feat}")
                st.pyplot(fig)
                plt.close(fig)

        # Correlation heatmap
        if len(num_cols) >= 2:
            st.subheader("Correlation Heatmap")
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        # Scatter between two features with color by target (if target present)
        st.subheader("Scatter explorer")
        if len(num_cols) >= 2:
            f1 = st.selectbox("X-axis", num_cols, index=0)
            f2 = st.selectbox("Y-axis", num_cols, index=1)
            fig, ax = plt.subplots(figsize=(6,4))
            if 'target' in [c.lower() for c in df.columns]:
                target_col_name = [c for c in df.columns if c.lower() == 'target'][0]
                sns.scatterplot(data=df, x=f1, y=f2, hue=target_col_name, ax=ax)
            else:
                sns.scatterplot(data=df, x=f1, y=f2, ax=ax)
            ax.set_title(f"{f1} vs {f2}")
            st.pyplot(fig)
            plt.close(fig)

        # Allow download of current dataframe
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download dataset CSV", data=csv, file_name="dataset_export.csv", mime="text/csv")

elif page == "Model Explainability (SHAP)":
    st.title("🧠 Model Explainability with SHAP")

    if model is None:
        st.error("Model not loaded — SHAP explanations are not available.")
    elif df is None:
        st.error("Dataset required for SHAP (to compute background). Upload dataset or place CSV in project root.")
    else:
        st.write("We'll compute SHAP values using a dataset-derived background set. This can take a little time (cached).")
        # ensure feature_cols in df
        if not all([c in df.columns for c in feature_cols]):
            st.error("Required feature columns not found in loaded dataset. Check that your dataset contains the expected features.")
        else:
            # prepare background and explainer
            X = df[feature_cols].fillna(df[feature_cols].median()).values
            # Use a small background sample for speed
            background = X[np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)]
            try:
                explainer = shap.Explainer(model, background, feature_names=feature_cols)
                st.success("SHAP explainer ready.")
            except Exception as e:
                st.error(f"Failed to create SHAP explainer: {e}")
                explainer = None

            if explainer is not None:
                st.subheader("Global explanation — SHAP summary plot")
                # compute shap values on a sample for speed
                sample_X = X[np.random.choice(X.shape[0], min(200, X.shape[0]), replace=False)]
                shap_values = explainer(sample_X)
                fig_summary = shap.plots.beeswarm(shap_values, show=False)
                # shap returns a matplotlib figure or plotting object; use matplotlib's current figure
                try:
                    plt.gcf().set_size_inches(10, 5)
                    st.pyplot(bbox_inches="tight", dpi=150)
                    plt.clf()
                except Exception:
                    st.write("Unable to render SHAP summary plot as a matplotlib figure in this environment.")

                st.subheader("Per-sample explanation")
                idx = st.number_input("Select sample index (0..n-1) from dataset for force plot", min_value=0, max_value=max(0, X.shape[0]-1), value=0)
                sample = X[idx].reshape(1, -1)
                shap_vals_sample = explainer(sample)
                st.write("Model prediction for sample (raw):", model.predict(sample))
                st.write("SHAP force plot (matplotlib fallback)")

                try:
                    # summary bar for the chosen sample
                    fig_force = shap.plots.bar(shap_vals_sample, max_display=20, show=False)
                    plt.gcf().set_size_inches(8, 3)
                    st.pyplot(bbox_inches="tight", dpi=150)
                    plt.clf()
                except Exception as e:
                    st.write("Could not render SHAP plots in this environment:", e)

                st.info("If you want the interactive JS force plot, run locally where shap's JS rendering is supported and embed with components.html.")

elif page == "About":
    st.title("About this App")
    st.markdown("""
    **Project:** Heart disease prediction (Streamlit)  
    **Features implemented:**  
    - Multi-page layout (Home, Prediction, EDA, SHAP explainability)  
    - Live dataset loading (upload or default file)  
    - Prediction using provided pickle model (scaling handled via dataset-derived StandardScaler)  
    - EDA charts (histogram, heatmap, scatter)  
    - SHAP explainability (global & per-sample)  

    **Notes / Tips**
    - Make sure `heart-disease-prediction-RF-model.pkl` and dataset `heart_cleveland_upload.csv` (or `heart_cleveland_upload.csv`) are present in the working directory (or upload dataset via the sidebar).  
    - If SHAP is slow, reduce background/sample sizes (the app uses modest defaults).  
    - For production or cloud deployment, pin compatible package versions in `requirements.txt`.
    """)

    st.markdown("**Author:** You — adapt and style as you like! 🎨")
    st.markdown("**Commands to run locally:**")
    st.code("streamlit run streamlit_app.py", language="bash")

