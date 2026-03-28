# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from reportlab.platypus import SimpleDocTemplate, Paragraph

# from utils.database import *
# from utils.prediction import predict
# from utils.chatbot import chatbot_response

# # -------------------------------
# # LOAD CSS
# # -------------------------------
# def load_css():
#     with open("assets/style.css") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# load_css()

# # -------------------------------
# # PAGE CONFIG
# # -------------------------------
# st.set_page_config(page_title="AI Health System", layout="wide")

# st.title("🏥 AI Preventive Health Risk Assessment System")

# # -------------------------------
# # DARK MODE
# # -------------------------------
# dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# if dark_mode:
#     st.markdown("""
#     <style>
#     body { background-color: #0f172a; color: white; }
#     .card { background-color: #1e293b; color: white; }
#     </style>
#     """, unsafe_allow_html=True)

# # -------------------------------
# # AUTH MENU
# # -------------------------------
# menu = ["Login", "Signup"]
# choice = st.sidebar.selectbox("Account", menu)

# # -------------------------------
# # SIGNUP
# # -------------------------------
# if choice == "Signup":
#     st.subheader("Create Account")
#     u = st.text_input("Username")
#     p = st.text_input("Password", type="password")

#     if st.button("Signup"):
#         add_user(u, p)
#         st.success("Account Created")

# # -------------------------------
# # LOGIN
# # -------------------------------
# elif choice == "Login":
#     u = st.text_input("Username")
#     p = st.text_input("Password", type="password")

#     if st.button("Login"):

#         if login_user(u, p):
#             st.success(f"Welcome {u} 👋")

#             # -------------------------------
#             # SIDEBAR NAVIGATION
#             # -------------------------------
#             st.sidebar.title("Navigation")
#             page = st.sidebar.radio(
#                 "Go to",
#                 ["Dashboard", "Profile", "Reports", "AI Assistant"]
#             )

#             # -------------------------------
#             # DASHBOARD
#             # -------------------------------
#             if page == "Dashboard":
#                 st.header("📊 Health Dashboard")

#                 col1, col2 = st.columns(2)

#                 with col1:
#                     st.markdown('<div class="card">', unsafe_allow_html=True)

#                     disease = st.selectbox("Select Disease", ["Heart", "Diabetes", "Liver"])

#                     val1 = st.slider("Parameter 1", 0.0, 300.0)
#                     val2 = st.slider("Parameter 2", 0.0, 300.0)
#                     val3 = st.slider("Parameter 3", 0.0, 300.0)
#                     val4 = st.slider("Parameter 4", 0.0, 300.0)

#                     features = np.array([[val1, val2, val3, val4]])

#                     predict_btn = st.button("Predict")

#                     st.markdown('</div>', unsafe_allow_html=True)

#                 with col2:
#                     st.markdown('<div class="card">', unsafe_allow_html=True)

#                     if predict_btn:
#                         prob = predict(disease, features)

#                         # Risk Output
#                         if prob < 30:
#                             st.markdown(f'<div class="low-risk">Low Risk: {prob:.2f}% 🟢</div>', unsafe_allow_html=True)
#                         elif prob < 70:
#                             st.markdown(f'<div class="medium-risk">Medium Risk: {prob:.2f}% 🟡</div>', unsafe_allow_html=True)
#                         else:
#                             st.markdown(f'<div class="high-risk">High Risk: {prob:.2f}% 🔴</div>', unsafe_allow_html=True)

#                         save_history(u, disease, prob)

#                         # Health Score
#                         st.subheader("🧠 Health Score")
#                         score = 100 - prob

#                         fig, ax = plt.subplots()
#                         ax.barh(["Health"], [score])
#                         ax.set_xlim(0, 100)
#                         st.pyplot(fig)

#                         # Chart
#                         st.subheader("📊 Risk Chart")
#                         fig, ax = plt.subplots()
#                         ax.bar([disease], [prob])
#                         st.pyplot(fig)

#                         # Recommendation
#                         st.subheader("💡 Recommendation")
#                         if prob > 70:
#                             st.error("Consult doctor immediately")
#                         elif prob > 40:
#                             st.warning("Improve lifestyle")
#                         else:
#                             st.success("Healthy")

#                         # Doctor Recommendation
#                         st.subheader("👨‍⚕️ Doctor Recommendation")
#                         if disease == "Heart":
#                             st.info("Consult Cardiologist ❤️")
#                         elif disease == "Diabetes":
#                             st.info("Consult Endocrinologist 🩸")
#                         else:
#                             st.info("Consult Hepatologist 🧪")

#                         # PDF
#                         if st.button("Download Report"):
#                             file = f"{u}_report.pdf"
#                             doc = SimpleDocTemplate(file)
#                             content = [
#                                 Paragraph(f"User: {u}", None),
#                                 Paragraph(f"Disease: {disease}", None),
#                                 Paragraph(f"Risk: {prob:.2f}%", None)
#                             ]
#                             doc.build(content)

#                             with open(file, "rb") as f:
#                                 st.download_button("Download PDF", f, file_name=file)

#                     st.markdown('</div>', unsafe_allow_html=True)

#             # -------------------------------
#             # PROFILE
#             # -------------------------------
#             elif page == "Profile":
#                 st.header("👤 User Profile")
#                 st.info(f"Username: {u}")

#             # -------------------------------
#             # REPORTS
#             # -------------------------------
#             elif page == "Reports":
#                 st.header("📈 Health Reports")

#                 history = get_history(u)

#                 if history:
#                     diseases = [h[0] for h in history]
#                     risks = [h[1] for h in history]

#                     fig, ax = plt.subplots()
#                     ax.plot(diseases, risks, marker='o')
#                     st.pyplot(fig)

#                 else:
#                     st.info("No history found")

#             # -------------------------------
#             # AI ASSISTANT
#             # -------------------------------
#             elif page == "AI Assistant":
#                 st.header("💬 AI Health Assistant")

#                 q = st.text_input("Ask your question")

#                 if st.button("Ask"):
#                     st.write(chatbot_response(q))

#         else:
#             st.error("Invalid Login")


import streamlit as st
import numpy as np
import pickle
import os
import sqlite3
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph

# -------------------------------
# SAFE MODEL LOADER
# -------------------------------
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return pickle.load(open(path, "rb"))

# LOAD MODELS
heart_model = load_model("models/heart_model.pkl")
heart_scaler = load_model("models/heart_scaler.pkl")

diabetes_model = load_model("models/diabetes_model.pkl")
diabetes_scaler = load_model("models/diabetes_scaler.pkl")

liver_model = load_model("models/liver_model.pkl")
liver_scaler = load_model("models/liver_scaler.pkl")

# -------------------------------
# DATABASE
# -------------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
c.execute("CREATE TABLE IF NOT EXISTS history (username TEXT, disease TEXT, risk REAL)")
conn.commit()

def add_user(u, p):
    c.execute("INSERT INTO users VALUES (?, ?)", (u, p))
    conn.commit()

def login_user(u, p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
    return c.fetchone()

def save_history(u, d, r):
    c.execute("INSERT INTO history VALUES (?, ?, ?)", (u, d, r))
    conn.commit()

def get_history(u):
    c.execute("SELECT disease, risk FROM history WHERE username=?", (u,))
    return c.fetchall()

# -------------------------------
# CHATBOT
# -------------------------------
def chatbot_response(q):
    q = q.lower()
    if "heart" in q:
        return "Heart disease involves arteries ❤️"
    elif "diabetes" in q:
        return "Diabetes is high blood sugar 🩸"
    elif "liver" in q:
        return "Liver disease affects detox 🧪"
    else:
        return "Consult a healthcare professional"

# -------------------------------
# UI STYLE
# -------------------------------
st.markdown("""
<style>
.card {background:white;padding:20px;border-radius:12px;box-shadow:0 4px 15px rgba(0,0,0,0.05);}
.low {background:#dcfce7;padding:10px;border-radius:10px;}
.med {background:#fef9c3;padding:10px;border-radius:10px;}
.high {background:#fee2e2;padding:10px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AI Health System", layout="wide")
st.title("🏥 AI Multi-Disease Health System")

# DARK MODE
dark = st.sidebar.toggle("🌙 Dark Mode")
if dark:
    st.markdown("<style>body{background:#0f172a;color:white;}</style>", unsafe_allow_html=True)

# AUTH
menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])

# SIGNUP
if menu == "Signup":
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Signup"):
        add_user(u, p)
        st.success("Account Created")

# LOGIN
if menu == "Login":
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):

        if login_user(u, p):
            st.success(f"Welcome {u}")

            page = st.sidebar.radio("Navigation", ["Dashboard", "Reports", "AI Assistant"])

            # ---------------- DASHBOARD ----------------
            if page == "Dashboard":

                disease = st.selectbox("Select Disease", ["Heart", "Diabetes", "Liver"])

                col1, col2 = st.columns(2)

                with col1:
                    val1 = st.slider("Parameter 1", 0.0, 300.0)
                    val2 = st.slider("Parameter 2", 0.0, 300.0)
                    val3 = st.slider("Parameter 3", 0.0, 300.0)
                    val4 = st.slider("Parameter 4", 0.0, 300.0)

                    predict_btn = st.button("Predict")

                with col2:
                    if predict_btn:
                        features = np.array([[val1, val2, val3, val4]])

                        # MODEL SWITCH
                        if disease == "Heart":
                            features = np.pad(features, ((0,0),(0,9)), 'constant')
                            scaler = heart_scaler
                            model = heart_model

                        elif disease == "Diabetes":
                            features = np.pad(features, ((0,0),(0,4)), 'constant')
                            scaler = diabetes_scaler
                            model = diabetes_model

                        else:
                            features = np.pad(features, ((0,0),(0,6)), 'constant')
                            scaler = liver_scaler
                            model = liver_model

                        scaled = scaler.transform(features)
                        prob = model.predict_proba(scaled)[0][1] * 100

                        # RISK UI
                        if prob < 30:
                            st.markdown(f'<div class="low">Low Risk: {prob:.2f}%</div>', unsafe_allow_html=True)
                        elif prob < 70:
                            st.markdown(f'<div class="med">Medium Risk: {prob:.2f}%</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="high">High Risk: {prob:.2f}%</div>', unsafe_allow_html=True)

                        save_history(u, disease, prob)

                        # GRAPH
                        fig, ax = plt.subplots()
                        ax.bar([disease], [prob])
                        st.pyplot(fig)

                        # PDF
                        if st.button("Download Report"):
                            file = f"{u}_{disease}.pdf"
                            doc = SimpleDocTemplate(file)
                            content = [
                                Paragraph(f"User: {u}", None),
                                Paragraph(f"Disease: {disease}", None),
                                Paragraph(f"Risk: {prob:.2f}%", None)
                            ]
                            doc.build(content)

                            with open(file, "rb") as f:
                                st.download_button("Download", f, file_name=file)

            # ---------------- REPORTS ----------------
            elif page == "Reports":
                history = get_history(u)

                if history:
                    diseases = [h[0] for h in history]
                    risks = [h[1] for h in history]

                    fig, ax = plt.subplots()
                    ax.plot(risks, marker='o')
                    st.pyplot(fig)

            # ---------------- CHATBOT ----------------
            elif page == "AI Assistant":
                q = st.text_input("Ask something")
                if st.button("Ask"):
                    st.write(chatbot_response(q))

        else:
            st.error("Invalid Login")
