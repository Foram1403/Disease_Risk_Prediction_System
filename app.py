import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph

from utils.database import *
from utils.prediction import predict
from utils.chatbot import chatbot_response

# -------------------------------
# LOAD CSS
# -------------------------------
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Health System", layout="wide")

st.title("🏥 AI Preventive Health Risk Assessment System")

# -------------------------------
# DARK MODE
# -------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    body { background-color: #0f172a; color: white; }
    .card { background-color: #1e293b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# AUTH MENU
# -------------------------------
menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Account", menu)

# -------------------------------
# SIGNUP
# -------------------------------
if choice == "Signup":
    st.subheader("Create Account")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Signup"):
        add_user(u, p)
        st.success("Account Created")

# -------------------------------
# LOGIN
# -------------------------------
elif choice == "Login":
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):

        if login_user(u, p):
            st.success(f"Welcome {u} 👋")

            # -------------------------------
            # SIDEBAR NAVIGATION
            # -------------------------------
            st.sidebar.title("Navigation")
            page = st.sidebar.radio(
                "Go to",
                ["Dashboard", "Profile", "Reports", "AI Assistant"]
            )

            # -------------------------------
            # DASHBOARD
            # -------------------------------
            if page == "Dashboard":
                st.header("📊 Health Dashboard")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)

                    disease = st.selectbox("Select Disease", ["Heart", "Diabetes", "Liver"])

                    val1 = st.slider("Parameter 1", 0.0, 300.0)
                    val2 = st.slider("Parameter 2", 0.0, 300.0)
                    val3 = st.slider("Parameter 3", 0.0, 300.0)
                    val4 = st.slider("Parameter 4", 0.0, 300.0)

                    features = np.array([[val1, val2, val3, val4]])

                    predict_btn = st.button("Predict")

                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)

                    if predict_btn:
                        prob = predict(disease, features)

                        # Risk Output
                        if prob < 30:
                            st.markdown(f'<div class="low-risk">Low Risk: {prob:.2f}% 🟢</div>', unsafe_allow_html=True)
                        elif prob < 70:
                            st.markdown(f'<div class="medium-risk">Medium Risk: {prob:.2f}% 🟡</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="high-risk">High Risk: {prob:.2f}% 🔴</div>', unsafe_allow_html=True)

                        save_history(u, disease, prob)

                        # Health Score
                        st.subheader("🧠 Health Score")
                        score = 100 - prob

                        fig, ax = plt.subplots()
                        ax.barh(["Health"], [score])
                        ax.set_xlim(0, 100)
                        st.pyplot(fig)

                        # Chart
                        st.subheader("📊 Risk Chart")
                        fig, ax = plt.subplots()
                        ax.bar([disease], [prob])
                        st.pyplot(fig)

                        # Recommendation
                        st.subheader("💡 Recommendation")
                        if prob > 70:
                            st.error("Consult doctor immediately")
                        elif prob > 40:
                            st.warning("Improve lifestyle")
                        else:
                            st.success("Healthy")

                        # Doctor Recommendation
                        st.subheader("👨‍⚕️ Doctor Recommendation")
                        if disease == "Heart":
                            st.info("Consult Cardiologist ❤️")
                        elif disease == "Diabetes":
                            st.info("Consult Endocrinologist 🩸")
                        else:
                            st.info("Consult Hepatologist 🧪")

                        # PDF
                        if st.button("Download Report"):
                            file = f"{u}_report.pdf"
                            doc = SimpleDocTemplate(file)
                            content = [
                                Paragraph(f"User: {u}", None),
                                Paragraph(f"Disease: {disease}", None),
                                Paragraph(f"Risk: {prob:.2f}%", None)
                            ]
                            doc.build(content)

                            with open(file, "rb") as f:
                                st.download_button("Download PDF", f, file_name=file)

                    st.markdown('</div>', unsafe_allow_html=True)

            # -------------------------------
            # PROFILE
            # -------------------------------
            elif page == "Profile":
                st.header("👤 User Profile")
                st.info(f"Username: {u}")

            # -------------------------------
            # REPORTS
            # -------------------------------
            elif page == "Reports":
                st.header("📈 Health Reports")

                history = get_history(u)

                if history:
                    diseases = [h[0] for h in history]
                    risks = [h[1] for h in history]

                    fig, ax = plt.subplots()
                    ax.plot(diseases, risks, marker='o')
                    st.pyplot(fig)

                else:
                    st.info("No history found")

            # -------------------------------
            # AI ASSISTANT
            # -------------------------------
            elif page == "AI Assistant":
                st.header("💬 AI Health Assistant")

                q = st.text_input("Ask your question")

                if st.button("Ask"):
                    st.write(chatbot_response(q))

        else:
            st.error("Invalid Login")
