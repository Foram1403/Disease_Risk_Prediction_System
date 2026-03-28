import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.database import *
from utils.chatbot import chatbot_response
from utils.prediction import predict
from utils.ui import apply_style

apply_style()

st.title("🏥 AI Preventive Health System")

menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

# ---------------- SIGNUP ----------------
if choice == "Signup":
    st.subheader("Create Account")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Signup"):
        add_user(u, p)
        st.success("Account Created")

# ---------------- LOGIN ----------------
if choice == "Login":
    st.subheader("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(u, p):
            st.success(f"Welcome {u}")

            disease = st.selectbox("Select Disease", ["Heart", "Diabetes", "Liver"])

            # INPUT
            val1 = st.slider("Value 1", 0.0, 300.0)
            val2 = st.slider("Value 2", 0.0, 300.0)
            val3 = st.slider("Value 3", 0.0, 300.0)
            val4 = st.slider("Value 4", 0.0, 300.0)

            features = np.array([[val1, val2, val3, val4]])

            # PREDICT
            if st.button("Predict"):
                prob = predict(disease, features)

                if prob < 30:
                    st.success(f"Low Risk {prob:.2f}%")
                elif prob < 70:
                    st.warning(f"Medium Risk {prob:.2f}%")
                else:
                    st.error(f"High Risk {prob:.2f}%")

                save_history(u, disease, prob)

                # GRAPH
                st.subheader("Risk Chart")
                fig, ax = plt.subplots()
                ax.bar([disease], [prob])
                st.pyplot(fig)

            # HISTORY
            st.subheader("📊 History Tracking")
            history = get_history(u)
            if history:
                diseases = [h[0] for h in history]
                risks = [h[1] for h in history]

                fig, ax = plt.subplots()
                ax.plot(diseases, risks, marker='o')
                st.pyplot(fig)

            # CHATBOT
            st.subheader("💬 Chatbot")
            q = st.text_input("Ask something")

            if st.button("Ask"):
                st.write(chatbot_response(q))

        else:
            st.error("Invalid login")
