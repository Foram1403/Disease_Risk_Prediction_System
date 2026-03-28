import streamlit as st

def apply_style():
    st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
