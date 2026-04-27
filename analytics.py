import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def show_analytics(df, target):
    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.pyplot(fig)

    st.subheader("📉 Target Distribution")
    st.bar_chart(df[target].value_counts())
