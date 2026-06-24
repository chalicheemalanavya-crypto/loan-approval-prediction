import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")

st.title("🏦 Loan Approval Prediction System")

income = st.number_input("Income")
credit_score = st.number_input("Credit Score")
loan_amount = st.number_input("Loan Amount")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = np.array([[income, credit_score, loan_amount, age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")