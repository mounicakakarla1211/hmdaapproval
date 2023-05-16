# app.py
import streamlit as st
import joblib
import numpy as np
from sklearn import datasets

def format_func(option):
    return loan_type_choices[option]

st.title("HMDA Loan Approval Predictor")

st.write("""
Enter the application details:
""")

loan_amount = st.text_input('Loan Amount', '')
income = st.text_input('Income', '')
loan_term = st.text_input('Loan Term', '')
combined_loan_to_value_ratio = st.text_input('Combined Loan To Value Ratio', '')
loan_type_choices = {1: "1", 2: "2", 3: "3"}
loan_type = st.selectbox('Loan Type',options=list(loan_type_choices.keys()), format_func=format_func)


#features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#iris_rf_model = joblib.load('iris_rf_model.pkl')
#prediction = iris_rf_model.predict(features)

st.subheader("Prediction:")
st.write(loan_type)
