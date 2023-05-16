# app.py
import streamlit as st
import joblib
import numpy as np
from sklearn import datasets

def format_func(choices,option):
    return choices[option]

st.title("HMDA Loan Approval Predictor")

st.write("""
Enter the application details:
""")
col1, col2, col3 = st.columns(3)
with col1:
    applicant_age_choices = {1: "1", 2: "2", 3: "3"}
    applicant_age = st.selectbox('Applicant Age',options=list(applicant_age_choices.keys()), format_func=format_func(applicant_age_choices))
    applicant_sex_choices = {1: "1", 2: "2", 3: "3"}
    applicant_sex = st.selectbox('Applicant Sex',options=list(applicant_sex_choices.keys()), format_func=applicant_sex_func)
    co_applicant_sex_choices = {1: "1", 2: "2", 3: "3"}
    co_applicant_sex = st.selectbox('Coapplicant Sex',options=list(co_applicant_sex_choices.keys()), format_func=co_applicant_sex_func)
    applicant_ethnicity_1_choices = {1: "1", 2: "2", 3: "3"}
    applicant_ethnicity_1 = st.selectbox('Applicant Ethnicity 1',options=list(applicant_ethnicity_1_choices.keys()), format_func=applicant_ethnicity_1_func)
    co_applicant_ethnicity_1_choices = {1: "1", 2: "2", 3: "3"}
    co_applicant_ethnicity_1 = st.selectbox('Coapplicant Ethnicity 1',options=list(co_applicant_ethnicity_1_choices.keys()), format_func=co_applicant_ethnicity_1_func)
    applicant_race_1_choices = {1: "1", 2: "2", 3: "3"}
    applicant_race_1 = st.selectbox('Applicant Race 1',options=list(applicant_race_1_choices.keys()), format_func=applicant_race_1_func)
    co_applicant_race_1_choices = {1: "1", 2: "2", 3: "3"}
    co_applicant_race_1 = st.selectbox('Coapplicant Race 1',options=list(co_applicant_race_1_choices.keys()), format_func=co_applicant_race_1_func)
    state_code_choices = {1: "1", 2: "2", 3: "3"}
    state_code = st.selectbox('State Code',options=list(state_code_choices.keys()), format_func=state_code_func)
    
with col2:
    income = st.text_input('Income', '')
    debt_to_income_ratio_choices = {1: "1", 2: "2", 3: "3"}
    debt_to_income_ratio = st.selectbox('Debt to Income Ratio',options=list(debt_to_income_ratio_choices.keys()), format_func=debt_to_income_ratio_func)
    applicant_credit_scoring_model_choices = {1: "1", 2: "2", 3: "3"}
    applicant_credit_scoring_model = st.selectbox('Applicant Credit Scoring Model',options=list(applicant_credit_scoring_model_choices.keys()), format_func=applicant_credit_scoring_model_func)
    co_applicant_credit_scoring_model_choices = {1: "1", 2: "2", 3: "3"}
    co_applicant_credit_scoring_model = st.selectbox('Coapplicant Credit Scoring Model',options=list(co_applicant_credit_scoring_model_choices.keys()), format_func=co_applicant_credit_scoring_model_func)
with col3:
    combined_loan_to_value_ratio = st.text_input('Combined Loan To Value Ratio', '')
    construction_method_choices = {1: "1", 2: "2"}
    construction_method = st.selectbox('Construction Method',options=list(construction_method_choices.keys()), format_func=loan_purpose_func)
    occupancy_type_choices = {1: "1", 2: "2", 3: "3"}
    occupancy_type = st.selectbox('Occupancy Type',options=list(occupancy_type_choices.keys()), format_func=occupancy_type_func)
    purchaser_type_choices = {1: "1", 2: "2", 3: "3"}
    purchaser_type = st.selectbox('Purchase Type',options=list(purchaser_type_choices.keys()), format_func=purchaser_type_func)
    business_or_commercial_purpose_choices = {1: "1", 2: "2", 3: "3"}
    business_or_commercial_purpose = st.selectbox('Business or Commericial Purpose',options=list(business_or_commercial_purpose_choices.keys()), format_func=business_or_commercial_purpose_func)
    loan_amount = st.text_input('Loan Amount', '')
    loan_type_choices = {1: "1", 2: "2", 3: "3"}
    loan_type = st.selectbox('Loan Type',options=list(loan_type_choices.keys()), format_func=loan_type_func)
    loan_purpose_choices = {1: "1", 2: "2", 4: "4", 31:"31", 32:"32"}
    loan_purpose = st.selectbox('Loan Purpose',options=list(loan_purpose_choices.keys()), format_func=loan_purpose_func)
    loan_term = st.text_input('Loan Term', '')
#features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#iris_rf_model = joblib.load('iris_rf_model.pkl')
#prediction = iris_rf_model.predict(features)

st.subheader("Prediction:")
st.write(loan_type)
