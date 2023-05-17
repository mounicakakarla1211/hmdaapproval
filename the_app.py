# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math
import pickle
from sklearn import datasets
from sklearn.preprocessing import RobustScaler

def applicant_age_func(option):
    return applicant_age_choices[option]
def applicant_sex_func(option):
    return applicant_sex_choices[option]
def co_applicant_sex_func(option):
    return co_applicant_sex_choices[option]
def applicant_ethnicity_1_func(option):
    return applicant_ethnicity_1_choices[option]
def co_applicant_ethnicity_1_func(option):
    return co_applicant_ethnicity_1_choices[option]
def applicant_race_1_func(option):
    return applicant_race_1_choices[option]
def co_applicant_race_1_func(option):
    return co_applicant_race_1_choices[option]
def state_code_func(option):
    return state_code_choices[option]

def debt_to_income_ratio_func(option):
    return debt_to_income_ratio_choices[option]
def applicant_credit_scoring_model_func(option):
    return applicant_credit_scoring_model_choices[option]
def co_applicant_credit_scoring_model_func(option):
    return co_applicant_credit_scoring_model_choices[option]

def construction_method_func(option):
    return construction_method_choices[option]
def occupancy_type_func(option):
    return occupancy_type_choices[option]
def purchaser_type_func(option):
    return purchaser_type_choices[option]
def business_or_commercial_purpose_func(option):
    return business_or_commercial_purpose_choices[option]
def loan_type_func(option):
    return loan_type_choices[option]
def loan_purpose_func(option):
    return loan_purpose_choices[option]
#The next function will add missing columns (in response to df_train)
def add_missing_dummy_columns(df, columns):
    missing_cols = set(columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
 
#The next function will delete extra columns (in response to df_train)
def fix_columns(df, columns):
    add_missing_dummy_columns(df, columns)
    # make sure we have all the columns we need
    assert(set(columns) - set(df.columns) == set())
    extra_cols = set(df.columns) - set(columns)
    if extra_cols:
        df = df[columns]
    return df
st.set_page_config(layout="wide")

st.title("HMDA Loan Approval Predictor")
col1, col2, col3,col4 = st.columns(4)
with col2:
    applicant_sex_choices = {"1": "1", "2": "2", "3": "3","6":"6"}
    applicant_sex = st.selectbox('Applicant Sex',options=list(applicant_sex_choices.keys()), format_func=applicant_sex_func)
    co_applicant_sex_choices = {"1": "1", "2": "2", "3": "3","5":"5","6":"6"}
    co_applicant_sex = st.selectbox('Coapplicant Sex',options=list(co_applicant_sex_choices.keys()), format_func=co_applicant_sex_func)
    applicant_ethnicity_1_choices = {"1": "1", "2": "2", "3": "3","11":"11","12":"12","13":"13","14":"14"}
    applicant_ethnicity_1 = st.selectbox('Applicant Ethnicity 1',options=list(applicant_ethnicity_1_choices.keys()), format_func=applicant_ethnicity_1_func)
    co_applicant_ethnicity_1_choices = {"1": "1", "2": "2", "3": "3","5":"5","11":"11","12":"12","13":"13","14":"14"}
    co_applicant_ethnicity_1 = st.selectbox('Coapplicant Ethnicity 1',options=list(co_applicant_ethnicity_1_choices.keys()), format_func=co_applicant_ethnicity_1_func)
    applicant_race_1_choices = {"1": "1", "2": "2", "3": "3","4": "4","5":"5","6":"6","21":"21","22":"22","23":"23","24":"24","25":"25","26":"26","27":"27","41":"41","42":"42","43":"43","44":"44"}
    applicant_race_1 = st.selectbox('Applicant Race 1',options=list(applicant_race_1_choices.keys()), format_func=applicant_race_1_func)
    co_applicant_race_1_choices = {"1": "1", "2": "2", "3": "3","4": "4","5":"5","6":"6","8":"8","21":"21","22":"22","23":"23","24":"24","25":"25","26":"26","27":"27","41":"41","42":"42","43":"43"}
    co_applicant_race_1 = st.selectbox('Coapplicant Race 1',options=list(co_applicant_race_1_choices.keys()), format_func=co_applicant_race_1_func)
    
with col3:
    income = st.text_input('Income', '')
    debt_to_income_ratio_choices = {"less20%" :"<20%","20%-less30%":"20%-<30%","30%-less36%":"30%-<36%","36":"36","37":"37","38":"38","39":"39","40":"40","41":"41","42":"42","43":"43","44":"44","45":"45","46":"46","47":"47","48":"48","49":"49","50%-60%":"50%-60%","greater60%":">60%","Missing":"Missing"}         
    debt_to_income_ratio = st.selectbox('Debt to Income Ratio',options=list(debt_to_income_ratio_choices.keys()), format_func=debt_to_income_ratio_func)
    applicant_credit_scoring_model_choices = {"1": "1", "2": "2", "3": "3","9":"9"}
    applicant_credit_scoring_model = st.selectbox('Applicant Credit Scoring Model',options=list(applicant_credit_scoring_model_choices.keys()), format_func=applicant_credit_scoring_model_func)
    co_applicant_credit_scoring_model_choices = {"1": "1", "2": "2", "3": "3","9":"9","10":"10"}
    co_applicant_credit_scoring_model = st.selectbox('Coapplicant Credit Scoring Model',options=list(co_applicant_credit_scoring_model_choices.keys()), format_func=co_applicant_credit_scoring_model_func)
with col1:
    applicant_age_choices = {"less25": "<25", "25-34": "25-34", "35-44": "35-44","45-54":"45-54","55-64":"55-64","65-74":"65-74",">74":"greater74"}
    applicant_age = st.selectbox('Applicant Age',options=list(applicant_age_choices.keys()), format_func=applicant_age_func)
    construction_method_choices = {"1": "1", "2": "2"}
    construction_method = st.selectbox('Construction Method',options=list(construction_method_choices.keys()), format_func=construction_method_func)
    occupancy_type_choices = {"1": "1", "2": "2", "3": "3"}
    occupancy_type = st.selectbox('Occupancy Type',options=list(occupancy_type_choices.keys()), format_func=occupancy_type_func)
    purchaser_type_choices = {"0":"0","1": "1", "2": "2", "3": "3","5":"5","6":"6","8":"8"}
    purchaser_type = st.selectbox('Purchase Type',options=list(purchaser_type_choices.keys()), format_func=purchaser_type_func)
    business_or_commercial_purpose_choices = {"1": "1", "2": "2"}
    business_or_commercial_purpose = st.selectbox('Business or Commericial Purpose',options=list(business_or_commercial_purpose_choices.keys()), format_func=business_or_commercial_purpose_func)
    state_code_choices = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}
    state_code = st.selectbox('State Code',options=list(state_code_choices.keys()), format_func=state_code_func)
with col4:
    combined_loan_to_value_ratio = st.text_input('Combined Loan To Value Ratio', '')
    loan_amount = st.text_input('Loan Amount', '')
    loan_type_choices = {"1": "1", "2": "2", "3": "3"}
    loan_type = st.selectbox('Loan Type',options=list(loan_type_choices.keys()), format_func=loan_type_func)
    loan_purpose_choices = {"1": "1", "2": "2", "4": "4", "31":"31", "32":"32"}
    loan_purpose = st.selectbox('Loan Purpose',options=list(loan_purpose_choices.keys()), format_func=loan_purpose_func)
    loan_term = st.text_input('Loan Term', '')    
if st.button('Submit'):
    income_log = math.log(int(income))
    loanamount_log = math.log(float(loan_amount))
    clvr = float(combined_loan_to_value_ratio)
    lt = float(loan_term)
    numVal = np.array([clvr, lt, income_log, loanamount_log])
    scaler = RobustScaler()
    numVal_Scaled = scaler.fit_transform(numVal.reshape(-1,1))
    numVal_Scaled  = numVal_Scaled.reshape(1,-1)
    df_num = pd.DataFrame(numVal_Scaled, index=[0],columns=["combined_loan_to_value_ratio","loan_term","income_log","loanamount_log"])
    df_cat = pd.DataFrame({"applicant_age":[applicant_age], "applicant_sex":[applicant_sex], "co_applicant_sex":[co_applicant_sex], 
                           "applicant_ethnicity_1":[applicant_ethnicity_1], "co_applicant_ethnicity_1":[co_applicant_ethnicity_1], 
                           "applicant_race_1":[applicant_race_1], "co_applicant_race_1":[co_applicant_race_1], "state_code":[state_code],
                           "debt_to_income_ratio":[debt_to_income_ratio], "applicant_credit_scoring_model":[applicant_credit_scoring_model], 
                           "co_applicant_credit_scoring_model":[co_applicant_credit_scoring_model], "loan_type":[loan_type], "loan_purpose":[loan_purpose], 
                           "construction_method":[construction_method], "occupancy_type":[occupancy_type],"purchaser_type":[purchaser_type],
                           "business_or_commercial_purpose":[business_or_commercial_purpose]})
    catDF = pd.get_dummies(df_cat,['loan_type', 'loan_purpose', 'construction_method', 'occupancy_type',
       'purchaser_type', 'applicant_credit_scoring_model',
       'co_applicant_credit_scoring_model', 'debt_to_income_ratio',
       'business_or_commercial_purpose', 'applicant_age', 'applicant_sex',
       'co_applicant_sex', 'applicant_ethnicity_1', 'co_applicant_ethnicity_1',
       'applicant_race_1', 'co_applicant_race_1', 'state_code'],dtype=int)    
    initial_column_df = pd.read_csv('model_catcolumns.csv')
    columns = initial_column_df['Features']
    fixed_d = fix_columns(catDF, columns)
    X = pd.concat([fixed_d, df_num],axis=1)
    hmdannmodel = pickle.load(open('nn_reg.pkl', 'rb'))
    prediction = hmdannmodel.predict(X)
    if prediction[0] == 1:
        st.subheader("Prediction: Approved")
    else:
        st.subheader("Prediction: Not Approved")
   

