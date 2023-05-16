# app.py
import streamlit as st
import joblib
import numpy as np
from sklearn import datasets

st.title("HMDA Loan Approval Predictor")

st.write("""
Enter the application details:
""")

sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 4.7)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.4)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#iris_rf_model = joblib.load('iris_rf_model.pkl')
#prediction = iris_rf_model.predict(features)

st.subheader("Prediction:")
#st.write(iris.target_names[prediction])
