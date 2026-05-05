
import streamlit as st
import pandas as pd
import joblib
import os
import sys
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Define Preprocessing & Pickle Fix
def preprocess_data(df):
    df = df.copy()
    if 'Attrition' in df.columns and df['Attrition'].dtype == 'object':
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    redundant_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 'Department', 'YearsWithCurrManager', 'JobLevel', 'PerformanceRating']
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns])
    X = df.drop('Attrition', axis=1) if 'Attrition' in df.columns else df
    y = df['Attrition'] if 'Attrition' in df.columns else None
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    return X, y, preprocessor

import __main__
__main__.preprocess_data = preprocess_data
sys.modules['__main__'].preprocess_data = preprocess_data

# 2. Model & Reference Data Loading
def load_resources():
    model = joblib.load(sorted([f for f in os.listdir('models/') if f.endswith('.pkl')])[-1])
    return model

model = load_resources()

# 3. Streamlit UI
st.set_page_config(page_title="HR Attrition Predictor", layout="wide")
st.title("Employee Attrition Risk Predictor")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab1:
    st.subheader("Quick Prediction (Top Features)")
    col1, col2, col3 = st.columns(3)
    with col1:
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        job_role = st.selectbox("Job Role", ["Sales Representative", "Research Director", "Healthcare Representative", "Laboratory Technician", "Human Resources", "Manager", "Manufacturing Director", "Research Scientist", "Sales Executive"])
        income = st.number_input("Monthly Income", 1000, 20000, 5000)
    with col2:
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
        years_role = st.slider("Years in Current Role", 0, 20, 2)
    with col3:
        total_years = st.slider("Total Working Years", 0, 40, 10)
        num_cos = st.slider("Num Companies Worked", 0, 10, 1)

    if st.button("Predict for this Employee"):
        input_dict = {f: 0 for f in ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']}
        input_dict.update({'BusinessTravel': travel, 'EducationField': 'Life Sciences', 'Gender': 'Male', 'JobRole': job_role, 'MaritalStatus': marital, 'OverTime': overtime})
        input_dict.update({'MonthlyIncome': income, 'TotalWorkingYears': total_years, 'YearsInCurrentRole': years_role, 'NumCompaniesWorked': num_cos})
        single_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(single_df)[0][1]
        if prob > 0.5: st.error(f"High Attrition Risk: {prob:.2%}")
        else: st.success(f"Low Attrition Risk: {prob:.2%}")

with tab2:
    st.subheader("Upload Data for Analysis")
    # List of 26 columns based on the training features (X)
    expected_columns = ['Age', 'BusinessTravel', 'DailyRate', 'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']
    template_df = pd.DataFrame(columns=expected_columns)
    
    st.download_button(
        label="Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="hr_attrition_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if st.button("Analyze Batch"):
            probs = model.predict_proba(data)[:, 1]
            data['Attrition_Probability'] = probs
            st.dataframe(data.sort_values(by='Attrition_Probability', ascending=False))
            
            st.subheader("Explaining Attrition Drivers (SHAP)")
            pre = model.named_steps['preprocessor']
            clf = model.named_steps['model']
            transformed_data = pre.transform(data)
            explainer = shap.LinearExplainer(clf, transformed_data)
            shap_values = explainer.shap_values(transformed_data)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, transformed_data, feature_names=np.concatenate([pre.named_transformers_['num'].get_feature_names_out(), pre.named_transformers_['cat'].get_feature_names_out()]), show=False)
            st.pyplot(fig)
