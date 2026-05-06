
import streamlit as st
import pandas as pd
import joblib
import os
import sys
import numpy as np
import shap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. DATABASE CONFIGURATION
# URI updated to include credentials for the app
DB_URI = st.secrets["SUPABASE_DB_URL"]
engine = create_engine(DB_URI)

def save_to_db(df):
    try:
        df.to_sql("attrition_predictions", engine, if_exists="append", index=False)
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

# 2. DEFINE PREPROCESSING
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
setattr(__main__, 'preprocess_data', preprocess_data)
sys.modules['__main__'].preprocess_data = preprocess_data

# 3. RESOURCE LOADING
def load_resources():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_path, 'models')
    if not os.path.exists(model_dir): return None
    files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not files: return None
    latest_file = sorted(files)[-1]
    return joblib.load(os.path.join(model_dir, latest_file))

model = load_resources()

# 4. STREAMLIT UI
st.set_page_config(page_title="HR Attrition Predictor", layout="wide")
st.title("Employee Attrition Risk Predictor with DB Logging")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

if model is not None:
    with tab1:
        st.subheader("Quick Prediction")
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        income = st.number_input("Monthly Income", 1000, 20000, 5000)

        if st.button("Predict and Save"):
            input_dict = {f: 0 for f in ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']}
            input_dict.update({'BusinessTravel': 'Travel_Rarely', 'EducationField': 'Life Sciences', 'Gender': 'Male', 'JobRole': 'Sales Executive', 'MaritalStatus': 'Single', 'OverTime': overtime, 'MonthlyIncome': income})
            single_df = pd.DataFrame([input_dict])

            prob = model.predict_proba(single_df)[0][1]
            single_df['Attrition_Probability'] = prob

            if save_to_db(single_df):
                st.success(f"Prediction ({prob:.2%}) saved to PostgreSQL!")

    with tab2:
        st.subheader("Batch Upload")
        uploaded_file = st.file_uploader("Choose CSV", type='csv')
        if uploaded_file and st.button("Process and Log"):
            data = pd.read_csv(uploaded_file)
            data['Attrition_Probability'] = model.predict_proba(data)[:, 1]
            if save_to_db(data):
                st.success("Batch processed and results logged to Database!")
                st.dataframe(data.head())
