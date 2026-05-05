
import streamlit as st
import pandas as pd
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Define the preprocessing logic used during training
def preprocess_data(df):
    df = df.copy()
    if 'Attrition' in df.columns and df['Attrition'].dtype == 'object':
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    redundant_cols = [
      'EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 
      'Department', 'YearsWithCurrManager', 'JobLevel', 'PerformanceRating'
    ]
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

# 2. CRITICAL FIX: Map the function into the expected module for joblib/pickle
import __main__
__main__.preprocess_data = preprocess_data
sys.modules['__main__'].preprocess_data = preprocess_data

# 3. Model Loading Logic
def load_latest_model():
    model_dir = 'models/'
    if not os.path.exists(model_dir):
        return None
    files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not files:
        return None
    latest_file = sorted(files)[-1]
    return joblib.load(os.path.join(model_dir, latest_file))

model = load_latest_model()

# 4. Streamlit UI
st.title("HR Employee Attrition Predictor")
st.write("Upload employee data (CSV) to predict attrition risk.")

uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(data.head())

    if st.button("Run Prediction"):
        if model:
            predictions = model.predict(data)
            probs = model.predict_proba(data)[:, 1]
            
            res_df = data.copy()
            res_df['Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
            res_df['Probability'] = probs
            
            st.subheader("Results")
            st.write(res_df[['Prediction', 'Probability']].head(20))
        else:
            st.error("Model could not be loaded.")
