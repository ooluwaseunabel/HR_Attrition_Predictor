
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Define the preprocessing function inside app.py so joblib can find it
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

# 2. Load the most recent model
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

# 3. App Interface
st.title("HR Employee Attrition Predictor")
st.write("Upload employee data to predict attrition risk.")

uploaded_file = st.file_uploader("Upload Employee Data CSV", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    if st.button("Predict Attrition"):
        if model is not None:
            # The model pipeline includes the preprocessor, so we pass raw data
            predictions = model.predict(data)
            probs = model.predict_proba(data)[:, 1]
            
            results = data.copy()
            results['Attrition_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
            results['Attrition_Probability'] = probs
            
            st.write("### Prediction Results")
            st.dataframe(results[['Attrition_Prediction', 'Attrition_Probability']])
        else:
            st.error("Model not found. Please check the 'models' directory.")
