
import streamlit as st
import pandas as pd
import joblib
import os

# 1. Load the most recent model
def load_latest_model():
    model_dir = 'models/'
    files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not files:
        return None
    latest_file = sorted(files)[-1]
    return joblib.load(os.path.join(model_dir, latest_file))

model = load_latest_model()

# 2. App Interface
st.title("HR Employee Attrition Predictor")
st.write("Enter employee details below to predict the likelihood of attrition.")

uploaded_file = st.file_uploader("Upload Employee Data CSV", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    if st.button("Predict Attrition"):
        if model is not None:
            predictions = model.predict(data)
            probs = model.predict_proba(data)[:, 1]
            results = data.copy()
            results['Attrition_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
            results['Attrition_Probability'] = probs
            st.write("Prediction Results:")
            st.dataframe(results[['Attrition_Prediction', 'Attrition_Probability']])
        else:
            st.error("Model not found. Please ensure the models directory contains a .pkl file.")
