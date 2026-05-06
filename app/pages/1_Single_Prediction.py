import streamlit as st
import pandas as pd
import sys
import os

# --- SYSTEM PATH FIX ---
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------------

from src.predict import predict_batch, numerical_features_dict, categorical_features_dict
from src.db import save_to_db

st.set_page_config(page_title="Single Prediction", layout="wide")
st.title("Single Prediction")

employee_data = {}
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

st.write("### Numerical Features")
col_idx = 0
for feature, (min_val, max_val, default_val) in numerical_features_dict.items():
    with cols[col_idx % 3]:
        employee_data[feature] = st.number_input(feature, min_value=min_val, max_value=max_val, value=default_val)
    col_idx += 1

st.write("### Categorical Features")
col_idx = 0 
for feature, options in categorical_features_dict.items():
    with cols[col_idx % 3]:
        employee_data[feature] = st.selectbox(feature, options)
    col_idx += 1

if st.button("Predict and Save"):
    input_df = pd.DataFrame([employee_data])
    try:
        result_df = predict_batch(input_df)
        prob = result_df['Attrition_Probability'].iloc[0]

        if save_to_db(result_df):
            # We use single braces here because it's just a string in Colab
            # but it will be an f-string in the final .py file.
            st.success(f"Prediction: {prob:.2%}")
            st.success("Result saved to PostgreSQL!")
        else:
            st.error("Failed to save prediction to database.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
