import streamlit as st
import pandas as pd
import sys
import os
import shap
import matplotlib.pyplot as plt

# --- SYSTEM PATH FIX ---
# This __file__ variable will work when Streamlit runs this as a script
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------------

from src.predict import predict_batch, numerical_features_dict, categorical_features_dict, feature_names_for_shap
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
        # predict_batch returns result_df, shap_values, X_transformed_for_shap
        result_df, shap_values, X_transformed_for_shap = predict_batch(input_df)
        prob = result_df['Attrition_Probability'].iloc[0]
        prediction = result_df['prediction'].iloc[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"High Attrition Risk (Probability: {prob:.2%})")
        else:
            st.success(f"Low Attrition Risk (Probability: {prob:.2%})")

        if save_to_db(result_df):
            st.success("Prediction saved to database!")
        else:
            st.error("Failed to save prediction to database.")

        st.subheader("Explanation (SHAP Values)")
        
        # Determine the single explanation
        # LinearExplainer shap_values can be a simple array for regression/binary
        shap_values_single = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        X_single = X_transformed_for_shap[0] if len(X_transformed_for_shap.shape) > 1 else X_transformed_for_shap

        with st.expander("View Detailed SHAP Explanation"):
            # Create the SHAP Explanation object
            # Note: shap_explainer.expected_value is usually available globally from src.predict
            from src.predict import shap_explainer
            
            fig, ax = plt.subplots(figsize=(10, 6))
            exp = shap.Explanation(
                values=shap_values_single, 
                base_values=shap_explainer.expected_value, 
                data=X_single, 
                feature_names=feature_names_for_shap
            )
            shap.waterfall_plot(exp, show=False)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
