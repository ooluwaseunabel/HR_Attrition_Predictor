import streamlit as st
import pandas as pd
import sys
import os
import shap
import matplotlib.pyplot as plt

# --- SYSTEM PATH FIX ---
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------------

ffrom src.predict import make_prediction, numerical_features_dict, categorical_features_dict, feature_names_for_shap
from src.db import save_to_db

st.set_page_config(page_title="Single Prediction", layout="wide")
st.title("Single Employee Prediction")

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
        # 1. Run prediction and get SHAP data
        result_df, shap_values, X_transformed_for_shap = predict_batch(input_df)

        prob = result_df['Attrition_Probability'].iloc[0]
        prediction = result_df['prediction'].iloc[0]

        # 2. ATTACH SHAP VALUES FOR DATABASE
        # We take the first row of SHAP values and store them as a list column
        # This is what ensures the 'shap_values' column exists when save_to_db is called
        result_df['shap_values'] = [shap_values[0].tolist()]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"High Attrition Risk (Probability: {prob:.2%})")
        else:
            st.success(f"Low Attrition Risk (Probability: {prob:.2%})")

        # 3. Save the dataframe (now containing predictions AND shap list)
        if save_to_db(result_df):
            st.success("Results and SHAP explanation successfully logged to database.")
        else:
            st.error("Failed to save results to database.")

        # 4. Display Visualization
        st.subheader("Explanation (SHAP Values)")

        shap_values_single = shap_values[0]
        X_single = X_transformed_for_shap[0]

        with st.expander("View Detailed SHAP Explanation", expanded=True):
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
