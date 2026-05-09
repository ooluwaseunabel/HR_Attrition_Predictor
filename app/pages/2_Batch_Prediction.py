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

from src.validation import validate_dataframe
from src.predict import predict_batch, numerical_features_dict, categorical_features_dict, feature_names_for_shap
from src.db import save_to_db

st.set_page_config(page_title="Batch Prediction", layout="wide")
st.title("Batch Prediction & Global Insights")

# Download template button
EXPECTED_COLUMNS = list(numerical_features_dict.keys()) + list(categorical_features_dict.keys())
template_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
st.download_button(
    "Download CSV Template",
    template_df.to_csv(index=False).encode('utf-8'),
    "attrition_template.csv",
    "text/csv",
    key='download_template_csv'
)

file = st.file_uploader("Upload CSV for Batch Processing", type=["csv"])

if file:
    df_uploaded = pd.read_csv(file)

    # Basic cleaning: Fill missing columns with defaults if necessary
    for col in categorical_features_dict.keys():
        if col not in df_uploaded.columns:
            df_uploaded[col] = categorical_features_dict[col][0]

    df_processed_for_validation = df_uploaded[EXPECTED_COLUMNS]
    errors = validate_dataframe(df_processed_for_validation)

    if errors:
        for err in errors:
            st.error(err)
    else:
        st.write("### Data Preview (Top 5 Rows)")
        st.dataframe(df_processed_for_validation.head())
        
        if st.button("Run Batch Prediction"):
            try:
                # 1. Run Prediction and SHAP calculation
                result_df, shap_values, X_transformed_for_shap = predict_batch(df_processed_for_validation.copy())

                # 2. ATTACH SHAP VALUES FOR DATABASE
                # We convert the entire matrix of SHAP values into a list of lists
                # result_df['shap_values'] will now hold the explanation for every row
                result_df['shap_values'] = [sv.tolist() for sv in shap_values]

                # 3. Save to Database (Handles result, SHAP, and automatic timestamp)
                if save_to_db(result_df):
                    st.success(f"Successfully processed {len(result_df)} records and logged to Database!")
                    
                    # Display results (excluding the messy shap_values column for the UI)
                    st.write("### Prediction Results")
                    st.dataframe(result_df.drop(columns=['shap_values']).head())

                    st.download_button(
                        "Download Results CSV",
                        result_df.drop(columns=['shap_values']).to_csv(index=False).encode('utf-8'),
                        "predictions.csv",
                        "text/csv"
                    )

                    # 4. Global Insights Visualization
                    st.subheader("Global Feature Importance")
                    st.info("This chart shows which factors contributed most to the overall attrition risk across this entire batch.")
                    
                    with st.expander("View SHAP Summary Plot", expanded=True):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Summary plot shows the magnitude of feature importance
                        shap.summary_plot(
                            shap_values, 
                            X_transformed_for_shap, 
                            feature_names=feature_names_for_shap, 
                            plot_type="bar", 
                            show=False
                        )
                        st.pyplot(fig)

                else:
                    st.error("Failed to save batch predictions to database.")
            except Exception as e:
                st.error(f"Prediction Error: {e}")