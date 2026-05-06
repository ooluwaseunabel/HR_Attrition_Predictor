
import streamlit as st
import pandas as pd
from src.validation import validate_dataframe
from src.predict import predict_batch, numerical_features_dict, categorical_features_dict
from src.db import save_to_db

st.set_page_config(page_title="Batch Prediction", layout="wide")
st.title("Batch Prediction")

# Download template button
EXPECTED_COLUMNS = list(numerical_features_dict.keys()) + list(categorical_features_dict.keys())
template_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
st.download_button(
    "Download CSV Template",
    template_df.to_csv(index=False).encode('utf-8'), # Encode for download
    "attrition_template.csv",
    "text/csv",
    key='download_template_csv'
)

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df_uploaded = pd.read_csv(file)

    # Preprocess uploaded data before validation to ensure consistent column sets
    # Add dummy columns for any missing but expected categorical features, filling with a default.
    for col in categorical_features_dict.keys():
        if col not in df_uploaded.columns:
            df_uploaded[col] = categorical_features_dict[col][0] # Default to first option

    # Filter to only the columns the model expects
    df_processed_for_validation = df_uploaded[EXPECTED_COLUMNS]

    errors = validate_dataframe(df_processed_for_validation)

    if errors:
        for err in errors:
            st.error(err)
    else:
        if st.button("Run Prediction"): # Unique key for this button
            try:
                result_df = predict_batch(df_processed_for_validation.copy()) # Pass a copy to avoid modifying original

                if save_to_db(result_df): # Save the dataframe with predictions
                    st.success("Batch processed and results logged to Database!")
                    st.dataframe(result_df.head())

                    st.download_button(
                        "Download Results",
                        result_df.to_csv(index=False).encode('utf-8'),
                        "predictions.csv",
                        "text/csv"
                    )
                else:
                    st.error("Failed to save batch predictions to database.")
            except ValueError as e:
                st.error(f"Prediction Error: {e}")

