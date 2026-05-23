import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
import json

# Setup Database connection
DB_URI = st.secrets["SUPABASE_DB_URL"]
engine = create_engine(DB_URI)

def save_to_db(df):
    """
    Saves predictions to Supabase.
    Fixes the TypeMismatch by explicitly JSON-encoding the shap_values.
    """
    try:
        df_to_save = df.copy()

        # FIX: Convert SHAP values column to a JSON-formatted string
        if 'shap_values' in df_to_save.columns:
            # We convert the list/array to a JSON string so PostgreSQL sees it as JSONB
            df_to_save['shap_values'] = df_to_save['shap_values'].apply(
                lambda x: json.dumps(x.tolist() if hasattr(x, 'tolist') else x)
            )

        df_to_save.to_sql("attrition_predictions", engine, if_exists="append", index=False)
        return True
    except Exception as e:
        # This will now log the detailed error if one still occurs
        st.error(f"Database Save Error: {e}")
        return False

def fetch_predictions():
    """Fetches historical data including the 'created_at' and 'shap_values' columns."""
    try:
        return pd.read_sql_table("attrition_predictions", engine)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
