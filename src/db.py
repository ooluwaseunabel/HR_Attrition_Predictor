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
    The 'created_at' column is handled automatically by the DB.
    """
    try:
        df_to_save = df.copy()
        
        # Convert SHAP list/array to a JSON string for PostgreSQL JSONB format
        if 'shap_values' in df_to_save.columns:
            df_to_save['shap_values'] = df_to_save['shap_values'].apply(
                lambda x: json.dumps(x.tolist() if hasattr(x, 'tolist') else x)
            )
            
        df_to_save.to_sql("attrition_predictions", engine, if_exists="append", index=False)
        return True
    except Exception as e:
        st.error(f"Database Save Error: {e}")
        return False

def fetch_predictions():
    """Fetches historical data including the new 'created_at' and 'shap_values' columns."""
    try:
        return pd.read_sql_table("attrition_predictions", engine)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
