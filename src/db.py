import streamlit as st
from sqlalchemy import create_engine
import pandas as pd

# Fetch the Supabase URL from Streamlit secrets
DB_URI = st.secrets["SUPABASE_DB_URL"]
engine = create_engine(DB_URI)

def save_to_db(df):
    """
    Saves the dataframe to the attrition_predictions table in PostgreSQL.
    """
    try:
        df.to_sql("attrition_predictions", engine, if_exists="append", index=False)
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

def fetch_predictions():
    """
    Fetches all predictions from the database.
    """
    try:
        df = pd.read_sql_table("attrition_predictions", engine)
        return df
    except Exception as e:
        st.error(f"Error fetching data from database: {e}")
        return pd.DataFrame()
