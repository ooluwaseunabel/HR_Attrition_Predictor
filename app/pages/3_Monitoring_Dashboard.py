
import streamlit as st
import pandas as pd
from src.db import fetch_predictions

st.set_page_config(page_title="Monitoring Dashboard", layout="wide")
st.title("Monitoring Dashboard")

df = fetch_predictions()

if not df.empty:
    st.subheader("Prediction Distribution")
    # Ensure 'prediction' column exists before attempting to value_counts
    if 'prediction' in df.columns:
        st.bar_chart(df["prediction"].value_counts())
    else:
        st.info("No 'prediction' column found in fetched data for distribution.")

    st.subheader("Average Risk Over Time")
    # Ensure 'timestamp' and 'Attrition_Probability' columns exist
    if 'timestamp' in df.columns and 'Attrition_Probability' in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Group by date to get daily average probability
        trend = df.groupby(df["timestamp"].dt.date)["Attrition_Probability"].mean()
        st.line_chart(trend)
    else:
        st.info("Missing 'timestamp' or 'Attrition_Probability' columns for trend analysis.")

    # If actuals exist, calculate accuracy
    if "actual_attrition" in df.columns and 'prediction' in df.columns:
        df_with_actuals = df.dropna(subset=["actual_attrition"])
        if not df_with_actuals.empty:
            accuracy = (df_with_actuals["prediction"] == df_with_actuals["actual_attrition"]).mean()
            st.metric("Model Accuracy (Live)", f"{accuracy:.2%}")
        else:
            st.info("No actual attrition data available for live accuracy calculation.")
    else:
        st.info("No 'actual_attrition' column found for live accuracy calculation.")
else:
    st.warning("No predictions logged yet. Make some predictions to see the dashboard!")

