import streamlit as st
import pandas as pd
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import shap

# --- SYSTEM PATH FIX ---
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------------

from src.db import fetch_predictions
from src.predict import feature_names_for_shap

st.set_page_config(page_title="HR Monitoring Dashboard", layout="wide")
st.title("HR Monitoring Dashboard")

df = fetch_predictions()

if not df.empty:
    # --- 1. METRICS ROW ---
    col1, col2, col3 = st.columns(3)
    total_preds = len(df)
    avg_risk = df["Attrition_Probability"].mean()
    col1.metric("Total Predictions", total_preds)
    col2.metric("Average Attrition Risk", f"{avg_risk:.2%}")

    if "actual_attrition" in df.columns:
        df_acc = df.dropna(subset=["actual_attrition"])
        if not df_acc.empty:
            acc = (df_acc["prediction"] == df_acc["actual_attrition"]).mean()
            col3.metric("Model Accuracy (Live)", f"{acc:.2%}")

    # --- 2. TREND ANALYSIS ---
    st.subheader("Average Risk Over Time")
    time_col = 'created_at' if 'created_at' in df.columns else 'timestamp'
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        trend = df.groupby(df[time_col].dt.date)["Attrition_Probability"].mean()
        st.line_chart(trend)

    # --- 3. GLOBAL SHAP INSIGHTS (Robust Version) ---
    st.subheader("What's Driving Attrition Globally?")

    if 'shap_values' in df.columns:
        try:
            processed_shaps = []
            expected_len = len(feature_names_for_shap)

            for x in df['shap_values']:
                try:
                    # Decode JSON if it's a string
                    val = json.loads(x) if isinstance(x, str) else x
                    # Flatten in case it's nested like [[...]]
                    val_flat = np.array(val).flatten()

                    # Only include if length matches current model features
                    if len(val_flat) == expected_len:
                        processed_shaps.append(val_flat)
                except:
                    continue

            if processed_shaps:
                all_shap_values = np.array(processed_shaps)
                with st.expander("Show Global Feature Importance Plot", expanded=True):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(
                        all_shap_values,
                        feature_names=feature_names_for_shap,
                        plot_type="bar",
                        show=False
                    )
                    st.pyplot(fig)
            else:
                st.warning("No valid SHAP values found that match the current model's feature count.")

        except Exception as e:
            st.error(f"Error processing SHAP values: {e}")
    else:
        st.info("Make some predictions with SHAP enabled to see global drivers.")

    # --- 4. RAW DATA LOG ---
    with st.expander("View Raw Prediction Logs"):
        display_cols = [c for c in df.columns if c != 'shap_values']
        st.dataframe(df[display_cols].sort_values(by=time_col, ascending=False))

else:
    st.warning("No predictions logged yet!")
