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

from src.predict import make_prediction, numerical_features_dict, categorical_features_dict
from src.db import save_to_db

st.set_page_config(page_title="Single Prediction", layout="wide")
st.title("Single Employee Attrition Prediction")

st.markdown("Input employee parameters below to calculate their real-time attrition risk score.")

# Group inputs using Streamlit tabs for cleaner layout
tab1, tab2, tab3 = st.tabs(["Personal Metrics", "Job Details", "Satisfaction & History"])

input_data = {}

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        input_data['Age'] = st.slider("Age", *numerical_features_dict['Age'])
        input_data['Gender'] = st.selectbox("Gender", categorical_features_dict['Gender'])
        input_data['MaritalStatus'] = st.selectbox("Marital Status", categorical_features_dict['MaritalStatus'])
    with col2:
        input_data['DistanceFromHome'] = st.slider("Distance From Home (miles)", *numerical_features_dict['DistanceFromHome'])
        input_data['Education'] = st.slider("Education Level", *numerical_features_dict['Education'])
        input_data['EducationField'] = st.selectbox("Education Field", categorical_features_dict['EducationField'])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        input_data['JobRole'] = st.selectbox("Job Role", categorical_features_dict['JobRole'])
        input_data['TotalWorkingYears'] = st.slider("Total Working Years", *numerical_features_dict['TotalWorkingYears'])
        input_data['NumCompaniesWorked'] = st.slider("Number of Companies Worked", *numerical_features_dict['NumCompaniesWorked'])
    with col2:
        input_data['MonthlyIncome'] = st.slider("Monthly Income ($)", *numerical_features_dict['MonthlyIncome'])
        input_data['DailyRate'] = st.slider("Daily Rate ($)", *numerical_features_dict['DailyRate'])
        input_data['HourlyRate'] = st.slider("Hourly Rate ($)", *numerical_features_dict['HourlyRate'])
        input_data['MonthlyRate'] = st.slider("Monthly Rate ($)", *numerical_features_dict['MonthlyRate'])
        input_data['OverTime'] = st.selectbox("Overtime Status", categorical_features_dict['OverTime'])

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        input_data['EnvironmentSatisfaction'] = st.slider("Environment Satisfaction", *numerical_features_dict['EnvironmentSatisfaction'])
        input_data['JobSatisfaction'] = st.slider("Job Satisfaction", *numerical_features_dict['JobSatisfaction'])
        input_data['JobInvolvement'] = st.slider("Job Involvement", *numerical_features_dict['JobInvolvement'])
        input_data['RelationshipSatisfaction'] = st.slider("Relationship Satisfaction", *numerical_features_dict['RelationshipSatisfaction'])
    with col2:
        input_data['WorkLifeBalance'] = st.slider("Work-Life Balance", *numerical_features_dict['WorkLifeBalance'])
        input_data['StockOptionLevel'] = st.slider("Stock Option Level", *numerical_features_dict['StockOptionLevel'])
        input_data['TrainingTimesLastYear'] = st.slider("Training Times Last Year", *numerical_features_dict['TrainingTimesLastYear'])
        input_data['YearsAtCompany'] = st.slider("Years At Company", *numerical_features_dict['YearsAtCompany'])
        input_data['YearsInCurrentRole'] = st.slider("Years In Current Role", *numerical_features_dict['YearsInCurrentRole'])
        input_data['YearsSinceLastPromotion'] = st.slider("Years Since Last Promotion", *numerical_features_dict['YearsSinceLastPromotion'])

# Run the inference engine
if st.button("Calculate Risk Profile"):
    try:
        df_input = pd.DataFrame([input_data])
        probabilities, predictions, shap_values = make_prediction(df_input)
        
        prob_val = probabilities[0]
        pred_val = predictions[0]
        
        st.write("---")
        if pred_val == 1:
            st.error(f"**High Risk of Attrition:** The system estimates a **{prob_val:.2%}** probability that this employee will depart.")
        else:
            st.success(f"**Low Risk of Attrition:** The system estimates a **{prob_val:.2%}** probability of departure.")
            
        # Format metrics record payload for database entry
        df_db = df_input.copy()
        df_db['Attrition_Probability'] = prob_val
        df_db['prediction'] = int(pred_val)
        df_db['shap_values'] = [shap_values[0].tolist()]
        
        if save_to_db(df_db):
            st.caption("Prediction metrics cleanly synchronized to remote storage logs.")
        else:
            st.caption("Database communication logging failure.")
            
    except Exception as e:
        st.error(f"Prediction Pipeline Error: {e}")
