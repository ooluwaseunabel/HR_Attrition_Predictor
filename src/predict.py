import pandas as pd
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
import shap
import numpy as np

# Numerical and categorical features
numerical_features_dict = {
    'Age': (18, 60, 30), 'DailyRate': (100, 1500, 800), 'DistanceFromHome': (1, 29, 10),
    'Education': (1, 5, 3), 'EnvironmentSatisfaction': (1, 4, 3), 'HourlyRate': (30, 100, 65),
    'JobInvolvement': (1, 4, 3), 'JobSatisfaction': (1, 4, 3), 'MonthlyIncome': (1000, 20000, 6500),
    'MonthlyRate': (2000, 27000, 14000), 'NumCompaniesWorked': (0, 9, 2), 'PercentSalaryHike': (11, 25, 15),
    'RelationshipSatisfaction': (1, 4, 3), 'StockOptionLevel': (0, 3, 1), 'TotalWorkingYears': (0, 40, 10),
    'TrainingTimesLastYear': (0, 6, 3), 'WorkLifeBalance': (1, 4, 3), 'YearsAtCompany': (0, 40, 7),
    'YearsInCurrentRole': (0, 18, 4), 'YearsSinceLastPromotion': (0, 15, 2)
}

categorical_features_dict = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'EducationField': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['No', 'Yes']
}

def preprocess_data(df):
    df = df.copy()
    if 'Attrition' in df.columns and df['Attrition'].dtype == 'object':
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    redundant_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 'Department', 'YearsWithCurrManager', 'JobLevel', 'PerformanceRating']
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns])
    return df.drop('Attrition', axis=1) if 'Attrition' in df.columns else df

def load_model_and_explainer():
    try:
        # Works on Streamlit Cloud
        base_path = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(base_path, '..'))
    except NameError:
        # Works in Google Colab
        repo_root = os.getcwd()

    model_dir_full_path = os.path.join(repo_root, 'models')

    # Multi-path search safety net to catch execution directory mismatches
    if not os.path.exists(model_dir_full_path):
        if os.path.exists(os.path.join(os.getcwd(), 'models')):
            model_dir_full_path = os.path.join(os.getcwd(), 'models')
        elif os.path.exists(os.path.join(os.getcwd(), '..', 'models')):
            model_dir_full_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'models'))

    if not os.path.exists(model_dir_full_path):
        print(f"CRITICAL: Models directory not found. Searched path: {model_dir_full_path}")
        return None, None, None

    files = [f for f in os.listdir(model_dir_full_path) if f.endswith('.pkl')]
    if not files: 
        print(f"CRITICAL: No .pkl files found in {model_dir_full_path}")
        return None, None, None

    latest_file = sorted(files)[-1]
    model_path = os.path.join(model_dir_full_path, latest_file)

    try:
        with open(model_path, 'rb') as f:
            smote_pipe = pickle.load(f)

        final_estimator = smote_pipe.named_steps['model']
        preprocessor = smote_pipe.named_steps['preprocessor']

        # Get feature names for SHAP
        num_names = preprocessor.named_transformers_['num'].get_feature_names_out()
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        all_feature_names = np.concatenate([num_names, cat_names])

        # Create SHAP explainer
        sample_data = {k: v[2] for k, v in numerical_features_dict.items()}
        sample_data.update({k: v[0] for k, v in categorical_features_dict.items()})
        shap_bg = preprocessor.transform(preprocess_data(pd.DataFrame([sample_data])))

        explainer = shap.LinearExplainer(final_estimator, shap_bg, feature_names=all_feature_names)
        return smote_pipe, explainer, all_feature_names
    except Exception as e:
        print(f"Error loading components: {e}")
        return None, None, None

# Initialize globals
model_pipeline, shap_explainer, feature_names_for_shap = load_model_and_explainer()

def make_prediction(df_input):
    global model_pipeline, shap_explainer, feature_names_for_shap
    
    if model_pipeline is None:
        model_pipeline, shap_explainer, feature_names_for_shap = load_model_and_explainer()
        
    if model_pipeline is None:
        raise ValueError("Model not loaded correctly. Please verify your 'models/' folder path.")
        
    X_processed = preprocess_data(df_input)
    probabilities = model_pipeline.predict_proba(X_processed)[:, 1]
    predictions = (probabilities > 0.5).astype(int)

    X_transformed = model_pipeline.named_steps['preprocessor'].transform(X_processed)
    shap_values = shap_explainer.shap_values(X_transformed)
    
    return probabilities, predictions, shap_values

def predict_batch(df_input):
    global model_pipeline, shap_explainer, feature_names_for_shap
    
    if model_pipeline is None:
        model_pipeline, shap_explainer, feature_names_for_shap = load_model_and_explainer()
        
    if model_pipeline is None:
        raise ValueError("Model not loaded correctly. Please verify your 'models/' folder path.")

    X_processed = preprocess_data(df_input)
    probabilities = model_pipeline.predict_proba(X_processed)[:, 1]
    predictions = (probabilities > 0.5).astype(int)

    # SHAP logic
    X_transformed = model_pipeline.named_steps['preprocessor'].transform(X_processed)
    shap_values = shap_explainer.shap_values(X_transformed)

    df_out = df_input.copy()
    df_out['Attrition_Probability'] = probabilities
    df_out['prediction'] = predictions
    return df_out, shap_values, X_transformed
