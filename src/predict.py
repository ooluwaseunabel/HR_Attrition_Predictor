import pandas as pd
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
import shap
import numpy as np

# Define numerical and categorical features
numerical_features_dict = {
    'Age': (18, 60, 30),
    'DailyRate': (100, 1500, 800),
    'DistanceFromHome': (1, 29, 10),
    'Education': (1, 5, 3),
    'EnvironmentSatisfaction': (1, 4, 3),
    'HourlyRate': (30, 100, 65),
    'JobInvolvement': (1, 4, 3),
    'JobSatisfaction': (1, 4, 3),
    'MonthlyIncome': (1000, 20000, 6500),
    'MonthlyRate': (2000, 27000, 14000),
    'NumCompaniesWorked': (0, 9, 2),
    'PercentSalaryHike': (11, 25, 15),
    'RelationshipSatisfaction': (1, 4, 3),
    'StockOptionLevel': (0, 3, 1),
    'TotalWorkingYears': (0, 40, 10),
    'TrainingTimesLastYear': (0, 6, 3),
    'WorkLifeBalance': (1, 4, 3),
    'YearsAtCompany': (0, 40, 7),
    'YearsInCurrentRole': (0, 18, 4),
    'YearsSinceLastPromotion': (0, 15, 2)
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
    """
    Cleans data, drops redundant columns, and defines a scaling/encoding pipeline.
    """
    df = df.copy()
    if 'Attrition' in df.columns and df['Attrition'].dtype == 'object':
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    redundant_cols = [
      'EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber',
      'Department', 'YearsWithCurrManager', 'JobLevel', 'PerformanceRating'
    ]
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns])
    X = df.drop('Attrition', axis=1) if 'Attrition' in df.columns else df
    return X

def load_model_and_explainer():
    """
    Loads model and initializes SHAP LinearExplainer.
    """
    # Find root path relative to this file (src/predict.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, '..'))
    model_dir_full_path = os.path.join(repo_root, 'models')

    if not os.path.exists(model_dir_full_path):
        return None, None, None

    files = [f for f in os.listdir(model_dir_full_path) if f.endswith('.pkl')]
    if not files:
        return None, None, None

    latest_file = sorted(files)[-1]
    model_path = os.path.join(model_dir_full_path, latest_file)

    try:
        with open(model_path, 'rb') as f:
            smote_pipe = pickle.load(f)

        final_estimator = smote_pipe.named_steps['model']
        preprocessor = smote_pipe.named_steps['preprocessor']

        # Get combined feature names for SHAP
        numerical_feature_names = preprocessor.named_transformers_['num'].get_feature_names_out()
        categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        all_feature_names = np.concatenate([numerical_feature_names, categorical_feature_names])

        # Create background data for SHAP using defaults
        sample_data = {}
        for k, (_,_,default) in numerical_features_dict.items():
            sample_data[k] = default
        for k, options in categorical_features_dict.items():
            sample_data[k] = options[0]

        sample_df = pd.DataFrame([sample_data])
        X_processed_bg = preprocess_data(sample_df)
        shap_bg_data = preprocessor.transform(X_processed_bg)

        explainer = shap.LinearExplainer(final_estimator, shap_bg_data, feature_names=all_feature_names)
        return smote_pipe, explainer, all_feature_names
    except Exception as e:
        return None, None, None

# Initialize Global objects
model_pipeline, shap_explainer, feature_names_for_shap = load_model_and_explainer()

def predict_batch(df_input):
    """
    Returns predictions and SHAP values.
    """
    if model_pipeline is None or shap_explainer is None:
        raise ValueError("Model or SHAP explainer not loaded.")

    original_input_df = df_input.copy()
    X_processed = preprocess_data(df_input)

    probabilities = model_pipeline.predict_proba(X_processed)[:, 1]
    predictions = (probabilities > 0.5).astype(int)

    X_transformed_for_shap = model_pipeline.named_steps['preprocessor'].transform(X_processed)
    shap_vals = shap_explainer.shap_values(X_transformed_for_shap)

    original_input_df['Attrition_Probability'] = probabilities
    original_input_df['prediction'] = predictions

    return original_input_df, shap_vals, X_transformed_for_shap
