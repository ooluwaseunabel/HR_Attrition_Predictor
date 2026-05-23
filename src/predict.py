import pandas as pd
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
import shap
import numpy as np

# Numerical and categorical features configuration dictionaries
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
    model_dir_full_path = None
    
    # 1. Search Chain: Map explicit structural variations across Cloud vs Local deployments
    possible_roots = [
        os.getcwd(),                                                 # App workspace root directory
        os.path.abspath(os.path.join(os.getcwd(), "..")),            # One level up from root
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), # Up relative to src/predict.py
        "/mount/src/hr_attrition_predictor"                         # Hardcoded production fallback
    ]
    
    for root in possible_roots:
        test_path = os.path.join(root, 'models')
        if os.path.exists(test_path) and any(f.endswith('.pkl') for f in os.listdir(test_path)):
            model_dir_full_path = test_path
            break

    if not model_dir_full_path:
        print("CRITICAL ERROR: 'models/' directory containing .pkl weights was not resolved.")
        return None, None, None

    files = [f for f in os.listdir(model_dir_full_path) if f.endswith('.pkl')]
    latest_file = sorted(files)[-1]
    model_path = os.path.join(model_dir_full_path, latest_file)

    try:
        with open(model_path, 'rb') as f:
            smote_pipe = pickle.load(f)

        final_estimator = smote_pipe.named_steps['model']
        preprocessor = smote_pipe.named_steps['preprocessor']

        # Construct target features matrix for SHAP processing
        num_names = preprocessor.named_transformers_['num'].get_feature_names_out()
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        all_feature_names = np.concatenate([num_names, cat_names])

        # Formulate background training context reference point
        sample_data = {k: v[2] for k, v in numerical_features_dict.items()}
        sample_data.update({k: v[0] for k, v in categorical_features_dict.items()})
        shap_bg = preprocessor.transform(preprocess_data(pd.DataFrame([sample_data])))

        explainer = shap.LinearExplainer(final_estimator, shap_bg, feature_names=all_feature_names)
        return smote_pipe, explainer, all_feature_names
    except Exception as e:
        print(f"Pipeline instantiation runtime error: {e}")
        return None, None, None

# Run initial boot assignment configuration
model_pipeline, shap_explainer, feature_names_for_shap = load_model_and_explainer()

def make_prediction(df_input):
    global model_pipeline, shap_explainer, feature_names_for_shap
    
    # FIX: Lazy loading safety mechanism. If startup failed, retry now on user click!
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
    
    # FIX: Lazy loading safety mechanism. If startup failed, retry now on user click!
    if model_pipeline is None:
        model_pipeline, shap_explainer, feature_names_for_shap = load_model_and_explainer()
        
    if model_pipeline is None:
        raise ValueError("Model not loaded correctly. Please verify your 'models/' folder path.")

    original_input_df = df_input.copy()
    X_processed = preprocess_data(df_input)

    probabilities = model_pipeline.predict_proba(X_processed)[:, 1]
    predictions = (probabilities > 0.5).astype(int)

    X_transformed_for_shap = model_pipeline.named_steps['preprocessor'].transform(X_processed)
    shap_vals = shap_explainer.shap_values(X_transformed_for_shap)

    original_input_df['Attrition_Probability'] = probabilities
    original_input_df['prediction'] = predictions

    return original_input_df, shap_vals, X_transformed_for_shap
