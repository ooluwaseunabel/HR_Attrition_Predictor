import pandas as pd
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline

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

def load_model():
    """
    Loads the latest trained model from the models/ directory.
    """
    model_dir = 'models'
    
    if not os.path.exists(model_dir):
        # Adjusted for structure where src is in the root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(repo_root, 'models')
        if not os.path.exists(model_dir):
            return None

    files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not files:
        return None

    latest_file = sorted(files)[-1]
    model_path = os.path.join(model_dir, latest_file)
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        return None

# Global model instance
model = load_model()

def predict_batch(df_input):
    """
    Performs prediction on a batch of input data using the loaded model.
    """
    if model is None:
        raise ValueError("Model not loaded.")

    X_processed = preprocess_data(df_input)
    probabilities = model.predict_proba(X_processed)[:, 1]
    
    df_input['Attrition_Probability'] = probabilities
    df_input['prediction'] = (probabilities > 0.5).astype(int)

    return df_input
