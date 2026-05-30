import pandas as pd

# --- FIX: Changed relative import to absolute import for Streamlit stability ---
from src.predict import numerical_features_dict, categorical_features_dict

def validate_dataframe(df):
    """
    Validates that the uploaded DataFrame matches the expected types and business rules.
    Returns a list of error strings. If empty, the validation passed.
    """
    errors = []

    # 1. Check for expected columns
    expected_cols = list(numerical_features_dict.keys()) + list(categorical_features_dict.keys())
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors  # Exit early if structural layout is broken

    # 2. Validate Numerical Ranges
    for col, (min_val, max_val, _) in numerical_features_dict.items():
        # Check data type
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric.")
            continue

        # Check range values
        out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
        if not out_of_bounds.empty:
            errors.append(f"Column '{col}' has values outside valid range [{min_val}, {max_val}].")

    # 3. Validate Categorical Options
    for col, valid_options in categorical_features_dict.items():
        invalid_rows = df[~df[col].isin(valid_options)]
        if not invalid_rows.empty:
            unique_invalid = invalid_rows[col].unique()
            errors.append(f"Column '{col}' contains invalid options: {unique_invalid}. Allowed: {valid_options}")

    return errors
