import pandas as pd
from .predict import numerical_features_dict, categorical_features_dict

def validate_dataframe(df: pd.DataFrame):
    """
    Validates the input DataFrame for batch prediction.
    Returns a list of error messages, or an empty list if validation passes.
    """
    errors = []
    expected_columns = list(numerical_features_dict.keys()) + list(categorical_features_dict.keys())

    # Check for missing columns
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Basic type checking and value range checking (simplified)
    for feature, (min_val, max_val, _) in numerical_features_dict.items():
        if feature in df.columns:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                errors.append(f"Column '{feature}' must be numeric.")
            elif (df[feature] < min_val).any() or (df[feature] > max_val).any():
                errors.append(f"Column '{feature}' contains values outside the expected range [{min_val}, {max_val}].")

    for feature, options in categorical_features_dict.items():
        if feature in df.columns:
            invalid_values = df[~df[feature].isin(options)][feature].unique()
            if len(invalid_values) > 0:
                errors.append(
                    f"Column '{feature}' contains invalid categorical values: "
                    f"{', '.join(map(str, invalid_values))}. Expected: {', '.join(options)}."
                )

    return errors
