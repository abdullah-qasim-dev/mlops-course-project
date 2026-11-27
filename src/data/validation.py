"""Basic schema and value checks for the AI4I 2020 dataset."""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


TARGET_COLUMN = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]

MODEL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
REQUIRED_COLUMNS = MODEL_FEATURE_COLUMNS + [TARGET_COLUMN] + FAILURE_MODE_COLUMNS


def validate_data(data: pd.DataFrame) -> None:
    """Raise a clear error if the raw dataset has an invalid schema or values."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    if data.empty:
        raise ValueError("Dataset is empty.")

    missing_values = data.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(f"Dataset contains missing values:\n{missing_values.to_string()}")

    duplicate_count = int(data.duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Dataset contains {duplicate_count} duplicate rows.")

    valid_types = {"L", "M", "H"}
    invalid_types = [
        value
        for value in data["Type"].unique()
        if not isinstance(value, str) or value not in valid_types
    ]
    if invalid_types:
        invalid_values = ", ".join(repr(value) for value in invalid_types)
        raise ValueError(f"'Type' must contain only the string values L, M, or H: {invalid_values}")

    numeric_columns = NUMERICAL_FEATURES + [TARGET_COLUMN] + FAILURE_MODE_COLUMNS
    non_numeric_columns = [
        column for column in numeric_columns if not is_numeric_dtype(data[column].dtype)
    ]
    if non_numeric_columns:
        raise TypeError(
            f"Expected numeric values in columns: {', '.join(non_numeric_columns)}"
        )

    numeric_values = data[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Numeric columns contain infinite or invalid values.")

    for column in [TARGET_COLUMN] + FAILURE_MODE_COLUMNS:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"'{column}' must contain only 0 or 1.")

    if (data[NUMERICAL_FEATURES] < 0).any().any():
        raise ValueError("Sensor measurements must not contain negative values.")
