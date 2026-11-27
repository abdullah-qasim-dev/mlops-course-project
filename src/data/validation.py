
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

REQUIRED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + MODEL_FEATURE_COLUMNS
    + [TARGET_COLUMN]
    + FAILURE_MODE_COLUMNS
)


def validate_data(data: pd.DataFrame) -> None:
    """Check the dataset columns and values."""

    # Check that all expected columns exist.
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )

    # Check that the dataset is not empty.
    if data.empty:
        raise ValueError("Dataset is empty.")

    # Check for missing values.
    missing_values = data.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise ValueError(
            f"Dataset contains missing values:\n{missing_values.to_string()}"
        )

    # Check for duplicate rows.
    duplicate_count = int(data.duplicated().sum())

    if duplicate_count:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate rows."
        )

    # Check machine type values.
    valid_types = {"L", "M", "H"}

    invalid_types = [
        value
        for value in data["Type"].unique()
        if not isinstance(value, str) or value not in valid_types
    ]

    if invalid_types:
        raise ValueError(
            "'Type' must contain only L, M, or H."
        )

    # Check that sensor and target columns are numeric.
    numeric_columns = (
        NUMERICAL_FEATURES
        + [TARGET_COLUMN]
        + FAILURE_MODE_COLUMNS
    )

    non_numeric_columns = [
        column
        for column in numeric_columns
        if not is_numeric_dtype(data[column].dtype)
    ]

    if non_numeric_columns:
        raise TypeError(
            f"Expected numeric values in columns: {', '.join(non_numeric_columns)}"
        )

    # Check for infinite or invalid numeric values.
    numeric_values = data[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Numeric columns contain infinite or invalid values."
        )

    # Target and failure mode columns must contain only 0 or 1.
    binary_columns = [TARGET_COLUMN] + FAILURE_MODE_COLUMNS

    for column in binary_columns:
        if not data[column].isin([0, 1]).all():
            raise ValueError(
                f"'{column}' must contain only 0 or 1."
            )

    # Sensor measurements should not be negative.
    if (data[NUMERICAL_FEATURES] < 0).any().any():
        raise ValueError(
            "Sensor measurements must not contain negative values."
        )
