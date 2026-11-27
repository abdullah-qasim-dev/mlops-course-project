
"""Tests for dataset validation."""

import pandas as pd
import pytest

from src.data.validation import REQUIRED_COLUMNS, validate_data


@pytest.fixture
def valid_data():
    """Create a small valid dataset for testing."""

    return pd.DataFrame([
        {
            "UDI": 1,
            "Product ID": "M14860",
            "Type": "M",
            "Air temperature": 300.0,
            "Process temperature": 310.0,
            "Rotational speed": 1500,
            "Torque": 40.0,
            "Tool wear": 100,
            "Machine failure": 0,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    ], columns=REQUIRED_COLUMNS)


def test_valid_data(valid_data):
    """Valid dataset should pass validation."""

    validate_data(valid_data)


def test_missing_columns(valid_data):
    """Dataset with missing columns should be rejected."""

    invalid_data = valid_data.drop(columns=["Torque"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_data(invalid_data)


def test_empty_dataset(valid_data):
    """Empty dataset should be rejected."""

    empty_data = valid_data.iloc[0:0]

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_data(empty_data)


def test_missing_values(valid_data):
    """Dataset containing missing values should be rejected."""

    invalid_data = valid_data.copy()
    invalid_data.loc[0, "Torque"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_data(invalid_data)


def test_duplicate_rows(valid_data):
    """Duplicate rows should be rejected."""

    invalid_data = pd.concat(
        [valid_data, valid_data],
        ignore_index=True
    )

    with pytest.raises(ValueError, match="duplicate rows"):
        validate_data(invalid_data)


def test_invalid_machine_type(valid_data):
    """Invalid machine type should be rejected."""

    invalid_data = valid_data.copy()
    invalid_data.loc[0, "Type"] = "X"

    with pytest.raises(ValueError, match="Type"):
        validate_data(invalid_data)


def test_negative_sensor_values(valid_data):
    """Negative sensor measurements should be rejected."""

    invalid_data = valid_data.copy()
    invalid_data.loc[0, "Torque"] = -10

    with pytest.raises(ValueError, match="negative values"):
        validate_data(invalid_data)


def test_invalid_target_values(valid_data):
    """Target column should contain only 0 or 1."""

    invalid_data = valid_data.copy()
    invalid_data.loc[0, "Machine failure"] = 2

    with pytest.raises(ValueError, match="only 0 or 1"):
        validate_data(invalid_data)


def test_infinite_sensor_values(valid_data):
    """Infinite sensor measurements should be rejected."""

    invalid_data = valid_data.copy()
    invalid_data.loc[0, "Torque"] = float("inf")

    with pytest.raises(ValueError, match="infinite or invalid"):
        validate_data(invalid_data)
