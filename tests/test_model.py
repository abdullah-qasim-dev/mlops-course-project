
"""Tests for the trained machine failure prediction model."""

from pathlib import Path

import joblib
import pandas as pd

from src.data.validation import MODEL_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_random_forest.joblib"


def test_model_file_exists():
    """Check that the trained model file exists."""

    assert MODEL_PATH.exists()


def test_model_loads_successfully():
    """Check that the trained model can be loaded."""

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_prediction():
    """Check that the model generates a valid prediction."""

    model = joblib.load(MODEL_PATH)

    sample_data = pd.DataFrame(
        [{
            "Type": "M",
            "Air temperature": 300,
            "Process temperature": 310,
            "Rotational speed": 1500,
            "Torque": 40,
            "Tool wear": 100,
        }],
        columns=MODEL_FEATURE_COLUMNS
    )

    prediction = model.predict(sample_data)

    assert len(prediction) == 1
    assert int(prediction[0]) in [0, 1]


def test_model_failure_probability():
    """Check that predicted failure probability is valid."""

    model = joblib.load(MODEL_PATH)

    sample_data = pd.DataFrame(
        [{
            "Type": "M",
            "Air temperature": 300,
            "Process temperature": 310,
            "Rotational speed": 1500,
            "Torque": 40,
            "Tool wear": 100,
        }],
        columns=MODEL_FEATURE_COLUMNS
    )

    probability = model.predict_proba(sample_data)[0][1]

    assert 0 <= probability <= 1
