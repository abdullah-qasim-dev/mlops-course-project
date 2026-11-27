
"""Automated tests for the Flask prediction API."""

import pytest

from src.serving.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client

# Sample input for machine failure prediction
VALID_INPUT = {
    "Type": "M",
    "Air temperature": 300,
    "Process temperature": 310,
    "Rotational speed": 1500,
    "Torque": 40,
    "Tool wear": 100
}


def test_health_endpoint(client):
    """Test whether the API is healthy."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid_input(client):
    """Test prediction with valid input."""
    response = client.post(
        "/predict",
        json=VALID_INPUT
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data
    assert "failure_probability" in data
    assert "maintenance_alert" in data
    assert "message" in data

    assert data["prediction"] in [0, 1]
    assert 0 <= data["failure_probability"] <= 1
    assert isinstance(data["maintenance_alert"], bool)


def test_predict_missing_features(client):
    """Test API response when a required feature is missing."""
    invalid_input = VALID_INPUT.copy()
    invalid_input.pop("Torque")

    response = client.post(
        "/predict",
        json=invalid_input
    )

    assert response.status_code == 400

    data = response.get_json()
    assert "missing_features" in data
    assert "Torque" in data["missing_features"]


def test_predict_invalid_machine_type(client):
    """Test API response when machine type is invalid."""
    invalid_input = VALID_INPUT.copy()
    invalid_input["Type"] = "X"

    response = client.post(
        "/predict",
        json=invalid_input
    )

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data
