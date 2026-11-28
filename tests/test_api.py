
"""Automated tests for the Flask prediction API."""

import pytest

from src.serving.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""

    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


# Sample input for direct machine failure prediction
VALID_INPUT = {
    "Type": "M",
    "Air temperature": 300,
    "Process temperature": 310,
    "Rotational speed": 1500,
    "Torque": 40,
    "Tool wear": 100,
}


# Sample response returned by Feast
VALID_ONLINE_FEATURES = {
    "machine_record_id": ["record_000001"],
    "Type": ["M"],
    "Air_temperature": [298.1],
    "Process_temperature": [308.6],
    "Rotational_speed": [1551],
    "Torque": [42.8],
    "Tool_wear": [0],
}


class FakeOnlineResponse:
    """Mock Feast online feature response."""

    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


def mock_feature_store(monkeypatch, online_data):
    """Replace Feast with a lightweight mock for API tests."""

    class FakeFeatureStore:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def get_online_features(self, features, entity_rows):
            assert "machine_features:Type" in features
            assert entity_rows == [
                {"machine_record_id": "record_000001"}
            ]

            return FakeOnlineResponse(online_data)

    monkeypatch.setattr(
        "src.serving.app.FeatureStore",
        FakeFeatureStore,
    )


def test_health_endpoint(client):
    """Test whether the API is healthy."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid_input(client):
    """Test prediction with valid direct input."""

    response = client.post(
        "/predict",
        json=VALID_INPUT,
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
        json=invalid_input,
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
        json=invalid_input,
    )

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data


def test_predict_from_store_valid_record(client, monkeypatch):
    """Test prediction using features retrieved from Feast."""

    mock_feature_store(
        monkeypatch,
        VALID_ONLINE_FEATURES,
    )

    response = client.post(
        "/predict-from-store",
        json={"machine_record_id": "record_000001"},
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["machine_record_id"] == "record_000001"
    assert data["feature_source"] == "Feast Online Store"
    assert data["prediction"] in [0, 1]
    assert 0 <= data["failure_probability"] <= 1
    assert isinstance(data["maintenance_alert"], bool)


def test_predict_from_store_missing_record_id(client):
    """Test API response when record ID is missing."""

    response = client.post(
        "/predict-from-store",
        json={},
    )

    assert response.status_code == 400

    data = response.get_json()
    assert "machine_record_id" in data["error"]


def test_predict_from_store_unknown_record(client, monkeypatch):
    """Test API response when record is absent from Feast."""

    missing_features = {
        "machine_record_id": ["unknown_record"],
        "Type": [None],
        "Air_temperature": [None],
        "Process_temperature": [None],
        "Rotational_speed": [None],
        "Torque": [None],
        "Tool_wear": [None],
    }

    mock_feature_store(
        monkeypatch,
        missing_features,
    )

    response = client.post(
        "/predict-from-store",
        json={"machine_record_id": "record_000001"},
    )

    assert response.status_code == 404

    data = response.get_json()
    assert "not found" in data["error"]


def test_predict_from_store_unavailable(client, monkeypatch):
    """Test API response when Feast cannot be reached."""

    class FailingFeatureStore:
        def __init__(self, repo_path):
            pass

        def get_online_features(self, features, entity_rows):
            raise RuntimeError("Feature store unavailable")

    monkeypatch.setattr(
        "src.serving.app.FeatureStore",
        FailingFeatureStore,
    )

    response = client.post(
        "/predict-from-store",
        json={"machine_record_id": "record_000001"},
    )

    assert response.status_code == 503

    data = response.get_json()
    assert "Unable to retrieve features" in data["error"]
