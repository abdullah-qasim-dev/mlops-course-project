
"""Flask API for predictive maintenance with direct and Feast-based prediction."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger
from feast import FeatureStore

from src.data.validation import MODEL_FEATURE_COLUMNS, NUMERICAL_FEATURES


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "final_random_forest.joblib"
FEATURE_STORE_REPO = PROJECT_ROOT / "feature_store" / "feature_repo"

# Load trained model pipeline
model = joblib.load(MODEL_PATH)

# Feast feature names
FEAST_FEATURES = [
    "machine_features:Type",
    "machine_features:Air_temperature",
    "machine_features:Process_temperature",
    "machine_features:Rotational_speed",
    "machine_features:Torque",
    "machine_features:Tool_wear",
]

# Map Feast feature names to the model's expected column names
FEAST_TO_MODEL_COLUMNS = {
    "Type": "Type",
    "Air_temperature": "Air temperature",
    "Process_temperature": "Process temperature",
    "Rotational_speed": "Rotational speed",
    "Torque": "Torque",
    "Tool_wear": "Tool wear",
}

# Create Flask application
app = Flask(__name__)

# Enable Swagger documentation
swagger = Swagger(app)


def generate_prediction(input_df: pd.DataFrame) -> dict:
    """Generate a machine failure prediction."""

    prediction = int(model.predict(input_df)[0])

    failure_probability = float(
        model.predict_proba(input_df)[0][1]
    )

    return {
        "prediction": prediction,
        "failure_probability": round(failure_probability, 4),
        "maintenance_alert": prediction == 1,
        "message": (
            "Maintenance required."
            if prediction == 1
            else "No maintenance alert."
        ),
    }


@app.route("/health", methods=["GET"])
def health():
    """
    Check API health.
    ---
    tags:
      - Health
    responses:
      200:
        description: API and model are running
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            model_loaded:
              type: boolean
              example: true
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict machine failure using directly supplied sensor values.
    ---
    tags:
      - Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - Type
            - Air temperature
            - Process temperature
            - Rotational speed
            - Torque
            - Tool wear
          properties:
            Type:
              type: string
              enum: [L, M, H]
              example: M
            Air temperature:
              type: number
              example: 300
            Process temperature:
              type: number
              example: 310
            Rotational speed:
              type: number
              example: 1500
            Torque:
              type: number
              example: 40
            Tool wear:
              type: number
              example: 100
    responses:
      200:
        description: Prediction generated successfully
      400:
        description: Invalid or missing input
    """

    input_data = request.get_json(silent=True)

    if not isinstance(input_data, dict):
        return jsonify({
            "error": "Request must contain valid JSON."
        }), 400

    # Check required features
    missing_features = [
        feature for feature in MODEL_FEATURE_COLUMNS
        if feature not in input_data
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features
        }), 400

    # Validate categorical feature
    if input_data["Type"] not in ["L", "M", "H"]:
        return jsonify({
            "error": "Type must be L, M, or H."
        }), 400

    # Validate numerical features
    try:
        for feature in NUMERICAL_FEATURES:
            input_data[feature] = float(input_data[feature])

            if not pd.notna(input_data[feature]) or input_data[feature] < 0:
                raise ValueError

    except (TypeError, ValueError):
        return jsonify({
            "error": "Numerical features must contain valid non-negative numbers."
        }), 400

    # Convert input into DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=MODEL_FEATURE_COLUMNS
    )

    return jsonify(generate_prediction(input_df)), 200


@app.route("/predict-from-store", methods=["POST"])
def predict_from_store():
    """
    Predict machine failure using features retrieved from Feast.
    ---
    tags:
      - Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - machine_record_id
          properties:
            machine_record_id:
              type: string
              example: record_000001
    responses:
      200:
        description: Prediction generated using Feast online features
      400:
        description: Invalid or missing record ID
      404:
        description: Record not found in Feast online store
      503:
        description: Feast feature retrieval failed
    """

    input_data = request.get_json(silent=True)

    if not isinstance(input_data, dict):
        return jsonify({
            "error": "Request must contain valid JSON."
        }), 400

    record_id = input_data.get("machine_record_id")

    if not isinstance(record_id, str) or not record_id.strip():
        return jsonify({
            "error": "A valid machine_record_id is required."
        }), 400

    # Retrieve features from Feast online store
    try:
        store = FeatureStore(repo_path=str(FEATURE_STORE_REPO))

        online_features = store.get_online_features(
            features=FEAST_FEATURES,
            entity_rows=[
                {"machine_record_id": record_id}
            ],
        ).to_dict()

        # Convert Feast feature names to model feature names
        retrieved_data = {
            model_column: online_features[feast_column][0]
            for feast_column, model_column in FEAST_TO_MODEL_COLUMNS.items()
        }

    except Exception:
        app.logger.exception("Feast online feature retrieval failed.")

        return jsonify({
            "error": "Unable to retrieve features from Feast online store."
        }), 503

    # Check whether the record exists
    if any(value is None for value in retrieved_data.values()):
        return jsonify({
            "error": "Record not found in Feast online store.",
            "machine_record_id": record_id
        }), 404

    # Validate retrieved categorical feature
    if retrieved_data["Type"] not in ["L", "M", "H"]:
        return jsonify({
            "error": "Retrieved machine Type is invalid."
        }), 503

    # Validate retrieved numerical features
    try:
        for feature in NUMERICAL_FEATURES:
            retrieved_data[feature] = float(retrieved_data[feature])

            if (
                not pd.notna(retrieved_data[feature])
                or retrieved_data[feature] < 0
            ):
                raise ValueError

    except (TypeError, ValueError):
        return jsonify({
            "error": "Retrieved sensor features contain invalid values."
        }), 503

    # Prepare model input
    input_df = pd.DataFrame(
        [retrieved_data],
        columns=MODEL_FEATURE_COLUMNS
    )

    # Generate prediction
    result = generate_prediction(input_df)
    result["machine_record_id"] = record_id
    result["feature_source"] = "Feast Online Store"

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )
