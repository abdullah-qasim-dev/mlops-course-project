
"""Flask API for predictive maintenance with Swagger documentation."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger

from src.data.validation import MODEL_FEATURE_COLUMNS, NUMERICAL_FEATURES

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "final_random_forest.joblib"

# Load trained model pipeline
model = joblib.load(MODEL_PATH)

# Create Flask application
app = Flask(__name__)

# Enable Swagger documentation
swagger = Swagger(app)


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
    Predict machine failure.
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

    # Generate prediction
    prediction = int(model.predict(input_df)[0])

    failure_probability = float(
        model.predict_proba(input_df)[0][1]
    )

    return jsonify({
        "prediction": prediction,
        "failure_probability": round(failure_probability, 4),
        "maintenance_alert": prediction == 1,
        "message": (
            "Maintenance required."
            if prediction == 1
            else "No maintenance alert."
        )
    }), 200


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )
