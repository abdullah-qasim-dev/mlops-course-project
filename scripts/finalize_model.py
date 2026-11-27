
"""Train and evaluate the final tuned Random Forest model."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.validation import validate_data
from src.data.preprocessing import create_preprocessor
from src.features.build_features import build_features
from src.models.evaluate import evaluate_model


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
TRACKING_DIR = ROOT_DIR / "mlruns"
MODEL_DIR = ROOT_DIR / "models"

EXPERIMENT_NAME = "AI4I Predictive Maintenance"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    """Train the final model and evaluate it on test data."""

    # Load and validate the raw dataset.
    data = pd.read_csv(RAW_DATA_PATH)
    validate_data(data)

    X, y = build_features(data)

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Create the final preprocessing and model pipeline.
    final_model = Pipeline([
        ("preprocessor", create_preprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=400,
                min_samples_split=5,
                min_samples_leaf=1,
                max_samples=0.7,
                max_features=None,
                max_depth=20,
                criterion="log_loss",
                class_weight={0: 1, 1: 5},
                bootstrap=True,
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
        ),
    ])

    # Train using the complete training set.
    final_model.fit(X_train, y_train)

    # Evaluate only once on the untouched test set.
    test_metrics, test_matrix = evaluate_model(
        final_model,
        X_test,
        y_test,
    )

    print("\nFinal test metrics:")
    print(test_metrics)

    print("\nFinal test confusion matrix:")
    print(test_matrix)

    # Save the final model locally.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "final_random_forest.joblib"
    joblib.dump(final_model, model_path)

    # Configure MLflow.
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    tracking_database = TRACKING_DIR / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_database.as_posix()}"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Log final model and test results in MLflow.
    with mlflow.start_run(run_name="final_random_forest"):

        mlflow.log_params({
            "n_estimators": 400,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "max_samples": 0.7,
            "max_features": "None",
            "max_depth": 20,
            "criterion": "log_loss",
            "class_weight": "{0: 1, 1: 5}",
            "bootstrap": True,
            "random_state": RANDOM_STATE,
        })

        mlflow.log_metrics({
            f"test_{name}": value
            for name, value in test_metrics.items()
        })

        mlflow.log_dict(
            {
                "labels": [0, 1],
                "matrix": test_matrix,
            },
            "test_confusion_matrix.json",
        )

        mlflow.sklearn.log_model(
            final_model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    print("\nFinal model saved at:")
    print(model_path)

    print("\nMLflow tracking directory:")
    print(TRACKING_DIR)


if __name__ == "__main__":
    main()
