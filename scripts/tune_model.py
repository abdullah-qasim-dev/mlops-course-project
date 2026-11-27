
"""Tune Random Forest and track results with MLflow."""

from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.validation import validate_data
from src.features.build_features import build_features
from src.models.config import RANDOM_STATE
from src.models.evaluate import evaluate_model
from src.models.tune import create_tuning_search


ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
TRACKING_DIR = ROOT_DIR / "mlruns"

EXPERIMENT_NAME = "AI4I Predictive Maintenance"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def main() -> None:
    """Tune Random Forest and evaluate it on validation data."""

    # Load and validate the raw dataset.
    data = pd.read_csv(RAW_DATA_PATH)
    validate_data(data)

    X, y = build_features(data)

    # Keep the test set separate for final evaluation.
    X_train, _X_test, y_train, _y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split the training data into fitting and validation sets.
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # Configure MLflow tracking.
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    tracking_database = TRACKING_DIR / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_database.as_posix()}"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Perform hyperparameter tuning using cross-validation.
    search = create_tuning_search()
    search.fit(X_fit, y_fit)

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        "\nBest cross-validation Average Precision:",
        search.best_score_,
    )

    # Evaluate the best model on validation data.
    tuned_model = search.best_estimator_

    validation_metrics, validation_matrix = evaluate_model(
        tuned_model,
        X_validation,
        y_validation,
    )

    print("\nTuned model validation metrics:")
    print(validation_metrics)

    print("\nValidation confusion matrix:")
    print(validation_matrix)

    # Log tuning results in MLflow.
    with mlflow.start_run(run_name="random_forest_tuning_extended"):

        mlflow.log_params(search.best_params_)

        mlflow.log_param("cv_folds", search.cv.n_splits)
        mlflow.log_param("n_iter", search.n_iter)
        mlflow.log_param("scoring", search.scoring)
        mlflow.log_param("random_state", RANDOM_STATE)

        mlflow.log_metric(
            "best_cv_average_precision",
            search.best_score_,
        )

        mlflow.log_metrics({
            f"validation_{name}": value
            for name, value in validation_metrics.items()
        })

        mlflow.log_dict(
            {
                "labels": [0, 1],
                "matrix": validation_matrix,
            },
            "validation_confusion_matrix.json",
        )

        # Save all cross-validation results.
        with TemporaryDirectory() as temporary_directory:
            results_path = (
                Path(temporary_directory) / "tuning_results.csv"
            )

            results = pd.DataFrame(search.cv_results_)
            results.to_csv(results_path, index=False)

            mlflow.log_artifact(str(results_path))

    print("\nTuning completed successfully.")
    print("The test set has not been used for evaluation.")

    print("\nMLflow tracking directory:")
    print(TRACKING_DIR)


if __name__ == "__main__":
    main()
