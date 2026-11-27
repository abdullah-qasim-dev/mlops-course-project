"""Train, compare, and evaluate baseline models with local MLflow tracking."""

from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.data.preprocessing import preprocess_data
from src.data.validation import validate_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate_model
from src.models.train import create_models, train_model


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
TRACKING_DIR = ROOT_DIR / "mlruns"
EXPERIMENT_NAME = "AI4I Predictive Maintenance"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def main() -> None:
    """Compare baseline models on validation data and test the selected model."""
    data = pd.read_csv(RAW_DATA_PATH)
    validate_data(data)
    X, y = build_features(data)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_fit_raw, X_validation_raw, y_fit, y_validation = train_test_split(
        X_train_raw,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    X_fit, X_validation, _ = preprocess_data(X_fit_raw, X_validation_raw)

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    tracking_database = TRACKING_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_database.as_posix()}")

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=(TRACKING_DIR / "artifacts").as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    validation_metrics = {}
    models = create_models()
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"validation_{model_name}"):
            parameters = {"model_name": model_name}
            parameters.update(
                {name: str(value) for name, value in model.get_params().items()}
            )
            mlflow.log_params(parameters)
            train_model(model, X_fit, y_fit)
            metrics, matrix = evaluate_model(model, X_validation, y_validation)
            validation_metrics[model_name] = metrics

            mlflow.log_metrics(
                {f"validation_{name}": value for name, value in metrics.items()}
            )
            mlflow.log_dict(
                {"labels": [0, 1], "matrix": matrix},
                "validation_confusion_matrix.json",
            )
            trusted_types = (
                ["sklearn.tree._tree.Tree"]
                if model_name in {"decision_tree", "random_forest"}
                else None
            )
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=trusted_types,
            )

        print(f"{model_name} validation metrics: {metrics}")

    best_model_name = max(
        validation_metrics,
        key=lambda name: validation_metrics[name]["average_precision"],
    )

    X_train, X_test, final_preprocessor = preprocess_data(
        X_train_raw, X_test_raw
    )
    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Final training and test feature columns do not match.")

    final_model = clone(models[best_model_name])
    train_model(final_model, X_train, y_train)

    # The test set is used only for the final evaluation, not model selection.
    test_metrics, test_matrix = evaluate_model(final_model, X_test, y_test)
    with mlflow.start_run(run_name=f"final_{best_model_name}"):
        parameters = {
            "model_name": best_model_name,
            "selected_by": "validation_average_precision",
        }
        parameters.update(
            {name: str(value) for name, value in final_model.get_params().items()}
        )
        mlflow.log_params(parameters)
        mlflow.log_metric(
            "selected_validation_average_precision",
            validation_metrics[best_model_name]["average_precision"],
        )
        mlflow.log_metrics(
            {f"test_{name}": value for name, value in test_metrics.items()}
        )
        mlflow.log_dict(
            {"labels": [0, 1], "matrix": test_matrix},
            "test_confusion_matrix.json",
        )
        trusted_types = (
            ["sklearn.tree._tree.Tree"]
            if best_model_name in {"decision_tree", "random_forest"}
            else None
        )
        mlflow.sklearn.log_model(
            final_model,
            name="model",
            skops_trusted_types=trusted_types,
        )
        # Log a temporary copy so the existing processed-data artifact stays unchanged.
        with TemporaryDirectory() as temporary_directory:
            preprocessor_path = Path(temporary_directory) / "preprocessor.joblib"
            joblib.dump(final_preprocessor, preprocessor_path)
            mlflow.log_artifact(
                str(preprocessor_path),
                artifact_path="preprocessor",
            )

    print(f"Selected model by validation Average Precision: {best_model_name}")
    print(f"Final test metrics: {test_metrics}")
    print(f"Final test confusion matrix (labels 0, 1): {test_matrix}")
    print(f"MLflow tracking directory: {TRACKING_DIR}")


if __name__ == "__main__":
    main()
