"""Validate, split, and preprocess the AI4I 2020 dataset."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import preprocess_data
from src.data.validation import IDENTIFIER_COLUMNS, FAILURE_MODE_COLUMNS, validate_data
from src.features.build_features import build_features


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    """Run the full preparation workflow and save its outputs."""
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    data = pd.read_csv(RAW_DATA_PATH)
    validate_data(data)
    X, y = build_features(data)

    # Split before fitting preprocessing so the test set cannot affect it.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train, X_test
    )

    overall_failure_rate = float(y.mean())
    train_failure_rate = float(y_train.mean())
    test_failure_rate = float(y_test.mean())
    rate_tolerance = 0.01  # One percentage point.
    if (
        abs(train_failure_rate - overall_failure_rate) > rate_tolerance
        or abs(test_failure_rate - overall_failure_rate) > rate_tolerance
    ):
        raise ValueError(
            "Failure rates differ from the full dataset by more than one percentage "
            f"point (full={overall_failure_rate:.2%}, "
            f"train={train_failure_rate:.2%}, test={test_failure_rate:.2%})."
        )

    if X_train_processed.isna().any().any() or X_test_processed.isna().any().any():
        raise ValueError("Preprocessing produced missing values.")

    if not X_train_processed.columns.equals(X_test_processed.columns):
        raise ValueError("Training and testing feature columns do not match.")

    excluded_columns = set(IDENTIFIER_COLUMNS + FAILURE_MODE_COLUMNS)
    if excluded_columns.intersection(X_train_processed.columns):
        raise ValueError("An identifier or failure-mode column was included in model inputs.")

    if len(X_train_processed) + len(X_test_processed) != len(data):
        raise ValueError("Training and test row counts do not match the input dataset.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train_processed.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test_processed.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_frame().to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_frame().to_csv(OUTPUT_DIR / "y_test.csv", index=False)
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")

    print(f"Training inputs: {X_train_processed.shape}")
    print(f"Testing inputs: {X_test_processed.shape}")
    print(f"Training targets: {y_train.shape}")
    print(f"Testing targets: {y_test.shape}")
    print(
        "Failure rates (full/train/test): "
        f"{overall_failure_rate:.2%}/{train_failure_rate:.2%}/{test_failure_rate:.2%}"
    )
    print(f"Prepared files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
