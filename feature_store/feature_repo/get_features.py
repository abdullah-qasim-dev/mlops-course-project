
"""Retrieve historical sensor features from Feast."""

from pathlib import Path

import pandas as pd
from feast import FeatureStore


REPO_PATH = Path(__file__).resolve().parent
DATA_PATH = REPO_PATH / "data" / "machine_features.parquet"
OUTPUT_PATH = REPO_PATH / "data" / "historical_features.parquet"


def retrieve_historical_features():
    # Load the original entity IDs and timestamps.
    data = pd.read_parquet(DATA_PATH)

    entity_df = data[["machine_record_id", "event_timestamp"]].copy()

    # Initialize Feast.
    store = FeatureStore(repo_path=str(REPO_PATH))

    # Retrieve all six model input features.
    features = [
        "machine_features:Type",
        "machine_features:Air_temperature",
        "machine_features:Process_temperature",
        "machine_features:Rotational_speed",
        "machine_features:Torque",
        "machine_features:Tool_wear",
    ]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=features,
    ).to_df()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(OUTPUT_PATH, index=False)

    print("Historical features retrieved successfully!")
    print("Shape:", training_df.shape)
    print("\nFirst 5 rows:")
    print(training_df.head().to_string())
    print("\nMissing values:")
    print(training_df.isnull().sum())
    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    retrieve_historical_features()
