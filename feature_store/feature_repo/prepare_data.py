
"""Prepare AI4I sensor data for the Feast feature store."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "feature_store"
    / "feature_repo"
    / "data"
    / "machine_features.parquet"
)

FEATURE_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def prepare_feature_data() -> None:
    """Create Feast input data with entity IDs and event timestamps."""

    data = pd.read_csv(INPUT_PATH)

    # Keep only the six model input features.
    features = data[FEATURE_COLUMNS].copy()

    # Generate a unique ID for every dataset record.
    features.insert(
        0,
        "machine_record_id",
        [f"record_{i:06d}" for i in range(1, len(features) + 1)],
    )

    # Generate demonstration timestamps at one-minute intervals.
    features["event_timestamp"] = pd.date_range(
        end=pd.Timestamp.now(tz="UTC"),
        periods=len(features),
        freq="min",
    )

    # Rename columns to match the Feast feature definitions.
    features = features.rename(
        columns={
            "Air temperature": "Air_temperature",
            "Process temperature": "Process_temperature",
            "Rotational speed": "Rotational_speed",
            "Tool wear": "Tool_wear",
        }
    )

    # Use data types matching the Feast schema.
    features["Type"] = features["Type"].astype(str)
    features["Air_temperature"] = features["Air_temperature"].astype("float32")
    features["Process_temperature"] = features["Process_temperature"].astype("float32")
    features["Rotational_speed"] = features["Rotational_speed"].astype("int64")
    features["Torque"] = features["Torque"].astype("float32")
    features["Tool_wear"] = features["Tool_wear"].astype("int64")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    print("Feast feature data created successfully.")
    print(f"Rows: {len(features)}")
    print(f"Columns: {features.columns.tolist()}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare_feature_data()
