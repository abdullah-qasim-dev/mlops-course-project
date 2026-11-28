
from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

from entities import machine_record


machine_source = FileSource(
    name="machine_sensor_source",
    path="data/machine_features.parquet",
    timestamp_field="event_timestamp",
)

machine_features = FeatureView(
    name="machine_features",
    entities=[machine_record],
    schema=[
        Field(name="Type", dtype=String),
        Field(name="Air_temperature", dtype=Float32),
        Field(name="Process_temperature", dtype=Float32),
        Field(name="Rotational_speed", dtype=Int64),
        Field(name="Torque", dtype=Float32),
        Field(name="Tool_wear", dtype=Int64),
    ],
    source=machine_source,
    ttl=timedelta(days=365),
    description="Machine sensor measurements used for failure prediction.",
)
