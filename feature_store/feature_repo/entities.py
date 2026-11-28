
from feast import Entity
from feast.value_type import ValueType

machine_record = Entity(
    name="machine_record_id",
    join_keys=["machine_record_id"],
    value_type=ValueType.STRING,
    description="Unique identifier for each machine sensor record.",
)
