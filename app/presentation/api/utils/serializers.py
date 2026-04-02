from dataclasses import asdict, is_dataclass
from typing import Any


def to_serializable(data: Any) -> Any:
    """
    Convert dataclass-based or standard Python objects into JSON-serializable data.
    """
    if is_dataclass(data):
        return asdict(data)

    if isinstance(data, dict):
        return {key: to_serializable(value) for key, value in data.items()}

    if isinstance(data, list):
        return [to_serializable(item) for item in data]

    return data