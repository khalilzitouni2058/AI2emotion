"""Infrastructure utilities for persisting analysis results."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class FileStorage:
    """Handles writing analysis results to files."""

    @staticmethod
    def _to_serializable(data: Any) -> Any:
        """
        Convert dataclass-based or standard Python objects into JSON-serializable data.
        """
        if is_dataclass(data):
            return asdict(data)

        if isinstance(data, dict):
            return {key: FileStorage._to_serializable(value) for key, value in data.items()}

        if isinstance(data, list):
            return [FileStorage._to_serializable(item) for item in data]

        return data

    def save_json(self, data: Any, output_path: str) -> Path:
        """
        Save data to a JSON file.

        Args:
            data: Dataclass instance, dict, or list
            output_path: Destination file path

        Returns:
            Path to the saved file
        """
        serializable_data = self._to_serializable(data)
        path = Path(output_path)

        with path.open("w", encoding="utf-8") as file:
            json.dump(serializable_data, file, indent=2, ensure_ascii=False)

        return path