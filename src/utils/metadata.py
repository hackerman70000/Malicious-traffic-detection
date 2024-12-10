import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ModelMetadata:
    """Metadata for trained models."""

    model_type: str
    framework_version: str
    training_date: str
    model_version: str
    input_features: int
    training_samples: int
    test_samples: int
    performance_metrics: Dict
    model_parameters: Dict
    additional_info: Optional[Dict] = None

    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: Path) -> "ModelMetadata":
        """Load metadata from JSON file."""
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return cls(**data)
