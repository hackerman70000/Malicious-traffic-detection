from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    """Configuration for data processing and paths."""

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    test_size: float = 0.2
    random_state: int = 42
    target_benign_ratio: float = 0.7
    chunk_size: int = 1000
    memory_limit: float = 0.75

    columns_to_drop: List[str] = field(
        default_factory=lambda: [
            "bidirectional_cwr_packets",
            "bidirectional_ece_packets",
            "bidirectional_urg_packets",
            "dst2src_cwr_packets",
            "dst2src_ece_packets",
            "src2dst_cwr_packets",
            "src2dst_ece_packets",
            "src2dst_urg_packets",
            "ip_version",
            "tunnel_id",
            "application_category_name",
            "application_confidence",
            "application_is_guessed",
            "application_name",
        ]
    )

    categorical_columns: List[str] = field(
        default_factory=lambda: [
            "application_category_name",
            "application_name",
            "dst_ip",
            "dst_mac",
            "dst_oui",
            "src_ip",
            "src_mac",
            "src_oui",
        ]
    )

    def __post_init__(self):
        """Convert string paths to Path objects and ensure directories exist."""
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        self.models_dir = Path(self.models_dir)

        for directory in [self.raw_data_dir, self.processed_data_dir, self.models_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""

    name: str = "xgboost_binary"
    model_type: str = "binary"
    framework: str = "xgboost"
    labels: List[str] = field(default_factory=lambda: ["Benign", "Malicious"])


@dataclass
class Config:
    """Main configuration class combining all config components."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        import json

        config_dict = {"data": self.data.__dict__, "model": self.model.__dict__}
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4, default=str)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        import json

        with open(path, "r") as f:
            config_dict = json.load(f)

        data_config = DataConfig(**config_dict["data"])
        model_config = ModelConfig(**config_dict["model"])
        return cls(data=data_config, model=model_config)


default_config = Config()
