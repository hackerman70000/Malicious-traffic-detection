from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class FileConfig:
    """Configuration for file operations and formats."""

    pickle_extension: str = ".pkl"
    supported_file_types: List[str] = field(default_factory=lambda: [".csv"])
    encoding: str = "utf-8"


@dataclass
class ProcessingConfig:
    """Configuration for data processing parameters."""

    unknown_category: str = "unknown"
    test_size: float = 0.2
    random_state: int = 42
    target_benign_ratio: float = 0.7
    min_class_ratio: float = 0.1
    chunk_size: int = 1000
    memory_limit: float = 0.75


@dataclass
class DataConfig:
    """Configuration for data processing and paths."""

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    files: FileConfig = field(default_factory=FileConfig)

    columns_to_drop: List[str] = field(
        default_factory=lambda: [
            "application_category_name",
            "application_name",
            "dst_ip",
            "dst_mac",
            "dst_oui",
            "src_ip",
            "src_mac",
            "src_oui",
            "id",
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
            "application_confidence",
            "application_is_guessed",
        ]
    )

    def __post_init__(self):
        """Convert string paths to Path objects and ensure directories exist."""
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        self.models_dir = Path(self.models_dir)

        for directory in [self.raw_data_dir, self.processed_data_dir, self.models_dir]:
            if not directory.resolve().is_relative_to(Path.cwd()):
                raise ValueError(
                    f"Path {directory} must be relative to current directory"
                )
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

        config_dict = {
            "data": {
                **self.data.__dict__,
                "processing": self.data.processing.__dict__,
                "files": self.data.files.__dict__,
            },
            "model": self.model.__dict__,
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4, default=str)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        import json

        with open(path, "r") as f:
            config_dict = json.load(f)

        processing_config = ProcessingConfig(
            **config_dict["data"].pop("processing", {})
        )
        files_config = FileConfig(**config_dict["data"].pop("files", {}))

        data_config = DataConfig(
            **config_dict["data"], processing=processing_config, files=files_config
        )
        model_config = ModelConfig(**config_dict["model"])

        return cls(data=data_config, model=model_config)


default_config = Config()
