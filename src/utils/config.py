from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelConfig:
    """Configuration settings for XGBoost models."""

    test_size: float = 0.2
    random_state: int = 42
    binary_params: Dict = field(default_factory=lambda: {})
    multiclass_params: Dict = field(default_factory=lambda: {})

    def __post_init__(self):
        self.binary_params = {"random_state": self.random_state, **self.binary_params}
        self.multiclass_params = {
            "random_state": self.random_state,
            "objective": "multi:softmax",
            "num_class": 9,
            **self.multiclass_params,
        }
