from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelConfig:
    """Configuration settings for XGBoost binary classifier."""

    test_size: float = 0.2
    random_state: int = 42
    model_params: Dict = field(
        default_factory=lambda: {
            "random_state": 42,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "objective": "binary:logistic",
            "tree_method": "hist",  # For faster training
            "enable_categorical": True,  # Enable categorical feature support
            "n_jobs": -1,  # Use all CPU cores
            "scale_pos_weight": 1,  # Will be adjusted if classes are imbalanced
            "subsample": 0.8,  # Prevent overfitting
            "colsample_bytree": 0.8,  # Prevent overfitting
            "min_child_weight": 1,
            "max_bin": 256,  # For memory efficiency
        }
    )

    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {"random_state": self.random_state}
        self.model_params["random_state"] = self.random_state
