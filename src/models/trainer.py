from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils.config import ModelConfig
from src.utils.visualization import plot_confusion_matrix


class ModelTrainer:
    """Handles model training, evaluation, and result storage."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.binary_labels = ["Benign", "Malicious"]
        self.multiclass_labels = [
            "Analysis",
            "Backdoor",
            "DoS",
            "Exploits",
            "Fuzzers",
            "Generic",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        ]

    def create_output_directories(self) -> Tuple[Path, Path]:
        """Create timestamped output directories for model artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path("models/trained")

        binary_dir = base_path / "binary" / f"xgboost_{timestamp}"
        multiclass_dir = base_path / "multiclass" / f"xgboost_{timestamp}"

        binary_dir.mkdir(parents=True, exist_ok=True)
        multiclass_dir.mkdir(parents=True, exist_ok=True)

        return binary_dir, multiclass_dir

    def train_and_evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: Dict,
        output_dir: Path,
        labels: List[str],
        model_type: str,
    ) -> None:
        """Train, evaluate, and save model results."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=labels)

        plot_confusion_matrix(
            cm,
            labels,
            f"{model_type} Classification Confusion Matrix",
            output_dir / "confusion_matrix.png",
        )

        model.save_model(output_dir / "model.json")

        with open(output_dir / "report.txt", "w") as f:
            f.write(f"XGBoost {model_type} Classification Results\n")
            f.write("=" * (len(model_type) + 31) + "\n\n")
            f.write(report)

    def train_all_models(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        X_attacks: pd.DataFrame,
        y_multiclass: pd.Series,
    ) -> Tuple[Path, Path]:
        """Train and evaluate both binary and multiclass models."""
        binary_dir, multiclass_dir = self.create_output_directories()

        self.train_and_evaluate_model(
            X,
            y_binary,
            self.config.binary_params,
            binary_dir,
            self.binary_labels,
            "Binary",
        )

        self.train_and_evaluate_model(
            X_attacks,
            y_multiclass,
            self.config.multiclass_params,
            multiclass_dir,
            self.multiclass_labels,
            "Multiclass",
        )

        return binary_dir, multiclass_dir
