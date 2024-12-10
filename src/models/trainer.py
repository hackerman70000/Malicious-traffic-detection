from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

from src.utils.config import ModelConfig
from src.utils.constants import ATTACK_TYPES, ModelType
from src.utils.metadata import ModelMetadata
from src.utils.visualization import plot_confusion_matrix


class ModelTrainer:
    """Handles model training, evaluation, and result storage."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.binary_labels = ["Benign", "Malicious"]

    def create_output_directories(self) -> Tuple[Path, Path]:
        """Create timestamped output directories for model artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path("models/trained")

        binary_dir = base_path / ModelType.BINARY.value / f"xgboost_{timestamp}"
        multiclass_dir = base_path / ModelType.MULTICLASS.value / f"xgboost_{timestamp}"

        binary_dir.mkdir(parents=True, exist_ok=True)
        multiclass_dir.mkdir(parents=True, exist_ok=True)

        return binary_dir, multiclass_dir

    def create_model_metadata(
        self,
        model_type: ModelType,
        model: xgb.XGBClassifier,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        performance_metrics: Dict,
    ) -> ModelMetadata:
        """Create metadata for trained model."""
        return ModelMetadata(
            model_type=model_type.value,
            framework_version=xgb.__version__,
            training_date=datetime.now().isoformat(),
            model_version=datetime.now().strftime("%Y.%m.%d"),
            input_features=X_train.shape[1],
            training_samples=X_train.shape[0],
            test_samples=X_test.shape[0],
            performance_metrics=performance_metrics,
            model_parameters=model.get_params(),
            additional_info={
                "description": f"XGBoost {model_type.value} classifier for network traffic analysis",
                "feature_names": list(X_train.columns),
                "target_labels": ATTACK_TYPES
                if model_type == ModelType.MULTICLASS
                else self.binary_labels,
            },
        )

    def train_and_evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: Dict,
        output_dir: Path,
        labels: List[str],
        model_type: ModelType,
    ) -> None:
        """Train, evaluate, and save model results."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions = model.predict(X_test)

        cm = confusion_matrix(y_test, predictions)
        report = classification_report(
            y_test, predictions, target_names=labels, output_dict=True
        )
        report_text = classification_report(y_test, predictions, target_names=labels)

        precision = precision_score(y_test, predictions, average="macro")
        print(f"\n{model_type.value.title()} Model Precision: {precision:.2%}")

        metadata = self.create_model_metadata(
            model_type=model_type,
            model=model,
            X_train=X_train,
            X_test=X_test,
            performance_metrics=report,
        )
        metadata.save(output_dir)

        model.save_model(output_dir / "model.json")

        plot_confusion_matrix(
            cm,
            labels,
            f"{model_type.value.title()} Classification Confusion Matrix",
            output_dir / "confusion_matrix.png",
        )

        with open(output_dir / "report.txt", "w") as f:
            f.write(f"XGBoost {model_type.value.title()} Classification Results\n")
            f.write("=" * (len(model_type.value) + 31) + "\n\n")
            f.write(report_text)

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
            ModelType.BINARY,
        )

        self.train_and_evaluate_model(
            X_attacks,
            y_multiclass,
            self.config.multiclass_params,
            multiclass_dir,
            ATTACK_TYPES,
            ModelType.MULTICLASS,
        )

        return binary_dir, multiclass_dir
