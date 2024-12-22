from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

from src.utils.config import ModelConfig
from src.utils.metadata import ModelMetadata
from src.utils.visualization import plot_confusion_matrix


class ModelTrainer:
    """Handles model training, evaluation, and result storage for binary classification."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.labels = ["Benign", "Malicious"]
        self.data_path = "data/processed/merged_data.csv"

    def create_output_directory(self) -> Path:
        """Create timestamped output directory for model artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("models/training/binary") / f"xgboost_{timestamp}_v1"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def create_model_metadata(
        self,
        model: xgb.XGBClassifier,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        performance_metrics: Dict,
    ) -> ModelMetadata:
        """Create metadata for trained model."""
        return ModelMetadata(
            model_type="binary",
            framework_version=xgb.__version__,
            training_date=datetime.now().isoformat(),
            model_version=datetime.now().strftime("%Y.%m.%d"),
            input_features=X_train.shape[1],
            training_samples=X_train.shape[0],
            test_samples=X_test.shape[0],
            performance_metrics=performance_metrics,
            model_parameters=model.get_params(),
            additional_info={
                "description": "XGBoost binary classifier for network traffic analysis",
                "feature_names": list(X_train.columns),
                "target_labels": self.labels,
            },
        )

    def prepare_data(self):
        """Load and prepare the dataset."""
        # Load data
        data = pd.read_csv(self.data_path)

        # Separate features and target
        X = data.drop("Label", axis=1)
        y = data["Label"]

        return X, y

    def train_model(self) -> Path:
        """Train, evaluate, and save model results."""
        output_dir = self.create_output_directory()

        # Prepare data
        X, y = self.prepare_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,  # Ensure balanced split
        )

        # Initialize and train model
        model = xgb.XGBClassifier(**self.config.binary_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(
            y_test, predictions, target_names=self.labels, output_dict=True
        )
        report_text = classification_report(
            y_test, predictions, target_names=self.labels
        )

        precision = precision_score(y_test, predictions, average="macro")
        print(f"\nBinary Model Precision: {precision:.2%}")

        # Create and save metadata
        metadata = self.create_model_metadata(
            model=model,
            X_train=X_train,
            X_test=X_test,
            performance_metrics=report,
        )
        metadata.save(output_dir)

        # Save model
        model.save_model(output_dir / "model.json")

        # Save confusion matrix plot
        plot_confusion_matrix(
            cm,
            self.labels,
            "Binary Classification Confusion Matrix",
            output_dir / "confusion_matrix.png",
        )

        # Save classification report
        with open(output_dir / "report.txt", "w") as f:
            f.write("XGBoost Binary Classification Results\n")
            f.write("=" * 35 + "\n\n")
            f.write(report_text)

        return output_dir
