import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

from src.utils.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_tree_visualization,
)


class ModelTrainer:
    """Handles model training, evaluation, and artifact management."""

    def __init__(self, config: "Config", feature_processor: "FeatureProcessor"):
        self.config = config
        self.feature_processor = feature_processor
        self.model: Optional[xgb.XGBClassifier] = None
        self.training_metadata: Dict = {}

    def _create_output_directory(self) -> Tuple[Path, Dict[str, Path]]:
        """Create timestamped output directory for model artifacts with subdirectories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            self.config.data.models_dir / "training" / f"xgboost_{timestamp}_v1"
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        subdirs = {
            "model": output_dir / "model",
            "metrics": output_dir / "metrics",
            "plots": output_dir / "plots",
            "plots/trees": output_dir / "plots/trees",
        }

        for dir_path in subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        return output_dir, subdirs

    def _prepare_training_data(
        self, data_path: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and prepare data for training."""
        logging.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        X, y = self.feature_processor.prepare_features(df, training=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=y,
        )

        return X_train, X_test, y_train, y_test

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> xgb.XGBClassifier:
        """Train the XGBoost model with early stopping."""
        logging.info("Training model...")

        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            n_estimators=1000,
            eval_metric=["error", "auc", "logloss"],
            early_stopping_rounds=10,
        )

        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

        if hasattr(self.model, "best_iteration"):
            logging.info(f"Best iteration: {self.model.best_iteration}")
            logging.info(f"Best score: {self.model.best_score}")

        return self.model

    def _save_artifacts(
        self,
        output_dir: Path,
        subdirs: Dict[str, Path],
        model: xgb.XGBClassifier,
        metrics: Dict,
        X_train: pd.DataFrame,
    ) -> None:
        """Save all model artifacts and metadata in organized subdirectories."""

        model.save_model(subdirs["model"] / "model.json")
        self.feature_processor.save_encoders(subdirs["model"])

        metadata = {
            "model_type": self.config.model.model_type,
            "framework_version": xgb.__version__,
            "training_date": datetime.now().isoformat(),
            "model_version": datetime.now().strftime("%Y.%m.%d"),
            "input_features": X_train.shape[1],
            "training_samples": X_train.shape[0],
            "performance_metrics": metrics["classification_report"],
            "model_parameters": self.model.get_params(),
            "feature_names": list(X_train.columns),
        }

        if hasattr(self.model, "best_iteration"):
            metadata["early_stopping"] = {
                "best_iteration": self.model.best_iteration,
                "best_score": self.model.best_score,
                "early_stopping_rounds": 10,
            }

        with open(subdirs["metrics"] / "metadata.json", "w") as f:
            import json

            json.dump(metadata, f, indent=4, default=str)

        with open(subdirs["metrics"] / "report.txt", "w") as f:
            f.write("XGBoost Binary Classification Results\n")
            f.write("=" * 35 + "\n\n")
            f.write(f"Dropped columns: {self.feature_processor.dropped_columns}\n\n")
            f.write(
                classification_report(
                    metrics["y_test"],
                    metrics["predictions"],
                    target_names=self.config.model.labels,
                )
            )

    def train_model(self) -> Path:
        """Train, evaluate, and save the model with all artifacts."""
        output_dir, subdirs = self._create_output_directory()

        data_path = self.config.data.processed_data_dir / "merged_data.csv"
        X_train, X_test, y_train, y_test = self._prepare_training_data(data_path)

        model = self._train_model(X_train, y_train, X_test, y_test)

        predictions = model.predict(X_test)
        metrics = {
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=self.config.model.labels,
                output_dict=True,
            ),
            "precision_score": precision_score(y_test, predictions, average="macro"),
            "y_test": y_test,
            "predictions": predictions,
        }

        logging.info(f"Model Precision: {metrics['precision_score']:.2%}")

        self._save_artifacts(output_dir, subdirs, model, metrics, X_train)

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            self.config.model.labels,
            "Binary Classification Confusion Matrix",
            subdirs["plots"] / "confusion_matrix.png",
        )

        plot_feature_importance(
            model,
            X_train.columns,
            subdirs["plots"] / "feature_importance.png",
            top_n=20,
        )

        plot_tree_visualization(model, subdirs["plots/trees"], num_trees=3)

        return output_dir
