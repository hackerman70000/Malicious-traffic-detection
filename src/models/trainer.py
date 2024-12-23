import json
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
        """Create timestamped output directory for model artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            self.config.data.models_dir
            / "training"
            / f"{self.config.model.name}_{timestamp}_v1"
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
        if X is None or y is None:
            raise ValueError("Feature processing failed to return valid X and y values")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.data.processing.test_size,
            random_state=self.config.data.processing.random_state,
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
        """Train the XGBoost model."""
        logging.info("Training model...")

        model_params = {
            "objective": self.config.model.objective,
            "n_estimators": self.config.model.n_estimators,
            "eval_metric": self.config.model.eval_metric,
            "early_stopping_rounds": self.config.model.early_stopping_rounds,
            "max_depth": self.config.model.max_depth,
            "learning_rate": self.config.model.learning_rate,
            "subsample": self.config.model.subsample,
            "colsample_bytree": self.config.model.colsample_bytree,
            "min_child_weight": self.config.model.min_child_weight,
            "gamma": self.config.model.gamma,
            "reg_alpha": self.config.model.reg_alpha,
            "reg_lambda": self.config.model.reg_lambda,
        }

        model_params = {k: v for k, v in model_params.items() if v != "default"}

        logging.info("Model parameters:")
        for param, value in model_params.items():
            logging.info(f"  {param}: {value}")

        self.model = xgb.XGBClassifier(**model_params)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=self.config.model.verbose,
        )

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
        """Save all model artifacts and metadata."""
        model.save_model(subdirs["model"] / "model.json")
        self.feature_processor.save_encoders(subdirs["model"])

        model_params = model.get_params()

        configured_params = {
            "objective": self.config.model.objective,
            "n_estimators": self.config.model.n_estimators,
            "max_depth": self.config.model.max_depth,
            "learning_rate": self.config.model.learning_rate,
            "subsample": self.config.model.subsample,
            "colsample_bytree": self.config.model.colsample_bytree,
            "min_child_weight": self.config.model.min_child_weight,
            "gamma": self.config.model.gamma,
            "reg_alpha": self.config.model.reg_alpha,
            "reg_lambda": self.config.model.reg_lambda,
            "early_stopping_rounds": self.config.model.early_stopping_rounds,
        }

        metadata = {
            "model_type": self.config.model.model_type,
            "framework_version": xgb.__version__,
            "training_date": datetime.now().isoformat(),
            "model_version": datetime.now().strftime("%Y.%m.%d"),
            "input_features": X_train.shape[1],
            "training_samples": X_train.shape[0],
            "performance_metrics": metrics["classification_report"],
            "configured_parameters": configured_params,
            "actual_parameters": model_params,
            "feature_names": list(X_train.columns),
        }

        if hasattr(model, "best_iteration"):
            metadata["early_stopping"] = {
                "best_iteration": model.best_iteration,
                "best_score": model.best_score,
            }

        with open(subdirs["metrics"] / "metadata.json", "w") as f:
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

    def train_model(self, data_path: Optional[Path] = None) -> Path:
        """Public method to train, evaluate, and save the model with all artifacts."""
        output_dir, subdirs = self._create_output_directory()

        data_path = data_path or (
            self.config.data.processed_data_dir / "merged_data.csv"
        )
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
