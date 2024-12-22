from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.config import ModelConfig
from src.utils.metadata import ModelMetadata
from src.utils.visualization import plot_confusion_matrix


class ModelTrainer:
    """Handles model training, evaluation, and result storage for binary classification."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.labels = ["Benign", "Malicious"]
        self.data_path = "data/processed/merged_data.csv"

        # Columns to drop (unique to either malicious or normal files)
        self.columns_to_drop = [
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

        self.categorical_columns = [
            "application_category_name",
            "application_name",
            "dst_ip",
            "dst_mac",
            "dst_oui",
            "src_ip",
            "src_mac",
            "src_oui",
        ]
        self.label_encoders = {}
        self.dropped_columns: List[str] = []  # Track actually dropped columns

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
                "categorical_features": self.categorical_columns,
                "dropped_columns": self.dropped_columns,
            },
        )

    def encode_categorical_features(
        self, df: pd.DataFrame, training: bool = True
    ) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        df = df.copy()

        for column in self.categorical_columns:
            if column in df.columns:
                if training:
                    # During training, fit new encoder
                    self.label_encoders[column] = LabelEncoder()
                    # Handle NaN values before encoding
                    df[column] = df[column].fillna("unknown")
                    df[column] = self.label_encoders[column].fit_transform(df[column])
                else:
                    # During inference, use existing encoder
                    df[column] = df[column].fillna("unknown")
                    known_values = set(self.label_encoders[column].classes_)
                    df[column] = df[column].map(
                        lambda x: "unknown" if x not in known_values else x
                    )
                    df[column] = self.label_encoders[column].transform(df[column])

        return df

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the dataset."""
        print("Loading data...")
        data = pd.read_csv(self.data_path)
        print(f"Initial data shape: {data.shape}")

        # Drop columns that are unique to either malicious or normal files
        print(f"Dropping {len(self.columns_to_drop)} columns...")
        self.dropped_columns = []  # Reset dropped columns list
        for col in self.columns_to_drop:
            if col in data.columns:
                data = data.drop(col, axis=1)
                self.dropped_columns.append(col)
        print(f"Dropped columns: {self.dropped_columns}")

        # Separate features and target
        X = data.drop("Label", axis=1)
        y = data["Label"]

        # Handle categorical features
        print("Encoding categorical features...")
        X = self.encode_categorical_features(X, training=True)

        # Convert all columns to float32 for better memory usage
        X = X.astype(np.float32)

        print(f"Final data shape: {X.shape}")
        print(f"Class distribution:\n{y.value_counts(normalize=True)}")

        return X, y

    def save_tree_visualization(
        self, model: xgb.XGBClassifier, output_dir: Path, num_trees: int = 3
    ):
        """Save visualization of the first few decision trees."""
        import matplotlib.pyplot as plt
        from xgboost import plot_tree

        print(f"\nSaving tree visualizations for first {num_trees} trees...")

        for i in range(min(num_trees, model.n_estimators)):
            plt.figure(figsize=(20, 10))
            plot_tree(model, num_trees=i)
            plt.title(f"Decision Tree {i}")
            plt.savefig(output_dir / f"tree_{i}.png", dpi=300, bbox_inches="tight")
            plt.close()

        # Also save feature importance plot
        plt.figure(figsize=(12, 6))
        feature_importance = pd.DataFrame(
            {
                "feature": model.feature_names_in_,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        plt.barh(
            feature_importance["feature"][:20], feature_importance["importance"][:20]
        )
        plt.title("Top 20 Feature Importance")
        plt.xlabel("F-score")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

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
            stratify=y,
        )

        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")

        # Initialize and train model
        print("Training model...")
        model = xgb.XGBClassifier(**self.config.model_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

        # Make predictions
        print("Making predictions...")
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

        # Save label encoders
        pd.to_pickle(self.label_encoders, output_dir / "label_encoders.pkl")

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
            f.write(f"Dropped columns: {self.dropped_columns}\n\n")
            f.write(report_text)

        self.save_tree_visualization(model, output_dir)

        # Create visualization directory for more detailed tree info
        viz_dir = output_dir / "tree_viz"
        viz_dir.mkdir(exist_ok=True)

        # Save detailed tree structure as text
        for i in range(min(3, model.n_estimators)):
            tree_dump = model.get_booster().get_dump(dump_format="text")[i]
            with open(viz_dir / f"tree_{i}_structure.txt", "w") as f:
                f.write(tree_dump)

        return output_dir
