import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

from mtd.utils.visualization import plot_confusion_matrix


class ModelRetrainer:
    """Handles retraining of an existing model with labeled CSV data."""

    def __init__(self, model_path: Path):
        """Load existing model and metadata from the given path."""
        self.model_path = model_path
        self.model = xgb.XGBClassifier()
        model_file = model_path / "model" / "model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {model_file}")
        self.model.load_model(model_file)

        metadata_path = model_path / "metrics" / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        if "feature_names" not in self.metadata:
            raise ValueError("Metadata missing required 'feature_names' field")

        self.feature_names = self.metadata["feature_names"]

    def get_current_version(self, dirname: str) -> int:
        """Extract the current version number from directory name."""
        match = re.search(r"_v(\d+)$", dirname)
        if match:
            return int(match.group(1))
        return 0

    def create_output_directory(self) -> Tuple[Path, str]:
        """Create versioned output directory for retrained model."""
        base_dir = self.model_path.parent.parent
        model_type = self.model_path.parent.name

        current_dir = self.model_path.name
        base_name = re.sub(r"_v\d+$", "", current_dir)

        current_version = self.get_current_version(current_dir)
        next_version = current_version + 1
        version_str = f"v{next_version}"

        output_dir = base_dir / model_type / f"{base_name}_{version_str}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating output directory for retrained model: {output_dir}")
        return output_dir, version_str

    def prepare_data(self, input_path: Union[Path, List[Path]]) -> pd.DataFrame:
        """Prepare data from CSV files for retraining."""
        if isinstance(input_path, list):
            dfs = []
            for path in input_path:
                df = pd.read_csv(path)

                if "Label" not in df.columns:
                    raise ValueError(f"Required 'Label' column not found in {path}")

                invalid_labels = set(df["Label"].unique()) - {0, 1}
                if invalid_labels:
                    raise ValueError(
                        f"Invalid labels found in {path}: {invalid_labels}. "
                        "Labels must be binary (0 or 1)"
                    )
                dfs.append(df)
            final_df = pd.concat(dfs, ignore_index=True)
        else:
            final_df = pd.read_csv(input_path)
            if "Label" not in final_df.columns:
                raise ValueError(f"Required 'Label' column not found in {input_path}")

            invalid_labels = set(final_df["Label"].unique()) - {0, 1}
            if invalid_labels:
                raise ValueError(
                    f"Invalid labels found in {input_path}: {invalid_labels}. "
                    "Labels must be binary (0 or 1)"
                )

        missing_cols = set(self.feature_names) - set(final_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features in input data: {missing_cols}")

        return final_df[self.feature_names + ["Label"]]

    def retrain_model(
        self,
        input_paths: Union[Path, List[Path]],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Path:
        """Retrain an existing model with new data and save all artifacts."""
        data = self.prepare_data(input_paths)

        unique_labels = data["Label"].unique()
        if len(unique_labels) < 2:
            raise ValueError(
                f"Training data contains only one class ({unique_labels[0]}). "
                "Both benign (0) and malicious (1) samples are required for training."
            )

        y = data.pop("Label")
        X = data

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training data distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Testing data distribution:\n{y_test.value_counts(normalize=True)}")

        print(f"\nRetraining model: {self.model_path}")
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions = self.model.predict(X_test)
        labels = ["Benign", "Malicious"]

        cm = confusion_matrix(y_test, predictions)
        report = classification_report(
            y_test,
            predictions,
            target_names=labels,
            output_dict=True,
        )
        report_text = classification_report(
            y_test,
            predictions,
            target_names=labels,
        )

        precision = precision_score(y_test, predictions, average="binary")
        print(f"\nPrecision Score: {precision:.2%}")

        output_dir, version = self.create_output_directory()

        model_dir = output_dir / "model"
        metrics_dir = output_dir / "metrics"
        plots_dir = output_dir / "plots"
        trees_dir = plots_dir / "trees"

        for dir_path in [model_dir, metrics_dir, plots_dir, trees_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.model.save_model(model_dir / "model.json")

        self.metadata.update(
            {
                "retrained_date": datetime.now().isoformat(),
                "precision": precision,
                "retrained_model_path": str(output_dir / "model.json"),
                "version": version,
                "feature_names": self.feature_names,
                "retraining_samples": len(X),
                "test_size": test_size,
                "random_state": random_state,
                "class_distribution": {
                    "training": y_train.value_counts().to_dict(),
                    "testing": y_test.value_counts().to_dict(),
                },
            }
        )

        with open(metrics_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

        with open(metrics_dir / "report.txt", "w") as f:
            f.write("Retrained Model Classification Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(report_text)

        plot_confusion_matrix(
            cm,
            labels,
            f"Confusion Matrix ({self.model_path.parent.name})",
            output_dir / "plots" / "confusion_matrix.png",
        )

        print(f"Retrained model saved at: {output_dir}")
        return output_dir
