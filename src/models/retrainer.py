import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

from src.utils.visualization import plot_confusion_matrix


class ModelRetrainer:
    """Handles retraining of an existing model."""

    def __init__(self, model_path: Path):
        """Load existing model and metadata from the given path."""
        self.model_path = model_path
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path / "model.json")

        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

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

    def retrain_model(
        self, X: pd.DataFrame, y: pd.Series, labels: Optional[List[str]] = None
    ) -> Path:
        """Retrain an existing model and save all artifacts."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Retraining model: {self.model_path}")
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions = self.model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)

        if labels is None:
            labels = ["Class " + str(i) for i in range(len(set(y)))]

        report = classification_report(
            y_test, predictions, target_names=labels, output_dict=True
        )
        report_text = classification_report(y_test, predictions, target_names=labels)

        precision = precision_score(y_test, predictions, average="macro")
        print(f"\nPrecision Score: {precision:.2%}")

        output_dir, version = self.create_output_directory()
        self.model.save_model(output_dir / "model.json")

        with open(output_dir / "report.txt", "w") as f:
            f.write("Retrained Model Classification Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(report_text)

        plot_confusion_matrix(
            cm,
            labels,
            f"Confusion Matrix ({self.model_path.parent.name})",
            output_dir / "confusion_matrix.png",
        )

        self.metadata.update(
            {
                "retrained_date": datetime.now().isoformat(),
                "precision": precision,
                "retrained_model_path": str(output_dir / "model.json"),
                "version": version,
            }
        )
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"Retrained model saved at: {output_dir}")
        return output_dir
