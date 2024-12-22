import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import xgboost as xgb
from nfstream import NFStreamer
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

from src.utils.visualization import plot_confusion_matrix


class ModelRetrainer:
    """Handles retraining of an existing model with new PCAP or CSV data."""

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

    def process_pcap(self, pcap_file: Path, label: int) -> pd.DataFrame:
        """Process PCAP file using NFStream and extract relevant features."""
        print(f"Processing PCAP file: {pcap_file}")

        streamer = NFStreamer(source=str(pcap_file), statistical_analysis=True)
        df = streamer.to_pandas()

        df["Label"] = label

        available_features = set(df.columns) & set(self.feature_names)
        missing_features = set(self.feature_names) - available_features

        if missing_features:
            raise ValueError(
                f"Missing required features in PCAP data: {missing_features}\n"
                f"These features were used in original model training but are not present in the new data."
            )

        return df[self.feature_names]

    def prepare_data(
        self, input_path: Union[Path, List[Path]], label: Optional[int] = None
    ) -> pd.DataFrame:
        """Prepare data from PCAP or CSV files for retraining."""
        if isinstance(input_path, list):
            dfs = []
            for path in input_path:
                if path.suffix == ".pcap":
                    if label is None:
                        raise ValueError("Label must be provided for PCAP files")
                    df = self.process_pcap(path, label)
                else:
                    df = pd.read_csv(path)
                dfs.append(df)
            final_df = pd.concat(dfs, ignore_index=True)
        else:
            if input_path.suffix == ".pcap":
                if label is None:
                    raise ValueError("Label must be provided for PCAP files")
                final_df = self.process_pcap(input_path, label)
            else:
                final_df = pd.read_csv(input_path)

        missing_cols = set(self.feature_names) - set(final_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features in input data: {missing_cols}")

        return final_df[self.feature_names]

    def retrain_model(
        self,
        input_paths: Union[Path, List[Path]],
        label: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Path:
        """Retrain an existing model with new data and save all artifacts."""
        X = self.prepare_data(input_paths, label)

        if "Label" in X.columns:
            y = X.pop("Label")
        else:
            if label is None:
                raise ValueError("No labels found in data and no label provided")
            y = pd.Series([label] * len(X))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Retraining model: {self.model_path}")
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions = self.model.predict(X_test)
        unique_classes = sorted(set(y_test) | set(predictions))
        labels = (
            ["Benign", "Malicious"]
            if len(unique_classes) > 1
            else ["Benign" if 0 in unique_classes else "Malicious"]
        )

        cm = confusion_matrix(
            y_test, predictions, labels=[0] if 0 in unique_classes else [1]
        )

        report = classification_report(
            y_test,
            predictions,
            target_names=labels,
            labels=[0] if 0 in unique_classes else [1],
            output_dict=True,
        )
        report_text = classification_report(
            y_test,
            predictions,
            target_names=labels,
            labels=[0] if 0 in unique_classes else [1],
        )

        precision = precision_score(
            y_test, predictions, average="binary", zero_division=1
        )
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
            output_dir / "confusion_matrix.png",
        )

        print(f"Retrained model saved at: {output_dir}")
        return output_dir
