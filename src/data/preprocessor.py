from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """Handles data loading and preprocessing operations."""

    def __init__(self, raw_data_path: str, raw_label_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.raw_label_path = Path(raw_label_path)
        self.processed_dir = Path("data/processed/UNSW-NB15")
        self.processed_path = self.processed_dir / "processed_data.csv"
        self.label_encoder_dict = {}

    def load_raw_data(self) -> pd.DataFrame:
        """Load and merge raw data and label files."""
        data = pd.read_csv(self.raw_data_path)
        labels = pd.read_csv(self.raw_label_path)
        return pd.merge(data, labels, left_index=True, right_index=True)

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoder_dict[col] = le

        return df

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """Save processed data to CSV file."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        print(f"Processed data saved to: {self.processed_path}")

    def prepare_datasets(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare binary and multiclass datasets."""

        if self.processed_path.exists():
            print("Loading preprocessed data...")
            df = pd.read_csv(self.processed_path)
        else:
            print("Processing raw data...")
            df = self.load_raw_data()
            df = self.encode_categorical_features(df)
            self.save_processed_data(df)

        X = df.drop("Label", axis=1)
        y = df["Label"]

        y_binary = (y > 0).astype(int)

        attack_mask = y > 0
        X_attacks = X[attack_mask]
        y_multiclass = y[attack_mask] - 1

        return X, y_binary, X_attacks, y_multiclass
