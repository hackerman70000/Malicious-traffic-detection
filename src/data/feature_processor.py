import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureProcessor:
    """Handles all feature engineering and preprocessing tasks."""

    def __init__(self, config: "Config"):
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._dropped_columns: List[str] = []

    def validate_data(self, df: pd.DataFrame, training: bool = True) -> None:
        """Validate the input data structure and content."""
        if training and "Label" not in df.columns:
            raise ValueError(
                "Label column not found in training data. "
                "The dataset must contain a 'Label' column for training."
            )

        if training:
            unique_labels = df["Label"].unique()
            if not set(unique_labels).issubset({0, 1}):
                raise ValueError(
                    f"Invalid label values found: {unique_labels}. "
                    "Labels must be binary (0 or 1)."
                )

            label_distribution = df["Label"].value_counts(normalize=True)
            if label_distribution.min() < 0.1:
                logging.warning(
                    f"Severe class imbalance detected. Class distribution:\n{label_distribution}"
                )

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values in the dataset."""
        initial_rows = len(df)
        initial_nulls = df.isnull().sum().sum()

        if initial_nulls > 0:
            logging.warning(
                f"Found {initial_nulls} missing values in {initial_rows} rows"
            )

            null_columns = df.columns[df.isnull().any()].tolist()
            for col in null_columns:
                null_count = df[col].isnull().sum()
                logging.info(f"Column '{col}' has {null_count} missing values")

            df_cleaned = df.dropna()

            dropped_rows = initial_rows - len(df_cleaned)
            logging.warning(
                f"Dropped {dropped_rows} rows ({dropped_rows/initial_rows:.2%} of data) "
                f"containing missing values"
            )
            logging.info(f"Remaining rows: {len(df_cleaned)}")

            remaining_nulls = df_cleaned.isnull().sum().sum()
            if remaining_nulls > 0:
                raise ValueError(
                    f"Still found {remaining_nulls} missing values after dropping rows"
                )

            return df_cleaned

        return df

    def prepare_features(
        self, df: pd.DataFrame, training: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Main method to prepare features for training or inference."""
        logging.info("Starting feature preparation...")

        self.validate_data(df, training)

        df = df.copy()

        df = self._handle_missing_values(df)

        y = None
        if "Label" in df.columns:
            if not training:
                logging.warning(
                    "Label column found in inference data - it will be ignored"
                )
            y = df["Label"]
            X = df.drop("Label", axis=1)
        else:
            if training:
                raise ValueError("Label column required for training")
            X = df.copy()

        X = self.drop_columns(X)
        X = self.encode_categorical_features(X, training=training)
        X = self.optimize_dtypes(X)

        logging.info(f"Feature preparation completed. Final shape: {X.shape}")
        if y is not None:
            logging.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

        return X, y

    def encode_categorical_features(
        self, df: pd.DataFrame, training: bool = True
    ) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        df = df.copy()

        for column in self.config.data.categorical_columns:
            if column in df.columns:
                if training:
                    self.label_encoders[column] = LabelEncoder()
                    df[column] = df[column].fillna("unknown")
                    df[column] = self.label_encoders[column].fit_transform(df[column])
                else:
                    if column not in self.label_encoders:
                        logging.warning(f"No encoder found for {column}, skipping...")
                        continue

                    df[column] = df[column].fillna("unknown")

                    df[column] = df[column].map(
                        lambda x: "unknown"
                        if x not in self.label_encoders[column].classes_
                        else x
                    )
                    df[column] = self.label_encoders[column].transform(df[column])

        return df

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop specified columns from the dataset."""
        self._dropped_columns = []
        for col in self.config.data.columns_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
                self._dropped_columns.append(col)
        return df

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        df = df.copy()

        float_cols = df.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            df[col] = df[col].astype(np.float32)

        int_cols = df.select_dtypes(include=["int64"]).columns
        for col in int_cols:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                else:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)

        return df

    @property
    def dropped_columns(self) -> List[str]:
        """Get list of columns that were dropped during processing."""
        return self._dropped_columns

    def save_encoders(self, path: Path) -> None:
        """Save label encoders to file."""
        pd.to_pickle(self.label_encoders, path / "label_encoders.pkl")

    def load_encoders(self, path: Path) -> None:
        """Load label encoders from file."""
        self.label_encoders = pd.read_pickle(path / "label_encoders.pkl")
