import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureProcessor:
    """Handles all feature engineering and preprocessing tasks."""

    def __init__(self, config: "Config"):
        self.config = config
        self._dropped_columns: List[str] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}

        import psutil

        if psutil.virtual_memory().percent > (
            self.config.data.processing.memory_limit * 100
        ):
            logging.warning(
                f"System memory usage above {self.config.data.processing.memory_limit * 100}%"
            )

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
            if label_distribution.min() < self.config.data.processing.min_class_ratio:
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

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop specified columns from the dataset."""
        self._dropped_columns = []
        for col in self.config.data.columns_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
                self._dropped_columns.append(col)
        return df

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency using pandas built-in optimizations."""
        return df.convert_dtypes()

    def save_encoders(self, path: Path) -> None:
        """Save label encoders to file with proper error handling."""
        try:
            encoders_path = (
                path / f"label_encoders{self.config.data.files.pickle_extension}"
            )
            if len(self.label_encoders) > 0:
                pd.to_pickle(self.label_encoders, encoders_path)
            else:
                logging.info(
                    "No encoders to save - all categorical columns were dropped."
                )
        except PermissionError as e:
            logging.error(f"Permission denied saving encoder file at {path}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error saving encoders to {path}: {e}")
            raise

    def load_encoders(self, path: Path) -> None:
        """Load label encoders from file with proper error handling."""
        try:
            encoders_path = (
                path / f"label_encoders{self.config.data.files.pickle_extension}"
            )
            if encoders_path.exists():
                self.label_encoders = pd.read_pickle(encoders_path)
            else:
                logging.info(
                    "No encoder file found - proceeding without loading encoders."
                )
                self.label_encoders = {}
        except FileNotFoundError:
            logging.info("No encoder file found - proceeding without loading encoders.")
            self.label_encoders = {}
        except PermissionError as e:
            logging.error(f"Permission denied accessing encoder file at {path}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading encoders from {path}: {e}")
            raise

    @property
    def dropped_columns(self) -> List[str]:
        """Get list of columns that were dropped during processing."""
        return self._dropped_columns

    def prepare_features(
        self, df: pd.DataFrame, training: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Main method to prepare features for training or inference."""
        logging.info("Starting feature preparation...")

        self.validate_data(df, training)

        df = df.copy()

        if training:
            columns_to_drop = set(self.config.data.columns_to_drop)
            available_columns = set(df.columns)
            columns_to_actually_drop = columns_to_drop.intersection(available_columns)
            if not columns_to_actually_drop:
                logging.warning(
                    "No columns from config.data.columns_to_drop found in the dataset"
                )

        df = self._handle_missing_values(df)

        y = None
        if "Label" in df.columns:
            if not training:
                logging.warning(
                    "Label column found in inference data - it will be ignored"
                )
            y = df.pop("Label")
        else:
            if training:
                raise ValueError("Label column required for training")

        X = self.drop_columns(df)
        X = self.optimize_dtypes(X)

        logging.info(f"Feature preparation completed. Final shape: {X.shape}")
        if y is not None:
            logging.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

        return X, y
