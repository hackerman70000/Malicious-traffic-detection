#!/usr/bin/env python3

"""
Retrain existing models with new data.
"""

# TO DO: Refactor
from pathlib import Path

from src.data.preprocessor import DataPreprocessor
from src.models.retrainer import ModelRetrainer
from src.utils.constants import ATTACK_TYPES


def main():
    """Main execution function."""
    print("Loading preprocessed data...")

    preprocessor = DataPreprocessor(
        "data/raw/UNSW-NB15/Data.csv", "data/raw/UNSW-NB15/Label.csv"
    )

    X, y_binary, X_attacks, y_multiclass = preprocessor.prepare_datasets()

    # Paths to the existing trained models - update these to your actual model paths
    # The paths should point to the directories containing your trained models
    # For now those are hardcoded, but later you can pass them as arguments
    binary_model_path = Path("models/development/v1/binary/xgboost_20241210_155649_v1")
    multiclass_model_path = Path(
        "models/development/v1/multiclass/xgboost_20241210_155649_v1"
    )

    if not (binary_model_path / "model.json").exists():
        raise FileNotFoundError(
            f"Binary model not found at {binary_model_path / 'model.json'}"
        )
    if not (multiclass_model_path / "model.json").exists():
        raise FileNotFoundError(
            f"Multiclass model not found at {multiclass_model_path / 'model.json'}"
        )

    print("\nRetraining binary model...")
    binary_retrainer = ModelRetrainer(binary_model_path)
    binary_output_dir = binary_retrainer.retrain_model(
        X,
        y_binary,
        labels=["Benign", "Malicious"],
    )
    print(f"Binary retrained model saved in: {binary_output_dir}")

    print("\nRetraining multiclass model...")
    multiclass_retrainer = ModelRetrainer(multiclass_model_path)
    multiclass_output_dir = multiclass_retrainer.retrain_model(
        X_attacks,
        y_multiclass,
        labels=ATTACK_TYPES,
    )
    print(f"Multiclass retrained model saved in: {multiclass_output_dir}")


if __name__ == "__main__":
    main()
