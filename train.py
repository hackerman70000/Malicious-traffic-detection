from pathlib import Path

from configs.default import default_config
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer


def main():
    """Main execution function."""

    data_dirs = [
        Path("data/raw/UNSW-NB15"),
        Path("data/processed/UNSW-NB15"),
        Path("models/training/binary"),
        Path("models/training/multiclass"),
    ]
    for dir_path in data_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    preprocessor = DataPreprocessor(
        "data/raw/UNSW-NB15/Data.csv", "data/raw/UNSW-NB15/Label.csv"
    )

    X, y_binary, X_attacks, y_multiclass = preprocessor.prepare_datasets()

    trainer = ModelTrainer(default_config)
    binary_dir, multiclass_dir = trainer.train_all_models(
        X, y_binary, X_attacks, y_multiclass
    )

    print(f"Binary classification results saved in: {binary_dir}")
    print(f"Multiclass classification results saved in: {multiclass_dir}")


if __name__ == "__main__":
    main()
