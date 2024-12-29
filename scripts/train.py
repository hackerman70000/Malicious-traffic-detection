#!/usr/bin/env python3

"""
Train a new model for malicious traffic detection.
"""

import argparse
import logging
from pathlib import Path

from mtd.data.feature_processor import FeatureProcessor
from mtd.models.trainer import ModelTrainer
from mtd.utils.config import Config, default_config


def setup_logging() -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a new model for malicious traffic detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", type=Path, help="Path to input CSV file with training data"
    )

    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=default_config.data.raw_data_dir,
        help="Directory containing raw data files",
    )

    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=default_config.data.processed_data_dir,
        help="Directory for processed data files",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_config.data.models_dir,
        help="Directory for saving model artifacts",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=default_config.data.processing.test_size,
        help="Proportion of data to use for testing",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=default_config.data.processing.random_state,
        help="Random state for reproducibility",
    )

    parser.add_argument(
        "--target-benign-ratio",
        type=float,
        default=default_config.data.processing.target_benign_ratio,
        help="Target ratio of benign traffic in the dataset",
    )

    parser.add_argument(
        "--min-class-ratio",
        type=float,
        default=default_config.data.processing.min_class_ratio,
        help="Minimum acceptable ratio for any class",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=default_config.model.name,
        help="Name of the model",
    )

    return parser.parse_args()


def validate_input_file(file_path: Path) -> None:
    """Validate the input file exists and is a CSV."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV file, got: {file_path}")


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Update configuration with command line arguments."""

    config.data.raw_data_dir = args.raw_data_dir
    config.data.processed_data_dir = args.processed_data_dir
    config.data.models_dir = args.models_dir

    config.data.processing.test_size = args.test_size
    config.data.processing.random_state = args.random_state
    config.data.processing.target_benign_ratio = args.target_benign_ratio
    config.data.processing.min_class_ratio = args.min_class_ratio

    config.model.name = args.model_name

    return config


def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    setup_logging()

    try:
        logging.info("Starting model training process...")

        logging.info("Using default configuration with CLI overrides")
        config = update_config_from_args(default_config, args)

        if args.input:
            validate_input_file(args.input)
            data_path = args.input
            logging.info(f"Using specified input file: {data_path}")
        else:
            data_path = config.data.processed_data_dir / "merged_data.csv"
            validate_input_file(data_path)
            logging.info(f"Using default input file: {data_path}")

        model_dir = config.data.models_dir / "training"
        model_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Model artifacts will be saved to: {model_dir}")

        logging.info("Initializing feature processor and model trainer")
        feature_processor = FeatureProcessor(config)
        trainer = ModelTrainer(config, feature_processor)

        logging.info("Starting model training...")
        output_dir = trainer.train_model(data_path)

        logging.info("Training completed successfully")
        logging.info(f"Model and artifacts saved in: {output_dir}")

    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
