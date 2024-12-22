import logging
from pathlib import Path

from src.data.feature_processor import FeatureProcessor
from src.models.trainer import ModelTrainer
from src.utils.config import default_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main execution function."""

    model_dir = Path("models/training")
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_processor = FeatureProcessor(default_config)
    trainer = ModelTrainer(default_config, feature_processor)

    output_dir = trainer.train_model()

    logging.info(f"Binary classification results saved in: {output_dir}")


if __name__ == "__main__":
    main()
