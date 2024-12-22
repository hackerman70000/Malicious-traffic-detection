from pathlib import Path

from configs.default import default_config
from src.models.trainer import ModelTrainer


def main():
    """Main execution function."""
    # Create necessary directories
    model_dir = Path("models/training/binary")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer with config
    trainer = ModelTrainer(default_config)

    # Train binary classification model
    output_dir = trainer.train_model()

    print(f"Binary classification results saved in: {output_dir}")


if __name__ == "__main__":
    main()
