#!/usr/bin/env python3

"""
Data preprocessing script for network traffic analysis.
Handles preprocessing of malicious and benign network traffic data.
"""

import argparse
import logging
import sys
from pathlib import Path

from mtd.data.preprocessor import MemoryEfficientPreprocessor
from mtd.utils.config import Config, default_config


def setup_logging(verbose: bool = False) -> None:
    """Configure logging settings."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess network traffic data for analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=Path, help="Path to custom configuration file")

    parser.add_argument(
        "--memory-limit",
        type=float,
        help="Memory usage limit (0-1)",
        default=default_config.data.processing.memory_limit,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Processing chunk size",
        default=default_config.data.processing.chunk_size,
    )

    parser.add_argument(
        "--target-ratio",
        type=float,
        help="Target ratio of benign traffic",
        default=default_config.data.processing.target_benign_ratio,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    """Load and update configuration based on arguments."""
    config = (
        Config.load(args.config)
        if args.config and args.config.exists()
        else default_config
    )

    config.data.processing.memory_limit = args.memory_limit
    config.data.processing.chunk_size = args.chunk_size
    config.data.processing.target_benign_ratio = args.target_ratio

    return config


def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    setup_logging(args.verbose)

    try:
        logging.info("Starting data preprocessing...")
        config = load_config(args)

        preprocessor = MemoryEfficientPreprocessor(config=config)

        logging.info("Processing files separately...")
        preprocessor.process_files_separately()

        logging.info("Merging files with target ratio...")
        preprocessor.merge_with_ratio()

        logging.info("Preprocessing completed successfully.")
        logging.info(f"Total rows processed: {preprocessor.stats.total_rows_processed}")
        logging.info(f"Output file: {preprocessor.final_output}")

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
