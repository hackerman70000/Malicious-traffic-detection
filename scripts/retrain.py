#!/usr/bin/env python3

"""
Retrain existing models with new PCAP or CSV data.
"""

import argparse
from pathlib import Path
from typing import List

from src.models.retrainer import ModelRetrainer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrain an existing model with new data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the existing model directory",
    )

    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to input PCAP or CSV file(s)",
    )

    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        help="Label for the input data (0 for benign, 1 for malicious). Required for PCAP files.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )

    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )

    return parser.parse_args()


def validate_inputs(paths: List[Path], label: int = None) -> None:
    """Validate input files and arguments."""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if path.suffix not in [".pcap", ".csv"]:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if path.suffix == ".pcap" and label is None:
            raise ValueError("Label must be provided when using PCAP files")


def main() -> None:
    """Main execution function."""
    args = parse_arguments()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    validate_inputs(args.input, args.label)

    print(f"Loading model from: {args.model_path}")
    retrainer = ModelRetrainer(args.model_path)

    print("\nRetraining model...")
    output_dir = retrainer.retrain_model(
        input_paths=args.input,
        label=args.label,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"\nRetrained model and artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
