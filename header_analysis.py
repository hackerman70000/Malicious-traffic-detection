import logging
from pathlib import Path
from typing import Set, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OVERFITTING_COLUMNS = {
    "dst_ip",
    "dst_mac",
    "dst_oui",
    "src_ip",
    "src_mac",
    "src_oui",
    "id",
}


def analyze_headers(
    malicious_dir: Path, normal_dir: Path
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Analyze headers from malicious and normal traffic files.

    Returns:
        Tuple containing:
        - common_headers: Headers present in both malicious and normal files
        - malicious_only: Headers only in malicious files
        - normal_only: Headers only in normal files
    """

    malicious_headers: Set[str] = set()
    normal_headers: Set[str] = set()

    def get_headers(file_path: Path) -> Set[str]:
        try:
            return set(pd.read_csv(file_path, nrows=0).columns)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return set()

    logging.info("Analyzing malicious traffic files...")
    for file_path in malicious_dir.glob("*.csv"):
        headers = get_headers(file_path)
        malicious_headers.update(headers)

    logging.info("\nAnalyzing normal traffic files...")
    for file_path in normal_dir.glob("*.csv"):
        headers = get_headers(file_path)
        normal_headers.update(headers)

    malicious_only = malicious_headers - normal_headers
    normal_only = normal_headers - malicious_headers
    common_headers = malicious_headers & normal_headers

    common_headers = common_headers - OVERFITTING_COLUMNS

    logging.info("\nHeader Analysis Summary:")
    logging.info("=" * 50)

    logging.info("\nHeaders only in malicious files:")
    for header in sorted(malicious_only):
        logging.info(f"- {header}")

    logging.info("\nHeaders only in normal files:")
    for header in sorted(normal_only):
        logging.info(f"- {header}")

    logging.info("\nCommon headers (after removing potential overfitting columns):")
    for header in sorted(common_headers):
        logging.info(f"- {header}")

    logging.info("\nColumns removed to prevent overfitting:")
    for header in sorted(OVERFITTING_COLUMNS):
        logging.info(f"- {header}")

    logging.info("\nTotal number of unique headers:")
    logging.info(f"Malicious files: {len(malicious_headers)}")
    logging.info(f"Normal files: {len(normal_headers)}")
    logging.info(f"Common headers (after filtering): {len(common_headers)}")
    logging.info(f"Total unique headers: {len(malicious_headers | normal_headers)}")

    return common_headers, malicious_only, normal_only


if __name__ == "__main__":
    analyze_headers(Path("data/raw/CTU-13"), Path("data/raw/normal"))
