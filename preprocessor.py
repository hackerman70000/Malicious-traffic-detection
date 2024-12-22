import csv
import gc
import logging
import os
import random
from pathlib import Path

import psutil

from header_analysis import analyze_headers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MemoryEfficientPreprocessor:
    def __init__(
        self,
        malicious_dir: str,
        normal_dir: str,
        output_dir: str,
        chunk_size: int = 1000,
        memory_limit: float = 0.75,
        target_benign_ratio: float = 0.7,
    ):
        self.malicious_dir = Path(malicious_dir)
        self.normal_dir = Path(normal_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.memory_limit = memory_limit
        self.target_benign_ratio = target_benign_ratio

        common_headers, malicious_only, normal_only = analyze_headers(
            self.malicious_dir, self.normal_dir
        )

        self.common_headers = common_headers
        self.columns_to_drop = malicious_only | normal_only
        self.all_fields = common_headers | {"Label"}

        logging.info("\nPreprocessor configuration:")
        logging.info(f"Number of common headers to keep: {len(self.common_headers)}")
        logging.info(f"Number of columns to drop: {len(self.columns_to_drop)}")

        # Output paths
        self.malicious_output_dir = self.output_dir / "malicious"
        self.benign_output_dir = self.output_dir / "benign"
        self.benign_output = self.benign_output_dir / "benign.csv"
        self.final_output = self.output_dir / "merged_data.csv"

        # Create output directories
        self.malicious_output_dir.mkdir(parents=True, exist_ok=True)
        self.benign_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_memory_usage(self) -> float:
        return psutil.Process().memory_percent()

    def _check_memory(self):
        if self._get_memory_usage() > self.memory_limit:
            logging.warning("High memory usage detected. Forcing garbage collection...")
            gc.collect()

    def _get_csv_files(self, directory: Path) -> list:
        return list(directory.glob("*.csv"))

    def _process_file(self, file_path: Path, label: int, writer: csv.DictWriter):
        """Process a single file."""
        try:
            with open(file_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)

                chunk = []
                for row in reader:
                    # Create new row with only the common headers
                    new_row = {
                        field: row.get(field, None) for field in self.common_headers
                    }
                    new_row["Label"] = label

                    chunk.append(new_row)

                    if len(chunk) >= self.chunk_size:
                        writer.writerows(chunk)
                        chunk.clear()
                        self._check_memory()

                if chunk:
                    writer.writerows(chunk)
                    chunk.clear()

                gc.collect()

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {str(e)}")

    def process_files_separately(self):
        """Process malicious and benign files into separate files."""
        sorted_fields = sorted(self.all_fields)

        # Process malicious traffic into separate files
        logging.info("Processing malicious traffic files...")
        for file_path in self._get_csv_files(self.malicious_dir):
            logging.info(f"Processing {file_path.name}")
            output_path = self.malicious_output_dir / file_path.name

            with open(output_path, "w", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=sorted_fields)
                writer.writeheader()
                self._process_file(file_path, label=1, writer=writer)

        # Process benign traffic into a single file
        logging.info("Processing benign traffic files...")
        with open(self.benign_output, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=sorted_fields)
            writer.writeheader()

            for file_path in self._get_csv_files(self.normal_dir):
                logging.info(f"Processing {file_path.name}")
                self._process_file(file_path, label=0, writer=writer)

        logging.info("Finished processing files separately")

    def count_lines(self, file_path: Path) -> int:
        """Count lines in a file efficiently."""
        count = 0
        with open(file_path, "rb") as f:
            while True:
                buffer = f.read(8192 * 1024)
                if not buffer:
                    break
                count += buffer.count(b"\n")
        return count - 1  # Subtract header line

    def merge_with_ratio(self):
        """Merge the separate files with the target ratio, maintaining malicious data distribution."""
        logging.info("Starting file merge with ratio balancing...")

        # Count benign samples
        benign_count = self.count_lines(self.benign_output)
        logging.info(f"Benign samples: {benign_count}")

        # Count malicious samples from each file
        malicious_files = list(self.malicious_output_dir.glob("*.csv"))
        malicious_counts = {}
        total_malicious = 0

        for file_path in malicious_files:
            count = self.count_lines(file_path)
            malicious_counts[file_path] = count
            total_malicious += count
            logging.info(f"Malicious file {file_path.name}: {count} samples")

        logging.info(f"Total malicious samples: {total_malicious}")

        # Calculate target number of malicious samples
        target_malicious = int(
            benign_count * (1 - self.target_benign_ratio) / self.target_benign_ratio
        )
        logging.info(f"Target total malicious samples: {target_malicious}")

        # Calculate sampling ratio for each file to maintain distribution
        file_ratios = {
            file_path: target_malicious * (count / total_malicious)
            for file_path, count in malicious_counts.items()
        }

        # Start merging
        with open(self.final_output, "w", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=sorted(self.all_fields))
            writer.writeheader()

            # Write all benign traffic
            logging.info("Writing benign traffic...")
            with open(self.benign_output, "r") as ben_file:
                ben_reader = csv.DictReader(ben_file)
                for row in ben_reader:
                    writer.writerow(row)

            # Sample and write malicious traffic from each file
            logging.info("Sampling and writing malicious traffic...")
            for file_path in malicious_files:
                target_samples = int(file_ratios[file_path])
                sampling_ratio = target_samples / malicious_counts[file_path]

                logging.info(f"Processing {file_path.name}")
                logging.info(f"Target samples: {target_samples}")
                logging.info(f"Sampling ratio: {sampling_ratio:.3f}")

                with open(file_path, "r") as mal_file:
                    mal_reader = csv.DictReader(mal_file)
                    for row in mal_reader:
                        if random.random() < sampling_ratio:
                            writer.writerow(row)

        logging.info("Finished merging files")
        logging.info(f"Final merged file: {self.final_output}")


def main():
    malicious_dir = "data/raw/CTU-13"
    normal_dir = "data/raw/normal"
    output_dir = "data/processed"
    chunk_size = 1000
    memory_limit = 0.75
    target_benign_ratio = 0.7

    os.makedirs(output_dir, exist_ok=True)

    preprocessor = MemoryEfficientPreprocessor(
        malicious_dir=malicious_dir,
        normal_dir=normal_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        memory_limit=memory_limit,
        target_benign_ratio=target_benign_ratio,
    )

    try:
        # Step 1: Process files separately
        preprocessor.process_files_separately()

        # Step 2: Merge with desired ratio
        preprocessor.merge_with_ratio()
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
