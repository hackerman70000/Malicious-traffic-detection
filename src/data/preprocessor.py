import csv
import gc
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class PreprocessingStats:
    """Statistics from preprocessing operation."""

    total_rows_processed: int = 0
    rows_dropped: int = 0
    processing_time: float = 0.0
    common_headers: Set[str] = field(default_factory=set)
    malicious_only_headers: Set[str] = field(default_factory=set)
    normal_only_headers: Set[str] = field(default_factory=set)


class MemoryEfficientPreprocessor:
    def __init__(self, config: "Config", stats: Optional[PreprocessingStats] = None):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.stats = stats or PreprocessingStats()

        self.malicious_dir = Path(self.config.data.raw_data_dir) / "CTU-13"
        self.normal_dir = Path(self.config.data.raw_data_dir) / "normal"
        self.output_dir = self.config.data.processed_data_dir

        self.malicious_output_dir = self.output_dir / "malicious"
        self.benign_output_dir = self.output_dir / "benign"
        self.benign_output = self.benign_output_dir / "benign.csv"
        self.final_output = self.output_dir / "merged_data.csv"

        self._setup_directories()
        self._validate_config()

        self._initialize_headers()

    def _setup_directories(self) -> None:
        """Setup and validate output directories."""
        for dir_path in [
            self.output_dir,
            self.malicious_output_dir,
            self.benign_output_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.config.data.processing.memory_limit < 1:
            raise ValueError("Memory limit must be between 0 and 1")
        if not 0 < self.config.data.processing.target_benign_ratio < 1:
            raise ValueError("Target benign ratio must be between 0 and 1")

    def _get_headers(self, file_path: Path) -> Set[str]:
        """Get headers from a CSV file."""
        try:
            with open(file_path, "r", newline="") as f:
                return set(next(csv.reader(f)))
        except Exception as e:
            logging.error(f"Error reading headers from {file_path}: {e}")
            return set()

    def _initialize_headers(self) -> None:
        """Analyze and initialize headers from all files."""
        logging.info("Analyzing file headers...")

        malicious_headers: Set[str] = set()
        normal_headers: Set[str] = set()

        for file_path in self.malicious_dir.glob("*.csv"):
            malicious_headers.update(self._get_headers(file_path))

        for file_path in self.normal_dir.glob("*.csv"):
            normal_headers.update(self._get_headers(file_path))

        self.stats.malicious_only_headers = malicious_headers - normal_headers
        self.stats.normal_only_headers = normal_headers - malicious_headers
        common_headers = malicious_headers & normal_headers

        self.stats.common_headers = common_headers - set(
            self.config.data.columns_to_drop
        )

        self.common_headers = self.stats.common_headers
        self.all_fields = self.common_headers | {"Label"}

    def _process_file(
        self, file_path: Path, label: int, writer: csv.DictWriter
    ) -> None:
        """Process a single file with chunk-based processing."""
        chunk = []
        try:
            with open(file_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    new_row = {
                        field: row.get(field, None) for field in self.common_headers
                    }
                    new_row["Label"] = label
                    chunk.append(new_row)
                    self.stats.total_rows_processed += 1

                    if len(chunk) >= self.config.data.processing.chunk_size:
                        writer.writerows(chunk)
                        chunk.clear()
                        gc.collect()

                if chunk:
                    writer.writerows(chunk)
                    chunk.clear()
                    gc.collect()

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {str(e)}")
            raise
        finally:
            chunk.clear()
            gc.collect()

    def process_files_separately(self) -> None:
        """Process malicious and benign files into separate files."""
        sorted_fields = sorted(self.all_fields)

        bar_format = "{desc:<30} {percentage:3.0f}%|{bar:50}{r_bar}"

        print("\nProcessing files:")

        malicious_files = list(self.malicious_dir.glob("*.csv"))
        for file_path in tqdm(
            malicious_files, desc="Malicious files", bar_format=bar_format
        ):
            output_path = self.malicious_output_dir / file_path.name
            with open(output_path, "w", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=sorted_fields)
                writer.writeheader()
                self._process_file(file_path, label=1, writer=writer)
            gc.collect()

        benign_files = list(self.normal_dir.glob("*.csv"))
        with open(self.benign_output, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=sorted_fields)
            writer.writeheader()

            for file_path in tqdm(
                benign_files, desc="Benign files", bar_format=bar_format
            ):
                self._process_file(file_path, label=0, writer=writer)
                gc.collect()

    def count_lines(self, file_path: Path) -> int:
        """Count lines in a file efficiently."""
        count = 0
        chunk_size = 8192 * 1024
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    count += chunk.count(b"\n")
            return count - 1
        finally:
            gc.collect()

    def merge_with_ratio(self) -> None:
        """Merge the separate files with the target ratio."""
        print("\nMerging files:")

        bar_format = "{desc:<30} {percentage:3.0f}%|{bar:50}{r_bar}"

        benign_count = self.count_lines(self.benign_output)
        malicious_files = list(self.malicious_output_dir.glob("*.csv"))
        malicious_counts = {f: self.count_lines(f) for f in malicious_files}
        total_malicious = sum(malicious_counts.values())

        target_malicious = int(
            benign_count
            * (1 - self.config.data.processing.target_benign_ratio)
            / self.config.data.processing.target_benign_ratio
        )

        file_ratios = {
            file_path: target_malicious * (count / total_malicious)
            for file_path, count in malicious_counts.items()
        }

        total_rows = benign_count + sum(malicious_counts.values())

        with open(self.final_output, "w", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=sorted(self.all_fields))
            writer.writeheader()

            with tqdm(
                total=total_rows, desc="Merging data", bar_format=bar_format
            ) as pbar:
                with open(self.benign_output, "r") as ben_file:
                    for row in csv.DictReader(ben_file):
                        writer.writerow(row)
                        pbar.update(1)
                gc.collect()

                for file_path in malicious_files:
                    sampling_ratio = (
                        file_ratios[file_path] / malicious_counts[file_path]
                    )

                    with open(file_path, "r") as mal_file:
                        for row in csv.DictReader(mal_file):
                            if random.random() < sampling_ratio:
                                writer.writerow(row)
                            pbar.update(1)
                    gc.collect()

        for file_path in malicious_files:
            file_path.unlink()
        self.benign_output.unlink()
        self.malicious_output_dir.rmdir()
        self.benign_output_dir.rmdir()

        gc.collect()
