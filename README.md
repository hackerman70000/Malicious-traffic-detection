# Malicious traffic detection

## Data

- [CTU-13 Dataset (Malicious)](https://www.stratosphereips.org/datasets-ctu13)
- [Normal Traffic Dataset](https://www.stratosphereips.org/datasets-normal)

## Usage

### PCAP to CSV conversion

#### Example 1: Malicious traffic

To process PCAP files in `data/malicious` and label all flows as malicious (`label=1`):

```bash
find data/malicious -name '*.pcap' | xargs -n 1 uv run python pcap2flow.py 1
```

#### Example 2: Benign traffic

To process PCAP files in `data/benign` and label all flows as benign (`label=0`):

```bash
find data/benign -name '*.pcap' | xargs -n 1 uv run python pcap2flow.py 0
```

### Data preprocessing

Run data preprocessing from project root:

```sh
uv run -m scripts.preprocess_data
```

### Model training and retraining

#### Initial training

Train a new model:

```sh
uv run -m scripts.train
```

#### Training Existing Model with New Data

Train an existing model with new data to create a versioned update. Input CSVs must include a binary 'Label' column (0 = benign, 1 = malicious).

```sh
uv run -m scripts.retrain --help

options:
  -h, --help            Show this help message and exit
  --model-path MODEL_PATH
                        Path to the existing model directory
  --input INPUT [INPUT ...]
                        Path(s) to input CSV file(s) with a 'Label' column (0 or 1)
  --test-size TEST_SIZE
                        Proportion of data for testing (default: 0.2)
  --random-state RANDOM_STATE
                        Random state for reproducibility (default: 42)
```

Example:

```sh
uv run -m scripts.retrain \
  --model-path models/development/xgboost_20241222_225105_v1 \
  --input data/processed/combined_flows.csv
```

**Key Notes:**

- Input CSVs must include binary 'Label' column (0 = benign, 1 = malicious).
- Training data must have both benign (0) and malicious (1) samples.
- Model versions increment automatically (v1 → v2, etc.).
- Trained model and artifacts save in a new versioned directory.

## Development

### Package management

[uv: Python packaging in Rust](https://github.com/astral-sh/uv)
