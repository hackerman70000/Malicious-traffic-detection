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

#### Training existing model with new data

Train an existing model with new data, creating a new versioned model:

```sh
uv run -m scripts.retrain \
  --model-path <path_to_model_directory> \
  --input <path_to_input_file> \
  --label <0_or_1>
```

Example:

```sh
# Train with new malicious traffic
uv run -m scripts.retrain \
  --model-path models/development/xgboost_20241222_225105_v1 \
  --input data/raw/malicious/traffic.pcap \
  --label 1

# Train with new benign traffic
uv run -m scripts.retrain \
  --model-path models/development/xgboost_20241222_225105_v1 \
  --input data/raw/normal/traffic.pcap \
  --label 0
```

Additional retraining options:

```sh
uv run -m scripts.retrain --help

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Path to the existing model directory
  --input INPUT [INPUT ...]
                        Path(s) to input PCAP or CSV file(s)
  --label {0,1}         Label for the input data (0 for benign, 1 for malicious)
  --test-size TEST_SIZE
                        Proportion of data to use for testing (default: 0.2)
  --random-state RANDOM_STATE
                        Random state for reproducibility (default: 42)
```

Note: When training with new data, the model version (v1, v2, etc.) is automatically incremented. The newly trained model and its artifacts are saved in a new directory with the updated version number.

## Development

### Package management

[uv: Python packaging in Rust](https://github.com/astral-sh/uv)
