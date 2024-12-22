# Malicious traffic detection

## Data

<https://www.unb.ca/cic/datasets/cic-unsw-nb15.html>

## Usage

### Data Preprocessing

Run data preprocessing from project root:

```sh
uv run -m scripts.preprocess_data
```

### Training

Train the model:

```sh
uv run -m scripts.train
```

Retrain existing model:

```sh
# TO BE IMPLEMENTED
# uv run -m scripts.retrain
```

## Development

### Package management

<https://github.com/astral-sh/uv>
