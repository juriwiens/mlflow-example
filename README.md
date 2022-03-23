# MLflow Example

Demonstrates the capabilities of [MLflow](https://www.mlflow.org) using a [keras classification model](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/) as an example.

## Installation

Install dependencies via [poetry](https://python-poetry.org/):

```shell
poetry install
```

## MLflow Tracking

When [train.py](./train.py) is executed, the training progress is logged as an experiment run via [automatic logging](https://www.mlflow.org/docs/latest/tracking.html#tensorflow-and-keras).
