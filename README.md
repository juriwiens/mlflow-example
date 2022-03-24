# MLflow Example

Demonstrates the capabilities of [MLflow](https://www.mlflow.org) using a [keras classification model](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/) as an example.

## Installation

Install dependencies via [poetry](https://python-poetry.org/):

```shell
poetry install
```

## MLflow Tracking

When [train.py](./train.py) is executed, the training progress is logged as an experiment run via [automatic logging](https://www.mlflow.org/docs/latest/tracking.html#tensorflow-and-keras):

```shell
poetry run python train.py
```

The training progress and logs can be inspected via a local web UI:

```shell
poetry run mlflow ui
```

By default, all data (backend data and artifacts) are stored on your local file system (see [docs](https://www.mlflow.org/docs/latest/tracking.html#how-runs-and-artifacts-are-recorded)). However, if you want to use the MLflow Model Registry, all backend data must be persisted in a [database-backed store](https://www.mlflow.org/docs/latest/tracking.html#backend-stores). A simple alternative variant is to configure the use of a SQLite database via the tracking URI, for example by setting it via the `MLFLOW_TRACKING_URI` environment variable:

```shell
MLFLOW_TRACKING_URI="sqlite:///mlflow.db" poetry run python train.py
```

In this case, the UI must be started with the SQLite URI as `backend-store-ui`:

```shell
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```
