from typing import Any, NamedTuple


class TrainingData(NamedTuple):
  features_train: Any | list
  label_train: Any | list
  features_test: Any | list
  label_test: Any | list


class CLIArguments(NamedTuple):
  load_model: bool
  save_model: bool
  save_test_data: bool
  predict: bool
  # Add any other arguments you need here

class TrainingDataMF(NamedTuple):
  train_df: Any | list
  test_df: Any | list