from typing import Any, NamedTuple, Union


class TrainingData(NamedTuple):
    features_train: Union[Any, list]
    label_train: Union[Any, list]
    features_test: Union[Any, list]
    label_test: Union[Any, list]


class CLIArguments(NamedTuple):
    load_model: bool
    save_model: bool
    save_test_data: bool
    predict: bool
    # Add any other arguments you need here


class TrainingDataMF(NamedTuple):
    train_df: Union[Any, list]
    test_df: Union[Any, list]
