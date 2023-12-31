"""Main module.

TODO: Try scaling of both varieties: normalization and standardization.
TODO: Add nicer logging
TODO: Add plotting capabilities.
TODO: Random number seeding for replicable results
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import sklearn as sl

import models
import preprocess
import validation

# Layout of the project
ROOT = Path(__file__).parent
DEFAULT_DATA_PATH = ROOT / "data"
DEFAULT_TRAINING_CSV = DEFAULT_DATA_PATH / "data_training.csv"
DEFAULT_TEST_CSV = DEFAULT_DATA_PATH / "data_test.csv"
DEFAULT_OUTPUT_PATH = ROOT / "output"
DEFAULT_PREDICTIONS_CSV = DEFAULT_OUTPUT_PATH / "P1_test_output.csv"

# Set up argument parser.
# TODO: add ability to change path, hyperparameters, etc.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", type=str, default=models.GRADIENT_BOOSTED_FOREST.name
)


# Container for results; allows us to wrap the driver code in a function and still inspect it
# after running things in interactive mode (i.e. with `python -i` or using PyCharm's
# Data Inspector).
class RunResult(NamedTuple):
    training_df: pd.DataFrame
    target_labels: pd.Series
    model_proxy: models.Model
    pipeline: sl.pipeline.Pipeline
    cross_validator: sl.model_selection.BaseCrossValidator
    test_df: pd.DataFrame
    predictions_df: pd.DataFrame


def run(
    model_name: str,
    training_csv: Path = DEFAULT_TRAINING_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    predictions_csv: Path = DEFAULT_PREDICTIONS_CSV,
):
    training_df = pd.read_csv(training_csv)

    # Reduce the problem to classifying the least common label, so 1 == False, 2 == True
    target_labels = training_df.Label == 1
    # XXX: PyCharm mistakenly thinks this returns a Bool, so can't do this all in the line above :/
    training_df.pop("Label")

    # Dynamically retrieve model.
    model = models.retrieve(model_name)
    print(f"Using {model_name}")

    full_pipeline = sl.pipeline.Pipeline(
        steps=[
            ("preprocessor", preprocess.COLUMN_TRANSFORMER),
            ("classifier", model.estimator),
        ]
    )

    # Acquire the best estimator with a parameter search.
    search_cv = validation.get_search_cv(
        full_pipeline=full_pipeline,
        parameter_grid=model.hyperparameters,
        training_df=training_df,
        target_labels=target_labels,  # noqa
    )

    # Reference for best estimator. Why the trailing underscore? Is this supposed to be protected?
    best_estimator = search_cv.best_estimator_

    # Make predictions, format them appropriately, and shove them into the output folder.
    test_df = pd.read_csv(test_csv)
    predictions = best_estimator.predict(test_df)
    predictions_df = pd.DataFrame({"Predictions": predictions})
    predictions_df.Predictions.replace({True: 1, False: 2}, inplace=True)
    predictions_df.to_csv(predictions_csv, header=False, index=False)

    # Return the search object for inspection purposes.
    return RunResult(
        training_df=training_df,
        target_labels=target_labels,
        model_proxy=model,
        pipeline=full_pipeline,
        cross_validator=search_cv,
        test_df=test_df,
        predictions_df=predictions_df,
    )


def get_results_df(run_result: RunResult):
    """Convenience method for getting CV result DF columns we really care about."""
    model_proxy = run_result.model_proxy
    # CV Search objects tend to put per-run information that I don't necessarily want to throw out.
    # This is just for when other people run the code and want to see the most important
    # metrics, and their corresponding hyperparameters.
    important_columns = [
        f"param_{hp_key}" for hp_key in model_proxy.hyperparameters.keys()
    ] + [f"mean_test_{cv_score_key}" for cv_score_key in validation.CV_SCORING.keys()]
    # Again, PyCharm is dumb and thinks that cv_results_ is not a member of BaseValidator
    return pd.DataFrame(run_result.cross_validator.cv_results_)[  # noqa
        important_columns
    ]


if __name__ == "__main__":
    args = parser.parse_args()
    result = run(model_name=args.model_name)
    search_df = get_results_df(result)
