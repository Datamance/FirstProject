"""Sandbox for trying stuff out."""

from pathlib import Path
from pprint import pprint
from typing import Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import sklearn as sl
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (f1_score, make_scorer, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline

# Won't work unless we do dense one-hot encoding -
# it looks like pandas makes it sparse again?
sl.set_config(transform_output="pandas")


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data"
# TODO: drop Feature 12 since it's redundant?
NUMERIC_FEATURE_COLS = ["Feature_1", "Feature_3"] + list(
    f"Feature_{n}" for n in range(8, 20)
)
CATEGORICAL_FEATURE_COLS = [
    "Feature_2",
    "Feature_4",
    "Feature_5",
    "Feature_6",  # Technically this is ordinal, but it might be worth encoding manually
    "Feature_7",
]


# Will automatically name the column transformers.
# Alternatively could directly construct ColumnTransformer
column_transformer = make_column_transformer(
    (preprocessing.MinMaxScaler(), NUMERIC_FEATURE_COLS),
    (preprocessing.OneHotEncoder(sparse_output=False), CATEGORICAL_FEATURE_COLS),
    remainder="passthrough",  # Shouldn't have to use this
)

# classifier = RandomForestClassifier()
classifier = GradientBoostingClassifier()

full_pipeline = Pipeline(
    steps=[
        ("preprocessor", column_transformer),
        ("classifier", classifier),
    ]
)


# Define custom scoring metrics
cv_scoring = {
    "precision": make_scorer(precision_score, average="weighted"),
    "recall": make_scorer(recall_score, average="weighted"),
    "f1_score": make_scorer(f1_score, average="weighted"),
    "auc_score": make_scorer(roc_auc_score, average="weighted"),
}

# For Random Forest
# parameter_grid = {"classifier__max_depth": list(range(5, 40))}
# For Gradient Boosting
parameter_grid = {
    "classifier__max_depth": list(range(4, 7)),
    "classifier__learning_rate": np.linspace(0.3, 0.5, 5),
    "classifier__n_estimators": list(range(90, 110)),
}


if __name__ == "__main__":
    training_df = pd.read_csv(DATA_PATH / "data_training.csv")
    target_labels = training_df.pop("Label") == 2

    # I think the number of folds is automatically 5
    # search_cv = RandomizedSearchCV(
    #     full_pipeline, parameter_grid, n_iter=20, scoring=cv_scoring, refit="f1_score"
    # )
    search_cv = GridSearchCV(
        full_pipeline,
        parameter_grid,
        n_jobs=-1,
        scoring=cv_scoring,
        refit="f1_score",
        verbose=3,
    )

    search_cv.fit(training_df, target_labels)

    important_columns = [
        "param_classifier__max_depth",
        "param_classifier__learning_rate",
        "param_classifier__n_estimators",
        "mean_test_f1_score",
        "rank_test_f1_score",
        "mean_test_auc_score",
        "rank_test_auc_score",
    ]

    cv_results = pd.DataFrame(search_cv.cv_results_)[important_columns]
