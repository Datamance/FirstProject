"""Cross Validation Code.

TODO: Try HalvingRandomSearchCV
"""
from typing import Dict

import pandas as pd
from sklearn.metrics import (f1_score, make_scorer, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Define custom scoring metrics
CV_SCORING = {
    "precision": make_scorer(precision_score, average="weighted"),
    "recall": make_scorer(recall_score, average="weighted"),
    "weighted_f1": make_scorer(f1_score, average="weighted"),
    "binary_f1": make_scorer(f1_score, average="binary"),
    "auc_score": make_scorer(roc_auc_score, average="weighted"),
}

# important_columns = [
#     "param_classifier__max_depth",
#     "param_classifier__learning_rate",
#     "param_classifier__n_estimators",
#     "mean_test_f1_score",
#     "rank_test_f1_score",
#     "mean_test_auc_score",
#     "rank_test_auc_score",
# ]


def get_search_cv(
    full_pipeline: Pipeline,
    parameter_grid: Dict,
    training_df: pd.DataFrame,
    target_labels: pd.Series,
    fit: bool = True,
):
    """Conducts the hyperparameter search while doing cross validation.

    To inspect results:
    >>> important_columns = ["mean_test_f1_score", "mean_test_auc_score"]
    >>> cv_results = pd.DataFrame(search_cv.cv_results_)[important_columns]
    """
    search_cv = GridSearchCV(
        full_pipeline,
        parameter_grid,
        n_jobs=-1,
        scoring=CV_SCORING,
        refit="weighted_f1",
        verbose=3,
    )

    if fit:
        search_cv.fit(training_df, target_labels)

    return search_cv
