"""Cross Validation Code.

TODO: Try HalvingRandomSearchCV
"""
from typing import Dict

import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import (f1_score, make_scorer, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, HalvingGridSearchCV,
                                     RandomizedSearchCV)
from sklearn.pipeline import Pipeline

# Define custom scoring metrics
CV_SCORING = {
    "precision": make_scorer(precision_score, average="weighted"),
    "recall": make_scorer(recall_score, average="weighted"),
    "f1_score": make_scorer(f1_score, average="weighted"),
    "auc_score": make_scorer(roc_auc_score, average="weighted"),
}


def get_search_cv(
    full_pipeline: Pipeline,
    parameter_grid: Dict,
    training_df: pd.DataFrame,
    target_labels: pd.Series,
    fit: bool = True,
):
    """Conducts the hyperparameter search while doing cross validation.

    TODO: parameterize to allow different hyperparameter search strategies.

    To inspect results:
    >>> important_columns = ["mean_test_f1_score", "mean_test_auc_score"]
    >>> cv_results = pd.DataFrame(search_cv.cv_results_)[important_columns]
    """
    # search_cv = GridSearchCV(
    #     full_pipeline,
    #     parameter_grid,
    #     n_jobs=-1,
    #     scoring=CV_SCORING,
    #     refit="f1_score",
    #     verbose=3,
    # )
    search_cv = HalvingGridSearchCV(
        full_pipeline,
        parameter_grid,
        n_jobs=-1,
        scoring="f1_weighted",
        refit="f1_weighted",
        verbose=3,
    )

    if fit:
        search_cv.fit(training_df, target_labels)

    return search_cv
