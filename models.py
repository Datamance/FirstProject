"""Models we want to test.

The purpose of these containers is to allow flexible replacement of models
and their corresponding CV parameters contingent on command line arguments.
"""

from typing import Dict, NamedTuple

import numpy as np
import sklearn as sl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


class Model(NamedTuple):
    name: str
    estimator: sl.base.BaseEstimator
    hyperparameters: Dict


RANDOM_FOREST = Model(
    name="Random Forest",
    estimator=RandomForestClassifier(),
    hyperparameters={"classifier__max_depth": list(range(5, 40))},
)

GRADIENT_BOOSTED_FOREST = Model(
    name="Gradient Boosted Forest",
    estimator=GradientBoostingClassifier(),
    hyperparameters={
        "classifier__max_depth": list(range(4, 7)),
        "classifier__learning_rate": np.linspace(0.3, 0.5, 5),
        "classifier__n_estimators": list(range(90, 110)),
    },
)

# Update this if you want to try other stuff.
MODEL_LIST = [RANDOM_FOREST, GRADIENT_BOOSTED_FOREST]

MODEL_REGISTRY = {model.name.lower().replace(" ", "_"): model for model in MODEL_LIST}


def retrieve(model_name: str) -> Model:
    """Retrieve something from the model registry."""
    # First, format to snake_case
    key = model_name.lower().replace(" ", "_")
    model = MODEL_REGISTRY.get(key, None)
    if model is not None:
        return model
    else:
        msg = f"No {model_name} in registry.\nAvailable keys: {MODEL_REGISTRY.keys()}"
        raise NameError(msg)
