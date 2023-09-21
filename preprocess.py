"""Preprocessor module.

TODO: try out other scaling/normalizing/standardizing modalities
TODO: parameterize construction of the column transformer.
"""

import sklearn as sl
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer, make_column_transformer

# Won't work unless we do dense one-hot encoding -
# it looks like pandas makes it sparse again?
sl.set_config(transform_output="pandas")

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
COLUMN_TRANSFORMER = make_column_transformer(
    (preprocessing.MinMaxScaler(), NUMERIC_FEATURE_COLS),
    (preprocessing.OneHotEncoder(sparse_output=False), CATEGORICAL_FEATURE_COLS),
    remainder="passthrough",  # Shouldn't have to use this
)
