import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class EnsureDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.columns]
        return pd.DataFrame(X, columns=self.columns)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns)



def build_preprocessor(num_cols):
    cat_pipe = Pipeline([
        ("ensure_df", EnsureDataFrame(["wine_type"])),
        ("ohe", OneHotEncoder(
            drop="if_binary",
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", cat_pipe, ["wine_type"])
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
