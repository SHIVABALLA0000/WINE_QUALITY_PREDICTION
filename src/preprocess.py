import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class EnsureDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.columns)

def build_preprocessor(num_cols):
    cat_pipe = Pipeline([
        ("ensure_df", EnsureDataFrame(["wine_type"])),
        ("ohe", OneHotEncoder(
            drop="if_binary",
            sparse_output=False,
            handle_unknown="ignore"
        ))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", cat_pipe, ["wine_type"])
        ],
        remainder="drop"
    )
