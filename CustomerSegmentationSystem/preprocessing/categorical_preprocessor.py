from typing import override
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class CategoricalColumnPreprocessor(BaseEstimator,
                                    TransformerMixin):

    def __init__(self):
        self.encoders_ = {}
        self.mode_values_ = {}
        self.is_already_fit_ = False
        self.columns_ = None

    def fit(self, x, y=None, **kwargs):

        if self.is_already_fit_:
            raise RuntimeError(f"Preprocessor is already fit!")

        dataset = None

        if isinstance(x, pd.DataFrame):
            dataset = x.copy()
        elif isinstance(x, np.ndarray) and x.ndim == 2:
            dataset = pd.DataFrame(data=x)
        else:
            raise TypeError(f"Expected a DataFrame or a 2D numpy array, got {type(x)} instead!")

        self.columns_ = dataset.columns

        for col in self.columns_:

            mode_value = dataset[col].mode()[0]
            self.mode_values_[col] = mode_value

            self.encoders_[col] = LabelEncoder()
            self.encoders_[col].fit_transform(dataset[col])

        self.is_already_fit_ = True
        return self

    def transform(self, x, y=None, **kwargs):

        if not self.is_already_fit_:
            raise RuntimeError("Preprocessor should be fit first before calling transform!")

        dataset = None

        if isinstance(x, pd.DataFrame):
            dataset = x.copy()
        elif isinstance(x, np.ndarray) and x.ndim == 2:
            dataset = pd.DataFrame(data=x)
        else:
            raise TypeError(f"Expected a DataFrame or a 2D numpy array, got {type(x)} instead!")

        for col in self.columns_:

            dataset[col] = np.where(
                (dataset[col] is None) or (dataset is pd.NA),
                self.mode_values_[col],
                dataset[col]
            )

            dataset[col] = self.encoders_[col].transform(dataset[col])

        return dataset.values

    @override
    def fit_transform(self, x, y=None, **fit_params):
        return self.fit(x, y).transform(x, y)

    def get_feature_names_out(self, feature_names=None):
        return self.columns_

