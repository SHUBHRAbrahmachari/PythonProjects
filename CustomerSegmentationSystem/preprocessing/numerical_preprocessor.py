from typing import override
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings(action="ignore")


class NumericalColumnPreprocessor(BaseEstimator,
                                  TransformerMixin):

    def __init__(self,
                 drop_duplicate_rows=True,
                 null_treatment_strategy="auto",
                 left_fence_percentile_limit=5,
                 right_fence_percentile_limit=95):

        self.drop_duplicate_rows = drop_duplicate_rows
        self.null_treatment_strategy = null_treatment_strategy
        self.null_value_replacements = {}
        self.is_outlier_present_for = {}
        self.winsorization_tuples = {}
        self.left_fence_percentile_limit = left_fence_percentile_limit
        self.right_fence_percentile_limit = right_fence_percentile_limit
        self.is_already_fit_ = False
        self.cols_ = None

    # check is a column has outliers present in it
    # if outliers are present and "auto" has been set, then replace null values with "median" else "mean"
    # then replace the outliers by fence replacements

    def fit(self, x, y=None, **kwargs):

        if self.is_already_fit_:
            raise RuntimeError(f"Preprocessor is already fit!")

        dataset = None

        if isinstance(x, np.ndarray) and x.ndim == 2:
            dataset = pd.DataFrame(x)
        elif isinstance(x, pd.DataFrame):
            dataset = x.copy()
        else:
            raise TypeError(f"Expected a DataFrame or a 2D numpy array, got {type(x)} instead!")

        self.cols_ = dataset.columns

        for col in self.cols_:
            # first we need to see whether any outliers exist or not, we need to remove null values and check
            values = dataset[col].dropna(inplace=False).values

            Ql, Q1, Q3, Qr = np.percentile(values, (self.left_fence_percentile_limit,
                                                    25,
                                                    75,
                                                    self.right_fence_percentile_limit))

            # then calculate interquartile-range, fence limits and replacements
            IQR = Q3 - Q1
            right_fence_limit = Q3 + 1.5 * IQR
            left_fence_limit = Q1 - 1.5 * IQR

            # if outliers are present then do set the values accordingly
            if np.any(a=((values > right_fence_limit) | (values < left_fence_limit)), axis=0):
                self.is_outlier_present_for[col] = True
                self.winsorization_tuples[col] = (left_fence_limit, Ql, right_fence_limit, Qr)
            else:
                self.is_outlier_present_for[col] = False

            # fix null value replacements accordingly
            if self.null_treatment_strategy == "auto":

                if self.is_outlier_present_for[col]:
                    self.null_value_replacements[col] = np.median(values)
                else:
                    self.null_value_replacements[col] = np.mean(values)

            elif self.null_treatment_strategy == "mean":
                self.null_value_replacements[col] = np.mean(values)

            else:
                self.null_value_replacements[col] = np.median(values)

        self.is_already_fit_ = True
        return self

    def transform(self, x, y=None, **kwargs):

        if not self.is_already_fit_:
            raise RuntimeError("Preprocessor should be fit first before transformation!")

        dataset = None

        if isinstance(x, pd.DataFrame):
            dataset = x.copy()
        elif isinstance(x, np.ndarray) and x.ndim == 2:
            dataset = pd.DataFrame(x)
        else:
            raise TypeError(f"Expected a DataFrame or a 2D numpy array, got {type(x)} instead!")

        for col in self.cols_:

            # first null value replacements
            dataset[col].replace(to_replace={np.nan: self.null_value_replacements[col]}, inplace=True)

            # then finally winsorization for null value treatment

            if self.is_outlier_present_for[col]:
                packet = self.winsorization_tuples[col]
                left_fence_limit = packet[0]
                left_fence_replacement = packet[1]
                right_fence_limit = packet[2]
                right_fence_replacement = packet[3]

                dataset[col] = np.where(
                    dataset[col] < left_fence_limit,
                    left_fence_replacement,
                    np.where(
                        dataset[col] > right_fence_limit,
                        right_fence_replacement,
                        dataset[col]
                    )
                )

        return dataset.values

    @override
    def fit_transform(self, x, y=None, **kwargs):
        return self.fit(x, y).transform(x, y)

    def get_feature_names_out(self, feature_names=None):
        if not self.is_already_fit_:
            raise RuntimeError("Preprocessor should be fit first!")

        return self.cols_
