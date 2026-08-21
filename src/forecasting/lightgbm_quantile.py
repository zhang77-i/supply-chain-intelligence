"""Quantile demand forecasting module.

This module provides a lightweight interface for probabilistic demand
forecasting. Quantile prediction is used because inventory decisions depend
on demand uncertainty rather than only point forecasts.
"""

from lightgbm import LGBMRegressor


class QuantileDemandForecaster:
    def __init__(self, quantile=0.5, **kwargs):
        self.model = LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            **kwargs,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
