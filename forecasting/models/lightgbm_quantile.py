import lightgbm as lgb


class QuantileForecaster:
    def __init__(self, alpha=0.5, params=None):
        self.alpha = alpha
        self.model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            **(params or {})
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
