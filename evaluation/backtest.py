import numpy as np


def rolling_backtest(series, model, horizon=1, window=28):
    predictions = []
    actuals = []

    for i in range(window, len(series) - horizon + 1):
        train = series[:i]
        test = series[i:i + horizon]

        prediction = model(train)

        predictions.extend(np.atleast_1d(prediction))
        actuals.extend(test)

    return np.array(predictions), np.array(actuals)
