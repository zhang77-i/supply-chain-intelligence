from collections import deque


def moving_average_forecast(series, window=28):
    if len(series) < window:
        raise ValueError("series length must be larger than window")
    return sum(series[-window:]) / window


def rolling_forecast(series, horizon, window=28):
    history = deque(series)
    result = []
    for _ in range(horizon):
        value = sum(list(history)[-window:]) / window
        result.append(value)
        history.append(value)
    return result
