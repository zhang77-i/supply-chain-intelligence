import numpy as np


def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def quantile_loss(y_true, y_pred, quantile=0.9):
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))
