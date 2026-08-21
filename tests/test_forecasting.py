import numpy as np

from src.forecasting.croston_sba import croston_sba_forecast


def test_croston_sba_non_negative():
    demand = np.array([0, 0, 10, 0, 5, 0])
    result = croston_sba_forecast(demand)
    assert result >= 0
