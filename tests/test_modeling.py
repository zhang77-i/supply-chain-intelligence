import numpy as np

from cainiao_inventory.baselines import croston_sba, tsb_forecast
from cainiao_inventory.modeling import (
    interpolate_row_quantiles,
    inventory_cost_components,
)


def test_intermittent_baselines_handle_all_zero() -> None:
    demand = np.zeros(20)
    assert croston_sba(demand) == 0
    assert tsb_forecast(demand) == 0


def test_quantile_interpolation_and_crossing_fix() -> None:
    predictions = np.array([[10.0, 8.0, 20.0]])
    result = interpolate_row_quantiles(
        predictions,
        np.array([0.2, 0.5, 0.8]),
        np.array([0.65]),
    )
    assert result[0] == 15.0


def test_asymmetric_inventory_cost() -> None:
    result = inventory_cost_components(
        actual=np.array([10.0, 5.0]),
        target=np.array([8.0, 8.0]),
        shortage_cost=np.array([4.0, 4.0]),
        overage_cost=np.array([1.0, 1.0]),
    )
    assert result["shortage_cost"] == 8.0
    assert result["overage_cost"] == 3.0
    assert result["total_cost"] == 11.0
