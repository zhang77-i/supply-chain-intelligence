import numpy as np
import pandas as pd

from cainiao_inventory.feature_engineering import (
    add_numpy_pandas_features,
    classify_demand,
    safe_divide,
)


def test_safe_divide_marks_zero_denominator_missing() -> None:
    result = safe_divide(np.array([2.0, 1.0]), np.array([4.0, 0.0]))
    assert result[0] == 0.5
    assert np.isnan(result[1])


def test_demand_pattern_classification() -> None:
    result = classify_demand(
        np.array([1.0, 1.0, 2.0, 2.0, np.nan]),
        np.array([0.2, 1.0, 0.2, 1.0, np.nan]),
        np.array([10, 10, 10, 10, 0]),
    )
    assert result.tolist() == [
        "smooth",
        "erratic",
        "intermittent",
        "lumpy",
        "all_zero",
    ]


def test_pandas_numpy_features_are_nonnegative_for_baselines() -> None:
    frame = pd.DataFrame(
        {
            "cart_uv_sum_14": [2.0],
            "pv_uv_sum_14": [4.0],
            "pay_uv_sum_14": [1.0],
            "qty_alipay_sum_14": [1.0],
            "qty_gmv_sum_14": [2.0],
            "demand_mean_7": [2.0],
            "demand_mean_28": [1.0],
            "observed_days_84": [70.0],
            "history_days_84": [84.0],
            "nonzero_days_84": [42.0],
            "nonzero_demand_std_84": [1.0],
            "nonzero_demand_mean_84": [2.0],
            "demand_sum_14": [-1.0],
        }
    )
    result = add_numpy_pandas_features(frame)
    assert result.loc[0, "cart_rate_14"] == 0.5
    assert result.loc[0, "baseline_last14"] == 0.0
    assert result.loc[0, "baseline_ma28"] == 14.0
