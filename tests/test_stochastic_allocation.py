import numpy as np
import pandas as pd

from cainiao_inventory.stochastic_allocation import solve_stochastic_allocation


def test_stochastic_allocation_reconciles_and_is_nonnegative() -> None:
    stores = pd.DataFrame(
        {
            "item_id": [1] * 5,
            "store_code": ["1", "2", "3", "4", "5"],
            "shortage_cost": [5.0] * 5,
            "overage_cost": [1.0] * 5,
        }
    )
    scenarios = {
        (1, code): np.array([index, index + 1, index + 2], dtype=float)
        for index, code in enumerate(["1", "2", "3", "4", "5"], start=1)
    }
    result, summary = solve_stochastic_allocation(
        pd.Series({1: 20.0}),
        stores,
        scenarios,
        quantity_scale=10,
        time_limit_seconds=5,
        workers=1,
    )
    assert np.isclose(result["stochastic_inventory_14d"].sum(), 20.0)
    assert (result["stochastic_inventory_14d"] >= 0).all()
    assert summary["max_absolute_reconciliation_gap"] == 0.0
