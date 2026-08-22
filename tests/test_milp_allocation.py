import numpy as np
import pandas as pd

from cainiao_inventory.allocation import solve_coordinated_allocation
from cainiao_inventory.milp_allocation import (
    solve_coordinated_allocation_milp,
)


def recommendations() -> pd.DataFrame:
    rows = [
        {
            "item_id": 1,
            "store_code": "all",
            "target_inventory_14d": 10.0,
            "shortage_cost": 1.0,
            "overage_cost": 1.0,
        }
    ]
    for code, target in zip(("1", "2", "3", "4", "5"), (3, 3, 3, 3, 3)):
        rows.append(
            {
                "item_id": 1,
                "store_code": code,
                "target_inventory_14d": float(target),
                "shortage_cost": 4.0,
                "overage_cost": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_milp_matches_cp_sat_objective_and_total() -> None:
    frame = recommendations()
    cp_result, cp_summary = solve_coordinated_allocation(
        frame,
        quantity_scale=10,
        time_limit_seconds=5,
        workers=1,
    )
    milp_result, milp_summary = solve_coordinated_allocation_milp(
        frame,
        quantity_scale=10,
        time_limit_seconds=5,
        workers=1,
    )
    assert np.isclose(cp_result["coordinated_inventory_14d"].sum(), 10.0)
    assert np.isclose(milp_result["milp_inventory_14d"].sum(), 10.0)
    assert np.isclose(
        cp_summary["objective_value_scaled"],
        milp_summary["objective_value_scaled"],
    )
    assert milp_summary["status"] == "OPTIMAL"
    assert milp_summary["max_absolute_reconciliation_gap"] == 0.0


def test_milp_enforces_optional_store_capacities() -> None:
    result, summary = solve_coordinated_allocation_milp(
        recommendations(),
        store_capacities={code: 2.0 for code in ("1", "2", "3", "4", "5")},
        time_limit_seconds=5,
        workers=1,
    )
    usage = result.groupby("store_code")["milp_inventory_14d"].sum()
    assert np.isclose(result["milp_inventory_14d"].sum(), 10.0)
    assert (usage <= 2.0 + 1e-9).all()
    assert summary["store_capacity_usage"] == {
        code: 2.0 for code in ("1", "2", "3", "4", "5")
    }
