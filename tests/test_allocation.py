import pandas as pd

from cainiao_inventory.allocation import solve_coordinated_allocation


def test_allocation_reconciles_store_sum_to_national_target() -> None:
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
    result, summary = solve_coordinated_allocation(
        pd.DataFrame(rows),
        quantity_scale=10,
        time_limit_seconds=5,
        workers=1,
    )
    assert result["coordinated_inventory_14d"].sum() == 10.0
    assert summary["max_absolute_reconciliation_gap"] == 0.0
