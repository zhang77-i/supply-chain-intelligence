"""Mixed-integer linear model for coordinated multi-warehouse inventory."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from .allocation import STORE_CODES, _complete_store_grid
from .config import PipelineConfig, load_config


STATUS_NAMES = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


def solve_coordinated_allocation_milp(
    recommendations: pd.DataFrame,
    *,
    quantity_scale: int = 10,
    cost_scale: int = 100,
    store_capacities: Mapping[str, float] | None = None,
    time_limit_seconds: float = 30.0,
    workers: int = 1,
    backend: str = "SCIP",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Solve integer allocation with continuous deviation variables.

    The public-data experiment leaves ``store_capacities`` unset because the
    Tianchi files do not provide warehouse capacities. The optional argument
    is intended for transparent scenario analysis with user-supplied values.
    """
    if quantity_scale <= 0 or cost_scale <= 0:
        raise ValueError("quantity_scale and cost_scale must be positive")

    capacities = {
        str(store): float(capacity)
        for store, capacity in (store_capacities or {}).items()
    }
    unknown_stores = sorted(set(capacities) - set(STORE_CODES))
    if unknown_stores:
        raise ValueError(f"unknown stores in capacity map: {unknown_stores}")
    if any(capacity < 0 for capacity in capacities.values()):
        raise ValueError("store capacities must be nonnegative")

    national, stores = _complete_store_grid(recommendations)
    national_target = national.set_index("item_id")["target_inventory_14d"]
    solver = pywraplp.Solver.CreateSolver(backend)
    if solver is None:
        raise RuntimeError(f"MILP backend is unavailable: {backend}")
    solver.SetTimeLimit(int(round(time_limit_seconds * 1000)))
    solver.SetNumThreads(int(workers))

    infinity = solver.infinity()
    allocations: dict[tuple[int, str], pywraplp.Variable] = {}
    variables_by_store: dict[str, list[pywraplp.Variable]] = {
        store: [] for store in STORE_CODES
    }
    objective = solver.Objective()

    for item_id, group in stores.groupby("item_id", sort=True):
        total = max(
            0,
            int(round(float(national_target.loc[item_id]) * quantity_scale)),
        )
        item_variables = []
        for row in group.itertuples(index=False):
            key = (int(item_id), str(row.store_code))
            independent = max(
                0,
                int(round(float(row.target_inventory_14d) * quantity_scale)),
            )
            allocated = solver.IntVar(0, total, f"x_{key[0]}_{key[1]}")
            shortage = solver.NumVar(0, infinity, f"u_{key[0]}_{key[1]}")
            overage = solver.NumVar(0, infinity, f"o_{key[0]}_{key[1]}")
            solver.Add(independent - allocated == shortage - overage)

            shortage_weight = max(
                1,
                int(round(float(row.shortage_cost) * cost_scale)),
            )
            overage_weight = max(
                1,
                int(round(float(row.overage_cost) * cost_scale)),
            )
            objective.SetCoefficient(shortage, shortage_weight)
            objective.SetCoefficient(overage, overage_weight)
            allocations[key] = allocated
            item_variables.append(allocated)
            variables_by_store[key[1]].append(allocated)
        solver.Add(solver.Sum(item_variables) == total)

    for store_code, capacity in capacities.items():
        scaled_capacity = int(round(capacity * quantity_scale))
        solver.Add(
            solver.Sum(variables_by_store[store_code]) <= scaled_capacity
        )

    objective.SetMinimization()
    started = time.perf_counter()
    status_code = solver.Solve()
    runtime = time.perf_counter() - started
    status = STATUS_NAMES.get(status_code, f"UNKNOWN_{status_code}")
    if status_code not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"MILP allocation failed with status {status}")

    stores["milp_inventory_14d"] = [
        allocations[(int(row.item_id), str(row.store_code))].solution_value()
        / quantity_scale
        for row in stores.itertuples(index=False)
    ]
    stores["allocation_delta"] = (
        stores["milp_inventory_14d"] - stores["target_inventory_14d"]
    )
    stores["national_target_inventory_14d"] = stores["item_id"].map(
        national_target
    )
    stores["rounded_national_target_inventory_14d"] = (
        stores["national_target_inventory_14d"] * quantity_scale
    ).round() / quantity_scale
    stores["allocated_sum_14d"] = stores.groupby("item_id")[
        "milp_inventory_14d"
    ].transform("sum")
    stores["reconciliation_gap"] = (
        stores["allocated_sum_14d"]
        - stores["rounded_national_target_inventory_14d"]
    )

    objective_value = float(objective.Value())
    best_bound = float(objective.BestBound())
    relative_gap = abs(objective_value - best_bound) / max(
        abs(objective_value),
        1.0,
    )
    capacity_usage = {
        store: float(
            stores.loc[stores["store_code"].eq(store), "milp_inventory_14d"].sum()
        )
        for store in capacities
    }
    summary: dict[str, object] = {
        "solver": f"OR-Tools MPSolver/{backend}",
        "status": status,
        "runtime_seconds": runtime,
        "relative_gap": relative_gap,
        "objective_value_scaled": objective_value,
        "best_bound_scaled": best_bound,
        "items": int(stores["item_id"].nunique()),
        "store_rows": int(len(stores)),
        "integer_variables": len(allocations),
        "continuous_deviation_variables": 2 * len(allocations),
        "quantity_resolution": 1 / quantity_scale,
        "max_absolute_reconciliation_gap": float(
            stores["reconciliation_gap"].abs().max()
        ),
        "store_capacities": capacities,
        "store_capacity_usage": capacity_usage,
    }
    return stores, summary


def _write_report(config: PipelineConfig, summary: dict[str, object]) -> None:
    report = f"""# 多仓库存分配 MILP 报告

- 求解器：{summary["solver"]}
- 状态：{summary["status"]}
- 商品数：{summary["items"]:,}
- 整数分配变量：{summary["integer_variables"]:,}
- 连续偏差变量：{summary["continuous_deviation_variables"]:,}
- 相对 gap：{summary["relative_gap"]:.6f}
- 最大全国—分仓加总误差：{summary["max_absolute_reconciliation_gap"]:.6f}
- 求解耗时：{summary["runtime_seconds"]:.3f} 秒

该模型使用整数库存分配变量和连续补少/补多偏差变量，目标函数与确定性
CP-SAT 协调模型保持一致。仓容约束仅在调用方提供容量时启用；天池公开数据
没有仓容字段，因此默认全量实验不虚构容量。

状态为 FEASIBLE 时必须连同 gap 报告，不能写成已证明全局最优。
"""
    (config.root / "reports" / "milp_allocation_report.md").write_text(
        report,
        encoding="utf-8",
    )


def run_milp_allocation(config_path: str | Path) -> None:
    config = load_config(config_path)
    connection = duckdb.connect()
    recommendations = connection.execute(
        "SELECT * FROM read_parquet(?)",
        [str(config.outputs["inventory_recommendations"])],
    ).fetchdf()
    settings = config.allocation
    allocation, summary = solve_coordinated_allocation_milp(
        recommendations,
        quantity_scale=int(settings.get("quantity_scale", 10)),
        cost_scale=int(settings.get("cost_scale", 100)),
        time_limit_seconds=float(settings.get("time_limit_seconds", 30)),
        workers=int(settings.get("workers", 1)),
    )
    output = config.outputs["coordinated_allocation_milp"]
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.register("milp_allocation_df", allocation)
    connection.execute(
        "COPY milp_allocation_df TO ? (FORMAT PARQUET)",
        [str(output)],
    )
    connection.close()
    allocation.head(200).to_csv(
        config.root / "reports" / "tables" / "milp_allocation_preview.csv",
        index=False,
    )
    (config.root / "reports" / "milp_allocation_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(config, summary)
