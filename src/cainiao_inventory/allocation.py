from __future__ import annotations

# Import CP-SAT before pandas on Windows.  This avoids a known protobuf DLL
# loading conflict in some Anaconda base environments.
from ortools.sat.python import cp_model

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import PipelineConfig, load_config


STORE_CODES = ("1", "2", "3", "4", "5")


def _complete_store_grid(
    recommendations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = recommendations.copy()
    data["store_code"] = data["store_code"].astype(str)
    national = data[data["store_code"] == "all"].copy()
    stores = data[data["store_code"].isin(STORE_CODES)].copy()

    grid = pd.MultiIndex.from_product(
        [national["item_id"].sort_values().unique(), STORE_CODES],
        names=["item_id", "store_code"],
    ).to_frame(index=False)
    stores = grid.merge(
        stores[
            [
                "item_id",
                "store_code",
                "shortage_cost",
                "overage_cost",
                "target_inventory_14d",
            ]
        ],
        on=["item_id", "store_code"],
        how="left",
    )
    stores["cold_start_fallback"] = stores["target_inventory_14d"].isna()
    stores["target_inventory_14d"] = stores["target_inventory_14d"].fillna(0.0)

    # Costs only weight reconciliation. Missing feature rows are rare; use the
    # observed median so they do not dominate the objective.
    stores["shortage_cost"] = stores["shortage_cost"].fillna(
        stores["shortage_cost"].median()
    )
    stores["overage_cost"] = stores["overage_cost"].fillna(
        stores["overage_cost"].median()
    )
    return national, stores


def solve_coordinated_allocation(
    recommendations: pd.DataFrame,
    *,
    quantity_scale: int = 10,
    cost_scale: int = 100,
    time_limit_seconds: float = 30.0,
    workers: int = 4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    national, stores = _complete_store_grid(recommendations)
    national_target = national.set_index("item_id")["target_inventory_14d"]

    model = cp_model.CpModel()
    variables: dict[tuple[int, str], cp_model.IntVar] = {}
    objective_terms = []

    for item_id, group in stores.groupby("item_id", sort=True):
        total = max(0, int(round(float(national_target.loc[item_id]) * quantity_scale)))
        item_vars = []
        for row in group.itertuples(index=False):
            key = (int(item_id), str(row.store_code))
            independent = max(
                0,
                int(round(float(row.target_inventory_14d) * quantity_scale)),
            )
            upper = max(total, independent)
            allocated = model.new_int_var(0, total, f"x_{item_id}_{row.store_code}")
            under = model.new_int_var(0, upper, f"under_{item_id}_{row.store_code}")
            over = model.new_int_var(0, upper, f"over_{item_id}_{row.store_code}")
            model.add(independent - allocated == under - over)
            shortage_weight = max(1, int(round(float(row.shortage_cost) * cost_scale)))
            overage_weight = max(1, int(round(float(row.overage_cost) * cost_scale)))
            objective_terms.extend(
                [shortage_weight * under, overage_weight * over]
            )
            variables[key] = allocated
            item_vars.append(allocated)
        model.add(sum(item_vars) == total)

    model.minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = 42

    started = time.perf_counter()
    status = solver.solve(model)
    runtime = time.perf_counter() - started
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT allocation failed with status {solver.status_name(status)}")

    stores["coordinated_inventory_14d"] = [
        solver.value(variables[(int(row.item_id), str(row.store_code))])
        / quantity_scale
        for row in stores.itertuples(index=False)
    ]
    stores["allocation_delta"] = (
        stores["coordinated_inventory_14d"] - stores["target_inventory_14d"]
    )
    stores["national_target_inventory_14d"] = stores["item_id"].map(
        national_target
    )
    stores["rounded_national_target_inventory_14d"] = (
        stores["national_target_inventory_14d"] * quantity_scale
    ).round() / quantity_scale
    stores["allocated_sum_14d"] = stores.groupby("item_id")[
        "coordinated_inventory_14d"
    ].transform("sum")
    stores["reconciliation_gap"] = (
        stores["allocated_sum_14d"]
        - stores["rounded_national_target_inventory_14d"]
    )
    stores["national_quantization_error"] = (
        stores["rounded_national_target_inventory_14d"]
        - stores["national_target_inventory_14d"]
    )

    summary: dict[str, object] = {
        "solver": "OR-Tools CP-SAT",
        "status": solver.status_name(status),
        "runtime_seconds": runtime,
        "items": int(stores["item_id"].nunique()),
        "store_rows": int(len(stores)),
        "cold_start_rows": int(stores["cold_start_fallback"].sum()),
        "quantity_resolution": 1 / quantity_scale,
        "max_absolute_reconciliation_gap": float(
            stores["reconciliation_gap"].abs().max()
        ),
        "max_absolute_quantization_error": float(
            stores["national_quantization_error"].abs().max()
        ),
        "mean_absolute_adjustment": float(stores["allocation_delta"].abs().mean()),
        "objective_value_scaled": float(solver.objective_value),
    }
    return stores, summary


def _write_report(config: PipelineConfig, summary: dict[str, object]) -> None:
    report = f"""# 全国—5区域仓库存协调报告

## 运行结论

- 求解器：{summary["solver"]}
- 状态：{summary["status"]}
- 商品数：{summary["items"]:,}
- 商品—仓分配行数：{summary["store_rows"]:,}
- 冷启动降级行数：{summary["cold_start_rows"]:,}
- 数量精度：{summary["quantity_resolution"]}
- 最大全国—分仓加总误差：{summary["max_absolute_reconciliation_gap"]:.6f}
- 最大总量离散化误差：{summary["max_absolute_quantization_error"]:.6f}
- 平均绝对调整量：{summary["mean_absolute_adjustment"]:.4f}
- 求解耗时：{summary["runtime_seconds"]:.3f} 秒

## 模型说明

以全国 Newsvendor 目标库存作为每个商品的总量约束，将库存分配到 5 个区域仓。
目标函数按真实补少/补多成本，加权惩罚协调后库存相对各仓独立 Newsvendor
建议的向下和向上偏离。该阶段使用公开数据中的真实成本，不虚构仓容、预算或
调拨成本。

CP-SAT按0.1件精度建模，各仓协调量对离散化后的全国目标严格加总一致；总量
离散化误差单独报告，不与模型约束误差混淆。

缺少推理特征的商品—仓组合采用零需求冷启动降级并显式标记。该做法保证输出
覆盖完整 5 仓网格，但冷启动行不应被解释为已获得充分历史证据的预测结果。
"""
    (config.root / "reports" / "multiwarehouse_allocation_report.md").write_text(
        report,
        encoding="utf-8",
    )


def run_allocation(config_path: str | Path) -> None:
    config = load_config(config_path)
    connection = duckdb.connect()
    recommendations = connection.execute(
        "SELECT * FROM read_parquet(?)",
        [str(config.outputs["inventory_recommendations"])],
    ).fetchdf()
    settings = config.allocation
    allocation, summary = solve_coordinated_allocation(
        recommendations,
        quantity_scale=int(settings.get("quantity_scale", 10)),
        cost_scale=int(settings.get("cost_scale", 100)),
        time_limit_seconds=float(settings.get("time_limit_seconds", 30)),
        workers=int(settings.get("workers", 4)),
    )
    output = config.outputs["coordinated_allocation"]
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.register("coordinated_allocation_df", allocation)
    connection.execute(
        "COPY coordinated_allocation_df TO ? (FORMAT PARQUET)",
        [str(output)],
    )
    connection.close()
    allocation.head(200).to_csv(
        config.root / "reports" / "tables" / "coordinated_allocation_preview.csv",
        index=False,
    )
    (config.root / "reports" / "allocation_run_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(config, summary)
