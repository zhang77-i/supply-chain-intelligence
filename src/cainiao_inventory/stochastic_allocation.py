"""Scenario-based stochastic allocation of national inventory to five stores."""

from __future__ import annotations

# Import CP-SAT before pandas on Windows to avoid protobuf DLL conflicts in
# some Anaconda environments.
from ortools.sat.python import cp_model

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .allocation import STORE_CODES, solve_coordinated_allocation
from .config import PipelineConfig, load_config


def inventory_cost_components(
    actual: np.ndarray,
    target: np.ndarray,
    shortage_cost: np.ndarray,
    overage_cost: np.ndarray,
) -> dict[str, float]:
    demand = np.asarray(actual, dtype=np.float64)
    plan = np.clip(np.asarray(target, dtype=np.float64), 0.0, None)
    shortage_units = np.maximum(demand - plan, 0.0)
    overage_units = np.maximum(plan - demand, 0.0)
    shortage = shortage_units * np.asarray(
        shortage_cost,
        dtype=np.float64,
    )
    overage = overage_units * np.asarray(
        overage_cost,
        dtype=np.float64,
    )
    return {
        "shortage_units": float(shortage_units.sum()),
        "overage_units": float(overage_units.sum()),
        "shortage_cost": float(shortage.sum()),
        "overage_cost": float(overage.sum()),
        "total_cost": float(shortage.sum() + overage.sum()),
    }


def build_residual_scenarios(
    historical: pd.DataFrame,
    current_stores: pd.DataFrame,
    *,
    scenario_count: int,
    random_seed: int,
) -> dict[tuple[int, str], np.ndarray]:
    """Bootstrap correlated five-store residual vectors from earlier folds."""
    history = historical[
        historical["store_code"].astype(str).isin(STORE_CODES)
    ].copy()
    history["residual"] = history["target_14d"] - history["lgbm_point"]
    residual_matrix = history.pivot_table(
        index=["fold_id", "cutoff_date", "item_id"],
        columns="store_code",
        values="residual",
        aggfunc="first",
    ).reindex(columns=STORE_CODES)
    residual_matrix = residual_matrix.dropna()
    if residual_matrix.empty:
        raise ValueError("No complete historical five-store residual vectors")

    # Winsorization prevents a single long-tail SKU from dominating every
    # scenario while retaining asymmetric and correlated forecast errors.
    lower = residual_matrix.quantile(0.01)
    upper = residual_matrix.quantile(0.99)
    residual_matrix = residual_matrix.clip(lower=lower, upper=upper, axis=1)
    residual_values = residual_matrix.to_numpy(dtype=np.float64)

    points = current_stores.pivot(
        index="item_id",
        columns="store_code",
        values="lgbm_point",
    ).reindex(columns=STORE_CODES)
    rng = np.random.default_rng(random_seed)
    scenarios: dict[tuple[int, str], np.ndarray] = {}
    for item_id, point_row in points.iterrows():
        selected = residual_values[
            rng.integers(0, len(residual_values), size=scenario_count)
        ]
        demand = np.maximum(
            point_row.to_numpy(dtype=np.float64)[None, :] + selected,
            0.0,
        )
        for store_index, store_code in enumerate(STORE_CODES):
            scenarios[(int(item_id), store_code)] = demand[:, store_index]
    return scenarios


def solve_stochastic_allocation(
    national_targets: pd.Series,
    stores: pd.DataFrame,
    scenarios: dict[tuple[int, str], np.ndarray],
    *,
    quantity_scale: int = 10,
    cost_scale: int = 100,
    time_limit_seconds: float = 45.0,
    workers: int = 4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Minimize expected asymmetric inventory cost under a total constraint."""
    data = stores.copy()
    data["store_code"] = data["store_code"].astype(str)
    data = data.sort_values(["item_id", "store_code"]).reset_index(drop=True)
    model = cp_model.CpModel()
    allocation: dict[tuple[int, str], cp_model.IntVar] = {}
    objective_terms = []
    scenario_count = len(next(iter(scenarios.values())))

    for item_id, group in data.groupby("item_id", sort=True):
        total = max(
            0,
            int(round(float(national_targets.loc[item_id]) * quantity_scale)),
        )
        item_variables = []
        for row in group.itertuples(index=False):
            key = (int(item_id), str(row.store_code))
            demand_values = np.maximum(
                np.rint(scenarios[key] * quantity_scale).astype(int),
                0,
            )
            maximum = max(total, int(demand_values.max(initial=0)))
            variable = model.new_int_var(0, total, f"x_{key[0]}_{key[1]}")
            allocation[key] = variable
            item_variables.append(variable)
            shortage_weight = max(
                1,
                int(round(float(row.shortage_cost) * cost_scale)),
            )
            overage_weight = max(
                1,
                int(round(float(row.overage_cost) * cost_scale)),
            )
            for scenario_index, demand in enumerate(demand_values):
                under = model.new_int_var(
                    0,
                    maximum,
                    f"u_{key[0]}_{key[1]}_{scenario_index}",
                )
                over = model.new_int_var(
                    0,
                    maximum,
                    f"o_{key[0]}_{key[1]}_{scenario_index}",
                )
                model.add(int(demand) - variable == under - over)
                objective_terms.extend(
                    [shortage_weight * under, overage_weight * over]
                )
        model.add(sum(item_variables) == total)

    model.minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = 42

    started = time.perf_counter()
    status = solver.solve(model)
    runtime = time.perf_counter() - started
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"Stochastic allocation failed: {solver.status_name(status)}"
        )

    data["stochastic_inventory_14d"] = [
        solver.value(allocation[(int(row.item_id), str(row.store_code))])
        / quantity_scale
        for row in data.itertuples(index=False)
    ]
    data["national_target_inventory_14d"] = data["item_id"].map(
        national_targets
    )
    data["allocated_sum_14d"] = data.groupby("item_id")[
        "stochastic_inventory_14d"
    ].transform("sum")
    rounded_total = (
        data["national_target_inventory_14d"] * quantity_scale
    ).round() / quantity_scale
    data["reconciliation_gap"] = data["allocated_sum_14d"] - rounded_total
    summary: dict[str, object] = {
        "status": solver.status_name(status),
        "runtime_seconds": runtime,
        "relative_gap": (
            abs(solver.objective_value - solver.best_objective_bound)
            / max(abs(solver.objective_value), 1.0)
        ),
        "items": int(data["item_id"].nunique()),
        "rows": int(len(data)),
        "scenario_count": scenario_count,
        "max_absolute_reconciliation_gap": float(
            data["reconciliation_gap"].abs().max()
        ),
    }
    return data, summary


def _complete_validation_fold(
    predictions: pd.DataFrame,
    fold_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = predictions[predictions["fold_id"].eq(fold_id)].copy()
    stores = current[current["store_code"].astype(str).isin(STORE_CODES)].copy()
    complete_items = stores.groupby("item_id")["store_code"].nunique()
    complete_items = complete_items[complete_items.eq(len(STORE_CODES))].index
    national = current[
        current["store_code"].astype(str).eq("all")
        & current["item_id"].isin(complete_items)
    ].copy()
    items = np.intersect1d(complete_items, national["item_id"].unique())
    return (
        national[national["item_id"].isin(items)].copy(),
        stores[stores["item_id"].isin(items)].copy(),
    )


def run_stochastic_backtest(config_path: str | Path) -> None:
    config = load_config(config_path)
    connection = duckdb.connect()
    predictions = connection.execute(
        "SELECT * FROM read_parquet(?)",
        [str(config.outputs["model_predictions"])],
    ).fetchdf()
    predictions["store_code"] = predictions["store_code"].astype(str)
    fold_ids = sorted(predictions["fold_id"].unique())
    settings = config.allocation
    scenario_count = int(settings.get("scenario_count", 7))
    decision_frames = []
    metric_rows = []
    solver_rows = []

    # Fold 1 has no earlier residuals, so folds 2..N form a strictly
    # leakage-safe evaluation.
    for fold_index, fold_id in enumerate(fold_ids[1:], start=1):
        historical = predictions[
            predictions["fold_id"].isin(fold_ids[:fold_index])
        ]
        national, stores = _complete_validation_fold(predictions, fold_id)
        national_targets = national.set_index("item_id")["lgbm_newsvendor"]

        deterministic_input = pd.concat(
            [
                national.rename(
                    columns={"lgbm_newsvendor": "target_inventory_14d"}
                ),
                stores.rename(
                    columns={"lgbm_newsvendor": "target_inventory_14d"}
                ),
            ],
            ignore_index=True,
        )
        deterministic, deterministic_summary = solve_coordinated_allocation(
            deterministic_input,
            quantity_scale=int(settings.get("quantity_scale", 10)),
            cost_scale=int(settings.get("cost_scale", 100)),
            time_limit_seconds=float(settings.get("time_limit_seconds", 30)),
            workers=int(settings.get("workers", 4)),
        )
        scenarios = build_residual_scenarios(
            historical,
            stores,
            scenario_count=scenario_count,
            random_seed=config.random_seed + fold_index,
        )
        stochastic, stochastic_summary = solve_stochastic_allocation(
            national_targets,
            stores,
            scenarios,
            quantity_scale=int(settings.get("quantity_scale", 10)),
            cost_scale=int(settings.get("cost_scale", 100)),
            time_limit_seconds=float(
                settings.get("stochastic_time_limit_seconds", 45)
            ),
            workers=int(settings.get("workers", 4)),
        )

        decisions = stores[
            [
                "fold_id",
                "cutoff_date",
                "item_id",
                "store_code",
                "target_14d",
                "shortage_cost",
                "overage_cost",
                "lgbm_newsvendor",
            ]
        ].merge(
            deterministic[
                ["item_id", "store_code", "coordinated_inventory_14d"]
            ],
            on=["item_id", "store_code"],
            how="left",
        ).merge(
            stochastic[
                ["item_id", "store_code", "stochastic_inventory_14d"]
            ],
            on=["item_id", "store_code"],
            how="left",
        )
        decision_frames.append(decisions)

        for method, column in (
            ("independent_newsvendor", "lgbm_newsvendor"),
            ("deterministic_coordinated", "coordinated_inventory_14d"),
            ("stochastic_coordinated", "stochastic_inventory_14d"),
        ):
            costs = inventory_cost_components(
                decisions["target_14d"].to_numpy(),
                decisions[column].to_numpy(),
                decisions["shortage_cost"].to_numpy(),
                decisions["overage_cost"].to_numpy(),
            )
            metric_rows.append(
                {
                    "fold_id": fold_id,
                    "method": method,
                    **costs,
                    "planned_units": float(decisions[column].sum()),
                }
            )
        solver_rows.append(
            {
                "fold_id": fold_id,
                "deterministic_status": deterministic_summary["status"],
                "deterministic_runtime_seconds": deterministic_summary[
                    "runtime_seconds"
                ],
                "stochastic_status": stochastic_summary["status"],
                "stochastic_runtime_seconds": stochastic_summary[
                    "runtime_seconds"
                ],
                "stochastic_relative_gap": stochastic_summary["relative_gap"],
                "items": stochastic_summary["items"],
                "scenario_count": stochastic_summary["scenario_count"],
            }
        )

    decisions = pd.concat(decision_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    solver_metrics = pd.DataFrame(solver_rows)
    output = config.outputs["stochastic_allocation"]
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.register("stochastic_allocation_df", decisions)
    connection.execute(
        "COPY stochastic_allocation_df TO ? (FORMAT PARQUET)",
        [str(output)],
    )
    connection.close()
    table_dir = config.root / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(table_dir / "stochastic_allocation_metrics.csv", index=False)
    solver_metrics.to_csv(
        table_dir / "stochastic_allocation_solver_metrics.csv",
        index=False,
    )

    average = metrics.groupby("method", as_index=False)["total_cost"].mean()
    cost_map = average.set_index("method")["total_cost"]
    improvement = (
        1.0
        - cost_map["stochastic_coordinated"]
        / cost_map["deterministic_coordinated"]
    )
    report = f"""# 场景驱动的随机多仓库存分配回测

## 方法

对每个验证折只使用更早折的 `实际需求 - LightGBM点预测` 残差，按完整
5仓残差向量进行自助抽样，从而保留区域误差的相关性。每个商品生成
{scenario_count} 个未来需求场景；在全国 Newsvendor 库存总量约束下，使用
OR-Tools CP-SAT 最小化各仓场景平均补少/补多成本。

第一折没有更早残差，故严格时间回测使用后续 {len(fold_ids) - 1} 折。

## 平均成本

{average.to_markdown(index=False)}

随机场景分配相对确定性协调的平均成本变化：
`{improvement:+.2%}`（正值表示成本下降）。

## 求解状态

{solver_metrics.to_markdown(index=False)}

## 边界

- 需求场景来自历史滚动回测残差，不使用当前折未来信息；
- 场景对跨仓残差相关性做经验保留，但不是完整概率分布校准；
- 全国目标库存仍由公开成本下的 Newsvendor 决策给出；
- 未虚构仓容、预算、调拨成本或平台上线收益；
- `FEASIBLE` 结果会同时报告相对 gap，不冒充已证明全局最优。
"""
    (config.root / "reports" / "stochastic_allocation_report.md").write_text(
        report,
        encoding="utf-8",
    )
    metadata = {
        "evaluated_folds": fold_ids[1:],
        "scenario_count": scenario_count,
        "average_total_cost": cost_map.to_dict(),
        "stochastic_vs_deterministic_cost_reduction": float(improvement),
    }
    (config.root / "reports" / "stochastic_allocation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
