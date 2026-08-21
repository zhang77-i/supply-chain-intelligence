from __future__ import annotations

import json
import platform
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .db import connect


def _one_row(connection: duckdb.DuckDBPyConnection, sql: str) -> dict:
    frame = connection.execute(sql).fetch_df()
    return frame.iloc[0].to_dict()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无记录_"
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else str(value)
        )
    headers = "| " + " | ".join(display.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in display.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([headers, rule, *rows])


def collect_audit(connection: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    overview = connection.execute(
        """
        SELECT '全国商品日表' AS dataset, COUNT(*) AS rows,
               COUNT(DISTINCT item_id) AS items, 1 AS locations,
               MIN(date) AS min_date, MAX(date) AS max_date
        FROM clean_item_feature
        UNION ALL
        SELECT '商品-分仓日表', COUNT(*), COUNT(DISTINCT item_id),
               COUNT(DISTINCT store_code), MIN(date), MAX(date)
        FROM clean_item_store_feature
        UNION ALL
        SELECT '成本配置表', COUNT(*), COUNT(DISTINCT item_id),
               COUNT(DISTINCT store_code), NULL, NULL
        FROM clean_config
        UNION ALL
        SELECT '完整日面板', COUNT(*), COUNT(DISTINCT item_id),
               COUNT(DISTINCT store_code), MIN(date), MAX(date)
        FROM daily_panel
        """
    ).fetch_df()

    quality = connection.execute(
        """
        SELECT '全国表重复主键' AS check_name, COUNT(*) AS issue_count
        FROM (
            SELECT item_id, date FROM raw_item_feature
            GROUP BY item_id, date HAVING COUNT(*) > 1
        )
        UNION ALL
        SELECT '分仓表重复主键', COUNT(*)
        FROM (
            SELECT item_id, store_code, date FROM raw_item_store_feature
            GROUP BY item_id, store_code, date HAVING COUNT(*) > 1
        )
        UNION ALL
        SELECT '配置表重复主键', COUNT(*)
        FROM (
            SELECT item_id, store_code FROM raw_config
            GROUP BY item_id, store_code HAVING COUNT(*) > 1
        )
        UNION ALL
        SELECT '全国负需求记录', COUNT(*) FROM clean_item_feature
        WHERE qty_alipay_njhs < 0
        UNION ALL
        SELECT '分仓负需求记录', COUNT(*) FROM clean_item_store_feature
        WHERE qty_alipay_njhs < 0
        UNION ALL
        SELECT '无效成本对', COUNT(*) FROM clean_config
        WHERE NOT valid_cost_pair
        """
    ).fetch_df()

    demand = connection.execute(
        """
        SELECT '全国原始记录' AS scope,
               COUNT(*) AS rows,
               ROUND(AVG(CASE WHEN qty_alipay_njhs = 0 THEN 1 ELSE 0 END), 4) AS zero_rate,
               ROUND(AVG(qty_alipay_njhs), 4) AS mean_demand,
               MAX(qty_alipay_njhs) AS max_demand
        FROM clean_item_feature
        UNION ALL
        SELECT '区域仓原始记录', COUNT(*),
               ROUND(AVG(CASE WHEN qty_alipay_njhs = 0 THEN 1 ELSE 0 END), 4),
               ROUND(AVG(qty_alipay_njhs), 4), MAX(qty_alipay_njhs)
        FROM clean_item_store_feature
        UNION ALL
        SELECT '完整日面板', COUNT(*),
               ROUND(AVG(CASE WHEN demand = 0 THEN 1 ELSE 0 END), 4),
               ROUND(AVG(demand), 4), MAX(demand)
        FROM daily_panel
        """
    ).fetch_df()

    panel = connection.execute(
        """
        SELECT
            COUNT(*) AS panel_rows,
            SUM(CASE WHEN is_observed THEN 1 ELSE 0 END) AS observed_rows,
            ROUND(AVG(CASE WHEN is_observed THEN 0 ELSE 1 END), 4) AS imputed_zero_rate,
            SUM(CASE WHEN is_after_last_observed THEN 1 ELSE 0 END) AS after_last_observed_rows
        FROM daily_panel
        """
    ).fetch_df()

    costs = connection.execute(
        """
        SELECT
            ROUND(MIN(critical_fractile), 4) AS min_alpha,
            ROUND(quantile_cont(critical_fractile, 0.25), 4) AS p25_alpha,
            ROUND(quantile_cont(critical_fractile, 0.50), 4) AS median_alpha,
            ROUND(quantile_cont(critical_fractile, 0.75), 4) AS p75_alpha,
            ROUND(MAX(critical_fractile), 4) AS max_alpha
        FROM clean_config
        WHERE valid_cost_pair
        """
    ).fetch_df()

    coverage = connection.execute(
        """
        SELECT
            COUNT(*) AS configured_series,
            SUM(CASE WHEN has_demand_history THEN 1 ELSE 0 END) AS observed_series,
            SUM(CASE WHEN NOT has_demand_history THEN 1 ELSE 0 END) AS no_history_series,
            ROUND(AVG(CASE WHEN has_demand_history THEN 1 ELSE 0 END), 4) AS history_coverage
        FROM config_coverage
        """
    ).fetch_df()

    consistency = connection.execute(
        """
        SELECT
            COUNT(*) AS item_days,
            ROUND(AVG(CASE WHEN demand_diff = 0 THEN 1 ELSE 0 END), 4) AS exact_match_rate,
            ROUND(AVG(ABS(demand_diff)), 4) AS mean_abs_diff,
            MAX(ABS(demand_diff)) AS max_abs_diff
        FROM national_store_daily_consistency
        """
    ).fetch_df()

    submission = connection.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM raw_sample_submission) AS submission_rows,
            (SELECT COUNT(DISTINCT item_id) FROM raw_sample_submission) AS submission_items,
            (
                SELECT COUNT(DISTINCT s.item_id)
                FROM raw_sample_submission s
                INNER JOIN dim_item i USING (item_id)
            ) AS overlapping_items
        """
    ).fetch_df()

    return {
        "overview": overview,
        "quality": quality,
        "demand": demand,
        "panel": panel,
        "costs": costs,
        "coverage": coverage,
        "consistency": consistency,
        "submission": submission,
    }


def write_audit_report(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    with connect(config, read_only=True) as connection:
        audit = collect_audit(connection)

    tables_dir = config.outputs["audit_tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in audit.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)

    overlap = int(audit["submission"].iloc[0]["overlapping_items"])
    submission_note = (
        "样例提交文件与 Part II 商品 ID 有交集，可用于格式核对。"
        if overlap > 0
        else "样例提交文件与 Part II 商品 ID 无交集，判定为不同赛季/版本；本项目不将其用于训练或回测。"
    )
    report = f"""# 菜鸟 Part II 初步数据审计报告

生成时间：{datetime.now().astimezone().isoformat(timespec="seconds")}

## 1. 数据资产概览

{_markdown_table(audit["overview"])}

原始三张 CSV 均无表头；导入层按照赛题字段说明显式定义了全国表 31 列、分仓表 32 列和成本表 3 列，未把首行误识别为字段名。

## 2. 数据质量检查

{_markdown_table(audit["quality"])}

## 3. 需求稀疏性

{_markdown_table(audit["demand"])}

区域仓需求比全国需求更稀疏。建模阶段会同时保留移动平均/Last-14 基线和适用于间歇性需求的需求类型标签，后续再比较 Croston/TSB 与分位数 LightGBM。

## 4. 日期补全面板

{_markdown_table(audit["panel"])}

当前面板规则：从每个“商品—仓”序列第一次出现之日起补齐到全局数据结束日；缺失行为量与需求暂记为 0，同时保留 `is_observed` 和 `is_after_last_observed` 标记。该假设会在模型消融中单独验证，不会静默丢弃原始缺失信息。

## 5. 补少/补多成本与临界分位

{_markdown_table(audit["costs"])}

每个库存单元使用 `alpha = A / (A + B)` 计算 Newsvendor 临界分位。不同库存单元的风险偏好存在显著差异，因此后续核心评估以成本而非单一 MAE 为准。

## 6. 成本配置与需求历史覆盖

{_markdown_table(audit["coverage"])}

配置表覆盖 963 个商品的“全国 + 5 区域仓”共 5,778 个库存单元。其中无历史需求的库存单元不会被伪造训练样本；其冷启动策略将在模型阶段单独处理。

## 7. 全国与分仓一致性

{_markdown_table(audit["consistency"])}

全国数据和 5 个区域仓数据是两个评测层级，不强制假设全国需求恒等于区域仓需求之和；差异作为层级协调阶段的输入，而不是在清洗时被覆盖。

## 8. 样例提交版本核验

{_markdown_table(audit["submission"])}

{submission_note}

## 9. 已完成的数据产物

- DuckDB 数据库：`data/interim/cainiao.duckdb`
- 滚动回测样本：`data/processed/backtest_samples.parquet`
- 最终日期推理快照：`data/processed/inference_snapshot.parquet`
- 回测折叠定义：`data/processed/backtests/fold_manifest.csv`
- 折叠样本映射：`data/processed/backtests/fold_assignments.parquet`

## 10. 下一阶段

1. 建立 Last-14、MA28、指数平滑、Croston/TSB 基线。
2. 训练点预测 LightGBM 与多分位 LightGBM。
3. 用 `A/(A+B)` 将需求分布转换为成本敏感目标库存。
4. 做全国—5 区域仓层级协调与随机库存分配。
5. 输出 MAE/WAPE、缺货成本、过量成本、总成本和分仓一致性等业务指标。
"""
    config.outputs["audit_report"].parent.mkdir(parents=True, exist_ok=True)
    config.outputs["audit_report"].write_text(report, encoding="utf-8")
    return audit


def write_run_metadata(
    config: PipelineConfig,
    samples: pd.DataFrame,
    fold_manifest: pd.DataFrame,
) -> None:
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "feature_rows": int(len(samples)),
        "labeled_rows": int(samples["cutoff_kind"].eq("backtest").sum()),
        "inference_rows": int(samples["cutoff_kind"].eq("inference").sum()),
        "backtest_folds": int(len(fold_manifest)),
        "random_seed": config.random_seed,
    }
    config.outputs["run_metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
