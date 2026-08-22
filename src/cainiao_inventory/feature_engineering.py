from __future__ import annotations

from datetime import timedelta

import duckdb
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .db import connect, execute_sql_file


def safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    result = np.full(num.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=result, where=np.abs(den) > 1e-12)
    return result


def classify_demand(adi: np.ndarray, cv2: np.ndarray, nonzero_days: np.ndarray) -> np.ndarray:
    labels = np.full(adi.shape, "lumpy", dtype=object)
    labels[(adi < 1.32) & (cv2 < 0.49)] = "smooth"
    labels[(adi < 1.32) & (cv2 >= 0.49)] = "erratic"
    labels[(adi >= 1.32) & (cv2 < 0.49)] = "intermittent"
    labels[np.asarray(nonzero_days) == 0] = "all_zero"
    return labels


def build_cutoff_table(
    connection: duckdb.DuckDBPyConnection,
    config: PipelineConfig,
) -> pd.DataFrame:
    min_date, max_date = connection.execute(
        "SELECT MIN(date), MAX(date) FROM daily_panel"
    ).fetchone()
    history_days = config.backtest["history_days"]
    horizon_days = config.backtest["horizon_days"]
    step_days = config.backtest["cutoff_step_days"]

    first_cutoff = min_date + timedelta(days=history_days - 1)
    last_labeled_cutoff = max_date - timedelta(days=horizon_days)
    cutoffs: list = []
    cursor = last_labeled_cutoff
    while cursor >= first_cutoff:
        cutoffs.append(cursor)
        cursor -= timedelta(days=step_days)
    cutoffs.sort()

    frame = pd.DataFrame(
        {
            "cutoff_date": [*cutoffs, max_date],
            "cutoff_kind": ["backtest"] * len(cutoffs) + ["inference"],
        }
    )
    connection.register("_model_cutoffs_df", frame)
    connection.execute(
        """
        CREATE OR REPLACE TABLE model_cutoffs AS
        SELECT CAST(cutoff_date AS DATE) AS cutoff_date, cutoff_kind
        FROM _model_cutoffs_df
        ORDER BY cutoff_date
        """
    )
    connection.unregister("_model_cutoffs_df")
    return frame


def add_numpy_pandas_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["cart_rate_14"] = safe_divide(frame["cart_uv_sum_14"], frame["pv_uv_sum_14"])
    frame["pay_rate_14"] = safe_divide(
        frame["pay_uv_sum_14"], frame["pv_uv_sum_14"]
    )
    frame["gmv_to_pay_rate_14"] = safe_divide(
        frame["qty_alipay_sum_14"], frame["qty_gmv_sum_14"]
    )
    frame["demand_trend_7_28"] = (
        safe_divide(frame["demand_mean_7"], frame["demand_mean_28"]) - 1.0
    )
    frame["observed_rate_84"] = safe_divide(
        frame["observed_days_84"], frame["history_days_84"]
    )
    frame["zero_rate_84"] = 1.0 - safe_divide(
        frame["nonzero_days_84"], frame["history_days_84"]
    )

    adi = safe_divide(frame["history_days_84"], frame["nonzero_days_84"])
    nonzero_cv = safe_divide(frame["nonzero_demand_std_84"], frame["nonzero_demand_mean_84"])
    cv2 = np.square(nonzero_cv)
    frame["adi_84"] = adi
    frame["cv2_84"] = cv2
    frame["demand_pattern"] = classify_demand(
        adi,
        cv2,
        frame["nonzero_days_84"].to_numpy(),
    )

    horizon = 14.0
    frame["baseline_last14"] = frame["demand_sum_14"].clip(lower=0.0)
    frame["baseline_ma28"] = (horizon * frame["demand_mean_28"]).clip(lower=0.0)
    frame["baseline_weighted"] = (
        0.65 * frame["baseline_last14"] + 0.35 * frame["baseline_ma28"]
    )
    return frame


def _write_feature_summary(frame: pd.DataFrame, config: PipelineConfig) -> None:
    summary = (
        frame.groupby(["cutoff_kind", "cutoff_date"], observed=True)
        .agg(
            rows=("item_id", "size"),
            items=("item_id", "nunique"),
            series=("store_code", "size"),
            mean_target_14d=("target_14d", "mean"),
            zero_target_rate=("target_14d", lambda values: float((values == 0).mean())),
        )
        .reset_index()
    )
    target = config.outputs["audit_tables"] / "feature_cutoff_summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(target, index=False)


def build_feature_datasets(config: PipelineConfig) -> pd.DataFrame:
    with connect(config) as connection:
        build_cutoff_table(connection, config)
        execute_sql_file(
            connection,
            config.root / "sql" / "05_build_feature_snapshots.sql",
            config,
        )
        frame = connection.execute(
            "SELECT * FROM feature_snapshots_raw ORDER BY cutoff_date, item_id, store_code"
        ).fetch_df()

    frame = add_numpy_pandas_features(frame)
    frame.insert(0, "sample_id", np.arange(len(frame), dtype=np.int64))
    labeled = frame.loc[frame["cutoff_kind"].eq("backtest")].copy()
    inference = frame.loc[frame["cutoff_kind"].eq("inference")].copy()

    for output in (
        config.outputs["feature_dataset"],
        config.outputs["inference_dataset"],
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(config.outputs["feature_dataset"], index=False)
    inference.to_parquet(config.outputs["inference_dataset"], index=False)
    _write_feature_summary(frame, config)
    return frame
