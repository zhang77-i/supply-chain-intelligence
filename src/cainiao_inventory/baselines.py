from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .db import connect

LOGGER = logging.getLogger(__name__)


def croston_sba(demand: np.ndarray, alpha: float = 0.1) -> float:
    values = np.asarray(demand, dtype=np.float64)
    nonzero = np.flatnonzero(values > 0)
    if len(nonzero) == 0:
        return 0.0
    first = int(nonzero[0])
    size = float(values[first])
    interval = float(first + 1)
    last_nonzero = first
    for index in nonzero[1:]:
        gap = float(index - last_nonzero)
        size = alpha * float(values[index]) + (1.0 - alpha) * size
        interval = alpha * gap + (1.0 - alpha) * interval
        last_nonzero = int(index)
    return max((1.0 - alpha / 2.0) * size / max(interval, 1e-12), 0.0)


def tsb_forecast(
    demand: np.ndarray,
    demand_alpha: float = 0.1,
    probability_alpha: float = 0.1,
) -> float:
    values = np.asarray(demand, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    nonzero = values > 0
    first = int(np.argmax(nonzero)) if nonzero.any() else -1
    if first < 0:
        return 0.0
    size = float(values[first])
    probability = 1.0
    for value in values[first + 1 :]:
        occurrence = 1.0 if value > 0 else 0.0
        probability = (
            probability_alpha * occurrence
            + (1.0 - probability_alpha) * probability
        )
        if value > 0:
            size = demand_alpha * float(value) + (1.0 - demand_alpha) * size
    return max(probability * size, 0.0)


def compute_intermittent_baselines(
    samples: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    output = config.outputs["intermittent_baselines"]
    if output.exists():
        cached = pd.read_parquet(output)
        if set(cached["sample_id"]) == set(samples["sample_id"]):
            return cached

    with connect(config, read_only=True) as connection:
        daily = connection.execute(
            """
            SELECT date, item_id, store_code, demand
            FROM daily_panel
            ORDER BY item_id, store_code, date
            """
        ).fetch_df()

    horizon = config.backtest["horizon_days"]
    croston_alpha = float(config.modeling["croston_alpha"])
    tsb_demand_alpha = float(config.modeling["tsb_demand_alpha"])
    tsb_probability_alpha = float(config.modeling["tsb_probability_alpha"])
    sample_groups = {
        key: group.sort_values("cutoff_date")
        for key, group in samples.groupby(["item_id", "store_code"], observed=True)
    }

    rows: list[dict] = []
    grouped = daily.groupby(["item_id", "store_code"], observed=True, sort=False)
    for group_number, (key, history) in enumerate(grouped, start=1):
        requested = sample_groups.get(key)
        if requested is None:
            continue
        dates = pd.to_datetime(history["date"]).to_numpy(dtype="datetime64[D]")
        demand = history["demand"].to_numpy(dtype=np.float64)
        for sample in requested.itertuples(index=False):
            cutoff = np.datetime64(pd.Timestamp(sample.cutoff_date).date())
            end = int(np.searchsorted(dates, cutoff, side="right"))
            observed = demand[:end]
            rows.append(
                {
                    "sample_id": int(sample.sample_id),
                    "baseline_croston_sba": horizon
                    * croston_sba(observed, alpha=croston_alpha),
                    "baseline_tsb": horizon
                    * tsb_forecast(
                        observed,
                        demand_alpha=tsb_demand_alpha,
                        probability_alpha=tsb_probability_alpha,
                    ),
                }
            )
        if group_number % 1000 == 0:
            LOGGER.info("Intermittent baselines: processed %s series", group_number)

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result
