#!/usr/bin/env python3
"""Run the optional leakage-safe PyTorch LSTM diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cainiao_inventory.config import load_config  # noqa: E402
from cainiao_inventory.db import connect  # noqa: E402
from cainiao_inventory.lstm_forecast import (  # noqa: E402
    build_sequences,
    regression_metrics,
    rolling_train_validation_split,
    train_lstm_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-fold LSTM forecast diagnostic without time leakage."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project.yaml",
    )
    return parser.parse_args()


def _format_metrics(metrics: dict[str, float]) -> str:
    return f"MAE={metrics['mae']:.4f}, WAPE={metrics['wape']:.4f}"


def main() -> None:
    config = load_config(parse_args().config)
    settings = config.deep_learning
    manifest = pd.read_csv(
        config.outputs["fold_manifest"], parse_dates=["validation_cutoff"]
    )
    validation_cutoff = pd.Timestamp(manifest["validation_cutoff"].max())
    max_series = int(settings.get("max_series", 200))
    if max_series <= 0:
        raise ValueError("deep_learning.max_series must be positive")

    with connect(config, read_only=True) as connection:
        cutoffs = connection.execute(
            """
            SELECT cutoff_date
            FROM model_cutoffs
            WHERE cutoff_kind = 'backtest' AND cutoff_date <= ?
            ORDER BY cutoff_date
            """,
            [validation_cutoff.date()],
        ).fetch_df()["cutoff_date"]
        daily = connection.execute(
            """
            WITH ranked AS (
                SELECT item_id
                FROM daily_panel
                WHERE store_code = 'all' AND date <= ?
                GROUP BY item_id
                ORDER BY SUM(demand) DESC, item_id
                LIMIT ?
            )
            SELECT date, item_id, store_code, demand
            FROM daily_panel
            WHERE store_code = 'all'
              AND item_id IN (SELECT item_id FROM ranked)
            ORDER BY item_id, date
            """,
            [validation_cutoff.date(), max_series],
        ).fetch_df()

    sequences = build_sequences(
        daily,
        list(pd.to_datetime(cutoffs)),
        lookback_days=int(settings.get("lookback_days", 56)),
        horizon_days=int(settings.get("horizon_days", 14)),
    )
    train, validation = rolling_train_validation_split(
        sequences, validation_cutoff
    )
    model, prediction, metadata = train_lstm_baseline(
        train,
        validation,
        hidden_size=int(settings.get("hidden_size", 32)),
        num_layers=int(settings.get("num_layers", 1)),
        dropout=float(settings.get("dropout", 0.0)),
        epochs=int(settings.get("epochs", 20)),
        batch_size=int(settings.get("batch_size", 64)),
        learning_rate=float(settings.get("learning_rate", 1e-3)),
        random_seed=config.random_seed,
    )
    horizon_days = int(settings.get("horizon_days", 14))
    baseline_last14 = validation.x[:, -horizon_days:].sum(axis=1)
    lstm_metrics = regression_metrics(validation.y, prediction)
    baseline_metrics = regression_metrics(validation.y, baseline_last14)
    relative_wape_improvement = 100 * (
        baseline_metrics["wape"] - lstm_metrics["wape"]
    ) / max(baseline_metrics["wape"], 1e-12)
    metadata.update(
        {
            "validation_cutoff": str(validation_cutoff.date()),
            "scope": "national top-demand series diagnostic",
            "max_series": max_series,
            "lstm_metrics": lstm_metrics,
            "last14_metrics": baseline_metrics,
            "relative_wape_improvement_percent": relative_wape_improvement,
        }
    )

    predictions = pd.DataFrame(
        {
            "cutoff_date": validation.cutoffs,
            "item_id": validation.item_ids,
            "store_code": validation.store_codes,
            "actual_demand_14d": validation.y,
            "last14_forecast_14d": baseline_last14,
            "lstm_forecast_14d": prediction,
        }
    )
    table_dir = config.root / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(table_dir / "lstm_validation_predictions.csv", index=False)
    (config.root / "reports" / "lstm_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model_dir = config.root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {"state_dict": model.state_dict(), "metadata": metadata},
        model_dir / "lstm_demand_forecaster.pt",
    )
    report = f"""# LSTM 需求预测诊断

- 验证截点：{validation_cutoff.date()}
- 范围：全国口径、历史需求最高的 {max_series} 条序列
- 训练样本：{len(train):,}
- 验证样本：{len(validation):,}
- LSTM：{_format_metrics(lstm_metrics)}
- Last-14：{_format_metrics(baseline_metrics)}
- 相对 Last-14 WAPE 改善：{relative_wape_improvement:.2f}%（负值表示退化）

这是单折、小范围、单变量的深度学习诊断，用于验证序列管线和时间切分。它不与
README 中全量六折 LightGBM 结果直接比较，也不会在未经成本敏感分位校准前替换
主实验库存目标。MILP 仍消费经过 Newsvendor 校准的库存建议。
"""
    (config.root / "reports" / "lstm_baseline_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
