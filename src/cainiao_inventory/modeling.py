from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .baselines import compute_intermittent_baselines
from .config import PipelineConfig, load_config

LOGGER = logging.getLogger(__name__)

CATEGORICAL_COLUMNS = [
    "item_id",
    "store_code",
    "scope",
    "cate_id",
    "cate_level_id",
    "brand_id",
    "supplier_id",
    "demand_pattern",
]

EXCLUDED_FEATURES = {
    "sample_id",
    "cutoff_date",
    "cutoff_kind",
    "target_14d",
    "baseline_last14",
    "baseline_ma28",
    "baseline_weighted",
    "baseline_croston_sba",
    "baseline_tsb",
}


def interpolate_row_quantiles(
    predictions: np.ndarray,
    quantiles: np.ndarray,
    target_quantiles: np.ndarray,
) -> np.ndarray:
    values = np.maximum.accumulate(
        np.asarray(predictions, dtype=np.float64),
        axis=1,
    )
    grid = np.asarray(quantiles, dtype=np.float64)
    targets = np.clip(np.asarray(target_quantiles, dtype=np.float64), grid[0], grid[-1])
    upper = np.searchsorted(grid, targets, side="right")
    upper = np.clip(upper, 1, len(grid) - 1)
    lower = upper - 1
    fraction = (targets - grid[lower]) / (grid[upper] - grid[lower])
    row = np.arange(len(values))
    return values[row, lower] + fraction * (
        values[row, upper] - values[row, lower]
    )


def inventory_cost_components(
    actual: np.ndarray,
    target: np.ndarray,
    shortage_cost: np.ndarray,
    overage_cost: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(actual, dtype=np.float64)
    plan = np.clip(np.asarray(target, dtype=np.float64), 0.0, None)
    shortage_units = np.maximum(y - plan, 0.0)
    overage_units = np.maximum(plan - y, 0.0)
    shortage = shortage_units * np.asarray(shortage_cost, dtype=np.float64)
    overage = overage_units * np.asarray(overage_cost, dtype=np.float64)
    return {
        "shortage_units": float(shortage_units.sum()),
        "overage_units": float(overage_units.sum()),
        "shortage_cost": float(shortage.sum()),
        "overage_cost": float(overage.sum()),
        "total_cost": float(shortage.sum() + overage.sum()),
    }


def evaluate_method(
    frame: pd.DataFrame,
    prediction_column: str,
    fold_id: str,
) -> dict[str, float | str]:
    actual = frame["target_14d"].to_numpy(dtype=np.float64)
    predicted = frame[prediction_column].to_numpy(dtype=np.float64)
    costs = inventory_cost_components(
        actual,
        predicted,
        frame["shortage_cost"].to_numpy(),
        frame["overage_cost"].to_numpy(),
    )
    absolute_error = np.abs(actual - predicted)
    return {
        "fold_id": fold_id,
        "method": prediction_column,
        "rows": len(frame),
        "mae": float(absolute_error.mean()),
        "wape": float(absolute_error.sum() / max(actual.sum(), 1e-12)),
        "mean_inventory_cost": costs["total_cost"] / max(len(frame), 1),
        **costs,
    }


def _prepare_frames(
    labeled: pd.DataFrame,
    inference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    combined = pd.concat(
        [
            labeled.assign(_dataset="labeled"),
            inference.assign(_dataset="inference"),
        ],
        ignore_index=True,
    )
    for column in CATEGORICAL_COLUMNS:
        if column in combined:
            combined[column] = combined[column].astype("category")
    numeric = combined.select_dtypes(include=[np.number]).columns
    combined[numeric] = combined[numeric].replace([np.inf, -np.inf], np.nan)
    feature_columns = [
        column
        for column in combined.columns
        if column not in EXCLUDED_FEATURES and column != "_dataset"
    ]
    prepared_labeled = combined.loc[combined["_dataset"].eq("labeled")].drop(
        columns="_dataset"
    )
    prepared_inference = combined.loc[combined["_dataset"].eq("inference")].drop(
        columns="_dataset"
    )
    return prepared_labeled, prepared_inference, feature_columns


def _model_parameters(config: PipelineConfig) -> dict:
    return {
        "n_estimators": int(config.modeling["n_estimators"]),
        "learning_rate": float(config.modeling["learning_rate"]),
        "num_leaves": int(config.modeling["num_leaves"]),
        "min_child_samples": int(config.modeling["min_child_samples"]),
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 0.1,
        "random_state": config.random_seed,
        "n_jobs": int(config.modeling["n_jobs"]),
        "verbosity": -1,
    }


def _fit_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    config: PipelineConfig,
    objective: str,
    alpha: float | None = None,
    n_estimators: int | None = None,
) -> lgb.LGBMRegressor:
    params = _model_parameters(config)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if objective == "quantile":
        params.update({"objective": "quantile", "alpha": float(alpha)})
    else:
        params.update({"objective": "regression_l1"})
    model = lgb.LGBMRegressor(**params)
    fit_kwargs = {}
    if x_valid is not None and y_valid is not None:
        fit_kwargs = {
            "eval_set": [(x_valid, y_valid)],
            "eval_metric": "l1",
            "callbacks": [
                lgb.early_stopping(
                    int(config.modeling["early_stopping_rounds"]),
                    verbose=False,
                )
            ],
        }
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def _aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "mae",
        "wape",
        "mean_inventory_cost",
        "shortage_units",
        "overage_units",
        "shortage_cost",
        "overage_cost",
        "total_cost",
    ]
    return (
        metrics.groupby("method", observed=True)[columns]
        .mean()
        .sort_values("mean_inventory_cost")
        .reset_index()
    )


def _markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(display.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def _write_modeling_report(
    aggregate: pd.DataFrame,
    config: PipelineConfig,
    runtime_seconds: float,
    recommendation_rows: int,
) -> None:
    best = aggregate.iloc[0]
    method_index = aggregate.set_index("method")
    best_baseline = aggregate.loc[
        aggregate["method"].str.startswith("baseline_")
    ].iloc[0]
    newsvendor_cost = float(
        method_index.loc["lgbm_newsvendor", "mean_inventory_cost"]
    )
    point_cost = float(
        method_index.loc["lgbm_point", "mean_inventory_cost"]
    )
    baseline_cost = float(best_baseline["mean_inventory_cost"])
    versus_point = 1.0 - newsvendor_cost / point_cost
    versus_baseline = 1.0 - newsvendor_cost / baseline_cost
    report = f"""# 菜鸟需求预测与成本敏感库存：模型报告

## 实验设置

- 25个历史截点中的最后6个作为滚动验证折；
- 每一折训练数据严格早于验证截点；
- 点预测模型：LightGBM L1；
- 概率预测：{len(config.modeling["quantiles"])}个固定分位LightGBM；
- SKU—仓级目标分位：`A/(A+B)`，在预测分位网格上插值；
- 对比Last-14、MA28、加权移动平均、Croston-SBA和TSB；
- 总运行时间：{runtime_seconds:.2f}秒。

## 六折平均结果

{_markdown(aggregate)}

当前按平均库存决策成本排序的最优方法为 `{best["method"]}`。该结论仅对应公开数据的滚动离线回测，不表述为线上业务收益。

## 关键结论

- 成本敏感分位预测相对 LightGBM 点预测，平均库存决策成本下降 {versus_point:.2%}；
- 相对最优传统基线 `{best_baseline["method"]}`，平均库存决策成本下降 {versus_baseline:.2%}；
- 点预测的 MAE/WAPE 更低，但 Newsvendor 决策成本更低，验证了“预测精度不等于决策质量”；
- 最终生成 {recommendation_rows:,} 个全国/区域库存单元的未来 14 天目标库存建议。

## 口径说明

- MAE/WAPE评价需求点预测；
- `mean_inventory_cost`使用每个库存单元真实给定的补少/补多成本；
- Quantile LightGBM在每行的临界分位处输出库存目标；
- 分位数交叉使用逐行单调化处理；
- 最终日期只有推荐结果，没有真实标签，因此不计入回测指标。

## 下一步

1. 按全国/区域仓、需求类型、类目拆分误差和成本；
2. 对极端需求做稳健损失与两阶段模型；
3. 完成全国—区域仓预测协调；
4. 引入总量约束后建立随机库存分配模型；
5. 对7个无历史库存单元设计冷启动降级策略。
"""
    (config.root / "reports" / "modeling_report.md").write_text(
        report,
        encoding="utf-8",
    )


def run_modeling(config_path: str | Path) -> None:
    started = time.perf_counter()
    config = load_config(config_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    labeled = pd.read_parquet(config.outputs["feature_dataset"])
    inference = pd.read_parquet(config.outputs["inference_dataset"])
    LOGGER.info("Computing Croston-SBA and TSB rolling baselines")
    intermittent = compute_intermittent_baselines(labeled, config)
    labeled = labeled.merge(
        intermittent,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    prepared, prepared_inference, features = _prepare_frames(labeled, inference)
    manifest = pd.read_csv(
        config.outputs["fold_manifest"],
        parse_dates=["train_cutoff_min", "train_cutoff_max", "validation_cutoff"],
    )
    quantiles = np.asarray(config.modeling["quantiles"], dtype=np.float64)
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    importance_rows: list[pd.DataFrame] = []
    point_iterations: list[int] = []
    quantile_iterations: dict[float, list[int]] = {
        float(q): [] for q in quantiles
    }

    for fold in manifest.itertuples(index=False):
        train_mask = prepared["cutoff_date"].le(fold.train_cutoff_max)
        valid_mask = prepared["cutoff_date"].eq(fold.validation_cutoff)
        train = prepared.loc[train_mask]
        valid = prepared.loc[valid_mask].copy()
        x_train, y_train = train[features], train["target_14d"]
        x_valid, y_valid = valid[features], valid["target_14d"]
        LOGGER.info(
            "%s: train=%s validation=%s",
            fold.fold_id,
            len(train),
            len(valid),
        )

        point_model = _fit_model(
            x_train,
            y_train,
            x_valid,
            y_valid,
            config,
            objective="point",
        )
        valid["lgbm_point"] = np.clip(
            point_model.predict(x_valid),
            0.0,
            None,
        )
        point_iterations.append(
            point_model.best_iteration_ or int(config.modeling["n_estimators"])
        )
        importance_rows.append(
            pd.DataFrame(
                {
                    "feature": features,
                    "gain": point_model.booster_.feature_importance(
                        importance_type="gain"
                    ),
                    "fold_id": fold.fold_id,
                }
            )
        )

        quantile_predictions = []
        for quantile in quantiles:
            model = _fit_model(
                x_train,
                y_train,
                x_valid,
                y_valid,
                config,
                objective="quantile",
                alpha=float(quantile),
            )
            quantile_predictions.append(
                np.clip(model.predict(x_valid), 0.0, None)
            )
            quantile_iterations[float(quantile)].append(
                model.best_iteration_ or int(config.modeling["n_estimators"])
            )
        valid["lgbm_newsvendor"] = interpolate_row_quantiles(
            np.column_stack(quantile_predictions),
            quantiles,
            valid["critical_fractile"].to_numpy(),
        )
        for method in [
            "baseline_last14",
            "baseline_ma28",
            "baseline_weighted",
            "baseline_croston_sba",
            "baseline_tsb",
            "lgbm_point",
            "lgbm_newsvendor",
        ]:
            metric_rows.append(evaluate_method(valid, method, fold.fold_id))
        prediction_frames.append(
            valid[
                [
                    "sample_id",
                    "cutoff_date",
                    "item_id",
                    "store_code",
                    "target_14d",
                    "shortage_cost",
                    "overage_cost",
                    "critical_fractile",
                    "baseline_last14",
                    "baseline_ma28",
                    "baseline_weighted",
                    "baseline_croston_sba",
                    "baseline_tsb",
                    "lgbm_point",
                    "lgbm_newsvendor",
                ]
            ].assign(fold_id=fold.fold_id)
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    aggregate = _aggregate_metrics(fold_metrics)
    importance = (
        pd.concat(importance_rows)
        .groupby("feature", observed=True)["gain"]
        .mean()
        .sort_values(ascending=False)
        .rename("mean_gain")
        .reset_index()
    )
    config.outputs["model_predictions"].parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(config.outputs["model_predictions"], index=False)
    reports = config.root / "reports" / "tables"
    reports.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(reports / "model_fold_metrics.csv", index=False)
    aggregate.to_csv(reports / "model_aggregate_metrics.csv", index=False)
    importance.to_csv(reports / "feature_importance.csv", index=False)

    LOGGER.info("Training final models and producing inventory recommendations")
    x_all, y_all = prepared[features], prepared["target_14d"]
    point_estimators = max(50, int(np.median(point_iterations)))
    point_model = _fit_model(
        x_all,
        y_all,
        None,
        None,
        config,
        objective="point",
        n_estimators=point_estimators,
    )
    inference_quantiles = []
    final_quantile_models = {}
    for quantile in quantiles:
        estimators = max(
            50,
            int(np.median(quantile_iterations[float(quantile)])),
        )
        model = _fit_model(
            x_all,
            y_all,
            None,
            None,
            config,
            objective="quantile",
            alpha=float(quantile),
            n_estimators=estimators,
        )
        final_quantile_models[float(quantile)] = model
        inference_quantiles.append(
            np.clip(
                model.predict(prepared_inference[features]),
                0.0,
                None,
            )
        )

    recommendations = inference[
        [
            "cutoff_date",
            "item_id",
            "store_code",
            "scope",
            "critical_fractile",
            "shortage_cost",
            "overage_cost",
        ]
    ].copy()
    recommendations["point_demand_forecast_14d"] = np.clip(
        point_model.predict(prepared_inference[features]),
        0.0,
        None,
    )
    recommendations["target_inventory_14d"] = interpolate_row_quantiles(
        np.column_stack(inference_quantiles),
        quantiles,
        recommendations["critical_fractile"].to_numpy(),
    )
    recommendations.to_parquet(
        config.outputs["inventory_recommendations"],
        index=False,
    )
    recommendations.head(200).to_csv(
        reports / "inventory_recommendations_preview.csv",
        index=False,
    )

    model_dir = config.root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "point_model": point_model,
            "quantile_models": final_quantile_models,
            "quantile_grid": quantiles,
            "features": features,
            "categorical_columns": CATEGORICAL_COLUMNS,
        },
        model_dir / "forecast_inventory_models.joblib",
        compress=3,
    )
    runtime = time.perf_counter() - started
    _write_modeling_report(
        aggregate,
        config,
        runtime,
        len(recommendations),
    )
    (config.root / "reports" / "model_run_metadata.json").write_text(
        json.dumps(
            {
                "runtime_seconds": runtime,
                "folds": len(manifest),
                "features": len(features),
                "quantiles": quantiles.tolist(),
                "prediction_rows": len(predictions),
                "recommendation_rows": len(recommendations),
                "best_method_by_mean_cost": aggregate.iloc[0]["method"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
