# 菜鸟 Part II 初步数据审计报告

生成时间：2026-07-28T20:10:56+08:00

## 1. 数据资产概览

| dataset | rows | items | locations | min_date | max_date |
| --- | --- | --- | --- | --- | --- |
| 全国商品日表 | 210549 | 963 | 1 | 2014-10-10 00:00:00 | 2015-12-27 00:00:00 |
| 商品-分仓日表 | 864772 | 963 | 5 | 2014-10-10 00:00:00 | 2015-12-27 00:00:00 |
| 成本配置表 | 5778 | 963 | 6 |  |  |
| 完整日面板 | 1376719 | 963 | 6 | 2014-10-10 00:00:00 | 2015-12-27 00:00:00 |

原始三张 CSV 均无表头；导入层按照赛题字段说明显式定义了全国表 31 列、分仓表 32 列和成本表 3 列，未把首行误识别为字段名。

## 2. 数据质量检查

| check_name | issue_count |
| --- | --- |
| 全国表重复主键 | 0 |
| 分仓表重复主键 | 0 |
| 配置表重复主键 | 0 |
| 全国负需求记录 | 0 |
| 分仓负需求记录 | 0 |
| 无效成本对 | 0 |

## 3. 需求稀疏性

| scope | rows | zero_rate | mean_demand | max_demand |
| --- | --- | --- | --- | --- |
| 全国原始记录 | 210549 | 0.4683 | 5.7924 | 11542.0 |
| 区域仓原始记录 | 864772 | 0.6778 | 1.4103 | 3578.0 |
| 完整日面板 | 1376719 | 0.7163 | 1.7717 | 11542.0 |

区域仓需求比全国需求更稀疏。建模阶段会同时保留移动平均/Last-14 基线和适用于间歇性需求的需求类型标签，后续再比较 Croston/TSB 与分位数 LightGBM。

## 4. 日期补全面板

| panel_rows | observed_rows | imputed_zero_rate | after_last_observed_rows |
| --- | --- | --- | --- |
| 1376719 | 1075321.0 | 0.2189 | 11659.0 |

当前面板规则：从每个“商品—仓”序列第一次出现之日起补齐到全局数据结束日；缺失行为量与需求暂记为 0，同时保留 `is_observed` 和 `is_after_last_observed` 标记。该假设会在模型消融中单独验证，不会静默丢弃原始缺失信息。

## 5. 补少/补多成本与临界分位

| min_alpha | p25_alpha | median_alpha | p75_alpha | max_alpha |
| --- | --- | --- | --- | --- |
| 0.179 | 0.3294 | 0.3672 | 0.7787 | 0.8276 |

每个库存单元使用 `alpha = A / (A + B)` 计算 Newsvendor 临界分位。不同库存单元的风险偏好存在显著差异，因此后续核心评估以成本而非单一 MAE 为准。

## 6. 成本配置与需求历史覆盖

| configured_series | observed_series | no_history_series | history_coverage |
| --- | --- | --- | --- |
| 5778 | 5771.0 | 7.0 | 0.9988 |

配置表覆盖 963 个商品的“全国 + 5 区域仓”共 5,778 个库存单元。其中无历史需求的库存单元不会被伪造训练样本；其冷启动策略将在模型阶段单独处理。

## 7. 全国与分仓一致性

| item_days | exact_match_rate | mean_abs_diff | max_abs_diff |
| --- | --- | --- | --- |
| 210549 | 1.0 | 0.0 | 0.0 |

全国数据和 5 个区域仓数据是两个评测层级，不强制假设全国需求恒等于区域仓需求之和；差异作为层级协调阶段的输入，而不是在清洗时被覆盖。

## 8. 样例提交版本核验

| submission_rows | submission_items | overlapping_items |
| --- | --- | --- |
| 6000 | 1000 | 0 |

样例提交文件与 Part II 商品 ID 无交集，判定为不同赛季/版本；本项目不将其用于训练或回测。

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
