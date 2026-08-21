# 场景驱动的随机多仓库存分配回测

## 方法

对每个验证折只使用更早折的 `实际需求 - LightGBM点预测` 残差，按完整
5仓残差向量进行自助抽样，从而保留区域误差的相关性。每个商品生成
7 个未来需求场景；在全国 Newsvendor 库存总量约束下，使用
OR-Tools CP-SAT 最小化各仓场景平均补少/补多成本。

第一折没有更早残差，故严格时间回测使用后续 5 折。

## 平均成本

| method                    |       total_cost |
|:--------------------------|-----------------:|
| deterministic_coordinated |      1.01569e+06 |
| independent_newsvendor    | 895655           |
| stochastic_coordinated    |      1.0066e+06  |

随机场景分配相对确定性协调的平均成本变化：
`+0.90%`（正值表示成本下降）。

## 求解状态

| fold_id   | deterministic_status   |   deterministic_runtime_seconds | stochastic_status   |   stochastic_runtime_seconds |   stochastic_relative_gap |   items |   scenario_count |
|:----------|:-----------------------|--------------------------------:|:--------------------|-----------------------------:|--------------------------:|--------:|-----------------:|
| fold_02   | OPTIMAL                |                        0.207134 | OPTIMAL             |                      5.72019 |                         0 |     792 |                7 |
| fold_03   | OPTIMAL                |                        0.213776 | OPTIMAL             |                      7.25789 |                         0 |     852 |                7 |
| fold_04   | OPTIMAL                |                        0.466455 | OPTIMAL             |                      7.01825 |                         0 |     905 |                7 |
| fold_05   | OPTIMAL                |                        0.255447 | OPTIMAL             |                      6.98987 |                         0 |     920 |                7 |
| fold_06   | OPTIMAL                |                        0.336065 | OPTIMAL             |                      7.85497 |                         0 |     948 |                7 |

## 边界

- 需求场景来自历史滚动回测残差，不使用当前折未来信息；
- 场景对跨仓残差相关性做经验保留，但不是完整概率分布校准；
- 全国目标库存仍由公开成本下的 Newsvendor 决策给出；
- 未虚构仓容、预算、调拨成本或平台上线收益；
- `FEASIBLE` 结果会同时报告相对 gap，不冒充已证明全局最优。
