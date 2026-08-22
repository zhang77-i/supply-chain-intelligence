# 菜鸟需求预测与多仓库存优化

基于阿里天池菜鸟 Part II 公开数据集的离线研究项目：把未来 14 天需求预测、成本敏感 Newsvendor 决策和全国—五仓库存协调串成一条可审计的 predict-then-optimize 流程。

## 方法

1. DuckDB 与 SQL 构建原始层、清洗层、日面板和数据审计；
2. 只使用截点及以前信息生成时序特征和未来 14 天标签；
3. 用 6 折滚动时间验证比较移动平均、Croston-SBA、TSB 与 LightGBM；
4. 用补少/补多成本比得到 Newsvendor 临界分位；
5. 用 CP-SAT 保证全国目标库存与五个区域仓分配一致；
6. 用显式 MILP 复现同一确定性协调目标，并支持调用方提供的仓容情景；
7. 用历史跨仓残差向量生成场景，做泄漏安全的随机协调回测。

另提供可选 PyTorch LSTM：用 56 天需求序列预测未来 14 天总需求，并严格要求
训练标签在验证截点前可观测。它目前是单折诊断基线，不冒充已经优于六折
LightGBM 的主模型。

## 已验证结果

| 实验 | 结果 |
| --- | ---: |
| 商品 / 区域仓 / 天数 | 963 / 5 / 444 |
| 成本敏感 LightGBM 相对点预测 | 决策成本降低 6.41% |
| 成本敏感 LightGBM 相对最佳传统基线 MA28 | 决策成本降低 25.32% |
| 确定性全国—五仓协调 | 4,815 组合，CP-SAT OPTIMAL |
| 7 场景随机协调相对同约束确定性协调 | 平均再降低 0.90% |

这些数字来自公开数据的滚动离线回测，不代表线上业务收益。独立 Newsvendor 因约束更少而成本更低，不能与全国—五仓协调方案作不公平结论。

## 快速验证

~~~bash
python -m pip install -r requirements.txt
python -m pytest -q
~~~

测试使用仓库内构造数据，不需要下载原始数据。若要重跑完整实验，请将天池数据包放在 data/raw/archive/CAINIAO Part II Data_20160509.zip，然后依次执行：

~~~bash
python scripts/run_pipeline.py
python scripts/run_modeling.py
python scripts/run_allocation.py
python scripts/run_milp_allocation.py
python scripts/run_stochastic_allocation.py
~~~

可选深度学习基线单独安装，避免给核心 MILP/CP-SAT 环境增加重量：

~~~bash
python -m pip install -r requirements-deep-learning.txt
python scripts/run_lstm_baseline.py
~~~

原始 CSV、DuckDB、Parquet 和训练模型不会提交到 Git。仓库保留 SQL、源码、测试、运行元数据与聚合报告，以便审阅口径。

## 目录

~~~text
sql/                      可审阅数据层
src/cainiao_inventory/    特征、模型、回测与优化
scripts/                  数据、预测、MILP/CP-SAT 与 LSTM 运行入口
tests/                    合成数据单元测试
reports/                  聚合结果和运行元数据
configs/project.yaml      数据路径与实验参数
~~~

## 结果边界

- 使用公开竞赛数据，不声称企业生产部署；
- 时间切分严格早于验证截点，不使用随机交叉验证；
- 预测精度与库存决策成本分开评价；
- FEASIBLE 结果必须同时报告 gap，不能写成已证明全局最优；
- 未虚构仓容、预算、调拨成本或公司利润。

MILP 使用整数库存分配变量、连续补少/补多偏差变量、商品总量平衡约束和可选仓容约束。天池数据没有仓容字段，因此已有全量结果不启用也不虚构容量；模型定义见 [MILP 数学模型](docs/milp_model.md)。

LSTM 的序列构造、滚动切分和适用边界见 [LSTM 基线说明](docs/lstm_baseline.md)。当前 MILP 仍消费经过 Newsvendor 成本校准的库存目标；LSTM 在完成全量多折与分位校准前不直接替换这一输入。

详见 [模型报告](reports/modeling_report.md)、[确定性协调报告](reports/multiwarehouse_allocation_report.md) 与 [随机协调报告](reports/stochastic_allocation_report.md)。
