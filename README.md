# Multi-Model Fusion for Academic Performance Prediction (OULAD)

面向学业成绩预测的可解释多模型融合框架——论文配套代码与实验产物。
Companion code and experimental artifacts for the paper *"Interpretable Multi-Model Fusion for Academic Performance Prediction"*.

## Overview 概述

本仓库复现论文全部数值结果：以逻辑回归、随机森林、XGBoost、LightGBM、CatBoost 为异质基学习器，
通过 Stacking（逻辑回归元学习器）与 MLP 元学习器实现决策级融合，在 OULAD 公开数据集
（32 593 条选课注册记录、46 维特征）上进行分层 5 折交叉验证（OOF 防泄漏），
并包含消融实验与 SHAP 可解释性分析。

**预测设定说明**：特征聚合覆盖课程全过程（含全部测评与 VLE 记录），属课程结束后的**事后判别**（retrospective prediction）。
面向在读早期预警部署时，需按干预时间窗重新截取特征。

## Key Results 主要结果

| Model / 模型 | Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | 0.8831 | 0.8792 | 0.9501 |
| Random Forest | 0.9080 | 0.9068 | 0.9688 |
| XGBoost | 0.9142 | 0.9119 | 0.9722 |
| LightGBM | 0.9136 | 0.9116 | 0.9719 |
| CatBoost (best single) | 0.9155 | 0.9139 | 0.9730 |
| Soft Voting | 0.9144 | 0.9128 | 0.9716 |
| Stacking (LR meta-learner) | 0.9155 | 0.9129 | 0.9727 |
| MLP meta-learner fusion | 0.9150 | 0.9128 | 0.9727 |

消融（融合 AUC）：仅静态特征 0.6565 → 静态+测评 0.9696 → 全特征（含 VLE）0.9727。
SHAP 前三位关键因子：测评参与次数、平均分、VLE 活跃天数。

## Data 数据

数据**不包含**在本仓库中（约 460 MB）。请从官方渠道下载 OULAD 并解压到 `data/`：

- 官方页面：https://analyse.kmi.open.ac.uk/open_dataset
- 引用：Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics Dataset // Scientific Data. 2017. Vol. 4. 170171. DOI: 10.1038/sdata.2017.103

所需文件清单见 `data/README.md`。

## Quickstart 快速开始

```bash
pip install -r requirements.txt

# 运行完整实验（约 6 分钟，输出写入 results/）
python src/run_experiment.py

# 重新生成论文四张配图（写入 figures/）
python src/make_figures.py
```

数据路径可用环境变量覆盖：`OULAD_DATA`（默认 `data/`）、`EXP_OUT`（默认 `results/`）。

## Repository Structure 目录结构

```
├── src/
│   ├── run_experiment.py   # 完整实验管线：特征工程 → 基模型 OOF → 融合 → SHAP → 消融
│   └── make_figures.py     # 论文配图生成（基于真实实验输出）
├── data/                   # OULAD 数据（需自行下载，见 data/README.md）
├── results/                # 实验产物：summary.json / base_fold_metrics.csv / shap_top.csv / run.log
├── figures/                # 论文四张配图
├── requirements.txt        # 依赖锁定（与论文 3.7 节一致）
└── LICENSE                 # MIT
```

## Reproducibility 可复现性

- 所有随机化步骤固定 `random_state=42`（StratifiedKFold 与全部模型初始化）。
- 数值结果以 `results/summary.json` 与 `results/run.log` 为准。
- CatBoost / LightGBM 在多线程下的浮点求和顺序差异可能使个别指标在第 4 位小数出现 ±1 的波动，不影响结论。

## Citation 引用

论文信息待发表后补充。数据请引用 OULAD 原始文献（见上）。

## License

代码采用 MIT License（见 LICENSE）。OULAD 数据遵循其原始发布条款。
