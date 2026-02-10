# 自动化参数调优训练指南

## 概述

`train_auto_tuned.py` 是一个完全自动化的参数调优训练脚本，使用 Optuna 贝叶斯优化自动搜索最优超参数。

## 核心特性

### 1. 自动化超参数搜索

脚本会自动优化以下参数：

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| learning_rate | 0.003 - 0.05 (log) | 学习率 |
| max_depth | 3 - 8 | 树的最大深度 |
| min_child_weight | 1 - 20 | 最小子节点权重 |
| subsample | 0.6 - 1.0 | 样本采样比例 |
| colsample_bytree | 0.6 - 1.0 | 特征采样比例 |
| reg_lambda | 0.1 - 10 (log) | L2 正则化系数 |
| reg_alpha | 0.0 - 5 (log) | L1 正则化系数 |
| gamma | 0.0 - 5 | 最小分裂增益 |
| scale_pos_weight | 1.0 - 5.0 | 正样本权重 |
| n_estimators | 300 - 1500 | 树的数量 |

### 2. 优化目标

默认优化 **F1 分数**，也可以配置为：
- `f1`: F1 分数（平衡精确率和召回率）
- `precision`: 精确率
- `recall`: 召回率
- `auc`: ROC-AUC

### 3. 时间序列交叉验证

使用严格的时间序列划分：
- **训练集**: 60%（最早的数据）
- **验证集**: 20%（中间的数据，用于 Optuna 优化）
- **测试集**: 20%（最新的数据，用于最终评估）

### 4. 自动阈值优化

在找到最优超参数后，自动优化决策阈值，默认目标精确率 ≥ 60%。

## 使用方法

### 基本使用

```bash
python scripts/train_auto_tuned.py
```

### 自定义配置

创建或编辑 `config/short_term_assault_config.json`：

```json
{
  "data": {
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "min_return_threshold": 0.04,
    "prediction_days": [3, 4, 5],
    "n_stocks": 150
  },
  "optuna": {
    "n_trials": 50,
    "timeout": 3600,
    "direction": "maximize",
    "metric": "f1",
    "cv_folds": 3,
    "early_stopping_rounds": 50
  },
  "threshold": {
    "target_precision": 0.60,
    "threshold_range": [0.15, 0.45],
    "threshold_step": 0.01
  }
}
```

### 配置说明

#### 数据配置 (data)
- `start_date`: 训练数据起始日期
- `end_date`: 训练数据结束日期
- `min_return_threshold`: 目标收益率阈值（0.04 表示 4%）
- `prediction_days`: 预测天数范围 [3, 4, 5] 表示 3-5 天
- `n_stocks`: 使用的股票数量

#### Optuna 配置 (optuna)
- `n_trials`: 试验次数（建议 30-100）
- `timeout`: 最大优化时间（秒）
- `direction`: 优化方向（maximize 或 minimize）
- `metric`: 优化指标（f1, precision, recall, auc）
- `cv_folds`: 交叉验证折数
- `early_stopping_rounds`: 早停轮数

#### 阈值配置 (threshold)
- `target_precision`: 目标精确率
- `threshold_range`: 阈值搜索范围
- `threshold_step`: 阈值搜索步长

## 输出文件

训练完成后会生成以下文件：

```
assets/models/
├── auto_tuned_model.pkl              # 最优模型 + 特征工程
├── auto_tuned_metadata.json          # 模型元数据和最优参数
└── optuna_study.pkl                  # Optuna 研究结果（包含所有试验）
```

## 性能对比

### 手动调优 vs 自动调优

| 指标 | 手动调优 | 自动调优 | 提升 |
|------|---------|---------|------|
| 调优时间 | 2-4 小时 | 10-30 分钟 | **75-87% 减少** |
| 精确率 | 37.14% | 待测试 | - |
| AUC | 0.5752 | 待测试 | - |
| 人力投入 | 高 | 低 | **完全自动化** |

## 优化建议

### 快速验证（10分钟）

```json
{
  "optuna": {
    "n_trials": 10,
    "timeout": 600
  }
}
```

### 标准调优（30分钟）

```json
{
  "optuna": {
    "n_trials": 50,
    "timeout": 3600
  }
}
```

### 精细调优（2小时）

```json
{
  "optuna": {
    "n_trials": 200,
    "timeout": 7200
  }
}
```

## 分析结果

### 查看最优参数

```python
import pickle
import json

# 加载元数据
with open('assets/models/auto_tuned_metadata.json', 'r') as f:
    metadata = json.load(f)

print("最优参数:")
for key, value in metadata['best_params'].items():
    print(f"  {key}: {value}")

print(f"\n最优分数: {metadata['optimization']['best_score']}")
print(f"决策阈值: {metadata['decision_threshold']}")

print(f"\n测试集性能:")
for metric, value in metadata['metrics'].items():
    print(f"  {metric}: {value}")
```

### 分析 Optuna 研究

```python
import pickle
import optuna
import matplotlib.pyplot as plt

# 加载研究
with open('assets/models/optuna_study.pkl', 'rb') as f:
    study = pickle.load(f)

# 查看优化历史
optuna.visualization.plot_optimization_history(study).show()

# 查看参数重要性
optuna.visualization.plot_param_importances(study).show()

# 查看参数关系
optuna.visualization.plot_parallel_coordinate(study).show()
```

## 常见问题

### Q1: 调优时间太长怎么办？

**A**: 减少 `n_trials` 或缩短 `timeout`：
```json
{
  "optuna": {
    "n_trials": 20,
    "timeout": 1800
  }
}
```

### Q2: 过拟合怎么办？

**A**: 增强正则化，在 `define_search_space` 中调整范围：
```python
'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20, log=True),
'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 10, log=True),
```

### Q3: 精确率太低怎么办？

**A**: 调整 `target_precision` 或 `metric`：
```json
{
  "threshold": {
    "target_precision": 0.70
  },
  "optuna": {
    "metric": "precision"
  }
}
```

### Q4: 如何平衡精确率和召回率？

**A**: 使用 `f1` 作为优化指标，这是自动平衡两者的最佳方式：
```json
{
  "optuna": {
    "metric": "f1"
  }
}
```

## 与其他训练脚本的对比

| 特性 | train_100_stocks_3years.py | train_final_optimized.py | **train_auto_tuned.py** |
|------|---------------------------|-------------------------|-------------------------|
| 参数调优 | ❌ 手动设置 | ❌ 手动设置 | ✅ **自动优化** |
| 调优方法 | 无 | 无 | **Optuna 贝叶斯优化** |
| 调优时间 | 依赖人工 | 依赖人工 | **10-30 分钟** |
| 最优参数 | 靠经验 | 靠经验 | **自动搜索** |
| 阈值优化 | ✅ 自动 | ✅ 自动 | ✅ 自动 |
| 人力投入 | 高 | 高 | **低** |
| 可复现性 | 差 | 差 | **优秀** |

## 总结

**自动化参数调优的优势：**

1. ✅ **节省时间**: 从数小时的人工调优缩短到 10-30 分钟
2. ✅ **更优结果**: 贝叶斯优化比网格搜索更高效
3. ✅ **避免偏差**: 完全自动，不受人为经验限制
4. ✅ **易于复现**: 每次运行都能得到一致的结果
5. ✅ **持续改进**: 可以随时增加试验次数继续优化

**推荐使用场景：**

- 首次训练模型
- 数据更新后重新训练
- 超参数空间较大
- 需要快速验证不同配置

**何时使用手动调优：**

- 已经有很好的基准参数
- 只需要微调个别参数
- 需要快速迭代测试
