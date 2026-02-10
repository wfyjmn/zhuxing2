# 对话4.txt - Optuna自动化参数调优训练器

## 概述

对话4.txt描述了一个完整的自动化参数调优训练系统，使用Optuna贝叶斯优化自动搜索最优超参数，并提供了多个优化建议以解决内存问题和模型性能问题。

## 核心功能

### 1. Optuna贝叶斯优化
- 自动搜索最优超参数
- 支持多种优化指标（AUC、F1、精确率）
- 时序交叉验证（60% / 20% / 20%）

### 2. 内存优化方案
- **Float32降维**: 节省50%内存
- **激进GC**: 防止内存泄漏
- **XGBoost内存优化**: max_bin=128
- **数据量控制**: 股票数量300→150，试验次数100→50

### 3. 优化目标调整
- **AUC优化**: 替代F1，避免"宁滥勿缺"
- **移除scale_pos_weight**: 避免过度加权
- **精确率硬性约束**: ≥60%

### 4. 大盘环境特征
- **大盘涨跌幅**: market_pct_chg
- **量价配合**: volume_ratio
- **技术指标**: MA、动量、RSI、波动率等

## 文件清单

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `assets/auto_tuned_trainer.py` | Python | 自动化参数调优训练器 |
| `config/optuna_auto_tuned_config.json` | JSON | Optuna配置文件 |
| `run_auto_tuned_training.sh` | Shell | 后台运行脚本 |
| `test_auto_tuned_trainer.py` | Python | 功能测试脚本 |
| `docs/对话4_需求与实现对照.md` | Markdown | 需求文档 |

## 快速开始

### 1. 安装依赖
```bash
pip install optuna xgboost pandas numpy
```

### 2. 测试功能
```bash
python test_auto_tuned_trainer.py
```

预期输出：
```
通过: 6/6

🎉 所有测试通过！AutoTunedTrainer已就绪
```

### 3. 运行训练

**方式1: 前台运行（快速测试）**
```bash
python assets/auto_tuned_trainer.py
```

**方式2: 后台运行（完整训练）**
```bash
chmod +x run_auto_tuned_training.sh
./run_auto_tuned_training.sh

# 查看日志
tail -f logs/auto_tuned_training.log
```

### 4. 查看结果
训练完成后，结果保存在：
- 模型文件: `assets/models/auto_tuned_model.pkl`
- 元数据: `assets/models/auto_tuned_metadata.json`
- Optuna研究: `assets/models/optuna_study.pkl`

## 配置说明

### Optuna配置
```json
{
  "optuna": {
    "n_trials": 50,        // 试验次数
    "timeout": 3600,       // 超时时间（秒）
    "metric": "auc",       // 优化指标
    "direction": "maximize"
  }
}
```

### 数据配置
```json
{
  "data": {
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "min_return_threshold": 0.04,  // 标签阈值
    "n_stocks": 150                // 股票数量
  }
}
```

### 阈值配置
```json
{
  "threshold": {
    "target_precision": 0.60,     // 目标精确率
    "threshold_range": [0.15, 0.45],
    "threshold_step": 0.01
  }
}
```

### 模型配置
```json
{
  "model": {
    "tree_method": "hist",
    "max_bin": 128,              // 内存优化参数
    "use_label_encoder": false
  }
}
```

## 预期性能

| 指标 | 目标范围 | 说明 |
|------|----------|------|
| AUC | 0.60 - 0.75 | 模型分离度 |
| 精确率 | ≥ 0.60 | 硬性约束 |
| 召回率 | 0.10 - 0.30 | 宁愿错过机会 |
| F1 | 0.20 - 0.40 | 平衡指标 |

## 关键优化说明

### 1. Float32内存优化
```python
def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
    return df
```

### 2. 激进GC策略
```python
def _force_gc(self):
    gc.collect()
    gc.collect()
```

### 3. AUC优化目标
```python
def objective(self, trial, ...):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred_proba)  # 使用AUC
```

### 4. 精确率硬性约束
```python
def optimize_threshold(self, y_val, y_val_pred_proba):
    for threshold in thresholds:
        if precision >= target_precision:  # 只收集满足精确率的阈值
            valid_thresholds.append(threshold)
```

## 常见问题

### Q1: 训练时出现OOM错误
**A**: 已通过Float32降维、激进GC、max_bin优化解决。如仍出现，可进一步减少n_stocks或n_trials。

### Q2: 精确率很低（<30%）
**A**: 检查配置文件中target_precision是否为0.60。可能需要：
- 放宽标签定义（降低min_return_threshold）
- 增加大盘环境特征
- 调整阈值搜索范围

### Q3: 召回率很低（<10%）
**A**: 这是正常的！根据对话4.txt的建议，宁愿错过100个机会（低召回），也不要抓100个坑（低精确）。精确率≥60%是硬性约束。

## 使用训练好的模型

```python
import pickle

# 加载模型
with open('assets/models/auto_tuned_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']

# 预测
df_features = ...  # 准备特征
X = df_features[feature_names].values
y_pred_proba = model.predict_proba(X)[:, 1]

# 使用最优阈值
threshold = 0.30  # 从metadata中读取
y_pred = (y_pred_proba >= threshold).astype(int)
```

## 与现有系统集成

### 1. 作为特征工程模块
```python
from assets.auto_tuned_trainer import AutoTunedTrainer

trainer = AutoTunedTrainer('config/optuna_auto_tuned_config.json')
X, y = trainer.extract_features_and_labels(...)
```

### 2. 作为训练脚本
```bash
./run_auto_tuned_training.sh
```

### 3. 作为模型服务
```python
# 加载模型
with open('assets/models/auto_tuned_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']

# 实时预测
predictions = model.predict_proba(new_features)
```

## 后续建议

1. **参数优化**: 尝试不同的搜索空间和优化指标
2. **特征工程**: 增加更多特征（行业特征、情绪指标等）
3. **集成学习**: 结合多个模型（XGBoost + LightGBM）
4. **实时预测**: 集成到主策略中进行实时选股
5. **A/B测试**: 对比不同配置的回测表现

## 总结

✅ **所有对话4.txt需求已完成并通过测试**

核心功能已就绪，包括：
- Optuna贝叶斯优化（自动参数调优）
- 内存优化（Float32 + GC + max_bin）
- 优化目标调整（AUC + 精确率约束）
- 大盘环境特征（market_pct_chg）
- 量价配合特征（volume_ratio）

系统已准备好进行自动化参数调优训练！

## 相关文档

- [对话4需求与实现对照](./对话4_需求与实现对照.md)
- [对话3需求与实现对照](./对话3_需求与实现对照.md)
- [对话2需求与实现对照](./对话2_需求与实现对照.md)
