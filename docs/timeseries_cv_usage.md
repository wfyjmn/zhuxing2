# 时间序列交叉验证和类别不平衡处理使用指南

## 概述

本文档介绍如何使用新实现的时间序列交叉验证（TimeSeriesCV）和类别不平衡处理（ImbalanceHandler）功能来解决模型过拟合和阈值选择问题。

## 核心功能

### 1. 时间序列交叉验证（TimeSeriesCV）

时间序列交叉验证避免数据泄露，遵循时间顺序进行模型评估。

#### 支持的策略

- **Forward-Chaining（扩展窗口）**：每个fold的训练集不断扩大
  ```
  Fold 1: [0..100] → [101..150]
  Fold 2: [0..150] → [151..200]
  Fold 3: [0..200] → [201..250]
  ```

- **Rolling Window（滑动窗口）**：每个fold的训练集大小固定
  ```
  Fold 1: [0..100] → [101..150]
  Fold 2: [50..150] → [151..200]
  Fold 3: [100..200] → [201..250]
  ```

#### 特性

- ✅ 避免数据泄露（训练集总是早于测试集）
- ✅ 计算统计显著性（Mean ± 95% CI）
- ✅ 提供稳定性评估（变异系数 CV）
- ✅ 生成详细的交叉验证报告

### 2. 类别不平衡处理（ImbalanceHandler）

提供多种处理类别不平衡的策略。

#### 支持的策略

1. **Class Weight（类别权重）**
   - 自动计算类别权重
   - 支持 balanced、inverse、sqrt_inv 方法

2. **Focal Loss（焦点损失）**
   - 降低简单样本的权重
   - 关注困难样本
   - 参数：alpha（平衡因子）、gamma（聚焦因子）

3. **Sample Reweighting（样本重权重）**
   - 基于类别平衡
   - 基于样本难度
   - 基于时间衰减（近期样本权重更高）

4. **Combined（组合策略）**
   - 同时使用类别权重和样本权重

## 使用示例

### 示例 1：基本时间序列交叉验证

```python
import sys
sys.path.insert(0, 'src')

from stock_system.timeseries_cv import PrecisionTimeSeriesCV
from sklearn.linear_model import LogisticRegression
import numpy as np

# 准备数据
X = np.random.randn(1000, 20)
y = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])

# 初始化交叉验证
cv = PrecisionTimeSeriesCV(
    n_splits=5,
    test_size=None,
    min_train_size=100,
    strategy='forward',  # 或 'rolling'
    confidence_level=0.95
)

# 执行交叉验证
model = LogisticRegression(max_iter=1000)
results = cv.cross_validate(
    model=model,
    X=X,
    y=y,
    threshold=0.5,
    verbose=True
)

# 查看结果
summary = results['summary']
print(f"平均精确率: {summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}")
print(f"95% CI: [{summary['precision']['lower_ci']:.4f}, {summary['precision']['upper_ci']:.4f}]")

# 生成报告
report = cv.generate_report(results, save_path='assets/cv_report.md')
```

### 示例 2：使用类别不平衡处理

```python
import sys
sys.path.insert(0, 'src')

from stock_system.timeseries_cv import PrecisionTimeSeriesCV
from stock_system.imbalance_handler import ImbalanceHandler
from sklearn.linear_model import LogisticRegression
import numpy as np

# 准备数据（类别不平衡）
X = np.random.randn(1000, 20)
y = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])

# 初始化不平衡处理器
handler = ImbalanceHandler(strategy='class_weight')

# 计算类别权重
class_weights = handler.compute_class_weights(y, method='balanced')
print(f"类别权重: {class_weights}")

# 计算样本权重
sample_weights = handler.compute_sample_weights(y, method='balanced')
print(f"样本权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

# 执行交叉验证（带样本权重）
cv = PrecisionTimeSeriesCV(n_splits=5, min_train_size=100)
model = LogisticRegression(max_iter=1000)

results = cv.cross_validate(
    model=model,
    X=X,
    y=y,
    fit_params={'sample_weight': sample_weights},
    threshold=0.5,
    verbose=True
)
```

### 示例 3：使用 Focal Loss

```python
from stock_system.imbalance_handler import ImbalanceHandler

# 初始化 Focal Loss 处理器
handler = ImbalanceHandler(
    strategy='focal_loss',
    alpha=0.25,  # 平衡因子
    gamma=2.0    # 聚焦因子
)

# 计算 Focal Loss 权重（需要预测概率）
y_true = np.array([0, 1, 1, 0, 1])
y_pred_proba = np.array([0.3, 0.8, 0.6, 0.2, 0.9])

focal_weights = handler.compute_focal_loss_weights(
    y_true, y_pred_proba,
    alpha=0.25,
    gamma=2.0
)
```

### 示例 4：集成到 train_assault_model.py

```python
from train_assault_model import train_with_timeseries_cv

# 使用时间序列交叉验证训练
results = train_with_timeseries_cv(
    df=data,
    feature_engineer=feature_engineer,
    config=config,
    n_splits=5,
    imbalance_strategy='class_weight',
    strategy='forward',
    threshold=0.5
)

# 查看结果
cv_results = results['cv_results']
summary = cv_results['summary']

print(f"平均精确率: {summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}")
print(f"平均召回率: {summary['recall']['mean']:.4f} ± {summary['recall']['std']:.4f}")
print(f"平均AP: {summary['ap']['mean']:.4f} ± {summary['ap']['std']:.4f}")
```

### 示例 5：比较不同不平衡处理策略

```python
from train_assault_model import compare_imbalance_strategies

# 比较不同策略
results = compare_imbalance_strategies(
    df=data,
    feature_engineer=feature_engineer,
    config=config,
    strategies=['class_weight', 'focal_loss', 'sample_reweight', 'combined']
)

# 查看比较结果
for strategy, metrics in results.items():
    print(f"\n{strategy}:")
    print(f"  精确率: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
    print(f"  召回率: {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"  AP: {metrics['ap_mean']:.4f} ± {metrics['ap_std']:.4f}")
    print(f"  稳定性(CV): {metrics['stability']:.4f}")
```

## 输出说明

### 交叉验证结果结构

```python
{
    'fold_results': [
        {
            'fold': 1,
            'train_size': 100,
            'test_size': 50,
            'train_positive_ratio': 0.05,
            'test_positive_ratio': 0.06,
            'precision': 0.75,
            'recall': 0.30,
            'ap': 0.65,
            'precision_at_k': 0.80,
            'n_positive_pred': 10,
            'n_positive_true': 3,
            'n_samples': 50
        },
        ...
    ],
    'summary': {
        'precision': {
            'mean': 0.73,
            'std': 0.05,
            'lower_ci': 0.68,
            'upper_ci': 0.78,
            'values': [0.75, 0.70, 0.74, ...]
        },
        'recall': { ... },
        'ap': { ... },
        'precision_at_k': { ... },
        'stability': {
            'precision': 0.07,
            'recall': 0.12,
            'ap': 0.08,
            'precision_at_k': 0.06
        }
    }
}
```

### 统计显著性解释

- **Mean**: 所有fold的平均值
- **Std**: 标准差
- **Lower CI / Upper CI**: 95% 置信区间
  - 如果区间较窄，说明结果稳定
  - 如果区间较宽，说明结果不稳定
- **Stability (CV)**: 变异系数（CV = std/mean）
  - CV < 0.1: 稳定 ✅
  - 0.1 ≤ CV < 0.2: 较不稳定 ⚠️
  - CV ≥ 0.2: 不稳定 ❌

## 最佳实践

### 1. 选择合适的策略

- **数据量充足**：使用 Forward-Chaining（充分利用历史数据）
- **数据量有限**：使用 Rolling Window（保持训练集大小）
- **市场变化快**：使用 Rolling Window + 时间衰减权重

### 2. 处理类别不平衡

- **正样本极少**：使用 Combined 策略（类别权重 + 样本权重）
- **关注困难样本**：使用 Focal Loss
- **时间敏感**：使用 Sample Reweighting（时间衰减）

### 3. 评估稳定性

- 检查 CV 值：CV < 0.1 表示稳定
- 检查 CI 宽度：区间越窄越好
- 比较各fold：不应有异常值

### 4. 避免过拟合

- 使用时间序列交叉验证（不是普通k-fold）
- 增加gap参数（训练集和测试集之间留间隔）
- 使用正则化
- 减少特征数量

## 常见问题

### Q1: 为什么使用时间序列交叉验证而不是普通k-fold？

A: 普通k-fold会随机打乱数据，导致数据泄露（用未来数据预测过去）。时间序列交叉验证遵循时间顺序，避免这个问题。

### Q2: 如何选择折数（n_splits）？

A:
- 数据充足（>1000样本）：n_splits=5~10
- 数据有限（<1000样本）：n_splits=3~5
- 确保每个fold的测试集至少有50个样本

### Q3: 如何处理极端不平衡（正样本<1%）？

A:
1. 使用 Combined 策略
2. 增加 min_train_size
3. 考虑数据增强
4. 使用 Focal Loss

### Q4: 交叉验证结果不稳定怎么办？

A:
1. 增加训练集大小（min_train_size）
2. 使用 Rolling Window 策略
3. 增加模型正则化
4. 减少特征数量
5. 检查数据质量

## 相关文件

- `src/stock_system/timeseries_cv.py`: 时间序列交叉验证实现
- `src/stock_system/imbalance_handler.py`: 类别不平衡处理实现
- `train_assault_model.py`: 训练脚本（集成新功能）
- `tests/test_precision_timeseries_cv.py`: 单元测试

## 参考文献

1. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation.
2. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection.
3. King, G., & Zeng, L. (2001). Logistic regression in rare events data.
