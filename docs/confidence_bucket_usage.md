# 置信度分桶和精确率回归测试使用指南

## 概述

本文档介绍如何使用置信度分桶分析和精确率回归测试功能，以减少误报、提升实际可用精确率，并防止模型性能回归。

## 核心功能

### 1. 置信度分桶分析（ConfidenceBucketAnalyzer）

分析不同置信度区间的精确率，帮助理解模型在不同置信度水平下的表现。

**功能**：
- 将预测概率分桶（均匀分桶、分位数分桶、自定义分桶）
- 计算每个桶的精确率、召回率等指标
- 生成置信度-精确率曲线
- 找出高精确率的置信度区间

### 2. 基于置信度的过滤器（ConfidenceBasedFilter）

根据置信度对预测进行分级决策。

**功能**：
- 高置信度预测（≥0.8）：自动交易
- 中等置信度预测（0.3-0.8）：人工复核
- 低置信度预测（<0.3）：拒绝

### 3. 精确率回归测试（PrecisionRegressionTest）

使用置信区间（CI）防止模型性能回归。

**功能**：
- 加载历史基准数据
- 计算当前模型的精确率
- 判断是否在置信区间内
- 如果低于CI下限，拒绝部署

## 使用示例

### 示例 1：置信度分桶分析

```python
import sys
sys.path.insert(0, 'src')

from stock_system.confidence_bucket import ConfidenceBucketAnalyzer
import numpy as np

# 准备数据
y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
y_proba = np.array([0.2, 0.8, 0.3, 0.9, 0.1, 0.7, 0.85, 0.25, 0.95, 0.15])

# 创建分析器（均匀分桶）
analyzer = ConfidenceBucketAnalyzer(
    n_buckets=5,
    bucket_type='uniform'
)

# 分析
results = analyzer.analyze(y_true, y_proba, threshold=0.5)

# 查看分桶结果
for bucket_name, bucket_data in results['bucket_results'].items():
    print(f"{bucket_name}:")
    print(f"  置信度区间: [{bucket_data['lower_bound']:.2f}, {bucket_data['upper_bound']:.2f})")
    print(f"  精确率: {bucket_data['precision']:.4f}")
    print(f"  样本数: {bucket_data['n_samples']}")

# 获取高精确率桶
high_precision_buckets = analyzer.get_high_precision_buckets(
    min_precision=0.7,
    min_samples=5
)
print(f"\n高精确率桶数量: {len(high_precision_buckets)}")

# 生成报告
report = analyzer.generate_report(save_path='assets/bucket_report.md')
```

### 示例 2：基于置信度的过滤

```python
from stock_system.confidence_bucket import ConfidenceBasedFilter
import numpy as np

# 创建过滤器
filter = ConfidenceBasedFilter(
    auto_trade_threshold=0.8,    # ≥0.8 自动交易
    manual_review_threshold=0.5, # 0.5-0.8 人工复核
    reject_threshold=0.3         # <0.3 拒绝
)

# 准备预测概率
y_proba = np.array([0.9, 0.7, 0.4, 0.2, 0.85])

# 过滤预测
decisions, stats = filter.filter(y_proba, return_labels=True)

print(f"自动交易: {stats['auto_trade']} 个")
print(f"人工复核: {stats['manual_review']} 个")
print(f"拒绝: {stats['reject']} 个")

# 评估过滤器效果
y_true = np.array([1, 1, 0, 0, 1])
evaluation = filter.evaluate_filter_effectiveness(y_true, y_proba, threshold=0.5)

print(f"\n整体精确率: {evaluation['overall_precision']:.4f}")
print(f"自动交易精确率: {evaluation['auto_trade_precision']:.4f}")
print(f"精确率提升: {evaluation['precision_improvement']:.4f}")
```

### 示例 3：精确率回归测试

```python
from stock_system.precision_regression_test import PrecisionRegressionTest
import numpy as np

# 创建回归测试器
regression_test = PrecisionRegressionTest(
    baseline_path='assets/precision_baseline.json',
    confidence_level=0.95,
    min_samples=50
)

# 第一次：保存基准
y_true_baseline = np.array([1, 1, 1, 0, 0, 0, 1, 0])
y_proba_baseline = np.array([0.9, 0.85, 0.9, 0.1, 0.15, 0.1, 0.8, 0.2])

regression_test.save_baseline(
    y_true_baseline, y_proba_baseline,
    threshold=0.5,
    metadata={'model_version': '1.0'},
    save_path='assets/precision_baseline.json'
)

# 第二次：运行测试
y_true_new = np.array([1, 0, 1, 0, 1, 0, 0, 1])
y_proba_new = np.array([0.8, 0.2, 0.75, 0.3, 0.9, 0.1, 0.2, 0.85])

result = regression_test.run_test(
    y_true_new, y_proba_new,
    metric='precision',
    strict_mode=True
)

print(f"当前精确率: {result['current_value']:.4f}")
print(f"基准均值: {result['baseline_mean']:.4f}")
print(f"95% CI: [{result['baseline_lower_ci']:.4f}, {result['baseline_upper_ci']:.4f}]")
print(f"性能变化: {result['performance_change']:+.4f}")
print(f"测试结果: {'通过' if result['passed'] else '失败'}")

# 生成报告
report = regression_test.generate_test_report(save_path='assets/regression_test_report.md')
```

### 示例 4：回归测试套件

```python
from stock_system.precision_regression_test import RegressionTestSuite
import numpy as np

# 创建测试套件
suite = RegressionTestSuite(
    baseline_path='assets/precision_baseline.json',
    confidence_level=0.95
)

# 添加测试
suite.add_test(
    test_name='precision_test',
    metric='precision',
    min_value=0.6,        # 精确率不低于60%
    max_degradation=0.05  # 性能下降不超过5%
)

suite.add_test(
    test_name='recall_test',
    metric='recall',
    min_value=0.3,
    max_degradation=0.1
)

# 准备数据
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
y_proba = np.array([0.9, 0.85, 0.2, 0.8, 0.15, 0.1, 0.9, 0.7, 0.3, 0.95])

# 运行测试套件
results = suite.run_suite(y_true, y_proba, threshold=0.5)

if results['status'] == 'baseline_created':
    print("已创建初始基准数据")
else:
    print(f"测试结果: {'通过' if results['all_passed'] else '失败'}")
    
    for test_name, test_result in results['test_results'].items():
        print(f"\n{test_name}:")
        print(f"  当前值: {test_result['current_value']:.4f}")
        print(f"  状态: {'通过' if test_result['overall_passed'] else '失败'}")

# 生成报告
report = suite.generate_suite_report(results, save_path='assets/suite_report.md')
```

### 示例 5：集成到训练流程

```python
import sys
sys.path.insert(0, 'src')

from stock_system.confidence_bucket import (
    ConfidenceBucketAnalyzer,
    ConfidenceBasedFilter
)
from stock_system.precision_regression_test import PrecisionRegressionTest

# 1. 训练模型后，进行置信度分桶分析
analyzer = ConfidenceBucketAnalyzer(n_buckets=10)
bucket_results = analyzer.analyze(y_test, y_pred_proba)

# 找出高精确率区间
high_precision_buckets = analyzer.get_high_precision_buckets(
    min_precision=0.75,
    min_samples=20
)

if high_precision_buckets:
    best_bucket = high_precision_buckets[0]
    print(f"最佳置信度区间: [{best_bucket['lower_bound']:.2f}, {best_bucket['upper_bound']:.2f})")
    print(f"该区间精确率: {best_bucket['precision']:.4f}")

# 2. 设置基于置信度的过滤器
filter = ConfidenceBasedFilter(
    auto_trade_threshold=best_bucket['lower_bound'],
    manual_review_threshold=best_bucket['lower_bound'] - 0.2,
    reject_threshold=best_bucket['lower_bound'] - 0.4
)

# 3. 运行回归测试，防止性能退化
regression_test = PrecisionRegressionTest(
    baseline_path='assets/precision_baseline.json'
)

try:
    regression_test.load_baseline()
    test_result = regression_test.run_test(
        y_test, y_pred_proba,
        metric='precision',
        strict_mode=True
    )
    
    if test_result['passed']:
        print("✅ 回归测试通过，可以部署模型")
    else:
        print("❌ 回归测试失败，需要优化模型")
        report = regression_test.generate_test_report()
        print(report)
        
except FileNotFoundError:
    print("首次运行，创建基准数据...")
    regression_test.save_baseline(y_test, y_pred_proba)
```

## 输出说明

### 置信度分桶结果

```python
{
    'bucket_results': {
        'bucket_0': {
            'bucket_id': 0,
            'lower_bound': 0.0,
            'upper_bound': 0.1,
            'n_samples': 50,
            'n_positive': 5,
            'positive_ratio': 0.1,
            'precision': 0.8,
            'recall': 0.4,
            'avg_confidence': 0.05,
            'true_positives': 4,
            'false_positives': 1,
            ...
        },
        ...
    },
    'confidence_precision_curve': DataFrame,
    'overall_stats': {
        'overall_precision': 0.75,
        'high_confidence_precision': 0.9,
        'low_confidence_precision': 0.6,
        ...
    }
}
```

### 回归测试结果

```python
{
    'metric': 'precision',
    'current_value': 0.72,
    'baseline_mean': 0.75,
    'baseline_lower_ci': 0.70,
    'baseline_upper_ci': 0.80,
    'passed': False,
    'strict_mode': True,
    'performance_change': -0.03,
    'relative_change': -0.04,
    'regression_level': 'moderate',
    'confidence_level': 0.95,
    'timestamp': '2024-02-03T23:00:00'
}
```

## 最佳实践

### 1. 置信度分桶分析

**选择合适的分桶策略**：
- 数据量大：使用均匀分桶（uniform）
- 数据量小：使用分位数分桶（quantile）
- 有特定需求：使用自定义分桶（custom）

**关注高精确率区间**：
- 找出精确率≥70%的区间
- 优先使用这些区间的预测
- 低精确率区间标记为人工复核

### 2. 基于置信度的过滤

**设置合理的阈值**：
- 自动交易阈值：≥0.8（高置信度）
- 人工复核阈值：0.5-0.8（中等置信度）
- 拒绝阈值：<0.5（低置信度）

**定期评估过滤器效果**：
- 监控自动交易的精确率
- 监控人工复核的精确率
- 调整阈值以优化效果

### 3. 精确率回归测试

**建立基准数据**：
- 使用至少5个历史版本的数据
- 计算均值和95%置信区间
- 定期更新基准

**严格模式 vs 宽松模式**：
- 严格模式：必须≥CI下限
- 宽松模式：可以低于CI下限，但需说明理由

**处理测试失败**：
- 严重回归：重新训练模型
- 中等回归：分析失败样本
- 轻微下降：监控观察

### 4. 集成策略

**训练时**：
1. 进行置信度分桶分析
2. 找出高精确率区间
3. 保存模型和置信度校准器

**部署前**：
1. 运行回归测试
2. 评估过滤器效果
3. 生成详细报告

**运行时**：
1. 对预测进行置信度分桶
2. 应用过滤器决策
3. 记录实际效果

## 常见问题

### Q1: 如何选择置信度阈值？

A: 
1. 进行置信度分桶分析
2. 找出精确率≥目标值（如70%）的区间
3. 选择该区间的下界作为阈值
4. 考虑业务需求平衡精确率和召回率

### Q2: 回归测试失败了怎么办？

A:
1. 检查性能下降程度
2. 分析失败样本的特征
3. 检查数据质量
4. 考虑调整模型超参数
5. 如果是严重回归，拒绝部署

### Q3: 如何提高自动交易的精确率？

A:
1. 提高自动交易阈值
2. 使用概率校准（Platt Scaling / Isotonic Regression）
3. 增加特征工程
4. 使用集成模型
5. 添加人工复核流程

### Q4: 基准数据应该多久更新一次？

A:
- 建议每季度更新一次
- 或在模型版本升级时更新
- 保持至少5个历史版本的数据

## 相关文件

- `src/stock_system/confidence_bucket.py`: 置信度分桶和过滤器实现
- `src/stock_system/precision_regression_test.py`: 精确率回归测试实现
- `tests/test_confidence_bucket.py`: 单元测试

## 参考文献

1. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.
2. Zadrozny, B., & Elkan, C. (2001). Obtaining calibrated probabilities from boosting.
3. Guo, C., et al. (2017). On calibration of modern neural networks.
