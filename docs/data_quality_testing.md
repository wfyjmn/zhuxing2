# 数据质量检测与回归测试使用指南

## 概述

本文档介绍 DeepQuant 系统中的数据质量检测和回归测试功能，包括：

1. **对抗性验证**（Adversarial Validation）- 检测数据泄露和分布差异
2. **标签稳定性测试**（Label Stability Test）- 检查标签质量和一致性
3. **固定数据集生成器**（Fixed Dataset Generator）- 生成确定性数据集用于回归测试
4. **PR回归测试**（PR Regression Test）- 自动化性能回归检测

这些功能帮助确保模型训练的数据质量和模型性能的稳定性。

---

## 一、对抗性验证（Adversarial Validation）

### 1.1 原理

对抗性验证是一种检测数据泄露和分布差异的方法。其核心思想是：

1. 将训练集和测试集合并，创建新标签（训练集=1，测试集=0）
2. 训练一个分类器来区分训练集和测试集
3. 如果分类器AUC > 0.6，说明存在显著分布差异（可能的数据泄露）

### 1.2 使用方法

```python
from stock_system.adversarial_validation import AdversarialValidation
import pandas as pd

# 初始化验证器
validator = AdversarialValidation(
    n_splits=5,          # 交叉验证折数
    auc_threshold=0.6,   # AUC阈值
    random_state=42      # 随机种子
)

# 执行验证
results = validator.validate(X_train, X_test)

# 查看结果
print(f"状态: {results['status']}")
print(f"AUC: {results['mean_auc']:.4f}")
print(f"潜在泄露特征: {results['leaky_features']}")

# 生成报告
report = validator.generate_report(save_path='assets/reports/adversarial_validation.md')
print(report)
```

### 1.3 结果解读

| 状态 | AUC范围 | 含义 | 建议操作 |
|------|---------|------|----------|
| `no_leak` | < 0.6 | 未检测到数据泄露 | 可以继续训练 |
| `suspicious` | 0.5 - 0.6 | 存在轻微分布差异 | 检查特征，观察模型性能 |
| `leak_detected` | ≥ 0.6 | 检测到数据泄露 | 移除泄露特征，重新划分数据集 |

### 1.4 特征重要性

验证器会返回特征重要性排序，高重要性的特征可能是泄露源：

```python
# 查看特征重要性
for feature_data in results['feature_importance'][:5]:
    print(f"{feature_data['feature']}: {feature_data['importance']:.4f}")
```

---

## 二、标签稳定性测试（Label Stability Test）

### 2.1 功能

标签稳定性测试用于检查：

- 样本数是否充足
- 正样本比例是否合理
- 训练集和测试集标签分布是否一致

### 2.2 使用方法

```python
from stock_system.adversarial_validation import LabelStabilityTest
import numpy as np

# 初始化测试器
tester = LabelStabilityTest(
    expected_positive_ratio=0.1,  # 预期正样本比例
    ratio_tolerance=0.02,         # 比例容忍度
    min_positive_samples=10       # 最小正样本数
)

# 执行测试
results = tester.test(y_train, y_test)

# 查看结果
print(f"训练集正样本比例: {results['train']['positive_ratio']:.2%}")
print(f"测试集正样本比例: {results['test']['positive_ratio']:.2%}")
print(f"整体通过: {results['overall_passed']}")

# 查看问题
if not results['train']['passed']:
    print("训练集问题:", results['train']['issues'])

# 生成报告
report = tester.generate_report(save_path='assets/reports/label_stability.md')
print(report)
```

### 2.3 一致性检查

测试器会使用卡方检验检查训练集和测试集的标签分布是否一致：

```python
# 查看一致性结果
consistency = results['consistency']
print(f"卡方检验p值: {consistency['p_value']:.4f}")
print(f"一致性通过: {consistency['consistency_passed']}")
```

---

## 三、固定数据集生成器（Fixed Dataset Generator）

### 3.1 用途

固定数据集生成器用于生成确定性的数据集，确保每次运行得到相同的数据。这对于：

- 回归测试（比较不同版本模型的性能）
- CI/CD流程（自动化测试）
- 调试和复现问题

非常有用。

### 3.2 生成回归测试数据集

```python
from stock_system.fixed_dataset_generator import FixedDatasetGenerator

# 初始化生成器（固定随机种子）
generator = FixedDatasetGenerator(seed=42)

# 生成回归测试数据集
X, y = generator.generate_regression_test_dataset(
    n_samples=5000,          # 样本数
    n_features=20,           # 特征数
    positive_ratio=0.1,      # 正样本比例
    signal_strength=0.7      # 信号强度（0-1）
)

# 保存数据集
generator.save_dataset(
    X, y,
    save_dir='assets/fixed_datasets',
    dataset_name='regression_test'
)
```

### 3.3 生成时间序列数据集

```python
# 生成时间序列数据集（模拟股票数据）
X, y, dates = generator.generate_time_series_dataset(
    n_samples=1000,         # 样本数
    n_features=15,          # 特征数
    positive_ratio=0.08,    # 正样本比例
    trend=0.001,            # 趋势强度
    noise_level=0.02        # 噪声水平
)

# 保存数据集
generator.save_dataset(
    X, y,
    save_dir='assets/fixed_datasets',
    dataset_name='timeseries_test'
)
```

### 3.4 加载数据集

```python
# 加载数据集
X, y, metadata = generator.load_dataset(
    save_dir='assets/fixed_datasets',
    dataset_name='regression_test'
)

print(f"样本数: {metadata['n_samples']}")
print(f"特征数: {metadata['n_features']}")
print(f"正样本数: {metadata['n_positive']}")
print(f"正样本比例: {metadata['positive_ratio']:.2%}")
```

### 3.5 可重现性验证

```python
# 使用相同种子生成两个数据集
generator1 = FixedDatasetGenerator(seed=42)
X1, y1 = generator1.generate_regression_test_dataset()

generator2 = FixedDatasetGenerator(seed=42)
X2, y2 = generator2.generate_regression_test_dataset()

# 验证数据是否相同
import numpy as np
np.testing.assert_array_equal(X1.values, X2.values)
np.testing.assert_array_equal(y1, y2)
print("✓ 数据集完全一致")
```

---

## 四、数据质量检查器（Data Quality Checker）

### 4.1 功能

数据质量检查器综合执行对抗性验证和标签稳定性测试，提供完整的数据质量报告。

### 4.2 使用方法

```python
from stock_system.adversarial_validation import DataQualityChecker

# 初始化检查器
checker = DataQualityChecker(
    auc_threshold=0.6,              # 对抗性验证AUC阈值
    expected_positive_ratio=0.1,    # 预期正样本比例
    ratio_tolerance=0.02            # 比例容忍度
)

# 执行检查
results = checker.check(X_train, y_train, X_test, y_test)

# 查看结果
print(f"总体通过: {results['overall_passed']}")
print(f"对抗性验证状态: {results['adversarial']['status']}")
print(f"标签稳定性状态: {results['label_stability']['overall_passed']}")

# 生成综合报告
report = checker.generate_comprehensive_report(
    save_path='assets/reports/data_quality.md'
)
print(report)
```

---

## 五、PR回归测试（PR Regression Test）

### 5.1 用途

PR回归测试用于在提交代码前自动化验证模型性能，防止性能退化。

### 5.2 测试流程

PR回归测试执行以下步骤：

1. **加载配置** - 读取模型配置文件
2. **加载数据集** - 从固定数据集目录加载数据
3. **数据质量检查** - 执行对抗性验证和标签稳定性测试
4. **训练模型** - 使用固定参数训练模型
5. **评估模型** - 计算精确率和召回率
6. **综合判断** - 判断是否通过测试

### 5.3 命令行使用

```bash
# 基本用法
python scripts/pr_regression_test.py

# 指定数据集
python scripts/pr_regression_test.py --dataset regression_test

# 自定义目标指标
python scripts/pr_regression_test.py \
    --target-precision 0.70 \
    --target-recall 0.30

# 严格模式（失败时退出）
python scripts/pr_regression_test.py --strict
```

### 5.4 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `regression_test` | 数据集名称（regression_test, timeseries_test, hard_test） |
| `--target-precision` | `0.70` | 目标精确率 |
| `--target-recall` | `0.30` | 目标召回率 |
| `--strict` | `False` | 严格模式（失败时退出码为1） |

### 5.5 测试结果

测试结果会保存在 `assets/pr_test_results/latest.json`：

```json
{
  "data_quality": {
    "adversarial": {
      "status": "no_leak",
      "mean_auc": 0.52
    },
    "label_stability": {
      "overall_passed": true
    }
  },
  "model_performance": {
    "precision": 0.7125,
    "recall": 0.3070,
    "precision_passed": true,
    "recall_passed": true,
    "passed": true
  },
  "overall_passed": true
}
```

---

## 六、CI/CD集成

### 6.1 GitHub Actions示例

```yaml
name: Model Regression Test

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run regression test
        run: |
          python scripts/pr_regression_test.py --strict
```

### 6.2 GitLab CI示例

```yaml
stages:
  - test

regression_test:
  stage: test
  script:
    - pip install -r requirements.txt
    - python scripts/pr_regression_test.py --strict
  only:
    - merge_requests
```

---

## 七、最佳实践

### 7.1 数据质量检查流程

在模型训练前，建议按照以下流程检查数据质量：

```python
# 1. 生成固定数据集
generator = FixedDatasetGenerator(seed=42)
X, y = generator.generate_regression_test_dataset(n_samples=5000)

# 2. 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 执行数据质量检查
checker = DataQualityChecker()
results = checker.check(X_train, y_train, X_test, y_test)

# 4. 根据结果决定是否继续
if results['overall_passed']:
    print("✓ 数据质量检查通过，可以开始训练")
else:
    print("✗ 数据质量检查失败，请修复问题")
    print(checker.generate_comprehensive_report())
```

### 7.2 定期回归测试

建议在以下场景执行回归测试：

- 提交代码前（手动或自动）
- 合并PR后（CI/CD）
- 修改模型结构或超参数后
- 更新数据管道后

### 7.3 性能阈值设置

设置合理的目标指标：

```python
# 保守设置（高精确率，低召回率）
target_precision = 0.75
target_recall = 0.25

# 平衡设置
target_precision = 0.70
target_recall = 0.30

# 激进设置（高召回率，低精确率）
target_precision = 0.65
target_recall = 0.40
```

根据实际交易策略选择合适的阈值。

---

## 八、常见问题

### Q1: 对抗性验证显示有泄露，但我知道数据没有问题，怎么办？

A: 有时数据本身的分布差异会导致误报。可以：

1. 检查泄露特征列表，确认这些特征是否包含未来信息
2. 如果确认没有泄露，可以调整`auc_threshold`到更高的值（如0.7）
3. 检查数据划分是否合理，是否存在时间泄露

### Q2: 标签稳定性测试失败，如何修复？

A: 根据失败原因采取不同措施：

- **样本数不足**：增加样本数或降低`min_positive_samples`
- **正样本比例异常**：调整数据采样策略或`expected_positive_ratio`
- **分布不一致**：重新划分数据集，确保训练集和测试集同分布

### Q3: 固定数据集生成器的种子如何选择？

A: 任何固定的种子都可以，建议：

- 使用常见的种子（如42、123、2024）
- 在团队中统一使用相同的种子
- 将种子记录在配置文件中

### Q4: PR回归测试失败后，如何定位问题？

A: 按照以下步骤排查：

1. 查看测试结果文件 `assets/pr_test_results/latest.json`
2. 检查数据质量检查是否通过
3. 检查模型性能是否达标
4. 如果数据质量检查失败，查看详细报告
5. 如果模型性能不达标，检查代码变更是否影响了模型

---

## 九、参考文档

- [置信度分桶与过滤器使用指南](./confidence_bucket_usage.md)
- [系统总体设计文档](./design.md)
- [API文档](./api.md)

---

## 十、更新日志

### V1.0 (2025-01-15)
- 新增对抗性验证功能
- 新增标签稳定性测试功能
- 新增固定数据集生成器
- 新增PR回归测试脚本
- 添加完整的使用文档
