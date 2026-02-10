# 深入数据泄露审计与在线监控使用指南

## 概述

本文档介绍 DeepQuant 系统中的深入数据泄露审计和在线监控功能，包括：

1. **标签一致性检查**（Label Consistency Checker）- 前向/后向标签一致性检查
2. **Lookahead特征检测**（Lookahead Feature Detector）- 检测特征是否包含未来信息
3. **特征漂移计算**（Feature Drift Calculator）- 检测特征分布漂移
4. **在线监控系统**（Online Monitoring System）- 滚动精确率监控和自动回撤

这些功能帮助确保数据质量、防止数据泄露，并提供实时的模型性能监控。

---

## 一、标签一致性检查（Label Consistency Checker）

### 1.1 功能

标签一致性检查器用于检查标签在时间序列上的稳定性，包括：

- **前向一致性检查**：检测标签在未来时间窗口内的稳定性
- **后向一致性检查**：检测标签在过去时间窗口内的稳定性
- **标签转换模式检测**：检测频繁的标签切换

### 1.2 使用方法

```python
from stock_system.deep_leak_audit import LabelConsistencyChecker
import numpy as np
import pandas as pd

# 初始化检查器
checker = LabelConsistencyChecker(
    window_size=5,           # 滑动窗口大小
    zscore_threshold=3.0,    # Z分数阈值
    max_change_ratio=0.5     # 最大变化比例
)

# 准备数据
y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
dates = pd.date_range(start='2024-01-01', periods=15, freq='D')

# 运行所有检查
results = checker.run_all_checks(y, dates)

# 查看结果
print(f"整体通过: {results['overall_passed']}")
print(f"前向异常数: {results['forward']['anomaly_count']}")
print(f"后向异常数: {results['backward']['anomaly_count']}")
print(f"总转换数: {results['transition']['statistics']['total_transitions']}")

# 生成报告
report = checker.generate_report(save_path='assets/reports/label_consistency.md')
print(report)
```

### 1.3 结果解读

| 检查类型 | 异常指标 | 含义 | 建议操作 |
|---------|---------|------|----------|
| 前向一致性 | `anomaly_count > 0` | 存在前向异常 | 检查标签生成逻辑 |
| 后向一致性 | `anomaly_count > 0` | 存在后向异常 | 考虑标签平滑处理 |
| 标签转换 | `rapid_switch_count > 0` | 存在快速切换 | 调整标签定义 |

### 1.4 检查原理

**前向一致性检查**：
- 计算标签的滚动平均和标准差
- 使用Z分数检测异常点
- 检测标签在未来时间窗口内的稳定性

**后向一致性检查**：
- 前向检查的反向操作
- 检测标签在过去时间窗口内的稳定性

**标签转换模式检测**：
- 统计标签切换次数
- 检测短时间内多次转换（快速切换）

---

## 二、Lookahead特征检测（Lookahead Feature Detector）

### 2.1 功能

Lookahead特征检测器用于检测特征是否包含未来信息（数据泄露），包括：

- **特征与未来标签的相关性检测**：通过检查特征与滞后标签的相关性
- **时间对齐检查**：检查特征和标签的时间对齐情况

### 2.2 使用方法

```python
from stock_system.deep_leak_audit import LookaheadFeatureDetector
import pandas as pd
import numpy as np

# 初始化检测器
detector = LookaheadFeatureDetector(
    lag_range=(1, 5),              # 滞后范围
    correlation_threshold=0.3,     # 相关性阈值
    significance_level=0.05        # 显著性水平
)

# 准备数据
X = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'future_feature': np.roll(y, shift=2)  # 包含未来信息的特征
})
y = np.random.choice([0, 1], size=100, p=[0.9, 0.1])

# 执行检测
results = detector.detect_lookahead(X, y)

# 查看结果
print(f"总特征数: {results['statistics']['total_features']}")
print(f"Lookahead特征数: {results['statistics']['lookahead_count']}")
print(f"Lookahead特征列表:")
for feature in results['lookahead_features']:
    print(f"  - {feature['feature']}: 最大相关性={feature['max_correlation']:.4f}")

# 生成报告
report = detector.generate_report(save_path='assets/reports/lookahead_features.md')
print(report)
```

### 2.3 检测原理

Lookahead检测通过以下步骤实现：

1. **滞后相关性计算**：计算特征与滞后标签的相关性
2. **显著性检验**：使用Pearson相关系数和p值判断显著性
3. **阈值判断**：超过阈值且显著的特征被标记为Lookahead

### 2.4 结果解读

| 相关性 | p值 | 含义 | 建议操作 |
|--------|-----|------|----------|
| 高（≥0.3） | < 0.05 | 可能的Lookahead | 移除或重新设计特征 |
| 低（<0.3） | ≥ 0.05 | 无Lookahead | 保留特征 |

---

## 三、特征漂移计算（Feature Drift Calculator）

### 3.1 功能

特征漂移计算器用于检测训练集和测试集之间的特征分布差异，包括：

- **PSI（Population Stability Index）**：衡量分布稳定性的指标
- **KS统计量**：Kolmogorov-Smirnov检验统计量
- **Jensen-Shannon距离**：衡量分布相似性的距离指标

### 3.2 使用方法

```python
from stock_system.deep_leak_audit import FeatureDriftCalculator
import pandas as pd
import numpy as np

# 初始化计算器
calculator = FeatureDriftCalculator(
    psi_threshold=0.2,    # PSI阈值
    ks_threshold=0.1,     # KS阈值
    js_threshold=0.1      # JS阈值
)

# 准备数据
X_train = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.normal(0, 1, 100),
})
X_test = pd.DataFrame({
    'feature1': np.random.normal(2, 1, 50),  # 分布偏移
    'feature2': np.random.normal(0, 1, 50),
})

# 计算漂移
results = calculator.calculate_drift(X_train, X_test)

# 查看结果
print(f"漂移特征数: {results['drift_summary']['drift_count']}")
print(f"高漂移特征数: {results['drift_summary']['high_drift_count']}")
print(f"高漂移特征: {results['drift_summary']['high_drift_features']}")

# 查看详细漂移信息
for feature, drift_info in results['feature_drift'].items():
    if drift_info['has_drift']:
        print(f"\n{feature}:")
        print(f"  PSI: {drift_info['psi']:.4f}")
        print(f"  KS统计量: {drift_info['ks_statistic']:.4f}")
        print(f"  JS距离: {drift_info['js_distance']:.4f}")
        print(f"  漂移等级: {drift_info['drift_level']}")

# 生成报告
report = calculator.generate_report(save_path='assets/reports/feature_drift.md')
print(report)
```

### 3.3 漂移指标解读

| 指标 | 阈值范围 | 含义 | 建议操作 |
|------|---------|------|----------|
| PSI | < 0.1 | 无漂移 | 正常 |
| PSI | 0.1 - 0.2 | 轻微漂移 | 监控 |
| PSI | ≥ 0.2 | 显著漂移 | 重新训练模型 |
| PSI | ≥ 0.4 | 高漂移 | 移除或替换特征 |

### 3.4 PSI等级

| PSI范围 | 等级 | 说明 |
|---------|------|------|
| PSI < 0.1 | 无漂移 | 分布稳定 |
| 0.1 ≤ PSI < 0.2 | 轻微漂移 | 可能需要关注 |
| 0.2 ≤ PSI < 0.4 | 中等漂移 | 建议重新训练 |
| PSI ≥ 0.4 | 高漂移 | 必须重新训练 |

---

## 四、在线监控系统（Online Monitoring System）

### 4.1 功能

在线监控系统提供实时的模型性能监控，包括：

- **滚动精确率监控**：实时监控模型精确率
- **自动回撤阈值**：自动检测性能退化并触发回撤
- **告警机制**：自定义告警回调函数

### 4.2 滚动精确率监控

```python
from stock_system.online_monitoring import RollingPrecisionMonitor
import numpy as np

# 初始化监控器
monitor = RollingPrecisionMonitor(
    window_size=50,           # 滚动窗口大小
    min_samples=10,           # 最小样本数
    alert_threshold=0.1       # 告警阈值（精确率下降比例）
)

# 模拟实时预测
for i in range(20):
    y_true = np.array([0, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    
    result = monitor.update(y_true, y_pred)
    
    if result['alert']:
        print(f"告警: {result['alert']['type']}")
        print(f"  原因: {result['alert']['reason']}")

# 查看性能趋势
trend = monitor.get_performance_trend(n_periods=10)
print(trend)

# 生成报告
report = monitor.generate_report(save_path='assets/reports/rolling_precision.md')
print(report)
```

### 4.3 自动回撤阈值

```python
from stock_system.online_monitoring import AutoRollbackThreshold
import numpy as np

# 初始化回撤管理器
rollback = AutoRollbackThreshold(
    precision_threshold=0.70,   # 精确率阈值
    min_samples=50,             # 最小样本数
    rolling_window=100,         # 滚动窗口大小
    degradation_tolerance=0.05  # 退化容忍度
)

# 模拟实时预测
for i in range(100):
    y_true = np.array([0, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    
    result = rollback.add_predictions(y_true, y_pred)
    
    if result['rollback_triggered']:
        print(f"回撤触发: {result['reason']}")
        # 执行回撤操作
        # rollback.trigger_rollback(result['reason'])

# 查看状态
status = rollback.get_status()
print(f"总预测数: {status['total_predictions']}")
print(f"回撤触发: {status['rollback_triggered']}")

# 生成报告
report = rollback.generate_report(save_path='assets/reports/auto_rollback.md')
print(report)
```

### 4.4 综合在线监控系统

```python
from stock_system.online_monitoring import OnlineMonitoringSystem
import numpy as np

# 定义告警回调
def alert_callback(alert):
    print(f"告警触发: {alert['type']}")
    if 'reason' in alert:
        print(f"原因: {alert['reason']}")
    # 发送邮件、短信等

# 初始化监控系统
system = OnlineMonitoringSystem(
    alert_callback=alert_callback
)

# 模拟实时预测
for i in range(50):
    y_true = np.array([0, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    
    result = system.process_predictions(y_true, y_pred)

# 查看系统状态
status = system.get_system_status()
print(status)

# 生成综合报告
report = system.generate_comprehensive_report(
    save_path='assets/reports/online_monitoring.md'
)
print(report)
```

---

## 五、完整审计流程示例

### 5.1 训练前审计

```python
from stock_system.deep_leak_audit import (
    LabelConsistencyChecker,
    LookaheadFeatureDetector,
    FeatureDriftCalculator
)

# 1. 标签一致性检查
label_checker = LabelConsistencyChecker()
label_results = label_checker.run_all_checks(y_train, dates_train)

if not label_results['overall_passed']:
    print("标签一致性检查失败，请修复标签问题")
    print(label_checker.generate_report())

# 2. Lookahead特征检测
lookahead_detector = LookaheadFeatureDetector()
lookahead_results = lookahead_detector.detect_lookahead(X_train, y_train)

if lookahead_results['statistics']['lookahead_count'] > 0:
    print(f"发现{lookahead_results['statistics']['lookahead_count']}个Lookahead特征")
    print("建议移除这些特征或重新设计")
    print(lookahead_detector.generate_report())

# 3. 特征漂移检查
drift_calculator = FeatureDriftCalculator()
drift_results = drift_calculator.calculate_drift(X_train, X_test)

if not drift_results['overall_passed']:
    print("特征漂移检测失败，请检查特征工程")
    print(drift_calculator.generate_report())
```

### 5.2 在线监控

```python
from stock_system.online_monitoring import OnlineMonitoringSystem

# 初始化监控系统
system = OnlineMonitoringSystem()

# 实时处理预测
for y_true, y_pred in streaming_predictions:
    result = system.process_predictions(y_true, y_pred)
    
    # 检查告警
    if result['monitor_result']['alert']:
        print("精确率告警:", result['monitor_result']['alert'])
    
    # 检查回撤
    if result['rollback_result']['rollback_triggered']:
        print("回撤触发:", result['rollback_result']['reason'])
        # 执行回撤操作

# 定期生成报告
system.generate_comprehensive_report(
    save_path=f'reports/monitoring_{datetime.now().strftime("%Y%m%d")}.md'
)
```

---

## 六、最佳实践

### 6.1 审计频率

建议在以下场景执行数据泄露审计：

- **训练前**：确保数据质量和特征正确性
- **模型更新后**：检查新模型的数据依赖
- **数据源变更后**：检测数据分布变化
- **定期审计**：每周或每月执行一次

### 6.2 在线监控配置

根据业务需求配置监控参数：

```python
# 保守配置（快速响应）
monitor = RollingPrecisionMonitor(
    window_size=30,           # 较小窗口
    min_samples=5,            # 较小样本数
    alert_threshold=0.05      # 较低阈值
)

# 平衡配置（推荐）
monitor = RollingPrecisionMonitor(
    window_size=50,           # 中等窗口
    min_samples=10,           # 中等样本数
    alert_threshold=0.1       # 中等阈值
)

# 宽松配置（减少误报）
monitor = RollingPrecisionMonitor(
    window_size=100,          # 较大窗口
    min_samples=20,           # 较大样本数
    alert_threshold=0.15      # 较高阈值
)
```

### 6.3 告警处理

建立告警处理流程：

1. **接收告警**：通过邮件、短信、Slack等方式
2. **确认告警**：检查是否为误报
3. **分析原因**：查看日志和指标
4. **采取措施**：
   - 暂停模型推理
   - 回滚到稳定版本
   - 重新训练模型
   - 修复数据管道
5. **记录事件**：更新知识库

### 6.4 回撤策略

根据业务场景选择回撤策略：

```python
# 策略1: 完全回撤（保守）
if rollback_triggered:
    # 停止所有预测
    # 回滚到上一个稳定版本
    # 通知相关人员

# 策略2: 部分回撤（平衡）
if rollback_triggered:
    # 降低预测置信度阈值
    # 增加人工审核
    # 监控性能恢复

# 策略3: 告警不回撤（激进）
if rollback_triggered:
    # 仅发送告警
    # 继续预测但增加监控频率
```

---

## 七、常见问题

### Q1: 标签一致性检查发现异常，但我知道标签是正确的，怎么办？

A: 可能是标签定义本身的特点。可以：

1. 检查异常点是否在业务逻辑上合理
2. 调整Z分数阈值和窗口大小
3. 如果确认标签正确，可以忽略告警

### Q2: Lookahead检测误报，特征没有包含未来信息，怎么办？

A: 可能是特征与标签的合法相关性。可以：

1. 检查特征生成逻辑，确认没有使用未来数据
2. 提高相关性阈值
3. 降低显著性水平（如从0.05提高到0.1）

### Q3: 特征漂移检测失败，但模型性能正常，怎么办？

A: 可能是特征分布的自然变化。可以：

1. 检查漂移程度（高漂移还是轻微漂移）
2. 监控模型性能是否受到影响
3. 如果性能稳定，可以继续监控
4. 如果性能下降，重新训练模型

### Q4: 在线监控频繁告警，如何减少误报？

A: 可以采取以下措施：

1. 调整告警阈值（提高阈值）
2. 增加窗口大小
3. 使用更稳定的滚动统计方法
4. 添加告警去重机制

---

## 八、参考文档

- [数据质量检测与回归测试使用指南](./data_quality_testing.md)
- [置信度分桶与过滤器使用指南](./confidence_bucket_usage.md)
- [系统总体设计文档](./design.md)

---

## 九、更新日志

### V1.0 (2025-01-15)
- 新增标签一致性检查功能
- 新增Lookahead特征检测功能
- 新增特征漂移计算功能
- 新增在线监控系统
- 新增滚动精确率监控
- 新增自动回撤阈值
- 添加完整的使用文档
- 添加29个测试用例，全部通过
