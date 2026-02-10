# 突击选股智能决策系统使用指南

## 概述

突击选股智能决策系统（Brain）是 DeepQuant 系统的核心大脑和决策机构，整合了：

- **数据泄露审计**（Data Leak Audit）- 深度审计数据质量
- **在线监控**（Online Monitoring）- 实时监控模型性能
- **智能决策**（Intelligent Decision）- 综合信息做出买卖决策
- **三重确认**（Triple Confirmation）- 资金/情绪/技术确认机制
- **置信度过滤**（Confidence Filter）- 基于置信度的决策过滤

系统架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    突击选股智能决策系统                         │
│                        (Brain)                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 数据泄露审计  │  │  在线监控    │  │  智能决策    │      │
│  │  - 标签一致性 │  │  - 精确率监控 │  │  - 预测      │      │
│  │  - Lookahead │  │  - 自动回撤   │  │  - 确认      │      │
│  │  - 特征漂移   │  │  - 告警机制   │  │  - 过滤      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│           ↓                  ↓                  ↓            │
│  ┌───────────────────────────────────────────────────┐      │
│  │              决策引擎 (Decision Engine)             │      │
│  │  - 综合信息分析                                    │      │
│  │  - 风险评估                                        │      │
│  │  - 策略执行                                        │      │
│  └───────────────────────────────────────────────────┘      │
│                          ↓                                    │
│  ┌───────────────────────────────────────────────────┐      │
│  │           交易输出 (Trading Output)                  │      │
│  │  - 买入 / 卖出 / 持有                               │      │
│  │  - 置信度                                          │      │
│  │  - 执行建议                                        │      │
│  └───────────────────────────────────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 一、系统架构

### 1.1 核心组件

| 组件 | 功能 | 说明 |
|------|------|------|
| **AssaultDecisionBrain** | 智能决策大脑 | 整合所有功能的核心决策系统 |
| **AssaultTradingSystemIntegrated** | 集成交易系统 | 整合决策和执行 |
| **LabelConsistencyChecker** | 标签一致性检查 | 检查标签稳定性 |
| **LookaheadFeatureDetector** | Lookahead检测 | 检测数据泄露 |
| **FeatureDriftCalculator** | 特征漂移计算 | 检测特征分布变化 |
| **RollingPrecisionMonitor** | 滚动精确率监控 | 实时监控精确率 |
| **AutoRollbackThreshold** | 自动回撤阈值 | 检测性能退化 |
| **OnlineMonitoringSystem** | 在线监控系统 | 综合监控系统 |
| **StockPredictor** | 股票预测器 | XGBoost模型预测 |
| **TripleConfirmation** | 三重确认 | 资金/情绪/技术确认 |
| **ConfidenceBasedFilter** | 置信度过滤器 | 基于置信度过滤决策 |

### 1.2 决策流程

```
输入数据
    ↓
┌──────────────────┐
│  1. 系统状态检查  │ ← 判断系统是否可用
└──────────────────┘
    ↓
┌──────────────────┐
│  2. 快速审计     │ ← 检查数据基本质量
└──────────────────┘
    ↓
┌──────────────────┐
│  3. 模型预测     │ ← 获取预测结果
└──────────────────┘
    ↓
┌──────────────────┐
│  4. 置信度过滤   │ ← 过滤低置信度预测
└──────────────────┘
    ↓
┌──────────────────┐
│  5. 三重确认     │ ← 资金/情绪/技术确认
└──────────────────┘
    ↓
┌──────────────────┐
│  6. 综合决策     │ ← 生成最终决策
└──────────────────┘
    ↓
输出：买入/卖出/持有
```

---

## 二、使用方法

### 2.1 基础使用

```python
from stock_system.assault_decision_brain import AssaultDecisionBrain

# 创建决策大脑
brain = AssaultDecisionBrain(
    config_path="config/short_term_assault_config.json",
    enable_deep_audit=True,    # 启用深度审计
    enable_online_monitoring=True  # 启用在线监控
)

# 准备数据
stock_data = pd.DataFrame({
    'feature1': [...],
    'feature2': [...],
    # ... 其他特征
})

# 做出决策
decision = brain.make_decision(
    stock_data,
    current_index=100,
    symbol="600000.SH"
)

print(f"决策: {decision['decision']}")
print(f"置信度: {decision['confidence']:.4f}")
print(f"原因: {decision['reason']}")
```

### 2.2 训练前审计

```python
# 执行训练前数据审计
audit_results = brain.run_pre_training_audit(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    dates=dates_train
)

# 查看审计结果
if audit_results['overall_passed']:
    print("✅ 数据审计通过，可以开始训练")
else:
    print("❌ 数据审计失败，请修复数据问题")
    print(f"失败模块: {list(audit_results['modules'].keys())}")
```

### 2.3 集成交易系统

```python
from stock_system.assault_decision_brain import AssaultTradingSystemIntegrated

# 创建集成交易系统
trading_system = AssaultTradingSystemIntegrated(
    config_path="config/short_term_assault_config.json"
)

# 训练前检查
can_train = trading_system.pre_training_check(
    X_train, y_train, X_test, y_test
)

if can_train:
    # 训练模型
    # ...
    
    # 执行交易
    trades = trading_system.execute_trading(
        stock_data,
        symbol="600000.SH"
    )
    
    print(f"执行 {len(trades)} 笔交易")
    
    # 生成报告
    report = trading_system.generate_final_report()
```

### 2.4 在线监控

```python
# 更新性能监控
y_true = np.array([0, 0, 1, 0, 0])
y_pred = np.array([0, 0, 1, 0, 0])

brain.update_performance_monitor(y_true, y_pred)

# 查看监控状态
status = brain.get_decision_status()

print(f"系统状态: {status['system_status']}")
print(f"性能状态: {status['performance_status']}")

if 'monitoring_status' in status:
    monitoring = status['monitoring_status']
    rollback = monitoring['rollback_status']
    print(f"基准精确率: {rollback.get('baseline_precision')}")
    print(f"当前精确率: {rollback.get('current_precision')}")
    print(f"回撤触发: {'是' if rollback['rollback_triggered'] else '否'}")
```

---

## 三、配置说明

### 3.1 系统配置

```python
brain = AssaultDecisionBrain(
    config_path="config/short_term_assault_config.json",
    enable_deep_audit=True,      # 启用深度审计
    enable_online_monitoring=True # 启用在线监控
)
```

### 3.2 审计配置

```python
from stock_system.deep_leak_audit import (
    LabelConsistencyChecker,
    LookaheadFeatureDetector,
    FeatureDriftCalculator
)

# 标签一致性检查
label_checker = LabelConsistencyChecker(
    window_size=5,              # 滑动窗口大小
    zscore_threshold=3.0,       # Z分数阈值
    max_change_ratio=0.5        # 最大变化比例
)

# Lookahead检测
lookahead_detector = LookaheadFeatureDetector(
    lag_range=(1, 5),           # 滞后范围
    correlation_threshold=0.3,  # 相关性阈值
    significance_level=0.05     # 显著性水平
)

# 特征漂移计算
drift_calculator = FeatureDriftCalculator(
    psi_threshold=0.2,          # PSI阈值
    ks_threshold=0.1,           # KS阈值
    js_threshold=0.1            # JS阈值
)
```

### 3.3 监控配置

```python
from stock_system.online_monitoring import (
    RollingPrecisionMonitor,
    AutoRollbackThreshold
)

# 滚动精确率监控
monitor = RollingPrecisionMonitor(
    window_size=50,             # 滚动窗口大小
    min_samples=10,             # 最小样本数
    alert_threshold=0.1         # 告警阈值
)

# 自动回撤阈值
rollback = AutoRollbackThreshold(
    precision_threshold=0.70,   # 精确率阈值
    min_samples=50,             # 最小样本数
    rolling_window=100,         # 滚动窗口大小
    degradation_tolerance=0.05  # 退化容忍度
)
```

---

## 四、完整示例

### 4.1 训练前审计 + 模型训练 + 在线交易

```python
import pandas as pd
import numpy as np
from stock_system.assault_decision_brain import AssaultTradingSystemIntegrated

# 1. 创建集成交易系统
trading_system = AssaultTradingSystemIntegrated(
    config_path="config/short_term_assault_config.json"
)

# 2. 准备数据
X_train = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    # ... 其他特征
})
y_train = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])

X_test = pd.DataFrame({
    'feature1': np.random.randn(200),
    'feature2': np.random.randn(200),
    # ... 其他特征
})
y_test = np.random.choice([0, 1], size=200, p=[0.9, 0.1])

# 3. 训练前审计
print("【步骤1】训练前数据审计")
can_train = trading_system.pre_training_check(
    X_train, y_train, X_test, y_test
)

if not can_train:
    print("❌ 数据审计失败，停止训练")
    exit(1)

print("✅ 数据审计通过")

# 4. 训练模型（伪代码）
print("【步骤2】训练模型")
# model.train(X_train, y_train)
print("✅ 模型训练完成")

# 5. 准备交易数据
stock_data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'close': np.random.randn(100) * 10 + 100,
    # ... 其他特征
})

# 6. 执行交易
print("【步骤3】执行交易")
trades = trading_system.execute_trading(
    stock_data,
    symbol="600000.SH"
)

print(f"✅ 执行 {len(trades)} 笔交易")

# 7. 生成报告
print("【步骤4】生成报告")
report = trading_system.generate_final_report()

with open('reports/trading_summary.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ 报告已生成")
```

### 4.2 自定义告警回调

```python
def custom_alert_callback(alert):
    """自定义告警回调"""
    print(f"\n🚨 系统告警!")
    print(f"  类型: {alert.get('type')}")
    print(f"  原因: {alert.get('reason')}")
    
    # 发送邮件
    # send_alert_email(alert)
    
    # 发送短信
    # send_alert_sms(alert)
    
    # 记录日志
    # log_alert(alert)

# 创建带有自定义回调的监控系统
from stock_system.assault_decision_brain import AssaultDecisionBrain
from stock_system.online_monitoring import OnlineMonitoringSystem

brain = AssaultDecisionBrain(
    config_path="config/short_term_assault_config.json",
    enable_deep_audit=True,
    enable_online_monitoring=True
)

# 替换告警回调
brain.monitoring_system.alert_callback = custom_alert_callback
```

---

## 五、最佳实践

### 5.1 训练前检查清单

- [ ] 执行数据泄露审计
- [ ] 检查标签一致性
- [ ] 检测Lookahead特征
- [ ] 计算特征漂移
- [ ] 生成审计报告
- [ ] 确认所有检查通过

### 5.2 在线监控检查清单

- [ ] 启用滚动精确率监控
- [ ] 配置自动回撤阈值
- [ ] 设置告警回调
- [ ] 定期查看监控状态
- [ ] 记录告警事件
- [ ] 及时响应告警

### 5.3 决策优化建议

1. **置信度过滤**：只对高置信度预测执行交易
2. **三重确认**：确保资金、情绪、技术都满足条件
3. **风险控制**：根据信号等级调整仓位和止损
4. **持续监控**：实时监控性能，及时调整策略

---

## 六、常见问题

### Q1: 如何处理数据审计失败？

A: 查看审计报告，定位失败原因：

```python
audit_results = brain.run_pre_training_audit(X_train, y_train)

if not audit_results['overall_passed']:
    for module_name, result in audit_results['modules'].items():
        if not result.get('overall_passed', True):
            print(f"模块 {module_name} 失败:")
            brain.generate_comprehensive_report(save_path='audit_report.md')
```

### Q2: 如何调整告警灵敏度？

A: 修改监控参数：

```python
# 降低告警阈值（更敏感）
monitor = RollingPrecisionMonitor(
    window_size=30,
    min_samples=5,
    alert_threshold=0.05  # 从0.1降低到0.05
)

# 提高告警阈值（减少误报）
monitor = RollingPrecisionMonitor(
    window_size=100,
    min_samples=20,
    alert_threshold=0.15  # 从0.1提高到0.15
)
```

### Q3: 如何触发回撤？

A: 回撤会自动触发，也可以手动触发：

```python
# 手动触发回撤
brain.monitoring_system.rollback_threshold.trigger_rollback(
    reason="手动触发回撤"
)

# 检查回撤状态
status = brain.get_decision_status()
if status['system_status'] == 'rollback':
    print("系统处于回撤状态")
```

### Q4: 如何查看决策历史？

A: 查看决策状态：

```python
status = brain.get_decision_status()

print(f"决策总数: {status['decision_count']}")
print(f"买入信号: {status['buy_signals']}")
print(f"卖出信号: {status['sell_signals']}")
print(f"持有信号: {status['hold_signals']}")
```

---

## 七、参考文档

- [深入数据泄露审计与在线监控使用指南](./deep_audit_and_monitoring.md)
- [数据质量检测与回归测试使用指南](./data_quality_testing.md)
- [置信度分桶与过滤器使用指南](./confidence_bucket_usage.md)
- [系统总体设计文档](./design.md)

---

## 八、更新日志

### V1.0 (2025-01-15)
- 集成数据泄露审计功能
- 集成在线监控功能
- 实现智能决策大脑
- 实现集成交易系统
- 添加完整的使用文档
- 添加7个测试用例，全部通过
