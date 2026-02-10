# 短期突击特征权重体系 v4.0 - 核心逻辑衔接优化文档

## 优化概述

本次优化解决了权重体系与预测模块、分桶分析器之间的衔接缺口，确保"代码能跑，策略落地不脱节"。

## 优化内容

### 1. 特征权重和预测模块的特征列表完全对齐

#### 1.1 新增印钞机专属权重分支

**权重调整方案**：
- 资金强度：40% → 35%（降低5%）
- 技术动量：25% → 20%（降低5%）
- 印钞机专属特征：0% → 10%（新增）

**印钞机专属特征列表**（14项）：
```
- is_sector_leader_cap          # 板块龙头标识（市值排名前10%）
- is_sector_leader_return       # 板块龙头标识（涨幅排名前5%）
- is_sector_leader              # 板块龙头综合标识
- continuous_inflow_days_count  # 连续资金流入天数
- has_continuous_inflow         # 有连续资金流入
- perfect_technical_pattern     # 技术形态完美（RSI>50且MACD金叉）
- return_certainty_score        # 收益确定性得分（0-2分）
- liquidity_qualified           # 流动性合格
- continuous_rise_days          # 连续上涨天数
- sustainability_qualified      # 持续性合格
- risk_qualified                # 风险合格
- money_machine_score           # 印钞机股票综合评分（0-5）
- is_money_machine              # 印钞机股票标记（得分≥5）
```

**权重分配**：
- 板块龙头综合标识：0.02（重点关注高确定性特征）
- 收益确定性得分：0.02（重点关注高确定性特征）
- 板块龙头标识（市值）：0.02
- 其他特征：各0.01

#### 1.2 补充预测模块缺失的特征

**新增特征**：
1. **rsi_signal**（权重0.03）
   - 归属维度：技术动量
   - 计算方式：基于RSI多周期组合的买卖信号
   - 阈值：1=买入信号，0=无信号，-1=卖出信号

2. **momentum_strength**（权重0.02）
   - 归属维度：技术动量
   - 计算方式：基于RSI、量价突破、攻击形态的综合强度
   - 阈值：0-1，越高越强

### 2. 过拟合差距的目标定义和联动机制

#### 2.1 明确过拟合差距的计算方式

```json
"overfitting_gap": {
  "calculation_method": "训练集精确率 - 测试集精确率"
}
```

#### 2.2 补充调整规则

**触发条件**：过拟合差距 > 20%

**优先调整策略**：
1. **降低技术动量维度权重**
   - 从20%降至15%
   - 原因：技术特征最容易过拟合

2. **调整XGBoost模型参数**
   - max_depth：从5降至3（降低模型复杂度）
   - reg_lambda：从3增至5（增强L2正则化）
   - reg_alpha：从0.5增至1.0（增强L1正则化）

#### 2.3 与分桶分析联动

**监控指标**：训练集高置信度桶 vs 测试集高置信度桶的精确率差距

**阈值**：>20%

**触发动作**：自动触发权重调整提示

### 3. RSI强化版策略与置信度联动

#### 3.1 不同市场环境下的置信度要求

| 市场环境 | RSI买入阈值 | 置信度要求 |
|---------|------------|----------|
| 牛市    | ≥65        | ≥0.75    |
| 震荡市  | ≥50        | ≥0.80    |
| 熊市    | ≥40        | ≥0.85    |

#### 3.2 背离检测与置信度联动

**顶背离处理**：
- 条件：价格创新高但RSI未创新高
- 动作：无论置信度多高，均降低仓位50%
- 原因：背离信号优先级高于置信度

**底背离处理**：
- 条件：价格创新低但RSI未创新低且置信度≥0.8
- 动作：可适当提高仓位（不超过单只股票上限）
- 原因：底背离+高置信度为强信号

## 配置文件结构

```json
{
  "strategy_name": "短期突击特征权重体系 v4.0",
  "version": "4.0",
  
  "alignment": {
    "predictor_features": [...],      // 预测模块的特征列表
    "printer_stock_features": [...]   // 印钞机专属特征列表
  },
  
  "feature_weights": {
    "capital_strength": {"weight": 0.35},
    "market_sentiment": {"weight": 0.35},
    "technical_momentum": {"weight": 0.20},
    "printer_stock_features": {"weight": 0.10}
  },
  
  "enhanced_rsi_strategy": {
    "confidence_linkage": {...},      // 置信度联动规则
    "divergence_detection": {...}     // 背离检测规则
  },
  
  "integration_with_modules": {
    "predictor_module": {...},        // 与预测模块的集成
    "bucket_analyzer": {...}          // 与分桶分析器的集成
  }
}
```

## 特征映射表

| 预测模块特征 | 权重体系归属 | 权重 |
|------------|------------|-----|
| main_capital_inflow_ratio | capital_strength.主力资金净流入占比 | 0.15 |
| large_order_buy_rate | capital_strength.大单净买入率 | 0.10 |
| capital_inflow_persistence | capital_strength.资金流入持续性 | 0.05 |
| northbound_capital_flow | capital_strength.北向资金流入 | 0.03 |
| elg_order_inflow | capital_strength.超大单流入 | 0.02 |
| capital_strength_index | 综合特征-资金强度指数 | - |
| sector_heat_index | market_sentiment.板块热度指数 | 0.12 |
| stock_sentiment_score | market_sentiment.个股情绪得分 | 0.10 |
| up_days_ratio | market_sentiment.市场广度指标 | 0.08 |
| sentiment_cycle_position | market_sentiment.情绪周期位置 | 0.05 |
| enhanced_rsi | technical_momentum.RSI强化版 | 0.08 |
| volume_price_breakout_strength | technical_momentum.量价突破强度 | 0.05 |
| intraday_attack_pattern | technical_momentum.分时图攻击形态 | 0.02 |
| momentum_index | technical_momentum.动量指数 | - |
| assault_composite_score | 综合特征-突击综合得分 | - |
| rsi_signal | technical_momentum.RSI信号 | 0.03 |
| momentum_strength | technical_momentum.动量强度 | 0.02 |

## 使用示例

### 1. 加载配置文件

```python
import json
with open('config/short_term_assault_config_v4.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
```

### 2. 获取特征权重

```python
feature_weights = config['feature_weights']
capital_weight = feature_weights['capital_strength']['weight']  # 0.35
```

### 3. 检查过拟合差距

```python
overfitting_gap = train_precision - test_precision
if overfitting_gap > config['optimization_goals']['overfitting_gap']['target']:
    # 触发调整规则
    adjustment_rules = config['optimization_goals']['overfitting_gap']['adjustment_rules']
    # 执行调整...
```

### 4. RSI策略与置信度联动

```python
import pandas as pd

def should_trigger_signal(rsi_value, confidence, market_environment):
    """判断是否应该触发信号"""
    strategy_config = config['enhanced_rsi_strategy']
    thresholds = strategy_config['dynamic_thresholds'][market_environment]
    
    if rsi_value >= thresholds['buy']:
        if confidence >= thresholds['confidence_threshold']:
            return True
    return False
```

### 5. 背离检测处理

```python
def handle_divergence(divergence_type, confidence):
    """处理背离信号"""
    divergence_config = config['enhanced_rsi_strategy']['divergence_detection']
    
    if divergence_type == 'top_divergence':
        # 无论置信度多高，均降低仓位50%
        return 0.5
    elif divergence_type == 'bottom_divergence':
        if confidence >= 0.8:
            # 可适当提高仓位
            return 1.2
    return 1.0
```

## 版本历史

### v4.0 (当前版本)
- 新增印钞机专属权重分支（权重10%）
- 补充预测模块缺失的特征（rsi_signal、momentum_strength）
- 明确过拟合差距的计算方式和调整规则
- RSI策略与置信度完全联动
- 与分桶分析器联动监控过拟合

### v3.0
- 初始版本
- 三大核心维度：资金强度、市场情绪、技术动量

## 注意事项

1. **特征对齐**：确保预测模块的特征列表与权重体系完全一致
2. **过拟合监控**：定期检查训练集和测试集的精确率差距
3. **置信度联动**：RSI策略必须与置信度要求结合使用
4. **权重调整**：过拟合差距超标时，优先降低技术动量维度权重

## 后续优化方向

1. 根据实际回测结果微调各维度的权重
2. 添加更多市场环境的识别逻辑
3. 优化印钞机专属特征的筛选标准
4. 增强与风控模块的联动

---

**文档版本**：v4.0  
**更新日期**：2024-01-05  
**作者**：DeepQuant Team
