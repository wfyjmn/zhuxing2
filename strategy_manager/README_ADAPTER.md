# strategy_manager 适配版 - 使用指南

## 概述

根据原有选股系统的特点，对 `strategy_manager` 模块进行了适配，简化了回测逻辑，整合了选股A/B/C程序。

### 主要改进

1. **简化回测引擎** (`simple_backtest.py`)
   - 不考虑实际持仓和资金管理
   - 只关注选股后的涨跌表现
   - 记录买入后N天的收益

2. **选股程序适配器** (`adapter.py`)
   - 封装选股A/B/C逻辑
   - 市场状态检测（20日均线）
   - 次日实盘数据对比
   - 错误检测和修正

3. **统一输出格式**
   - 标准化选股结果
   - 支持行业板块分类
   - 便于后续处理

## 快速开始

### 1. 市场状态检测

```python
from strategy_manager import Config, MarketStateDetector

config = Config()
detector = MarketStateDetector(config)

# 检测市场状态
market_info = detector.detect_market_state()
print(f"市场状态: {market_info['state']}")
print(f"描述: {market_info['description']}")

# 推荐策略
recommended = detector.recommend_strategy(market_info['state'])
print(f"推荐策略: {recommended}")
```

### 2. 运行选股程序

```python
from strategy_manager import Config, ScreenerAdapter
import pandas as pd

config = Config()
adapter = ScreenerAdapter(config)

# 加载数据（假设有选股数据）
data = pd.read_csv('your_stock_data.csv')

# 运行选股A（含市场状态判断）
result_a = adapter.run_screener_a(data)

# 运行选股B（风险过滤）
result_b = adapter.run_screener_b(data)

# 运行选股C（市场感知 + 风险过滤 + 行业分类）
result_c = adapter.run_screener_c(data, enable_industry=True)
```

### 3. 回测和对比

```python
from strategy_manager import Config, ScreenerAdapter

config = Config()
adapter = ScreenerAdapter(config)

# 回测选股结果
result = adapter.backtest_and_compare(
    selected_df=result_a,
    buy_date="20240101",
    hold_days=5
)

# 查看统计指标
stats = result['stats']
print(f"胜率: {stats['win_rate']}%")
print(f"平均收益: {stats['avg_return']}%")

# 查看详细报告
print(result['report'])
```

### 4. 错误检测和修正

```python
from strategy_manager import Config, ScreenerAdapter

config = Config()
adapter = ScreenerAdapter(config)

# 检测回测中的错误
detection = adapter.detect_and_correct_errors(backtest_df)

if detection['errors']:
    print("发现问题:")
    for error in detection['errors']:
        print(f"  - {error}")

if detection['suggestions']:
    print("改进建议:")
    for suggestion in detection['suggestions']:
        print(f"  - {suggestion}")
```

## 核心模块说明

### SimpleBacktestEngine - 简化回测引擎

**特点:**
- 不考虑持仓管理
- 只记录买入后N天的价格变化
- 计算简单统计指标

**主要方法:**
```python
engine = SimpleBacktestEngine(config)

# 回测选股结果
backtest_df = engine.backtest_selection(
    selected_df=data,
    buy_date="20240101",
    hold_days=5
)

# 计算统计指标
stats = engine.calculate_stats(backtest_df)

# 生成报告
report = engine.generate_report(backtest_df, stats, "策略名称")
```

### MarketStateDetector - 市场状态检测器

**特点:**
- 使用20日均线判断市场状态
- 支持牛市/震荡市/熊市分类
- 根据市场状态推荐策略

**主要方法:**
```python
detector = MarketStateDetector(config)

# 检测市场状态
market_info = detector.detect_market_state()
# 返回: {'state': 'bull', 'ma20': 3000, 'deviation_pct': 5.2, ...}

# 推荐策略
strategy = detector.recommend_strategy('bull')
# 返回: 'momentum'
```

### ScreenerAdapter - 选股程序适配器

**特点:**
- 整合选股A/B/C逻辑
- 统一输出格式
- 支持市场状态判断
- 支持行业板块分类

**主要方法:**
```python
adapter = ScreenerAdapter(config)

# 选股A：市场状态判断 → 策略选择 → 基础筛选
result_a = adapter.run_screener_a(data, market_state='bull')

# 选股B：风险过滤 + 基础筛选
result_b = adapter.run_screener_b(data)

# 选股C：市场感知 + 风险过滤 + 行业分类
result_c = adapter.run_screener_c(data, enable_industry=True)

# 回测和对比
result = adapter.backtest_and_compare(result_a, "20240101", hold_days=5)

# 错误检测
detection = adapter.detect_and_correct_errors(backtest_df)
```

## 数据格式要求

### 选股数据格式

选股数据应包含以下列：

| 列名 | 说明 | 类型 |
|------|------|------|
| ts_code | 股票代码 | str |
| name | 股票名称 | str |
| close | 收盘价 | float |
| pct_chg | 涨跌幅(%) | float |
| turnover_rate | 换手率(%) | float |
| volume_ratio | 量比 | float |
| industry | 行业 | str (可选) |

### 回测结果格式

回测结果包含以下列：

| 列名 | 说明 | 类型 |
|------|------|------|
| ts_code | 股票代码 | str |
| name | 股票名称 | str |
| buy_date | 买入日期 | str |
| buy_price | 买入价 | float |
| sell_date | 卖出日期 | str |
| sell_price | 卖出价 | float |
| return_pct | 收益率(%) | float |
| holding_days | 持有天数 | int |

## 演示脚本

运行演示脚本查看完整示例：

```bash
python strategy_manager/demo_adapter.py
```

演示内容包括：
1. 市场状态检测
2. 选股程序运行（A/B/C）
3. 回测和对比
4. 错误检测和修正

## 与原有系统集成

### 1. 替换原有回测逻辑

原有的验证跟踪模块可以使用 `SimpleBacktestEngine`：

```python
# 原有代码
# validation_track.py

from strategy_manager import SimpleBacktestEngine, Config

# 初始化
config = Config()
engine = SimpleBacktestEngine(config)

# 回测
backtest_df = engine.backtest_selection(
    selected_df=selected_stocks,
    buy_date=buy_date,
    hold_days=5
)

# 统计
stats = engine.calculate_stats(backtest_df)
```

### 2. 整合选股逻辑

在主控制器中整合选股A/B/C：

```python
# main_controller.py

from strategy_manager import ScreenerAdapter, MarketStateDetector

def run_stock_selection():
    config = Config()

    # 市场状态检测
    detector = MarketStateDetector(config)
    market_info = detector.detect_market_state()

    # 根据市场状态选择策略
    adapter = ScreenerAdapter(config)

    # 运行选股C（组合策略）
    result = adapter.run_screener_c(data, enable_industry=True)

    # 回测对比
    backtest_result = adapter.backtest_and_compare(
        selected_df=result,
        buy_date=buy_date,
        hold_days=5
    )

    # 错误检测
    detection = adapter.detect_and_correct_errors(
        backtest_result['backtest_df']
    )

    return result, backtest_result, detection
```

### 3. 日志和报告

使用适配器的报告功能：

```python
print(backtest_result['report'])

# 或者保存到文件
with open('backtest_report.txt', 'w', encoding='utf-8') as f:
    f.write(backtest_result['report'])
```

## 注意事项

1. **Tushare Token**: 需要配置 `.env` 文件或通过代码传入
   ```bash
   TUSHARE_TOKEN=your_token_here
   ```

2. **数据格式**: 确保选股数据包含必要的列名

3. **市场状态**: 使用20日均线判断，需要至少60天的历史数据

4. **回测限制**: 简化版回测不考虑交易成本、滑点等实际因素

5. **行业分类**: 需要数据中包含 `industry` 列

## 后续优化方向

1. **增加风险过滤**: 整合跌停历史、解禁数据、龙虎榜数据
2. **参数优化**: 支持自动优化选股参数
3. **组合策略**: 支持多个策略的组合使用
4. **实时对比**: 支持与实盘数据的实时对比
5. **可视化**: 增加收益曲线、胜率等图表展示

## 联系方式

如有问题或建议，请联系开发团队。
