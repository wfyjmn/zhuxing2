# DeepQuant V2.0 - 整合指南

## 概述

DeepQuant V2.0 整合了策略管理器适配器，提供了更强大的选股、回测和错误检测功能。

## 新增功能

### 1. 增强的主控制器 (`main_controller_v2.py`)

**新增命令行选项：**

```bash
# 运行完整流程（使用适配器）
python main_controller_v2.py full --use-adapter

# 仅运行选股（启用回测和错误检测）
python main_controller_v2.py select --use-adapter --enable-backtest --hold-days 5 --detect-errors

# 仅运行市场状态检测
python main_controller_v2.py detect-market

# 运行测试模式
python main_controller_v2.py test
```

**选项说明：**

| 选项 | 说明 |
|------|------|
| `--use-adapter` | 使用策略管理器适配器 |
| `--no-adapter` | 禁用适配器，使用原有模式 |
| `--screeners A B C` | 选择运行的选股程序 |
| `--enable-backtest` | 启用回测功能 |
| `--hold-days N` | 回测持有天数 |
| `--detect-errors` | 启用错误检测 |
| `--verbose` | 显示详细输出 |

### 2. 数据格式检查工具 (`data_format_checker.py`)

检查选股数据格式是否符合要求：

```bash
# 基本检查
python data_format_checker.py your_data.csv

# 详细检查
python data_format_checker.py your_data.csv --verbose
```

**必需字段：**
- ts_code: 股票代码
- name: 股票名称
- close: 收盘价
- pct_chg: 涨跌幅(%)
- turnover_rate: 换手率(%)
- volume_ratio: 量比

**可选字段：**
- industry: 行业
- pe_ttm: 市盈率
- pb: 市净率
- roe: 净资产收益率

### 3. 功能测试脚本 (`test_adapter_integration.py`)

验证所有功能是否正常工作：

```bash
python test_adapter_integration.py
```

测试内容：
1. 依赖环境检查
2. 策略管理器模块检查
3. 市场状态检测
4. 选股程序测试
5. 回测功能测试
6. 错误检测测试

### 4. 增强的选股集合程序 (`run_all_screeners.py`)

原有脚本已增强，支持以下新功能：

```bash
# 市场状态检测
python scripts/run_all_screeners.py --detect-market-state

# 启用回测
python scripts/run_all_screeners.py --enable-backtest --hold-days 5

# 错误检测
python scripts/run_all_screeners.py --detect-errors

# 使用适配器
python scripts/run_all_screeners.py --use-adapter

# 完整功能
python scripts/run_all_screeners.py \
  --detect-market-state \
  --enable-backtest \
  --hold-days 5 \
  --detect-errors \
  --use-adapter \
  --verbose
```

## 快速开始

### 1. 检查环境

```bash
# 运行测试脚本
python test_adapter_integration.py
```

确保所有测试通过。

### 2. 检查数据格式

```bash
# 检查你的选股数据
python data_format_checker.py your_selection_data.csv
```

如果有缺失字段，请按照提示添加。

### 3. 运行选股（新方式）

```bash
# 使用新的主控制器
python main_controller_v2.py select --use-adapter --enable-backtest --detect-errors
```

### 4. 查看结果

选股结果会保存在 `output/` 目录：
- `selected_stocks_A_YYYYMMDD.csv` - 选股A结果
- `selected_stocks_B_YYYYMMDD.csv` - 选股B结果
- `selected_stocks_C_YYYYMMDD.csv` - 选股C结果

## 集成到现有流程

### 替换原有的选股调用

**原有代码：**
```python
# 运行第1轮筛选
result = subprocess.run(
    [sys.executable, '柱形选股-筛选.py'],
    ...
)
```

**新代码：**
```python
from strategy_manager import Config, ScreenerAdapter, MarketStateDetector

config = Config()
detector = MarketStateDetector(config)

# 检测市场状态
market_info = detector.detect_market_state()

# 使用适配器
adapter = ScreenerAdapter(config)
result = adapter.run_screener_c(data, enable_industry=True)

# 回测
backtest_result = adapter.backtest_and_compare(
    selected_df=result,
    buy_date=buy_date,
    hold_days=5
)

# 错误检测
detection = adapter.detect_and_correct_errors(
    backtest_result['backtest_df']
)
```

### 替换原有的验证跟踪

**原有代码：**
```python
# 需要手动计算收益
returns = (sell_price / buy_price - 1) * 100
```

**新代码：**
```python
from strategy_manager import SimpleBacktestEngine

engine = SimpleBacktestEngine(config)
backtest_df = engine.backtest_selection(
    selected_df=selected_stocks,
    buy_date=buy_date,
    hold_days=5
)

stats = engine.calculate_stats(backtest_df)
print(f"胜率: {stats['win_rate']}%")
print(f"平均收益: {stats['avg_return']}%")
```

## 参数调优

### 选股A 参数

```python
# 牛市参数
bull_params = {
    'min_pct_chg': 3.0,      # 降低涨幅要求
    'turnover_min': 2.0,
    'volume_ratio_min': 1.3
}

# 熊市参数
bear_params = {
    'min_pct_chg': 6.0,      # 提高涨幅要求
    'turnover_min': 4.0,
    'volume_ratio_min': 2.0
}
```

### 回测参数

```python
# 短线回测
short_term = {
    'hold_days': 3,         # 持有3天
}

# 中线回测
medium_term = {
    'hold_days': 10,        # 持有10天
}
```

## 常见问题

### Q1: 如何禁用适配器，使用原有模式？

```bash
python main_controller_v2.py select --no-adapter
```

或继续使用原有的 `main_controller.py`。

### Q2: 回测失败怎么办？

检查：
1. Tushare Token 是否配置
2. 买入日期是否有数据
3. 网络连接是否正常

### Q3: 数据格式不通过怎么办？

1. 使用 `data_format_checker.py` 检查具体缺失的字段
2. 添加必需字段
3. 确保字段类型正确（数值类型应为 float/int）

### Q4: 如何只运行特定的选股程序？

```bash
# 只运行选股A和C
python main_controller_v2.py select --screeners A C --use-adapter
```

## 性能优化建议

1. **减少API调用**
   - 使用缓存机制
   - 批量获取数据

2. **异步处理**
   - 多线程选股
   - 并行回测

3. **数据缓存**
   - 缓存历史数据
   - 避免重复获取

## 后续优化方向

1. **实盘对接**
   - 自动下单
   - 实时监控

2. **智能通知**
   - 微信/邮件通知
   - 异常预警

3. **可视化**
   - 收益曲线
   - 策略对比

4. **机器学习**
   - 自动参数优化
   - 策略推荐

## 技术支持

如有问题，请查看：
- `strategy_manager/README_ADAPTER.md` - 适配器详细文档
- `strategy_manager/demo_adapter.py` - 演示脚本
- `test_adapter_integration.py` - 测试脚本

## 版本历史

### V2.0 (2024)
- 整合策略管理器适配器
- 新增市场状态检测
- 新增回测功能
- 新增错误检测
- 增强的命令行选项

### V1.0
- 基础选股功能
- 验证跟踪
- 参数优化
