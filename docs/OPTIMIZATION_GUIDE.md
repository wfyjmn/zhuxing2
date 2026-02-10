# 选股程序优化说明

## 优化概述

针对选股A（`ai_stock_screener.py`）和选股B（`ai_stock_screener_v2.py`）中发现的潜在问题，进行了全面优化，提升了程序的稳定性和性能。

## 优化版本

- ✅ `ai_stock_screener_optimized.py` - 选股A优化版
- ✅ `ai_stock_screener_v2_optimized.py` - 选股B优化版

---

## 主要优化内容

### 1. API调用效率优化

#### 问题
- `pro.daily()` 批量获取所有股票，可能触发Tushare频率限制
- 循环中逐个筛选，没有使用分批处理

#### 解决方案
```python
# 添加API调用配置
API_CONFIG = {
    'retry_times': 3,           # 重试次数
    'retry_delay': 1,           # 重试间隔（秒）
    'request_delay': 0.3,       # 请求间隔（秒）
    'batch_size': 1000,         # 批量获取数量
}

# 实现分批获取函数
def get_daily_data_batch(ts_codes, start_date, end_date, batch_size=None):
    """分批获取历史数据，避免频率限制"""
    if batch_size is None:
        batch_size = API_CONFIG['batch_size']

    all_data = []
    total = len(ts_codes)

    for i in range(0, total, batch_size):
        batch = ts_codes[i:i + batch_size]
        print(f"    - 正在获取第{i+1}-{min(i+batch_size, total)}/{total}只股票的历史数据...")

        try:
            df = api_call_with_retry(
                pro.daily,
                ts_code=batch,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"    ❌ 获取批次数据失败: {e}")
            continue

    if len(all_data) == 0:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)
```

#### 优化效果
- ✅ 避免一次性请求过多数据触发频率限制
- ✅ 支持大批量股票数据处理
- ✅ 提供详细的进度提示

---

### 2. 数据处理优化

#### 问题
- 代码中存在重复查询：`df[df['ts_code'] == ts_code].iloc[0]`
- 数据处理效率有待提升

#### 解决方案
```python
# 优化：使用更高效的数据处理方式
df_hist = df_hist.sort_values(['ts_code', 'trade_date'])

# 计算每只股票的5日平均成交量
df_hist_5d = df_hist.groupby('ts_code')['vol'].rolling(5).mean().reset_index()
df_hist_5d.columns = ['ts_code', 'vol_5d']
df_hist_5d = df_hist_5d.dropna().groupby('ts_code').last()

df = df.merge(df_hist_5d[['vol_5d']], on='ts_code', how='left')
```

#### 优化效果
- ✅ 减少重复查询
- ✅ 使用向量化操作提升性能
- ✅ 避免使用iloc逐行查询

---

### 3. 换手率计算修正（重要）

#### 问题
选股B中的换手率计算公式错误：

```python
# ❌ 错误的计算方式
'换手率(%)': round(row_data['vol'] / row_data['amount'] * 100, 2)
```

这个公式是错误的，因为：
- `vol` 是成交量（股）
- `amount` 是成交额（元）
- 比例关系不正确

#### 解决方案
```python
# ✅ 正确的方式：直接从 daily_basic 获取 turnover_rate
df_daily_basic = api_call_with_retry(
    pro.daily_basic,
    trade_date=trade_date,
    fields='ts_code,pe_ttm,total_mv,circ_mv,turnover_rate'  # 添加turnover_rate
)

# 使用Tushare提供的换手率（单位：%）
if 'turnover_rate' in df.columns:
    df['turnover_rate'] = df['turnover_rate'].fillna(0)
else:
    # 如果没有换手率数据，则跳过
    print("    ⚠️  未获取到换手率数据")
```

#### 优化效果
- ✅ 使用官方提供的换手率数据，准确性更高
- ✅ 避免错误计算导致选股结果偏差
- ✅ 提升数据可靠性

---

### 4. 异常处理增强

#### 问题
- 网络波动可能导致API调用失败
- 缺少重试机制
- 没有请求间隔

#### 解决方案
```python
def api_call_with_retry(func, *args, **kwargs):
    """
    带重试机制的API调用
    """
    for attempt in range(API_CONFIG['retry_times']):
        try:
            result = func(*args, **kwargs)
            time.sleep(API_CONFIG['request_delay'])  # 请求间隔
            return result
        except Exception as e:
            if attempt < API_CONFIG['retry_times'] - 1:
                print(f"  ⚠️  API调用失败（第{attempt+1}次尝试）: {e}")
                print(f"  ⏳  {API_CONFIG['retry_delay']}秒后重试...")
                time.sleep(API_CONFIG['retry_delay'])
            else:
                print(f"  ❌ API调用失败（已达最大重试次数）: {e}")
                raise
    return None
```

#### 优化效果
- ✅ 自动重试失败的请求
- ✅ 避免临时网络问题导致程序中断
- ✅ 提供清晰的错误提示
- ✅ 请求间隔避免触发频率限制

---

## 使用建议

### 1. 逐步迁移

建议先使用优化版本进行测试，验证结果正确性后再替换原版本：

```bash
# 测试优化版
cd /workspace/projects
python scripts/ai_stock_screener_optimized.py
python scripts/ai_stock_screener_v2_optimized.py

# 对比结果
diff assets/data/selected_stocks_20260210.csv \
     <（优化版输出的文件）
```

### 2. 替换原版本

确认优化版本运行稳定后，可以替换原版本：

```bash
# 备份原版本
mv scripts/ai_stock_screener.py scripts/ai_stock_screener_backup.py
mv scripts/ai_stock_screener_v2.py scripts/ai_stock_screener_v2_backup.py

# 使用优化版本
mv scripts/ai_stock_screener_optimized.py scripts/ai_stock_screener.py
mv scripts/ai_stock_screener_v2_optimized.py scripts/ai_stock_screener_v2.py
```

### 3. 配置参数调整

根据实际情况调整API配置：

```python
API_CONFIG = {
    'retry_times': 3,           # 网络不稳定时可增加到5
    'retry_delay': 1,           # 根据需要调整
    'request_delay': 0.3,       # 避免触发频率限制
    'batch_size': 1000,         # 可调整为500或2000
}
```

---

## 性能对比

| 指标 | 原版本 | 优化版本 | 改进 |
|------|--------|----------|------|
| API失败重试 | ❌ 不支持 | ✅ 支持 | 稳定性大幅提升 |
| 批量处理 | ❌ 不支持 | ✅ 支持 | 避免频率限制 |
| 换手率准确性 | ❌ 错误 | ✅ 正确 | 数据准确性提升 |
| 错误提示 | ❌ 基础 | ✅ 详细 | 便于问题排查 |
| 请求间隔 | ❌ 无 | ✅ 0.3秒 | 避免触发限制 |

---

## 注意事项

1. **换手率数据源**：优化版直接使用 `daily_basic` 接口的 `turnover_rate` 字段，这是Tushare官方提供的准确数据。

2. **分批处理**：批量获取历史数据时，会显示进度信息，请耐心等待。

3. **重试机制**：默认重试3次，如果网络不稳定，可以增加到5次。

4. **请求间隔**：每次API调用后等待0.3秒，避免触发频率限制。

---

## 后续优化建议

1. **缓存机制**：可以添加本地缓存，避免重复获取相同数据
2. **并行处理**：使用多线程或异步IO加速数据获取
3. **配置文件**：将API配置移到配置文件中，便于调整
4. **日志记录**：添加详细的日志记录，便于问题追踪

---

## 总结

优化版本解决了原版本存在的四大问题：
- ✅ API调用效率低 → 使用分批处理和重试机制
- ✅ 数据处理冗余 → 优化数据处理逻辑
- ✅ 换手率计算错误 → 使用官方数据源
- ✅ 缺少异常处理 → 增强异常处理能力

建议先测试优化版本，确认运行稳定后再替换原版本。

---

**优化日期**: 2026年2月10日
**优化版本**: v2.0
**作者**: Agent搭建专家
