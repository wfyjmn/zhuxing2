# 真实股票数据使用指南

## 概述

本指南说明如何使用真实股票数据运行柱形突击选股系统。

## 步骤 1: 获取 Tushare Token

1. 访问 [Tushare 官网](https://tushare.pro/register)
2. 注册账号并登录
3. 点击右上角头像 -> 个人中心
4. 在"接口Token"页面点击"获取Token"
5. 复制生成的 Token（格式类似：xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx）

## 步骤 2: 配置 Token

### 方法 1: 使用环境变量（推荐）

在终端中执行：

```bash
export TUSHARE_TOKEN=你的token_here
```

### 方法 2: 使用 .env 文件

1. 复制模板文件：

```bash
cp config/.env.template config/.env
```

2. 编辑 `config/.env` 文件，将 Token 替换为你的真实 Token：

```bash
TUSHARE_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 步骤 3: 运行真实数据选股

### 基本运行

```bash
python scripts/run_real_data_assault.py
```

### 自定义参数运行

```bash
python scripts/run_real_data_assault.py \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --limit 50
```

### 参数说明

- `--start-date`: 数据开始日期（默认：2023-01-01）
- `--end-date`: 数据结束日期（默认：2024-12-31）
- `--limit`: 限制股票数量，用于测试（默认：50，设为 None 表示不限制）

## 步骤 4: 查看结果

运行完成后，会生成以下文件：

### 1. 原始数据
- **文件**: `assets/data/real_stock_data.csv`
- **内容**: 包含所有股票的历史行情、特征和标签

### 2. 选股结果
- **文件**: `assets/results/real_data_selection_results.csv`
- **内容**: 高置信度选股结果

查看选股结果：

```bash
head -20 assets/results/real_data_selection_results.csv
```

## 数据筛选规则

系统会自动过滤以下股票：

1. **科创板** (688xxx): 风险较高
2. **创业板** (300xxx, 301xxx): 风险较高
3. **ST股**: 特别处理股票，风险较高
4. **北交所** (BJxxx): 交易规则不同

## 流程说明

真实数据选股流程包含以下步骤：

1. **数据获取**: 从 Tushare 获取历史行情数据
2. **特征工程**: 生成 42 个特征（资金强度、市场情绪、技术动量）
3. **标签生成**: 计算未来收益，生成目标标签
4. **模型训练**: 使用随机森林训练模型
5. **模型评估**: 计算精确率、召回率、AUC 等指标
6. **置信度分桶**: 按预测概率分桶分析
7. **选股**: 输出高置信度股票（预测概率 > 0.7）

## 性能优化建议

### 1. 增加数据量

获取更多股票和更长时间的数据：

```bash
python scripts/run_real_data_assault.py \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --limit 500
```

### 2. 调整标签阈值

编辑 `scripts/run_real_data_assault.py`，修改标签生成逻辑：

```python
# 原代码：10日内涨幅>5%为正样本
stock_data['target'] = (stock_data['future_return_10d'] > 0.05).astype(int)

# 修改为：10日内涨幅>3%为正样本（更宽松）
stock_data['target'] = (stock_data['future_return_10d'] > 0.03).astype(int)

# 或修改为：10日内涨幅>10%为正样本（更严格）
stock_data['target'] = (stock_data['future_return_10d'] > 0.10).astype(int)
```

### 3. 调整选股阈值

修改高置信度选股的阈值：

```python
# 原代码：预测概率 > 0.7
high_conf_mask = y_pred_proba > 0.7

# 修改为：预测概率 > 0.6（更宽松）
high_conf_mask = y_pred_proba > 0.6

# 或修改为：预测概率 > 0.8（更严格）
high_conf_mask = y_pred_proba > 0.8
```

## 常见问题

### Q1: Token 配置正确，但提示"未配置 Token"

**A**: 检查 .env 文件路径是否正确，确保文件在项目根目录的 `config/` 目录下。

### Q2: 获取数据很慢

**A**: 这是正常的，Tushare API 有调用频率限制。建议：
- 先用少量股票测试（--limit 10）
- 调整 `config/tushare_config.json` 中的 `max_workers` 参数

### Q3: 出现"数据不足"错误

**A**: 可能原因：
- 时间范围太短
- 股票数量太少
- 某些股票在该时间段无数据

建议增加时间范围和股票数量。

### Q4: 模型性能很差

**A**: 可能原因：
- 数据量不足
- 特征工程质量
- 标签阈值设置不合理

建议：
- 增加数据量
- 调整标签阈值
- 尝试不同的模型

## 安全提示

⚠️ **重要**: Token 是您的私密凭证，请勿：

1. 将 Token 提交到 Git 仓库
2. 在公开场合分享
3. 在不安全的环境中使用

.gitignore 已配置自动忽略 `.env` 文件。

## 下一步

完成真实数据选股后，您可以：

1. **分析结果**: 查看选股结果的统计特性
2. **回测验证**: 使用历史数据验证策略有效性
3. **参数优化**: 调整模型参数和选股阈值
4. **实盘模拟**: 使用模拟交易验证策略

## 技术支持

如有问题，请检查：

1. Tushare Token 是否正确
2. 网络连接是否正常
3. 数据时间范围是否合理
4. Python 依赖是否完整安装

祝投资顺利！
