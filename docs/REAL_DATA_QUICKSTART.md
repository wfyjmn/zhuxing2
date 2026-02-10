# 柱形突击选股系统 - 真实数据快速开始

## 快速开始

### 1. 配置 Tushare Token

```bash
# 方式1: 使用环境变量（推荐）
export TUSHARE_TOKEN=你的token_here

# 方式2: 创建配置文件
cp config/.env.template config/.env
# 编辑 config/.env，将 Token 替换为你的真实 Token
```

### 2. 验证配置

```bash
python scripts/check_config.py
```

### 3. 运行真实数据选股

```bash
# 基本运行（使用 50 只股票，2023-2024 年数据）
python scripts/run_real_data_assault.py

# 自定义参数
python scripts/run_real_data_assault.py \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --limit 100
```

## 查看结果

```bash
# 查看选股结果（前20行）
head -20 assets/results/real_data_selection_results.csv

# 使用 Pandas 查看
python -c "
import pandas as pd
df = pd.read_csv('assets/results/real_data_selection_results.csv')
print(df[['ts_code', 'name', 'trade_date', 'predicted_prob', 'future_return_10d']].head(20))
"
```

## 详细文档

- [真实数据使用指南](docs/real_data_usage_guide.md) - 完整的使用说明
- [快速开始指南](#快速开始) - 3分钟上手
- [常见问题](docs/real_data_usage_guide.md#常见问题) - 问题排查

## 文件说明

### 配置文件
- `config/.env.template` - 环境变量模板
- `config/.env` - 实际环境变量（需要创建）

### 运行脚本
- `scripts/check_config.py` - 检查配置是否正确
- `scripts/run_real_data_assault.py` - 运行真实数据选股

### 输出文件
- `assets/data/real_stock_data.csv` - 原始数据
- `assets/results/real_data_selection_results.csv` - 选股结果

## 数据筛选规则

系统自动过滤以下股票：
- ❌ 科创板 (688xxx)
- ❌ 创业板 (300xxx, 301xxx)
- ❌ ST股
- ❌ 北交所 (BJxxx)

## 选股策略

### 特征维度
1. **资金强度 (40%)** - 主力资金、大单、资金持续性
2. **市场情绪 (35%)** - 板块热度、个股情绪、市场广度
3. **技术动量 (25%)** - RSI、量价突破、攻击形态

### 选股标准
- 预测概率 > 0.7（高置信度）
- 10日内涨幅 > 5%（作为训练标签）

## 性能指标

运行后会显示：
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数
- AUC
- 各置信度分桶的精确率

## 获取 Token

1. 访问 https://tushare.pro/register
2. 注册账号并登录
3. 点击右上角头像 -> 个人中心
4. 在"接口Token"页面获取 Token

## 安全提示

⚠️ Token 是您的私密凭证，请勿提交到 Git 仓库。

## 技术支持

如有问题，请先运行：
```bash
python scripts/check_config.py
```

检查配置是否正确。
