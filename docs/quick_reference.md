# DeepQuant 快速参考卡

> **面向程序员的快速参考**：核心参数、常用命令、代码片段

---

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置Token
```bash
# 方式1: 编辑配置文件
vim config/.env
# 添加: TUSHARE_TOKEN=your_token_here

# 方式2: 命令行
echo "TUSHARE_TOKEN=your_token_here" > config/.env
```

### 运行训练
```bash
# 默认参数
python scripts/run_real_data_assault.py

# 自定义参数
python scripts/run_real_data_assault.py \
    --start-date 2022-01-01 \
    --end-date 2024-12-31 \
    --limit 500
```

---

## 📊 核心参数速查

### 数据参数
```python
# 数据范围
start_date: str = '2023-01-01'  # 开始日期
end_date: str = '2025-12-30'    # 结束日期

# 股票筛选
limit_stocks: int = 300         # 股票数量
train_ratio: float = 0.8        # 训练集比例

# 过滤规则（自动执行）
exclude_kcb: bool = True        # 排除科创板(688)
exclude_gem: bool = True        # 排除创业板(300/301)
exclude_st: bool = True         # 排除ST股
exclude_bj: bool = True         # 排除北交所(BJ)
```

### 标签参数
```python
# 标签定义
future_window: int = 10         # 未来窗口(天)
positive_threshold: float = 0.05  # 正样本阈值(+5%)
negative_threshold: float = -0.03 # 负样本阈值(-3%)

# 样本划分
lookback_window: int = 20       # 特征计算窗口
min_samples: int = 100          # 最小样本数
```

### 模型参数
```python
# RandomForest (当前使用)
model = RandomForestClassifier(
    n_estimators=100,           # 树的数量: 50-500
    max_depth=10,               # 最大深度: 5-15
    min_samples_split=2,        # 最小分裂样本: 2-10
    min_samples_leaf=1,         # 叶节点最小样本: 1-5
    max_features='sqrt',        # 最大特征数: sqrt/log2
    class_weight='balanced',    # 类别权重
    random_state=42             # 随机种子
)

# XGBoost (可选)
model = xgb.XGBClassifier(
    max_depth=6,                # 最大深度: 3-10
    learning_rate=0.1,          # 学习率: 0.01-0.3
    n_estimators=100,           # 迭代次数: 50-300
    subsample=0.8,              # 样本采样: 0.6-1.0
    colsample_bytree=0.8,       # 特征采样: 0.6-1.0
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)
```

### 选股参数
```python
# 阈值设置
confidence_threshold: float = 0.6  # 预测概率阈值: 0.6-0.8
min_probability: float = 0.5       # 最小概率

# 仓位管理
max_positions: int = 20            # 最大持仓数: 10-30
position_size: float = 0.05        # 单股仓位: 0.02-0.1

# 风控参数
stop_loss: float = 0.08            # 止损比例: 0.05-0.15
take_profit: float = 0.15          # 止盈比例: 0.10-0.20
```

---

## 🔧 常用代码片段

### 数据加载
```python
from stock_system.data_collector import MarketDataCollector

collector = MarketDataCollector()

# 获取股票列表
stock_list = collector.get_stock_list()

# 获取单只股票数据
daily_data = collector.get_daily_data(
    ts_code='000001.SZ',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# 批量获取
all_data = []
for _, stock in stock_list.head(100).iterrows():
    data = collector.get_daily_data(stock['ts_code'], start_date, end_date)
    data['ts_code'] = stock['ts_code']
    all_data.append(data)
```

### 特征工程
```python
from stock_system.assault_features import AssaultFeatureEngineer

engineer = AssaultFeatureEngineer()

# 创建所有特征
df = engineer.create_all_features(df)

# 单独创建特征
df = engineer.create_capital_strength_features(df)
df = engineer.create_market_sentiment_features(df)
df = engineer.create_technical_momentum_features(df)
```

### 模型训练
```python
from sklearn.ensemble import RandomForestClassifier

# 提取特征和标签
exclude_cols = ['ts_code', 'name', 'trade_date', 'target', 
                'future_return_5d', 'future_return_10d', 'future_return_20d']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].fillna(0)
y = df['target'].values

# 训练模型
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X, y)
```

### 模型预测
```python
from stock_system.predictor import StockPredictor

predictor = StockPredictor()

# 预测
result = predictor.predict(test_data)

# 查看结果
print(result[['ts_code', 'trade_date', 'predicted_label', 'predicted_prob']].head())

# 高置信度选股
high_conf = result[result['predicted_prob'] > 0.7]
print(f"高置信度股票: {len(high_conf)} 只")
```

### 模型评估
```python
from sklearn.metrics import classification_report, roc_auc_score

# 分类报告
print(classification_report(y_test, y_pred))

# AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")

# 置信度分桶
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = y_pred_proba > threshold
    if mask.sum() > 0:
        precision = (y_test[mask] == 1).sum() / mask.sum()
        avg_return = test_data[mask]['future_return_10d'].mean()
        print(f"阈值>{threshold}: {mask.sum()}只 | 精确率:{precision:.2%} | 收益:{avg_return:.2%}")
```

### 特征重要性分析
```python
import pandas as pd

# 特征重要性
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

# 打印Top 20
print(feature_importance.head(20))

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.barh(feature_importance['feature'][:20], feature_importance['importance'][:20])
plt.title('Top 20 特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.tight_layout()
plt.show()
```

---

## 📁 核心文件速查

### 主程序
```
scripts/run_real_data_assault.py    # 主训练脚本
src/main.py                         # 主入口
```

### 核心模块
```
src/stock_system/
├── data_collector.py               # 数据采集器
├── assault_features.py             # 特征工程
├── predictor.py                    # 预测器
├── confidence_bucket.py            # 置信度分析
└── assault_decision_brain.py       # 决策大脑
```

### 配置文件
```
config/
├── .env                            # 环境变量
├── tushare_config.json             # Tushare配置
├── model_config.json               # 模型配置
└── short_term_assault_config.json  # 策略配置
```

---

## 🎯 特征工程详解

### 资金强度特征 (40%权重)

| 特征 | 代码 | 阈值 |
|------|------|------|
| 主力资金净流入占比 | `main_capital_inflow_ratio` | >5% |
| 大单净买入率 | `large_order_buy_rate` | >30% |
| 资金流入持续性 | `capital_inflow_persistence` | ≥0.66 |
| 北向资金流入 | `northbound_capital_flow` | 板块前20% |

### 市场情绪特征 (35%权重)

| 特征 | 代码 | 阈值 |
|------|------|------|
| 板块热度指数 | `sector_heat_index` | >0.1 |
| 个股情绪得分 | `stock_sentiment_score` | >0.7 |
| 上涨天数占比 | `up_days_ratio` | >0.6 |
| 情绪周期位置 | `sentiment_cycle_position` | 上升初期 |

### 技术动量特征 (25%权重)

| 特征 | 代码 | 阈值 |
|------|------|------|
| 增强RSI | `enhanced_rsi` | >60 |
| 量价突破强度 | `volume_price_breakout_strength` | >2 |
| 盘中攻击形态 | `intraday_attack_pattern` | 存在明显攻击波 |

---

## 🐛 常见问题排查

### 问题1: Token未配置
```
错误: ValueError('TUSHARE_TOKEN is not set in environment variables')

解决:
1. 检查 config/.env 文件是否存在
2. 检查文件中是否包含 TUSHARE_TOKEN
3. 运行检查脚本: python scripts/check_config.py
```

### 问题2: 特征数量为0
```
错误: ValueError('No features found')

解决:
1. 检查数据是否包含价格和成交量字段
2. 检查特征工程代码是否正确执行
3. 打印列名: print(df.columns.tolist())
```

### 问题3: 测试集为空
```
错误: ValueError('Test set is empty')

解决:
1. 检查数据时间范围是否足够
2. 检查 train_ratio 参数
3. 使用80%/20%的时间序列划分
```

### 问题4: 模型训练失败
```
错误: RuntimeError('Model training failed')

解决:
1. 检查数据是否有缺失值: df.isnull().sum()
2. 检查特征是否为数值类型: df.dtypes
3. 检查标签是否平衡: y.value_counts()
```

---

## 📈 性能优化技巧

### 1. 数据加载优化
```python
# 多线程加载
from concurrent.futures import ThreadPoolExecutor

def fetch_stock(stock_info):
    collector = MarketDataCollector()
    return collector.get_daily_data(stock_info['ts_code'], start_date, end_date)

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_stock, stock) for stock in stock_list]
    all_data = [f.result() for f in futures]
```

### 2. 特征选择
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=30)
X_selected = selector.fit_transform(X_train, y_train)
```

### 3. 超参数调优
```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300)
    }
    model = xgb.XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## 🎨 自定义扩展

### 添加新特征
```python
def create_custom_feature(df):
    df = df.copy()
    
    # 例如：布林带
    df['bb_upper'] = df['close'].rolling(20).mean() + 2*df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2*df['close'].rolling(20).std()
    
    return df
```

### 添加新模型
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=100
)
model.fit(X_train, y_train)
```

### 添加新策略
```python
class CustomStrategy:
    def generate_signals(self, df):
        df['signal'] = 0
        df.loc[df['close'] > df['ma20'], 'signal'] = 1
        df.loc[df['close'] < df['ma20'], 'signal'] = -1
        return df
```

---

## 📞 获取帮助

- 完整文档: `docs/technical_documentation.md`
- 快速开始: `docs/REAL_DATA_QUICKSTART.md`
- 使用指南: `docs/real_data_usage_guide.md`
- 示例代码: `assets/reports/`

---

**版本**: v1.0  
**更新**: 2026-02-04
