# DeepQuant 智能选股系统 - 技术文档

> **面向程序员的技术文档**：提供核心程序、参数配置和训练流程的详细说明

---

## 目录

1. [系统架构概览](#系统架构概览)
2. [核心程序清单](#核心程序清单)
3. [参数配置文件](#参数配置文件)
4. [训练流程详解](#训练流程详解)
5. [关键参数说明](#关键参数说明)
6. [代码优化建议](#代码优化建议)
7. [扩展开发指南](#扩展开发指南)

---

## 1. 系统架构概览

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      数据采集层                               │
│  MarketDataCollector (Tushare API)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     特征工程层                                │
│  AssaultFeatureEngineer (资金+情绪+技术)                     │
│  - 资金强度特征 (40%)                                        │
│  - 市场情绪特征 (35%)                                        │
│  - 技术动量特征 (25%)                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     模型训练层                                │
│  StockPredictor (XGBoost / RandomForest)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     决策与评估层                              │
│  ConfidenceBucketAnalyzer / AssaultDecisionBrain              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 技术栈

| 组件 | 技术栈 | 版本 |
|------|--------|------|
| 数据采集 | Tushare | 最新 |
| 数据处理 | Pandas, NumPy | 2.x, 1.24+ |
| 机器学习 | XGBoost, scikit-learn | 1.7+, 1.3+ |
| 模型集成 | LangGraph, LangChain | 1.0+ |

---

## 2. 核心程序清单

### 2.1 主训练脚本

**文件**: `scripts/run_real_data_assault.py`

**功能**: 完整的训练流程入口

**核心流程**:
```python
def run_real_data_pipeline(start_date, end_date, limit_stocks):
    # 1. 数据加载
    stock_data = load_real_stock_data(start_date, end_date, limit_stocks)
    
    # 2. 特征工程
    stock_data = process_real_data(stock_data)
    
    # 3. 数据划分
    train_data, test_data = split_data(stock_data, ratio=0.8)
    
    # 4. 模型训练
    model = train_model(train_data)
    
    # 5. 模型评估
    metrics = evaluate_model(model, test_data)
    
    # 6. 选股结果
    results = select_stocks(model, test_data, threshold=0.6)
```

**关键参数**:
```bash
--start-date: 开始日期 (默认: 2023-01-01)
--end-date: 结束日期 (默认: 2025-12-30)
--limit: 股票数量 (默认: 300)
```

**使用示例**:
```bash
# 基本运行
python scripts/run_real_data_assault.py

# 自定义参数
python scripts/run_real_data_assault.py \
    --start-date 2022-01-01 \
    --end-date 2024-12-31 \
    --limit 500
```

### 2.2 数据采集器

**文件**: `src/stock_system/data_collector.py`

**类**: `MarketDataCollector`

**核心方法**:
```python
class MarketDataCollector:
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        # 返回: ts_code, name, industry 等信息
        
    def get_daily_data(self, ts_code, start_date, end_date) -> pd.DataFrame:
        """获取日线数据"""
        # 返回: open, high, low, close, volume, amount 等
        
    def get_money_flow(self, ts_code, start_date, end_date) -> pd.DataFrame:
        """获取资金流向数据"""
        # 返回: buy_xxx_vol, sell_xxx_vol, net_xxx_vol 等
```

**配置文件**: `config/tushare_config.json`
```json
{
  "token": "your_token_here",
  "timeout": 30,
  "retry_count": 3,
  "max_workers": 5,
  "rate_limit_delay": 0.1,
  "cache_expiry_hours": 24
}
```

**优化点**:
- 多线程批量获取数据
- 智能缓存机制（24小时有效期）
- 自动重试机制（3次）

### 2.3 特征工程

**文件**: `src/stock_system/assault_features.py`

**类**: `AssaultFeatureEngineer`

**核心方法**:
```python
class AssaultFeatureEngineer:
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建所有特征"""
        df = self.create_capital_strength_features(df)   # 资金强度
        df = self.create_market_sentiment_features(df)   # 市场情绪
        df = self.create_technical_momentum_features(df) # 技术动量
        df = self.create_derived_features(df)             # 衍生特征
        return df
```

**特征分类**:

#### 2.3.1 资金强度特征 (权重40%)

| 特征名称 | 计算方法 | 阈值 |
|---------|---------|------|
| main_capital_inflow_ratio | OBV变化率 / 成交额 | >5%为强流入 |
| large_order_buy_rate | 大单买入额 / 总买入额 | >30%为强信号 |
| capital_inflow_persistence | 连续净流入天数 / 3 | ≥0.66为佳 |
| northbound_capital_flow | 相对强度（代理） | 板块前20% |

#### 2.3.2 市场情绪特征 (权重35%)

| 特征名称 | 计算方法 | 阈值 |
|---------|---------|------|
| sector_heat_index | 板块热度指数 | >0.1为过热 |
| stock_sentiment_score | 个股情绪得分 | >0.7为积极 |
| up_days_ratio | 上涨天数 / 总天数 | >0.6为普涨 |
| sentiment_cycle_position | 情绪周期位置 | 上升初期/主升段 |

#### 2.3.3 技术动量特征 (权重25%)

| 特征名称 | 计算方法 | 阈值 |
|---------|---------|------|
| enhanced_rsi | RSI(6)*0.4 + RSI(12)*0.3 + RSI(24)*0.3 | >60为强 |
| volume_price_breakout_strength | 成交量倍数 × 涨幅 | >2为强突破 |
| intraday_attack_pattern | 攻击形态识别 | 存在明显攻击波 |

### 2.4 预测器

**文件**: `src/stock_system/predictor.py`

**类**: `StockPredictor`

**核心方法**:
```python
class StockPredictor:
    def __init__(self, config_path: str = None):
        """初始化预测器"""
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.threshold = self.config.get('xgboost', {}).get('threshold', 0.5)
        
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """预测股票涨跌"""
        # 1. 准备特征
        X = self._prepare_features(data)
        
        # 2. 模型预测
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        
        # 3. 返回结果
        result = data.copy()
        result['predicted_label'] = predictions
        result['predicted_prob'] = probabilities
        return result
```

**配置文件**: `config/model_config.json`
```json
{
  "xgboost": {
    "model_path": "models/xgboost_model.pkl",
    "model_metadata_path": "models/xgboost_metadata.json",
    "threshold": 0.5,
    "params": {
      "objective": "binary:logistic",
      "eval_metric": "auc",
      "max_depth": 6,
      "eta": 0.1,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "seed": 42
    }
  },
  "data": {
    "train_features": ["main_capital_inflow_ratio", "large_order_buy_rate", ...]
  }
}
```

**当前实现**: 使用 RandomForestClassifier
```python
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
```

### 2.5 置信度分桶分析器

**文件**: `src/stock_system/confidence_bucket.py`

**类**: `ConfidenceBucketAnalyzer`

**功能**: 按预测概率分桶，分析不同置信度的精确率

**核心方法**:
```python
class ConfidenceBucketAnalyzer:
    def analyze(self, y_true, y_proba, threshold=0.5) -> Dict:
        """分析置信度分桶"""
        # 1. 将预测概率分桶
        buckets = self._create_buckets(y_proba)
        
        # 2. 计算每个桶的精确率
        bucket_metrics = []
        for bucket in buckets:
            precision = calculate_precision(bucket)
            bucket_metrics.append(precision)
        
        # 3. 返回结果
        return {
            'buckets': bucket_metrics,
            'overall_precision': overall_precision
        }
```

---

## 3. 参数配置文件

### 3.1 策略配置

**文件**: `config/short_term_assault_config.json`

**核心配置**:

```json
{
  "strategy_name": "短期突击特征权重体系",
  "core_philosophy": "少错过，不犯错，全身而退",
  "version": "3.0",
  
  "optimization_goals": {
    "recall": {
      "target": 0.80,
      "weight": 0.40
    },
    "precision": {
      "target": 0.50,
      "weight": 0.35
    },
    "overfitting_gap": {
      "target": 0.20,
      "weight": 0.15
    }
  },
  
  "feature_weights": {
    "capital_strength": {"weight": 0.40},
    "market_sentiment": {"weight": 0.35},
    "technical_momentum": {"weight": 0.25}
  }
}
```

### 3.2 模型配置

**文件**: `config/model_config.json`

**XGBoost 参数**:
```json
{
  "xgboost": {
    "params": {
      "objective": "binary:logistic",
      "eval_metric": "auc",
      "max_depth": 6,
      "eta": 0.1,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "min_child_weight": 1,
      "gamma": 0,
      "lambda": 1,
      "alpha": 0,
      "scale_pos_weight": 1,
      "seed": 42
    },
    "train_params": {
      "num_boost_round": 100,
      "early_stopping_rounds": 10,
      "verbose_eval": 10
    }
  }
}
```

### 3.3 环境变量配置

**文件**: `config/.env`

```bash
# Tushare API Token
TUSHARE_TOKEN=your_token_here

# 模型配置
MODEL_PATH=models/xgboost_model.pkl
CACHE_DIR=assets/data/market_cache
LOG_LEVEL=INFO
```

---

## 4. 训练流程详解

### 4.1 完整流程图

```
┌─────────────┐
│ 参数配置     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 数据加载     │
│ - 获取股票列表 │
│ - 过滤股票   │
│ - 采集数据   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 特征工程     │
│ - 资金强度   │
│ - 市场情绪   │
│ - 技术动量   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 标签生成     │
│ - 计算未来收益 │
│ - 生成正负样本 │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 数据划分     │
│ - 80%训练    │
│ - 20%测试    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 模型训练     │
│ - RandomForest │
│ - XGBoost   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 模型评估     │
│ - 分类报告   │
│ - AUC     │
│ - 置信度分桶 │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 选股结果     │
│ - 阈值筛选   │
│ - 统计分析   │
└─────────────┘
```

### 4.2 关键代码片段

#### 4.2.1 数据加载

```python
def load_real_stock_data(start_date, end_date, limit_stocks):
    # 1. 初始化数据采集器
    collector = MarketDataCollector()
    
    # 2. 获取股票列表
    stock_list = collector.get_stock_list()
    
    # 3. 过滤股票（排除科创板、创业板、ST股、北交所）
    stock_list = stock_list[
        (~stock_list['ts_code'].str.startswith('688')) &
        (~stock_list['ts_code'].str.startswith('300')) &
        (~stock_list['ts_code'].str.startswith('301')) &
        (~stock_list['ts_code'].str.startswith('BJ')) &
        (~stock_list['name'].str.contains('ST'))
    ]
    
    # 4. 限制数量
    if limit_stocks:
        stock_list = stock_list.head(limit_stocks)
    
    # 5. 批量获取数据
    all_data = []
    for _, stock in stock_list.iterrows():
        daily_data = collector.get_daily_data(stock['ts_code'], start_date, end_date)
        daily_data['ts_code'] = stock['ts_code']
        daily_data['name'] = stock['name']
        all_data.append(daily_data)
    
    return pd.concat(all_data, ignore_index=True)
```

#### 4.2.2 特征工程

```python
def process_real_data(stock_data):
    # 1. 排序
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    
    # 2. 为每只股票分别创建特征
    result_list = []
    for ts_code, group in stock_data.groupby('ts_code'):
        group = group.sort_values('trade_date').reset_index(drop=True)
        feature_engineer = AssaultFeatureEngineer()
        group = feature_engineer.create_all_features(group)
        result_list.append(group)
    
    # 3. 合并
    stock_data = pd.concat(result_list, ignore_index=True)
    
    # 4. 生成标签（未来收益）
    for window in [5, 10, 20]:
        stock_data[f'future_return_{window}d'] = (
            stock_data.groupby('ts_code')['close']
            .pct_change(window)
            .shift(-window)
        )
    
    # 5. 生成目标标签
    stock_data['target'] = (stock_data['future_return_10d'] > 0.05).astype(int)
    
    # 6. 清理数据
    stock_data = stock_data.dropna()
    
    return stock_data
```

#### 4.2.3 模型训练

```python
def train_model(train_data):
    # 1. 提取特征和标签
    exclude_columns = ['ts_code', 'name', 'trade_date', 'target', ...]
    feature_columns = [col for col in train_data.columns if col not in exclude_columns]
    
    X_train = train_data[feature_columns].fillna(0)
    y_train = train_data['target'].values
    
    # 2. 创建模型
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=100,      # 树的数量
        max_depth=10,          # 树的最大深度
        random_state=42,       # 随机种子
        class_weight='balanced' # 处理类别不平衡
    )
    
    # 3. 训练
    model.fit(X_train, y_train)
    
    return model
```

#### 4.2.4 模型评估

```python
def evaluate_model(model, test_data):
    # 1. 预测
    X_test = test_data[feature_columns].fillna(0)
    y_test = test_data['target'].values
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. 评估指标
    from sklearn.metrics import classification_report, roc_auc_score
    
    print(classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # 3. 置信度分桶分析
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = y_pred_proba > threshold
        avg_return = test_data[mask]['future_return_10d'].mean()
        up_rate = (test_data[mask]['future_return_10d'] > 0).mean()
        print(f"阈值 > {threshold}: {mask.sum()} 只 | 收益: {avg_return:.2%} | 胜率: {up_rate:.2%}")
```

---

## 5. 关键参数说明

### 5.1 数据参数

| 参数 | 类型 | 默认值 | 说明 | 建议 |
|------|------|--------|------|------|
| start_date | str | 2023-01-01 | 数据开始日期 | 建议2-3年 |
| end_date | str | 2025-12-30 | 数据结束日期 | 建议-6个月 |
| limit_stocks | int | 300 | 股票数量 | 建议300-500 |
| train_ratio | float | 0.8 | 训练集比例 | 建议0.8 |
| lookback_window | int | 20 | 回看窗口 | 建议20 |

### 5.2 标签参数

| 参数 | 类型 | 默认值 | 说明 | 建议 |
|------|------|--------|------|------|
| future_window | int | 10 | 未来窗口（天） | 建议5-20 |
| positive_threshold | float | 0.05 | 正样本阈值 | 建议0.03-0.1 |
| negative_threshold | float | -0.03 | 负样本阈值 | 建议-0.05-0 |

### 5.3 模型参数（RandomForest）

| 参数 | 类型 | 默认值 | 说明 | 建议 |
|------|------|--------|------|------|
| n_estimators | int | 100 | 树的数量 | 100-500 |
| max_depth | int | 10 | 树的最大深度 | 5-15 |
| min_samples_split | int | 2 | 分裂最小样本数 | 2-10 |
| min_samples_leaf | int | 1 | 叶节点最小样本数 | 1-5 |
| max_features | str | 'sqrt' | 最大特征数 | sqrt/log2 |
| class_weight | str | 'balanced' | 类别权重 | balanced |

### 5.4 模型参数（XGBoost）

| 参数 | 类型 | 默认值 | 说明 | 建议 |
|------|------|--------|------|------|
| max_depth | int | 6 | 树的最大深度 | 3-10 |
| eta (learning_rate) | float | 0.1 | 学习率 | 0.01-0.3 |
| subsample | float | 0.8 | 样本采样比例 | 0.6-1.0 |
| colsample_bytree | float | 0.8 | 特征采样比例 | 0.6-1.0 |
| min_child_weight | int | 1 | 最小子节点权重 | 1-10 |
| gamma | float | 0 | 最小分裂增益 | 0-5 |
| lambda (L2) | float | 1 | L2正则化 | 0-10 |
| alpha (L1) | float | 0 | L1正则化 | 0-10 |

### 5.5 选股参数

| 参数 | 类型 | 默认值 | 说明 | 建议 |
|------|------|--------|------|------|
| confidence_threshold | float | 0.6 | 预测概率阈值 | 0.6-0.8 |
| max_positions | int | 20 | 最大持仓数 | 10-30 |
| position_size | float | 0.05 | 单股仓位 | 0.02-0.1 |
| stop_loss | float | 0.08 | 止损比例 | 0.05-0.15 |

---

## 6. 代码优化建议

### 6.1 数据采集优化

**当前问题**:
- 单线程逐个获取数据，效率低
- 缓存策略简单，可能过期

**优化建议**:
```python
# 1. 使用多线程/异步获取
from concurrent.futures import ThreadPoolExecutor

def fetch_stock_data(stock_info):
    """获取单只股票数据"""
    collector = MarketDataCollector()
    return collector.get_daily_data(stock_info['ts_code'], start_date, end_date)

# 并行获取
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_stock_data, stock) for _, stock in stock_list.iterrows()]
    all_data = [future.result() for future in futures]

# 2. 优化缓存策略
# 按日期和股票代码分级缓存
# 支持增量更新
```

### 6.2 特征工程优化

**当前问题**:
- 特征数量较多（54个），可能存在冗余
- 部分特征使用简单代理，准确度不高

**优化建议**:
```python
# 1. 特征选择
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=30)
X_selected = selector.fit_transform(X_train, y_train)

# 2. 特征重要性分析
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# 3. 特征降维
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_train)

# 4. 特征交叉
# 例如：资金强度 × 技术动量
X['capital_momentum'] = X['main_capital_inflow_ratio'] * X['enhanced_rsi']
```

### 6.3 模型优化

**当前问题**:
- 仅使用单模型（RandomForest）
- 未进行超参数调优
- 未处理时间序列特性

**优化建议**:
```python
# 1. 使用 XGBoost
import xgboost as xgb

model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc'
)

# 2. 超参数调优（Optuna）
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    model = xgb.XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 3. 集成学习
from sklearn.ensemble import VotingClassifier

model1 = RandomForestClassifier(n_estimators=100, max_depth=10)
model2 = xgb.XGBClassifier(max_depth=6, learning_rate=0.1)
model3 = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.1)

ensemble = VotingClassifier(
    estimators=[('rf', model1), ('xgb', model2), ('lgb', model3)],
    voting='soft'
)

# 4. 时间序列交叉验证
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc')

# 5. 在线学习
# 使用滚动窗口训练，定期更新模型
```

### 6.4 评估优化

**当前问题**:
- 仅使用静态指标（准确率、AUC）
- 未考虑时间衰减
- 未计算实际交易成本

**优化建议**:
```python
# 1. 回测框架
def backtest(predictions, test_data, threshold=0.6, cost=0.001):
    """回测"""
    # 选择股票
    selected = predictions[predictions['predicted_prob'] > threshold]
    
    # 计算收益
    returns = []
    for _, row in selected.iterrows():
        actual_return = row['future_return_10d']
        net_return = actual_return - cost * 2  # 买入+卖出成本
        returns.append(net_return)
    
    # 计算指标
    total_return = np.mean(returns)
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(returns)
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': sum(1 for r in returns if r > 0) / len(returns)
    }

# 2. 时间衰减
# 近期的数据权重更高
def calculate_time_decay_weights(dates, current_date, decay_factor=0.95):
    """计算时间衰减权重"""
    days_diff = (current_date - dates).dt.days
    weights = decay_factor ** (days_diff / 30)  # 每月衰减
    return weights

# 3. 样本外测试
# 使用滚动窗口进行样本外测试
def rolling_window_test(data, window_size=252, test_size=63):
    """滚动窗口测试"""
    results = []
    for i in range(0, len(data) - window_size - test_size, test_size):
        train_data = data.iloc[i:i+window_size]
        test_data = data.iloc[i+window_size:i+window_size+test_size]
        
        model = train_model(train_data)
        result = evaluate_model(model, test_data)
        results.append(result)
    
    return results
```

### 6.5 代码结构优化

**当前问题**:
- 主训练脚本过长（400+ 行）
- 缺乏模块化

**优化建议**:
```python
# 1. 拆分为多个模块
# models/
#   - trainer.py: 模型训练
#   - evaluator.py: 模型评估
#   - predictor.py: 模型预测
# 
# features/
#   - capital_features.py: 资金强度特征
#   - sentiment_features.py: 市场情绪特征
#   - momentum_features.py: 技术动量特征
#
# utils/
#   - data_loader.py: 数据加载
#   - data_preprocessor.py: 数据预处理
#   - metrics.py: 评估指标

# 2. 使用配置类
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    start_date: str = '2023-01-01'
    end_date: str = '2025-12-30'
    limit_stocks: int = 300
    train_ratio: float = 0.8
    lookback_window: int = 20
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

# 3. 使用 Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', DataPreprocessor()),
    ('feature_selector', FeatureSelector(k=30)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier())
])

# 4. 添加日志和监控
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("开始训练模型...")
```

---

## 7. 扩展开发指南

### 7.1 添加新特征

```python
# src/stock_system/features/custom_features.py

def create_custom_feature(df: pd.DataFrame) -> pd.DataFrame:
    """创建自定义特征"""
    df = df.copy()
    
    # 例如：布林带
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    
    # 例如：K线形态识别
    df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1)
    df['hammer'] = ((df['close'] > df['open']) & 
                   (df['open'] - df['low']) > 2 * (df['high'] - df['close']))
    
    return df

# 在 AssaultFeatureEngineer 中集成
class AssaultFeatureEngineer:
    def create_all_features(self, df):
        df = self.create_capital_strength_features(df)
        df = self.create_market_sentiment_features(df)
        df = self.create_technical_momentum_features(df)
        df = create_custom_feature(df)  # 新增
        return df
```

### 7.2 添加新模型

```python
# src/stock_system/models/lightgbm_predictor.py

import lightgbm as lgb

class LightGBMPredictor:
    """LightGBM 预测器"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.model = None
        
    def train(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            early_stopping_rounds=10
        )
        
    def predict(self, X):
        """预测"""
        return self.model.predict(X)

# 在主脚本中集成
def train_with_multiple_models(X_train, y_train):
    """使用多模型训练"""
    models = {}
    
    # RandomForest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    
    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    models['xgboost'] = xgb_model
    
    # LightGBM
    lgb_model = LightGBMPredictor()
    lgb_model.train(X_train, y_train, X_val, y_val)
    models['lightgbm'] = lgb_model
    
    return models
```

### 7.3 添加新策略

```python
# src/stock_system/strategies/mean_reversion.py

class MeanReversionStrategy:
    """均值回归策略"""
    
    def __init__(self, window=20, std_threshold=2):
        self.window = window
        self.std_threshold = std_threshold
        
    def generate_signals(self, df):
        """生成交易信号"""
        # 计算均值和标准差
        mean = df['close'].rolling(self.window).mean()
        std = df['close'].rolling(self.window).std()
        
        # 生成信号
        df['z_score'] = (df['close'] - mean) / std
        df['signal'] = 0
        
        # 买入信号：价格低于均值2个标准差
        df.loc[df['z_score'] < -self.std_threshold, 'signal'] = 1
        
        # 卖出信号：价格高于均值2个标准差
        df.loc[df['z_score'] > self.std_threshold, 'signal'] = -1
        
        return df

# 在决策大脑中集成
class AssaultDecisionBrain:
    def __init__(self):
        self.predictor = StockPredictor()
        self.strategies = {
            'mean_reversion': MeanReversionStrategy(),
            'trend_following': TrendFollowingStrategy(),
            'momentum': MomentumStrategy()
        }
        
    def make_decision(self, stock_data):
        """综合决策"""
        # 1. 模型预测
        prediction = self.predictor.predict(stock_data)
        
        # 2. 策略信号
        strategy_signals = {}
        for name, strategy in self.strategies.items():
            signals = strategy.generate_signals(stock_data)
            strategy_signals[name] = signals
        
        # 3. 综合决策
        final_decision = self.combine_signals(prediction, strategy_signals)
        
        return final_decision
```

### 7.4 添加数据源

```python
# src/stock_system/data_sources/akshare_collector.py

import akshare as ak

class AkshareDataCollector:
    """AKShare 数据采集器"""
    
    def get_stock_list(self):
        """获取股票列表"""
        return ak.stock_info_a_code_name()
    
    def get_daily_data(self, ts_code, start_date, end_date):
        """获取日线数据"""
        symbol = ts_code.split('.')[0]
        return ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', '')
        )

# 在数据采集器中集成
class MarketDataCollector:
    def __init__(self, data_source='tushare'):
        if data_source == 'tushare':
            self.collector = TushareDataCollector()
        elif data_source == 'akshare':
            self.collector = AkshareDataCollector()
        else:
            raise ValueError(f"不支持的数据源: {data_source}")
```

---

## 8. 附录

### 8.1 文件结构

```
workspace/projects/
├── config/                          # 配置文件
│   ├── .env                         # 环境变量
│   ├── tushare_config.json           # Tushare配置
│   ├── model_config.json             # 模型配置
│   └── short_term_assault_config.json # 策略配置
├── src/
│   ├── stock_system/                 # 核心模块
│   │   ├── data_collector.py         # 数据采集
│   │   ├── assault_features.py       # 特征工程
│   │   ├── predictor.py              # 预测器
│   │   ├── confidence_bucket.py      # 置信度分析
│   │   └── assault_decision_brain.py # 决策大脑
│   └── main.py                       # 主入口
├── scripts/                         # 脚本
│   ├── run_real_data_assault.py      # 真实数据训练
│   └── check_config.py              # 配置检查
├── assets/                          # 资源文件
│   ├── data/                        # 数据文件
│   ├── models/                      # 模型文件
│   └── reports/                     # 报告文件
└── docs/                            # 文档
```

### 8.2 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置Token
echo "TUSHARE_TOKEN=your_token_here" > config/.env

# 3. 运行训练
python scripts/run_real_data_assault.py --limit 100

# 4. 查看结果
cat assets/results/real_data_selection_results.csv
```

### 8.3 常见问题

**Q1: 如何更换模型？**
A: 修改 `scripts/run_real_data_assault.py` 中的模型定义

**Q2: 如何调整特征权重？**
A: 修改 `config/short_term_assault_config.json` 中的 `feature_weights`

**Q3: 如何增加股票数量？**
A: 使用 `--limit` 参数，如 `--limit 500`

**Q4: 如何使用其他数据源？**
A: 实现新的数据采集器类，并在 `MarketDataCollector` 中集成

---

## 9. 联系与贡献

- **项目地址**: [待补充]
- **问题反馈**: [待补充]
- **贡献指南**: [待补充]

---

**文档版本**: v1.0  
**最后更新**: 2026-02-04  
**维护者**: DeepQuant Team
