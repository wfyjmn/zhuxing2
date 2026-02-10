# DeepQuant 程序员文档索引

> **面向程序员的完整文档集**：从快速开始到深度开发

---

## 📚 文档导航

### 1. 快速开始

| 文档 | 路径 | 说明 |
|------|------|------|
| 快速开始指南 | `docs/REAL_DATA_QUICKSTART.md` | 5分钟上手指南 |
| 快速参考卡 | `docs/quick_reference.md` | 参数速查、代码片段 |
| 交互式脚本 | `scripts/interactive_training.py` | 一键配置和训练 |

### 2. 技术文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 技术文档 | `docs/technical_documentation.md` | 核心程序、参数配置、训练流程详解 |
| 架构设计 | [待补充] | 系统架构、模块设计 |
| API文档 | [待补充] | 函数接口、类方法 |

### 3. 使用指南

| 文档 | 路径 | 说明 |
|------|------|------|
| 真实数据使用指南 | `docs/real_data_usage_guide.md` | 数据采集、配置、使用说明 |
| 训练报告模板 | `assets/reports/model_training_report_template.md` | 训练报告模板 |

### 4. 分析报告

| 文档 | 路径 | 说明 |
|------|------|------|
| 小规模测试分析 | `assets/reports/real_stock_selection_analysis.md` | 50只股票测试报告 |
| 选股策略分类 | `assets/reports/selected_stocks_by_strategy.md` | 策略分类报告 |
| 大规模训练报告 | `assets/reports/large_scale_training_report.md` | 300只股票训练报告 |

---

## 🚀 快速开始

### 方式1: 交互式脚本（推荐）

```bash
# 运行交互式脚本
python scripts/interactive_training.py

# 按照提示选择操作:
# 1. 检查配置
# 2. 交互式配置
# 3. 快速开始
# 4. 测试模型
# 5. 查看统计信息
```

### 方式2: 命令行

```bash
# 1. 检查配置
python scripts/interactive_training.py --check

# 2. 快速开始
python scripts/interactive_training.py --start

# 3. 使用自定义参数
python scripts/run_real_data_assault.py \
    --start-date 2023-01-01 \
    --end-date 2025-12-30 \
    --limit 300 \
    --threshold 0.6
```

### 方式3: 配置文件

```bash
# 1. 编辑配置
vim config/.env
# 添加: TUSHARE_TOKEN=your_token_here

# 2. 运行训练
python scripts/run_real_data_assault.py

# 3. 查看结果
cat assets/results/real_data_selection_results.csv
```

---

## 📊 核心概念

### 特征体系

```
短期突击特征权重体系
├── 资金强度 (40%)
│   ├── 主力资金净流入占比
│   ├── 大单净买入率
│   ├── 资金流入持续性
│   └── 北向资金流入
├── 市场情绪 (35%)
│   ├── 板块热度指数
│   ├── 个股情绪得分
│   ├── 上涨天数占比
│   └── 情绪周期位置
└── 技术动量 (25%)
    ├── 增强RSI
    ├── 量价突破强度
    └── 盘中攻击形态
```

### 模型配置

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| n_estimators | 100 | 树的数量 | 50-500 |
| max_depth | 10 | 最大深度 | 5-15 |
| threshold | 0.6 | 预测阈值 | 0.5-0.8 |
| future_window | 10 | 未来窗口(天) | 5-20 |

### 选股策略

```
选股流程:
1. 数据采集 (300只股票)
2. 特征工程 (54个特征)
3. 模型预测 (RandomForest)
4. 置信度分桶 (阈值0.6)
5. 选股输出 (平均收益7.5%)
```

---

## 🔧 核心模块

### 数据采集器

```python
from stock_system.data_collector import MarketDataCollector

collector = MarketDataCollector()
stock_list = collector.get_stock_list()
daily_data = collector.get_daily_data('000001.SZ', '2023-01-01', '2023-12-31')
```

### 特征工程

```python
from stock_system.assault_features import AssaultFeatureEngineer

engineer = AssaultFeatureEngineer()
df = engineer.create_all_features(df)
```

### 预测器

```python
from stock_system.predictor import StockPredictor

predictor = StockPredictor()
result = predictor.predict(test_data)
```

---

## 📈 常见操作

### 1. 更新数据

```bash
# 使用新的时间范围
python scripts/run_real_data_assault.py \
    --start-date 2024-01-01 \
    --end-date 2025-12-30
```

### 2. 增加股票数量

```bash
# 从300只增加到500只
python scripts/run_real_data_assault.py --limit 500
```

### 3. 调整阈值

```bash
# 提高阈值到0.7（更精确，但召回率降低）
python scripts/run_real_data_assault.py --threshold 0.7
```

### 4. 切换模型

```python
# 在 run_real_data_assault.py 中修改
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# 使用XGBoost
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
```

---

## 🐛 故障排查

### 问题1: Token未配置

```bash
# 检查配置
python scripts/interactive_training.py --check

# 设置Token
echo "TUSHARE_TOKEN=your_token_here" > config/.env
```

### 问题2: 特征数量为0

```python
# 检查数据
print(df.columns.tolist())

# 检查特征工程
df = engineer.create_all_features(df)
print(df.columns.tolist())
```

### 问题3: 模型过拟合

```python
# 减少模型复杂度
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,  # 从10降到8
    min_samples_leaf=3  # 从1增加到3
)
```

---

## 📝 开发指南

### 添加新特征

```python
# 1. 在 assault_features.py 中添加特征
def create_custom_feature(df):
    df['custom_feature'] = ...
    return df

# 2. 在 create_all_features 中调用
df = create_custom_feature(df)
```

### 添加新模型

```python
# 1. 创建新的预测器类
class CustomPredictor:
    def train(self, X, y):
        self.model = ...
        
    def predict(self, X):
        return self.model.predict(X)

# 2. 在主脚本中使用
predictor = CustomPredictor()
predictor.train(X_train, y_train)
```

### 添加新策略

```python
# 1. 创建策略类
class CustomStrategy:
    def generate_signals(self, df):
        df['signal'] = ...
        return df

# 2. 在决策大脑中集成
from assault_decision_brain import AssaultDecisionBrain

brain = AssaultDecisionBrain()
brain.strategies['custom'] = CustomStrategy()
```

---

## 📞 获取帮助

- **快速参考**: `docs/quick_reference.md`
- **技术文档**: `docs/technical_documentation.md`
- **使用指南**: `docs/real_data_usage_guide.md`
- **GitHub Issues**: [待补充]

---

## 📦 项目结构

```
workspace/projects/
├── config/                          # 配置文件
│   ├── .env                         # 环境变量
│   ├── tushare_config.json           # Tushare配置
│   ├── model_config.json             # 模型配置
│   └── short_term_assault_config.json # 策略配置
├── src/
│   ├── stock_system/                 # 核心模块
│   │   ├── data_collector.py         # 数据采集器
│   │   ├── assault_features.py       # 特征工程
│   │   ├── predictor.py              # 预测器
│   │   ├── confidence_bucket.py      # 置信度分析
│   │   └── assault_decision_brain.py # 决策大脑
│   └── main.py                       # 主入口
├── scripts/                         # 脚本
│   ├── run_real_data_assault.py      # 主训练脚本
│   ├── interactive_training.py       # 交互式脚本
│   └── check_config.py              # 配置检查
├── assets/                          # 资源文件
│   ├── data/                        # 数据文件
│   ├── models/                      # 模型文件
│   ├── results/                     # 结果文件
│   └── reports/                     # 报告文件
├── docs/                            # 文档
│   ├── technical_documentation.md    # 技术文档
│   ├── quick_reference.md           # 快速参考
│   ├── real_data_usage_guide.md     # 使用指南
│   └── REAL_DATA_QUICKSTART.md      # 快速开始
└── requirements.txt                 # 依赖列表
```

---

## 🎯 下一步

### 初学者

1. 阅读 `docs/REAL_DATA_QUICKSTART.md`
2. 运行 `python scripts/interactive_training.py`
3. 查看 `docs/quick_reference.md`

### 进阶用户

1. 阅读 `docs/technical_documentation.md`
2. 修改配置文件
3. 添加自定义特征

### 高级开发者

1. 研究核心模块代码
2. 开发新模型和策略
3. 优化系统性能

---

**文档版本**: v1.0  
**最后更新**: 2026-02-04  
**维护者**: DeepQuant Team
