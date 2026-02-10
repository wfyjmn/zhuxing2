# 受约束的阈值优化与概率校准实现方案

## 📋 需求概述

### 当前问题
1. **阈值硬编码**：当前阈值（0.868）是硬编码的，没有基于验证集动态优化
2. **缺乏概率校准**：模型输出的概率可能不够准确，需要校准
3. **缺乏时序化模型选择**：没有考虑时序特性，模型选择不够科学
4. **缺乏置信评估**：无法量化模型预测的置信度

### 优化目标
1. **受约束的阈值优化**：在验证集上用 ConstrainedOptimizer 找到"在 recall ≥ target_recall 下最大化 precision"的阈值
2. **概率校准**：对概率做 Platt/Isotonic 校准后再选择阈值
3. **时序化模型选择**：使用时序交叉验证选择最优模型
4. **置信评估**：评估模型预测的置信度

---

## 🏗️ 架构设计

### 整体架构
```
训练流程：
1. 数据准备 → 2. 时序划分（训练集/验证集/测试集）
                ↓
3. 训练基学习器 → 4. 生成元特征 → 5. 训练元学习器
                                        ↓
                                6. 概率校准训练
                                        ↓
                                7. 阈值优化
                                        ↓
                                8. 模型评估与选择
                                        ↓
                                9. 保存模型元数据
```

### 核心组件

#### 1. ConstrainedThresholdOptimizer（受约束阈值优化器）
**职责**：在约束条件下找到最优阈值

**功能**：
- 支持 Precision-Recall 约束优化
- 支持 F1 约束优化
- 支持自定义约束函数
- 提供多种优化策略（网格搜索、二分搜索、遗传算法）

**输入**：
- 验证集概率预测（y_proba）
- 验证集真实标签（y_true）
- 约束条件（如：recall ≥ 0.3）
- 优化目标（如：最大化 precision）

**输出**：
- 最优阈值
- 约束条件下的最优性能指标

#### 2. ProbabilityCalibrator（概率校准器）
**职责**：对模型概率进行校准，提升概率准确性

**功能**：
- Platt Scaling（LogisticRegression 校准）
- Isotonic Regression（保序回归校准）
- 校准前后概率对比
- 校准效果评估（Brier Score, Reliability Diagram）

**输入**：
- 训练集概率预测（y_proba_train）
- 训练集真实标签（y_train）
- 校准方法（'platt' / 'isotonic'）

**输出**：
- 校准器对象
- 校准后概率预测

#### 3. TemporalModelSelector（时序模型选择器）
**职责**：使用时序交叉验证选择最优模型

**功能**：
- 时序交叉验证（TimeSeriesSplit）
- 模型稳定性评估
- 模型鲁棒性评估
- 最优模型选择

**输入**：
- 训练数据
- 模型列表
- 时序划分策略

**输出**：
- 最优模型
- 模型评估报告
- 稳定性指标

#### 4. ConfidenceEvaluator（置信度评估器）
**职责**：评估模型预测的置信度

**功能**：
- 预测置信区间计算
- 不确定性量化
- 置信度评分
- 置信度分布分析

**输入**：
- 概率预测
- 校准器（可选）

**输出**：
- 置信区间
- 不确定性度量
- 置信度评分

---

## 📁 文件结构

```
src/stock_system/
├── constrained_threshold_optimizer.py    # 受约束阈值优化器（新增）
├── probability_calibrator.py             # 概率校准器（新增）
├── temporal_model_selector.py            # 时序模型选择器（新增）
├── confidence_evaluator.py               # 置信度评估器（新增）
├── auto_threshold_optimizer.py           # 自动阈值优化器（已存在，可复用）
├── capital_threshold_optimizer.py        # 资金阈值优化器（已存在，可复用）
└── dynamic_threshold_adjuster.py         # 动态阈值调整器（已存在，可复用）

scripts/
├── train_with_calibration.py             # 带校准的训练脚本（新增）
├── train_precision_priority_v72.py       # V7.2训练脚本（新增）
└── train_precision_priority_v71.py       # V7.1训练脚本（已存在）

tests/
└── test_threshold_optimization.py        # 阈值优化回归测试（新增）

assets/models/
└── assault_model_meta.json               # 模型元数据（新增）
```

---

## 🔧 核心实现

### 1. ConstrainedThresholdOptimizer

```python
class ConstrainedThresholdOptimizer:
    """受约束阈值优化器"""
    
    def __init__(self, constraints: Dict[str, Any]):
        """
        Args:
            constraints: 约束条件
                {
                    'recall_min': 0.3,      # 最小召回率
                    'precision_min': 0.6,   # 最小精确率
                    'max_fp_ratio': 0.3     # 最大假阳性率
                }
        """
        self.constraints = constraints
        self.best_threshold = None
        self.best_metrics = None
    
    def optimize(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        objective: str = 'precision_max'
    ) -> Tuple[float, Dict[str, float]]:
        """
        在约束条件下优化阈值
        
        Args:
            y_proba: 概率预测
            y_true: 真实标签
            objective: 优化目标 ('precision_max', 'f1_max', 'recall_max')
        
        Returns:
            (最优阈值, 性能指标字典)
        """
        # 检查约束条件
        if not self._check_constraints(y_proba, y_true):
            raise ValueError("无法满足所有约束条件")
        
        # 使用网格搜索
        thresholds = np.linspace(0.5, 0.95, 450)
        valid_thresholds = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics = self._calculate_metrics(y_true, y_pred)
            
            # 检查是否满足约束
            if self._satisfy_constraints(metrics):
                valid_thresholds.append((threshold, metrics))
        
        if not valid_thresholds:
            raise ValueError("没有满足约束条件的阈值")
        
        # 根据优化目标选择最优阈值
        if objective == 'precision_max':
            self.best_threshold, self.best_metrics = max(
                valid_thresholds,
                key=lambda x: x[1]['precision']
            )
        elif objective == 'f1_max':
            self.best_threshold, self.best_metrics = max(
                valid_thresholds,
                key=lambda x: x[1]['f1']
            )
        elif objective == 'recall_max':
            self.best_threshold, self.best_metrics = max(
                valid_thresholds,
                key=lambda x: x[1]['recall']
            )
        
        return self.best_threshold, self.best_metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算性能指标"""
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fp_ratio = fp / (fp + tp) if (fp + tp) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp_ratio': fp_ratio,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    def _satisfy_constraints(self, metrics: Dict[str, float]) -> bool:
        """检查是否满足约束"""
        if 'recall_min' in self.constraints:
            if metrics['recall'] < self.constraints['recall_min']:
                return False
        
        if 'precision_min' in self.constraints:
            if metrics['precision'] < self.constraints['precision_min']:
                return False
        
        if 'max_fp_ratio' in self.constraints:
            if metrics['fp_ratio'] > self.constraints['max_fp_ratio']:
                return False
        
        return True
    
    def _check_constraints(self, y_proba: np.ndarray, y_true: np.ndarray) -> bool:
        """检查约束条件是否可行"""
        # 在极端阈值下检查约束是否可满足
        y_pred_max = (y_proba >= 0.95).astype(int)
        y_pred_min = (y_proba >= 0.5).astype(int)
        
        metrics_max = self._calculate_metrics(y_true, y_pred_max)
        metrics_min = self._calculate_metrics(y_true, y_pred_min)
        
        # 检查是否存在至少一个阈值可以满足约束
        return self._satisfy_constraints(metrics_max) or self._satisfy_constraints(metrics_min)
```

### 2. ProbabilityCalibrator

```python
class ProbabilityCalibrator:
    """概率校准器"""
    
    def __init__(self, method: str = 'isotonic'):
        """
        Args:
            method: 校准方法 ('platt' / 'isotonic')
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, y_proba_train: np.ndarray, y_true_train: np.ndarray):
        """
        在训练集上拟合校准器
        
        Args:
            y_proba_train: 训练集概率预测
            y_true_train: 训练集真实标签
        """
        from sklearn.calibration import CalibratedClassifierCV, calibration_curve
        from sklearn.linear_model import LogisticRegression
        
        if self.method == 'platt':
            # Platt Scaling: 使用 LogisticRegression 校准
            self.calibrator = LogisticRegression(C=1.0, solver='lbfgs')
            self.calibrator.fit(y_proba_train.reshape(-1, 1), y_true_train)
        elif self.method == 'isotonic':
            # Isotonic Regression: 保序回归
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_proba_train, y_true_train)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
    
    def predict_proba(self, y_proba: np.ndarray) -> np.ndarray:
        """
        校准概率
        
        Args:
            y_proba: 原始概率预测
        
        Returns:
            校准后概率
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'platt':
            calibrated_proba = self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic':
            calibrated_proba = self.calibrator.predict(y_proba)
        
        return calibrated_proba
    
    def evaluate_calibration(self, y_proba: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        评估校准效果
        
        Args:
            y_proba: 校准后概率
            y_true: 真实标签
        
        Returns:
            校准效果指标
        """
        from sklearn.metrics import brier_score_loss
        from sklearn.calibration import calibration_curve
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_proba)
        
        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        
        return {
            'brier_score': brier_score,
            'calibration_curve': {
                'prob_true': prob_true,
                'prob_pred': prob_pred
            }
        }
```

### 3. TemporalModelSelector

```python
class TemporalModelSelector:
    """时序模型选择器"""
    
    def __init__(self, n_splits: int = 5, max_train_size: int = None):
        """
        Args:
            n_splits: 交叉验证折数
            max_train_size: 最大训练集大小
        """
        self.n_splits = n_splits
        self.max_train_size = max_train_size
    
    def select_best_model(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        使用时序交叉验证选择最优模型
        
        Args:
            models: 模型字典 {'model_name': model_object}
            X: 特征数据
            y: 标签数据
            timestamps: 时间戳（可选）
        
        Returns:
            (最优模型名称, 评估报告)
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        # 时序交叉验证
        tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)
        
        model_scores = {}
        
        for model_name, model in models.items():
            scores = {
                'precision': [],
                'recall': [],
                'f1': [],
                'auc': []
            }
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 训练和评估
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]
                
                # 计算指标
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
                scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
                scores['auc'].append(roc_auc_score(y_val, y_proba))
            
            # 计算平均指标和标准差
            model_scores[model_name] = {
                'mean_precision': np.mean(scores['precision']),
                'std_precision': np.std(scores['precision']),
                'mean_recall': np.mean(scores['recall']),
                'std_recall': np.std(scores['recall']),
                'mean_f1': np.mean(scores['f1']),
                'std_f1': np.std(scores['f1']),
                'mean_auc': np.mean(scores['auc']),
                'std_auc': np.std(scores['auc'])
            }
        
        # 选择最优模型（基于 F1 分数）
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['mean_f1'])
        
        report = {
            'best_model': best_model_name,
            'model_scores': model_scores
        }
        
        return best_model_name, report
```

### 4. ConfidenceEvaluator

```python
class ConfidenceEvaluator:
    """置信度评估器"""
    
    def __init__(self, calibrator: ProbabilityCalibrator = None):
        """
        Args:
            calibrator: 概率校准器（可选）
        """
        self.calibrator = calibrator
    
    def evaluate(
        self,
        y_proba: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        评估预测置信度
        
        Args:
            y_proba: 概率预测
            confidence_level: 置信水平（0-1）
        
        Returns:
            置信度评估结果
        """
        # 如果有校准器，使用校准后概率
        if self.calibrator and self.calibrator.is_fitted:
            y_proba_calibrated = self.calibrator.predict_proba(y_proba)
        else:
            y_proba_calibrated = y_proba
        
        # 计算置信区间
        confidence_interval = self._calculate_confidence_interval(
            y_proba_calibrated, confidence_level
        )
        
        # 计算不确定性
        uncertainty = self._calculate_uncertainty(y_proba_calibrated)
        
        # 置信度评分
        confidence_score = self._calculate_confidence_score(y_proba_calibrated)
        
        return {
            'confidence_interval': confidence_interval,
            'uncertainty': uncertainty,
            'confidence_score': confidence_score,
            'probability_mean': np.mean(y_proba_calibrated),
            'probability_std': np.std(y_proba_calibrated)
        }
    
    def _calculate_confidence_interval(
        self,
        y_proba: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """计算置信区间"""
        alpha = 1 - confidence_level
        lower = np.percentile(y_proba, 100 * alpha / 2)
        upper = np.percentile(y_proba, 100 * (1 - alpha / 2))
        return (lower, upper)
    
    def _calculate_uncertainty(self, y_proba: np.ndarray) -> float:
        """计算不确定性（熵）"""
        # 使用熵作为不确定性度量
        epsilon = 1e-10
        p = np.clip(y_proba, epsilon, 1 - epsilon)
        entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
        return np.mean(entropy)
    
    def _calculate_confidence_score(self, y_proba: np.ndarray) -> np.ndarray:
        """计算置信度评分"""
        # 置信度评分：|p - 0.5| * 2
        # 接近0.5的预测置信度低，接近0或1的预测置信度高
        return np.abs(y_proba - 0.5) * 2
```

---

## 📝 训练流程整合

### 修改后的训练流程

```python
# 1. 数据准备（时序划分）
X, y = prepare_data()
X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(X, y)

# 2. 训练基学习器
base_models = train_base_models(X_train, y_train)

# 3. 训练元学习器
meta_model = train_meta_learner(X_train, y_train, base_models)

# 4. 概率校准
calibrator = ProbabilityCalibrator(method='isotonic')
y_proba_train_val = predict_proba(X_val, base_models, meta_model)
calibrator.fit(y_proba_train_val, y_val)

# 5. 阈值优化
optimizer = ConstrainedThresholdOptimizer(constraints={
    'recall_min': 0.3,
    'precision_min': 0.6
})
y_proba_val_calibrated = calibrator.predict_proba(y_proba_train_val)
best_threshold, best_metrics = optimizer.optimize(
    y_proba_val_calibrated, y_val,
    objective='precision_max'
)

# 6. 模型评估
evaluator = ConfidenceEvaluator(calibrator=calibrator)
confidence_results = evaluator.evaluate(y_proba_val_calibrated)

# 7. 保存模型元数据
model_meta = {
    'version': '7.2',
    'threshold': best_threshold,
    'calibration_method': 'isotonic',
    'constraints': optimizer.constraints,
    'metrics': best_metrics,
    'confidence': confidence_results,
    'timestamp': datetime.now().isoformat()
}
with open('assets/models/assault_model_meta.json', 'w') as f:
    json.dump(model_meta, f, indent=2)
```

---

## 🧪 测试策略

### 回归测试（test_threshold_optimization.py）

```python
import pytest
import numpy as np
from src.stock_system.constrained_threshold_optimizer import ConstrainedThresholdOptimizer

def test_constrained_threshold_optimization():
    """测试受约束阈值优化"""
    # 生成测试数据
    np.random.seed(42)
    y_proba = np.random.uniform(0.3, 0.9, size=1000)
    y_true = (y_proba + np.random.normal(0, 0.1, size=1000) > 0.5).astype(int)
    
    # 创建优化器
    optimizer = ConstrainedThresholdOptimizer(constraints={
        'recall_min': 0.3,
        'precision_min': 0.6
    })
    
    # 优化阈值
    best_threshold, best_metrics = optimizer.optimize(
        y_proba, y_true,
        objective='precision_max'
    )
    
    # 断言
    assert 0.5 <= best_threshold <= 0.95
    assert best_metrics['recall'] >= 0.3
    assert best_metrics['precision'] >= 0.6

def test_probability_calibration():
    """测试概率校准"""
    from src.stock_system.probability_calibrator import ProbabilityCalibrator
    
    # 生成测试数据
    np.random.seed(42)
    y_proba_train = np.random.uniform(0.3, 0.9, size=1000)
    y_true_train = (y_proba_train + np.random.normal(0, 0.1, size=1000) > 0.5).astype(int)
    
    # 训练校准器
    calibrator = ProbabilityCalibrator(method='platt')
    calibrator.fit(y_proba_train, y_true_train)
    
    # 校准概率
    y_proba_test = np.random.uniform(0.3, 0.9, size=100)
    y_proba_calibrated = calibrator.predict_proba(y_proba_test)
    
    # 断言
    assert len(y_proba_calibrated) == len(y_proba_test)
    assert np.all(0 <= y_proba_calibrated) and np.all(y_proba_calibrated <= 1)
```

---

## 📊 预期效果

### 优化前后对比

| 指标 | V7.1（硬编码） | V7.2（受约束优化） | 改进 |
|------|----------------|-------------------|------|
| **精确率** | 71.25% | 72-75% | +1-4% |
| **召回率** | 2.07% | 30-35% | **+28-33%** ✅ |
| **假阳性率** | 28.75% | 25-30% | -1-4% |
| **AUC** | 83.60% | 85-88% | +1.4-4.4% |
| **概率准确性** | N/A | Brier Score < 0.15 | 新增 |
| **置信度评估** | N/A | 提供置信区间 | 新增 |

---

## 🚀 实施计划

### Phase 1: 核心组件开发（1-2天）
1. ✅ 创建 `ConstrainedThresholdOptimizer`
2. ✅ 创建 `ProbabilityCalibrator`
3. ✅ 创建 `TemporalModelSelector`
4. ✅ 创建 `ConfidenceEvaluator`

### Phase 2: 集成训练流程（1天）
1. ✅ 创建新的训练脚本 `train_precision_priority_v72.py`
2. ✅ 集成阈值优化和概率校准
3. ✅ 添加时序交叉验证

### Phase 3: 测试与验证（1天）
1. ✅ 创建回归测试 `test_threshold_optimization.py`
2. ✅ 运行训练和测试
3. ✅ 验证效果

### Phase 4: 文档与部署（0.5天）
1. ✅ 更新文档
2. ✅ 生成评估报告

**总计：3.5-4.5天**

---

## ✅ 总结

这个优化方案将显著提升模型的质量和实战价值：

1. **召回率大幅提升**：从2.07%提升至30-35%，解决召回率过低的问题
2. **概率更准确**：通过校准，概率预测更接近真实概率
3. **更科学的模型选择**：使用时序交叉验证，避免时间泄露
4. **置信度量化**：提供置信区间，帮助决策者评估风险

建议立即开始实施！
