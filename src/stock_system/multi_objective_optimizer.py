"""
多目标优化框架
使用Optuna进行多目标优化：精确率、召回率、交易成本
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Callable
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Install with: pip install optuna")


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(
        self, 
        data: pd.DataFrame,
        features: List[str],
        target: str,
        config: Dict[str, Any] = None
    ):
        """
        初始化多目标优化器
        
        Args:
            data: 数据集
            features: 特征列表
            target: 目标变量
            config: 配置参数
        """
        self.data = data
        self.features = features
        self.target = target
        self.config = config or {}
        
        # 默认配置
        self.method = self.config.get('method', 'optuna')
        self.sampler = self.config.get('sampler', 'TPESampler')
        self.n_trials = self.config.get('n_trials', 100)
        self.random_state = self.config.get('random_state', 42)
        
        # 优化目标和约束
        self.optimization_metrics = self.config.get('optimization_metrics', {
            'primary': 'precision',
            'constraint': 'recall',
            'penalty': 'trade_count'
        })
        self.trade_cost_penalty = self.config.get('trade_cost_penalty', 0.001)
        self.recall_constraint = self.config.get('recall_constraint', 0.7)
        
    def optimize(
        self,
        generate_signals_func: Callable,
        study_name: str = "multi_objective_optimization"
    ) -> Tuple[Dict[str, Any], float]:
        """
        执行多目标优化
        
        Args:
            generate_signals_func: 生成信号的函数，接受(data, features, params)返回信号序列
            study_name: study名称
        
        Returns:
            (最佳参数, 最佳得分)
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available, falling back to grid search")
            return self._grid_search_optimization(generate_signals_func)
        
        def objective(trial):
            # 超参数搜索空间
            params = self._get_search_space(trial)
            
            # 应用阈值生成信号
            try:
                signals = generate_signals_func(self.data, self.features, params)
            except Exception as e:
                return -1000  # 惩罚异常
            
            # 计算多个指标
            precision = precision_score(self.data[self.target], signals, zero_division=0)
            recall = recall_score(self.data[self.target], signals, zero_division=0)
            trade_count = signals.sum()
            
            # 多目标：最大化精确率，召回率≥70%，最小化交易次数
            if recall < self.recall_constraint:
                # 不满足召回率约束，给予重罚
                return -1000 + recall * 10
            
            # 目标函数：精确率 - 交易成本惩罚
            trade_cost_penalty = trade_count * self.trade_cost_penalty
            score = precision - trade_cost_penalty
            
            return score
        
        # 创建研究
        if self.sampler == 'TPESampler':
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        elif self.sampler == 'RandomSampler':
            sampler = optuna.samplers.RandomSampler(seed=self.random_state)
        else:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=study_name
        )
        
        # 优化
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params, study.best_value
    
    def _get_search_space(self, trial) -> Dict[str, Any]:
        """定义参数搜索空间"""
        params = {}
        
        # RSI相关阈值
        params['rsi_threshold'] = trial.suggest_float('rsi_threshold', 40, 70)
        params['rsi_period'] = trial.suggest_categorical('rsi_period', [6, 12, 14, 24])
        
        # 成交量相关阈值
        params['volume_threshold'] = trial.suggest_float('volume_threshold', 1.2, 3.0)
        
        # 资金相关阈值
        params['capital_threshold'] = trial.suggest_float('capital_threshold', 0.03, 0.10)
        
        # 情绪相关阈值
        params['sentiment_threshold'] = trial.suggest_float('sentiment_threshold', 60, 90)
        
        # 模型阈值
        params['model_threshold'] = trial.suggest_float('model_threshold', 0.3, 0.7)
        
        return params
    
    def _grid_search_optimization(
        self,
        generate_signals_func: Callable
    ) -> Tuple[Dict[str, Any], float]:
        """网格搜索优化（当Optuna不可用时）"""
        print("Performing grid search optimization...")
        
        # 定义参数网格
        rsi_thresholds = np.arange(40, 71, 10)
        model_thresholds = np.arange(0.3, 0.8, 0.1)
        
        best_params = {}
        best_score = -1
        
        for rsi_th in rsi_thresholds:
            for model_th in model_thresholds:
                params = {
                    'rsi_threshold': rsi_th,
                    'model_threshold': model_th,
                    'volume_threshold': 2.0,
                    'capital_threshold': 0.05,
                    'sentiment_threshold': 75
                }
                
                try:
                    signals = generate_signals_func(self.data, self.features, params)
                    
                    precision = precision_score(self.data[self.target], signals, zero_division=0)
                    recall = recall_score(self.data[self.target], signals, zero_division=0)
                    trade_count = signals.sum()
                    
                    if recall < self.recall_constraint:
                        continue
                    
                    score = precision - trade_count * self.trade_cost_penalty
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                except:
                    continue
        
        return best_params, best_score
    
    def optimize_with_pareto(
        self,
        generate_signals_func: Callable,
        study_name: str = "pareto_optimization"
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[pd.DataFrame]]:
        """
        帕累托前沿优化（多目标，不使用惩罚项）
        
        Args:
            generate_signals_func: 生成信号的函数
            study_name: study名称
        
        Returns:
            (帕累托前沿参数列表, 帕累托前沿数据框列表)
        """
        if not OPTUNA_AVAILABLE:
            raise ValueError("Optuna is required for Pareto optimization")
        
        # 多目标优化：最大化精确率和召回率
        def objective(trial):
            params = self._get_search_space(trial)
            
            try:
                signals = generate_signals_func(self.data, self.features, params)
            except:
                return -1.0, -1.0  # 惩罚异常
            
            precision = precision_score(self.data[self.target], signals, zero_division=0)
            recall = recall_score(self.data[self.target], signals, zero_division=0)
            
            return precision, recall
        
        # 创建多目标研究
        study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            study_name=study_name
        )
        
        # 优化
        study.optimize(objective, n_trials=self.n_trials)
        
        # 获取帕累托前沿
        pareto_front = study.best_trials
        
        # 整理结果
        pareto_params = []
        pareto_metrics = []
        
        for trial in pareto_front:
            params = trial.params
            precision, recall = trial.values
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            pareto_params.append(params)
            pareto_metrics.append({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return pareto_params, pd.DataFrame(pareto_metrics)


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(
        self,
        model_class,
        param_space: Dict[str, Tuple],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any] = None
    ):
        """
        初始化超参数优化器
        
        Args:
            model_class: 模型类（如XGBClassifier）
            param_space: 参数搜索空间，如 {'max_depth': (3, 8), 'learning_rate': (0.01, 0.1)}
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            config: 配置参数
        """
        self.model_class = model_class
        self.param_space = param_space
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.config = config or {}
        
        self.n_trials = self.config.get('n_trials', 100)
        self.eval_metric = self.config.get('eval_metric', 'f1')
        self.recall_constraint = self.config.get('recall_constraint', 0.7)
        
    def optimize(self, use_optuna: bool = True) -> Tuple[Dict[str, Any], float]:
        """
        优化超参数
        
        Args:
            use_optuna: 是否使用Optuna
        
        Returns:
            (最佳参数, 最佳得分)
        """
        if use_optuna and OPTUNA_AVAILABLE:
            return self._optimize_with_optuna()
        else:
            return self._optimize_with_random_search()
    
    def _optimize_with_optuna(self) -> Tuple[Dict[str, Any], float]:
        """使用Optuna优化"""
        def objective(trial):
            params = {}
            
            # 从搜索空间获取参数
            for param_name, (low, high) in self.param_space.items():
                if isinstance(low, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
            
            # 训练模型
            model = self.model_class(**params, random_state=42)
            model.fit(self.X_train, self.y_train)
            
            # 预测
            y_pred = model.predict(self.X_val)
            y_proba = model.predict_proba(self.X_val)[:, 1]
            
            # 计算指标
            precision = precision_score(self.y_val, y_pred, zero_division=0)
            recall = recall_score(self.y_val, y_pred, zero_division=0)
            f1 = f1_score(self.y_val, y_pred, zero_division=0)
            
            # 召回率约束
            if recall < self.recall_constraint:
                return -1000
            
            if self.eval_metric == 'precision':
                return precision
            elif self.eval_metric == 'recall':
                return recall
            elif self.eval_metric == 'f1':
                return f1
            else:
                return f1
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params, study.best_value
    
    def _optimize_with_random_search(self) -> Tuple[Dict[str, Any], float]:
        """使用随机搜索优化"""
        print("Using random search for hyperparameter optimization...")
        
        best_params = {}
        best_score = -1
        
        # 随机采样
        for _ in range(min(self.n_trials, 50)):  # 限制次数
            params = {}
            
            for param_name, (low, high) in self.param_space.items():
                if isinstance(low, int):
                    params[param_name] = np.random.randint(low, high + 1)
                else:
                    params[param_name] = np.random.uniform(low, high)
            
            # 训练和评估
            model = self.model_class(**params, random_state=42)
            model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_val)
            recall = recall_score(self.y_val, y_pred, zero_division=0)
            
            if recall >= self.recall_constraint:
                if self.eval_metric == 'precision':
                    score = precision_score(self.y_val, y_pred, zero_division=0)
                elif self.eval_metric == 'recall':
                    score = recall
                else:
                    score = f1_score(self.y_val, y_pred, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        return best_params, best_score


def create_sample_generate_signals_function():
    """创建示例的信号生成函数"""
    def generate_signals(data, features, params):
        """生成交易信号"""
        # 简单示例：基于多个特征的加权组合
        signals = pd.Series(0, index=data.index)
        
        # RSI信号
        if 'rsi_6' in data.columns:
            rsi_signal = (data['rsi_6'] > params.get('rsi_threshold', 50)).astype(int)
            signals += rsi_signal
        
        # 模型概率信号
        if 'model_probability' in data.columns:
            model_signal = (data['model_probability'] > params.get('model_threshold', 0.5)).astype(int)
            signals += model_signal
        
        # 成交量信号
        if 'volume_ratio' in data.columns:
            volume_signal = (data['volume_ratio'] > params.get('volume_threshold', 2.0)).astype(int)
            signals += volume_signal
        
        # 综合信号（只要满足任意条件）
        final_signal = (signals > 0).astype(int)
        
        return final_signal
    
    return generate_signals
