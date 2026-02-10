"""
基于约束的优化算法
在召回率约束下优化精确率和F1分数
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Callable
from scipy.optimize import minimize
from sklearn.metrics import precision_score, recall_score, f1_score


class ConstrainedOptimizer:
    """约束优化器"""
    
    def __init__(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        model,
        config: Dict[str, Any] = None
    ):
        """
        初始化约束优化器
        
        Args:
            X_val: 验证集特征
            y_val: 验证集标签
            model: 训练好的模型（需要有predict_proba方法）
            config: 配置参数
        """
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.config = config or {}
        
        # 配置参数
        self.recall_target = self.config.get('min_recall', 0.70)
        self.threshold_bounds = self.config.get('threshold_bounds', [0.3, 0.7])
        self.algorithm = self.config.get('algorithm', 'Nelder-Mead')
        self.max_iterations = self.config.get('max_iterations', 100)
        
    def optimize_threshold_for_precision(
        self,
        recall_constraint: float = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        在召回率约束下优化精确率
        
        Args:
            recall_constraint: 召回率约束，默认使用配置中的值
        
        Returns:
            (最优阈值, 性能指标字典)
        """
        if recall_constraint is None:
            recall_constraint = self.recall_target
        
        # 获取模型预测概率
        y_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        def objective(threshold):
            """目标函数：最大化精确率，约束召回率"""
            y_pred = (y_proba > threshold).astype(int)
            
            try:
                precision = precision_score(self.y_val, y_pred, zero_division=0)
                recall = recall_score(self.y_val, y_pred, zero_division=0)
            except:
                return 1000  # 惩罚项
            
            # 目标:最大化精确率，约束:召回率≥目标值
            if recall >= recall_constraint:
                # 如果满足约束，最大化精确率（最小化负精确率）
                return -precision
            else:
                # 如果不满足约束，给予重罚
                penalty = 10 * (recall_constraint - recall)
                return -precision + penalty
        
        # 初始阈值
        initial_threshold = 0.5
        
        # 优化
        result = minimize(
            objective,
            initial_threshold,
            bounds=[self.threshold_bounds],
            method=self.algorithm,
            options={'maxiter': self.max_iterations, 'xatol': 1e-4}
        )
        
        optimal_threshold = result.x[0]
        
        # 计算最优阈值下的性能指标
        y_pred_optimal = (y_proba > optimal_threshold).astype(int)
        metrics = self._calculate_metrics(self.y_val, y_pred_optimal)
        
        return optimal_threshold, metrics
    
    def optimize_threshold_for_f1(
        self,
        recall_constraint: float = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        在召回率约束下优化F1分数
        
        Args:
            recall_constraint: 召回率约束
        
        Returns:
            (最优阈值, 性能指标字典)
        """
        if recall_constraint is None:
            recall_constraint = self.recall_target
        
        # 获取模型预测概率
        y_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        def objective(threshold):
            """目标函数：最大化F1分数，约束召回率"""
            y_pred = (y_proba > threshold).astype(int)
            
            try:
                precision = precision_score(self.y_val, y_pred, zero_division=0)
                recall = recall_score(self.y_val, y_pred, zero_division=0)
                f1 = f1_score(self.y_val, y_pred, zero_division=0)
            except:
                return 1000
            
            # 目标:最大化F1，约束:召回率≥目标值
            if recall >= recall_constraint:
                return -f1
            else:
                # 不满足约束，给予重罚
                penalty = 20 * (recall_constraint - recall)
                return -f1 + penalty
        
        # 初始阈值
        initial_threshold = 0.5
        
        # 优化
        result = minimize(
            objective,
            initial_threshold,
            bounds=[self.threshold_bounds],
            method=self.algorithm,
            options={'maxiter': self.max_iterations, 'xatol': 1e-4}
        )
        
        optimal_threshold = result.x[0]
        
        # 计算最优阈值下的性能指标
        y_pred_optimal = (y_proba > optimal_threshold).astype(int)
        metrics = self._calculate_metrics(self.y_val, y_pred_optimal)
        
        return optimal_threshold, metrics
    
    def multi_objective_optimization(
        self,
        weights: Dict[str, float] = None,
        constraints: Dict[str, float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        多目标优化：平衡精确率、召回率、F1
        
        Args:
            weights: 各指标的权重 {'precision': 0.4, 'recall': 0.3, 'f1': 0.3}
            constraints: 约束条件 {'recall': 0.7}
        
        Returns:
            (最优阈值, 性能指标字典)
        """
        if weights is None:
            weights = {'precision': 0.4, 'recall': 0.3, 'f1': 0.3}
        
        if constraints is None:
            constraints = {'recall': self.recall_target}
        
        # 获取模型预测概率
        y_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        def objective(threshold):
            """多目标函数"""
            y_pred = (y_proba > threshold).astype(int)
            
            try:
                precision = precision_score(self.y_val, y_pred, zero_division=0)
                recall = recall_score(self.y_val, y_pred, zero_division=0)
                f1 = f1_score(self.y_val, y_pred, zero_division=0)
            except:
                return 1000
            
            # 检查约束
            penalty = 0
            for metric, constraint_value in constraints.items():
                if metric == 'recall' and recall < constraint_value:
                    penalty += 20 * (constraint_value - recall)
                # 可以添加其他约束
            
            # 计算加权得分
            score = (
                weights['precision'] * precision +
                weights['recall'] * recall +
                weights['f1'] * f1
            )
            
            return -score + penalty
        
        # 初始阈值
        initial_threshold = 0.5
        
        # 优化
        result = minimize(
            objective,
            initial_threshold,
            bounds=[self.threshold_bounds],
            method=self.algorithm,
            options={'maxiter': self.max_iterations, 'xatol': 1e-4}
        )
        
        optimal_threshold = result.x[0]
        
        # 计算最优阈值下的性能指标
        y_pred_optimal = (y_proba > optimal_threshold).astype(int)
        metrics = self._calculate_metrics(self.y_val, y_pred_optimal)
        
        return optimal_threshold, metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算性能指标"""
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # 计算支持度（各类样本数）
            unique, counts = np.unique(y_pred, return_counts=True)
            pred_positive = dict(zip(unique, counts)).get(1, 0)
            pred_negative = dict(zip(unique, counts)).get(0, 0)
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pred_positive': int(pred_positive),
                'pred_negative': int(pred_negative)
            }
        except Exception as e:
            return {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'error': str(e)
            }
    
    def grid_search_with_constraint(
        self,
        threshold_range: Tuple[float, float] = None,
        n_points: int = 100,
        recall_constraint: float = None,
        objective: str = 'f1'  # 'f1' or 'precision'
    ) -> Tuple[float, Dict[str, float]]:
        """
        带约束的网格搜索
        
        Args:
            threshold_range: 阈值搜索范围 (min, max)
            n_points: 搜索点数
            recall_constraint: 召回率约束
            objective: 优化目标
        
        Returns:
            (最优阈值, 性能指标字典)
        """
        if threshold_range is None:
            threshold_range = self.threshold_bounds
        
        if recall_constraint is None:
            recall_constraint = self.recall_target
        
        # 获取模型预测概率
        y_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        # 生成网格
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
        
        best_threshold = 0.5
        best_score = -1
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            
            try:
                precision = precision_score(self.y_val, y_pred, zero_division=0)
                recall = recall_score(self.y_val, y_pred, zero_division=0)
                f1 = f1_score(self.y_val, y_pred, zero_division=0)
                
                # 检查约束
                if recall < recall_constraint:
                    continue
                
                # 选择目标
                if objective == 'f1':
                    score = f1
                elif objective == 'precision':
                    score = precision
                else:
                    score = f1
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
            except:
                continue
        
        return best_threshold, best_metrics
    
    def precision_recall_tradeoff(
        self,
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        分析精确率-召回率权衡
        
        Args:
            n_points: 分析点数
        
        Returns:
            包含不同阈值下性能指标的数据框
        """
        # 获取模型预测概率
        y_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        # 生成阈值范围
        thresholds = np.linspace(0.1, 0.9, n_points)
        
        results = []
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            
            try:
                precision = precision_score(self.y_val, y_pred, zero_division=0)
                recall = recall_score(self.y_val, y_pred, zero_division=0)
                f1 = f1_score(self.y_val, y_pred, zero_division=0)
                
                results.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            except:
                continue
        
        return pd.DataFrame(results)
    
    def find_optimal_operating_point(
        self,
        precision_weight: float = 0.5,
        recall_weight: float = 0.5,
        min_recall: float = None,
        min_precision: float = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        找到最优操作点
        
        Args:
            precision_weight: 精确率权重
            recall_weight: 召回率权重
            min_recall: 最小召回率约束
            min_precision: 最小精确率约束
        
        Returns:
            (最优阈值, 性能指标字典)
        """
        # 获取精确率-召回率曲线
        pr_curve = self.precision_recall_tradeoff()
        
        # 计算综合得分
        pr_curve['score'] = (
            precision_weight * pr_curve['precision'] +
            recall_weight * pr_curve['recall']
        )
        
        # 应用约束
        if min_recall is not None:
            pr_curve = pr_curve[pr_curve['recall'] >= min_recall]
        
        if min_precision is not None:
            pr_curve = pr_curve[pr_curve['precision'] >= min_precision]
        
        if len(pr_curve) == 0:
            return 0.5, {'error': 'No threshold satisfies constraints'}
        
        # 找到最高分的阈值
        best_row = pr_curve.loc[pr_curve['score'].idxmax()]
        
        best_threshold = best_row['threshold']
        best_metrics = {
            'precision': best_row['precision'],
            'recall': best_row['recall'],
            'f1': best_row['f1'],
            'score': best_row['score']
        }
        
        return best_threshold, best_metrics
