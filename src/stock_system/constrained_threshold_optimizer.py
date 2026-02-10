"""
受约束阈值优化器
在满足召回率约束的条件下，最大化精确率
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class ConstrainedThresholdOptimizer:
    """受约束阈值优化器
    
    在满足约束条件（如 recall ≥ target_recall）的情况下，
    找到能最大化目标指标（如 precision）的最优阈值
    """
    
    def __init__(
        self,
        constraints: Dict[str, float],
        search_range: Tuple[float, float] = (0.5, 0.95),
        step_size: float = 0.001,
        min_samples: int = 50
    ):
        """
        初始化受约束阈值优化器
        
        Args:
            constraints: 约束条件字典
                - recall_min: 最小召回率
                - precision_min: 最小精确率
                - max_fp_ratio: 最大假阳性率
                - min_f1: 最小F1分数
            search_range: 阈值搜索范围
            step_size: 搜索步长
            min_samples: 最小样本数
        """
        self.constraints = constraints
        self.search_range = search_range
        self.step_size = step_size
        self.min_samples = min_samples
        
        self.best_threshold = None
        self.best_metrics = None
        self.optimization_history = []
    
    def optimize(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        objective: str = 'precision_max',
        method: str = 'grid_search'
    ) -> Tuple[float, Dict[str, float]]:
        """
        在约束条件下优化阈值
        
        Args:
            y_proba: 概率预测（未校准或已校准）
            y_true: 真实标签
            objective: 优化目标
                - 'precision_max': 最大化精确率
                - 'recall_max': 最大化召回率
                - 'f1_max': 最大化F1分数
                - 'minimize_fp': 最小化假阳性率
            method: 优化方法
                - 'grid_search': 网格搜索
                - 'binary_search': 二分搜索（仅适用于单一约束）
        
        Returns:
            (最优阈值, 性能指标字典)
        """
        # 检查数据
        if len(y_proba) != len(y_true):
            raise ValueError("y_proba 和 y_true 长度不一致")
        
        # 检查约束条件是否可行
        if not self._check_constraints_feasibility(y_proba, y_true):
            raise ValueError("无法满足所有约束条件，请调整约束参数")
        
        # 选择优化方法
        if method == 'grid_search':
            best_threshold, best_metrics = self._grid_search(y_proba, y_true, objective)
        elif method == 'binary_search':
            best_threshold, best_metrics = self._binary_search(y_proba, y_true, objective)
        else:
            raise ValueError(f"未知的优化方法: {method}")
        
        self.best_threshold = best_threshold
        self.best_metrics = best_metrics
        
        return best_threshold, best_metrics
    
    def _grid_search(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        objective: str
    ) -> Tuple[float, Dict[str, float]]:
        """网格搜索最优阈值"""
        # 生成候选阈值
        thresholds = np.arange(self.search_range[0], self.search_range[1], self.step_size)
        
        valid_thresholds = []
        
        for threshold in thresholds:
            # 生成预测
            y_pred = (y_proba >= threshold).astype(int)
            
            # 计算指标
            metrics = self._calculate_metrics(y_true, y_pred, y_proba)
            
            # 检查样本数
            if metrics['n_selected'] < self.min_samples:
                continue
            
            # 检查约束条件
            if self._satisfy_constraints(metrics):
                valid_thresholds.append((threshold, metrics))
            
            # 记录历史
            self.optimization_history.append({
                'threshold': threshold,
                'metrics': metrics,
                'satisfies_constraints': self._satisfy_constraints(metrics)
            })
        
        if not valid_thresholds:
            raise ValueError(f"没有满足约束条件的阈值。请尝试放宽约束条件。\n"
                           f"当前约束: {self.constraints}")
        
        # 根据优化目标选择最优阈值
        best_threshold, best_metrics = self._select_best_threshold(valid_thresholds, objective)
        
        return best_threshold, best_metrics
    
    def _binary_search(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        objective: str
    ) -> Tuple[float, Dict[str, float]]:
        """二分搜索最优阈值（仅适用于单一约束）"""
        # 检查是否适用于二分搜索
        if len(self.constraints) != 1:
            raise ValueError("二分搜索仅适用于单一约束条件")
        
        constraint_name = list(self.constraints.keys())[0]
        constraint_value = list(self.constraints.values())[0]
        
        # 二分搜索
        low, high = self.search_range
        best_threshold = high
        best_metrics = None
        found_valid = False
        
        iteration = 0
        max_iterations = 100
        
        while high - low > self.step_size and iteration < max_iterations:
            iteration += 1
            mid = (low + high) / 2
            
            # 评估中间阈值
            y_pred = (y_proba >= mid).astype(int)
            metrics = self._calculate_metrics(y_true, y_pred, y_proba)
            
            if metrics['n_selected'] < self.min_samples:
                # 样本太少，降低阈值
                high = mid
                continue
            
            # 检查约束
            if self._satisfy_constraints(metrics):
                # 满足约束，尝试降低阈值以提升目标
                high = mid
                best_threshold = mid
                best_metrics = metrics
                found_valid = True
            else:
                # 不满足约束，提高阈值
                low = mid
        
        if not found_valid:
            raise ValueError(f"二分搜索未找到满足约束的阈值。约束条件: {self.constraints}")
        
        return best_threshold, best_metrics
    
    def _select_best_threshold(
        self,
        valid_thresholds: list,
        objective: str
    ) -> Tuple[float, Dict[str, float]]:
        """根据优化目标选择最优阈值"""
        if objective == 'precision_max':
            best_threshold, best_metrics = max(
                valid_thresholds,
                key=lambda x: x[1]['precision']
            )
        elif objective == 'recall_max':
            best_threshold, best_metrics = max(
                valid_thresholds,
                key=lambda x: x[1]['recall']
            )
        elif objective == 'f1_max':
            best_threshold, best_metrics = max(
                valid_thresholds,
                key=lambda x: x[1]['f1']
            )
        elif objective == 'minimize_fp':
            best_threshold, best_metrics = min(
                valid_thresholds,
                key=lambda x: x[1]['n_false_positive']
            )
        else:
            raise ValueError(f"未知的优化目标: {objective}")
        
        return best_threshold, best_metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """计算性能指标"""
        # 基本指标
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # 计算衍生指标
        n_selected = tp + fp
        fp_ratio = fp / n_selected if n_selected > 0 else 0
        fn_ratio = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # AUC
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = 0.5
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'n_selected': n_selected,
            'n_true_positive': tp,
            'n_false_positive': fp,
            'n_true_negative': tn,
            'n_false_negative': fn,
            'fp_ratio': fp_ratio,
            'fn_ratio': fn_ratio
        }
    
    def _satisfy_constraints(self, metrics: Dict[str, float]) -> bool:
        """检查是否满足所有约束条件"""
        # 检查召回率约束
        if 'recall_min' in self.constraints:
            if metrics['recall'] < self.constraints['recall_min']:
                return False
        
        # 检查精确率约束
        if 'precision_min' in self.constraints:
            if metrics['precision'] < self.constraints['precision_min']:
                return False
        
        # 检查假阳性率约束
        if 'max_fp_ratio' in self.constraints:
            if metrics['fp_ratio'] > self.constraints['max_fp_ratio']:
                return False
        
        # 检查F1分数约束
        if 'min_f1' in self.constraints:
            if metrics['f1'] < self.constraints['min_f1']:
                return False
        
        return True
    
    def _check_constraints_feasibility(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray
    ) -> bool:
        """检查约束条件是否可行"""
        # 在极端阈值下检查约束是否可满足
        # 低阈值（0.5）
        y_pred_low = (y_proba >= 0.5).astype(int)
        metrics_low = self._calculate_metrics(y_true, y_pred_low, y_proba)
        
        # 高阈值（0.95）
        y_pred_high = (y_proba >= 0.95).astype(int)
        metrics_high = self._calculate_metrics(y_true, y_pred_high, y_proba)
        
        # 检查是否存在至少一个阈值可以满足约束
        return self._satisfy_constraints(metrics_low) or self._satisfy_constraints(metrics_high)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        if not self.optimization_history:
            return {"error": "未运行优化"}
        
        # 统计满足约束的阈值数量
        valid_count = sum(1 for h in self.optimization_history if h['satisfies_constraints'])
        total_count = len(self.optimization_history)
        
        return {
            'best_threshold': self.best_threshold,
            'best_metrics': self.best_metrics,
            'constraints': self.constraints,
            'valid_thresholds_count': valid_count,
            'total_searched': total_count,
            'valid_ratio': valid_count / total_count if total_count > 0 else 0
        }
    
    def plot_optimization_curve(self, save_path: str = None):
        """绘制优化曲线"""
        import matplotlib.pyplot as plt
        
        if not self.optimization_history:
            print("未运行优化，无法绘制曲线")
            return
        
        # 提取数据
        thresholds = [h['threshold'] for h in self.optimization_history]
        precisions = [h['metrics']['precision'] for h in self.optimization_history]
        recalls = [h['metrics']['recall'] for h in self.optimization_history]
        valid = [h['satisfies_constraints'] for h in self.optimization_history]
        
        # 绘制
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 图1: Precision-Recall 曲线
        ax1.plot(recalls, precisions, 'b-', linewidth=2, label='Precision-Recall')
        valid_recalls = [recalls[i] for i in range(len(recalls)) if valid[i]]
        valid_precisions = [precisions[i] for i in range(len(precisions)) if valid[i]]
        ax1.scatter(valid_recalls, valid_precisions, c='green', alpha=0.3, s=10, label='Valid Thresholds')
        
        # 标记最优阈值
        if self.best_threshold is not None:
            ax1.scatter(
                [self.best_metrics['recall']],
                [self.best_metrics['precision']],
                c='red', s=200, marker='*',
                label=f'Best Threshold: {self.best_threshold:.4f}'
            )
        
        # 绘制约束线
        if 'recall_min' in self.constraints:
            ax1.axvline(x=self.constraints['recall_min'], color='orange', 
                       linestyle='--', label=f'Min Recall: {self.constraints["recall_min"]:.2f}')
        if 'precision_min' in self.constraints:
            ax1.axhline(y=self.constraints['precision_min'], color='purple', 
                       linestyle='--', label=f'Min Precision: {self.constraints["precision_min"]:.2f}')
        
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 图2: 阈值 vs 性能
        ax2.plot(thresholds, precisions, 'r-', linewidth=2, label='Precision')
        ax2.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
        
        # 标记有效区域
        valid_thresholds_list = [thresholds[i] for i in range(len(thresholds)) if valid[i]]
        if valid_thresholds_list:
            ax2.fill_between(valid_thresholds_list, 0, 1, alpha=0.1, color='green', label='Valid Region')
        
        # 标记最优阈值
        if self.best_threshold is not None:
            ax2.axvline(x=self.best_threshold, color='blue', linestyle='--', linewidth=2,
                       label=f'Best: {self.best_threshold:.4f}')
        
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel('Performance', fontsize=12)
        ax2.set_title('Threshold vs Performance', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"优化曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()


class MultiObjectiveThresholdOptimizer:
    """多目标阈值优化器
    
    支持多个优化目标的加权优化
    """
    
    def __init__(
        self,
        constraints: Dict[str, float],
        objective_weights: Dict[str, float] = None
    ):
        """
        Args:
            constraints: 约束条件
            objective_weights: 目标权重
                {
                    'precision': 0.5,
                    'recall': 0.3,
                    'f1': 0.2
                }
        """
        self.constraints = constraints
        self.objective_weights = objective_weights or {
            'precision': 0.5,
            'recall': 0.3,
            'f1': 0.2
        }
        
        # 归一化权重
        total_weight = sum(self.objective_weights.values())
        self.objective_weights = {
            k: v / total_weight 
            for k, v in self.objective_weights.items()
        }
    
    def optimize(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        多目标优化
        
        Args:
            y_proba: 概率预测
            y_true: 真实标签
        
        Returns:
            (最优阈值, 性能指标字典)
        """
        # 使用基础优化器
        optimizer = ConstrainedThresholdOptimizer(constraints=self.constraints)
        
        # 网格搜索
        best_threshold, best_metrics = optimizer.optimize(
            y_proba, y_true,
            objective='precision_max',  # 这里不重要，因为会重新计算
            method='grid_search'
        )
        
        # 重新评估所有有效阈值，计算多目标得分
        valid_thresholds = [
            (h['threshold'], h['metrics'])
            for h in optimizer.optimization_history
            if h['satisfies_constraints']
        ]
        
        if not valid_thresholds:
            raise ValueError("没有满足约束条件的阈值")
        
        # 计算多目标得分
        def calculate_score(metrics: Dict[str, float]) -> float:
            score = 0
            if 'precision' in self.objective_weights:
                score += metrics['precision'] * self.objective_weights['precision']
            if 'recall' in self.objective_weights:
                score += metrics['recall'] * self.objective_weights['recall']
            if 'f1' in self.objective_weights:
                score += metrics['f1'] * self.objective_weights['f1']
            return score
        
        # 选择得分最高的阈值
        best_threshold, best_metrics = max(
            valid_thresholds,
            key=lambda x: calculate_score(x[1])
        )
        
        # 添加得分到指标
        best_metrics['multi_objective_score'] = calculate_score(best_metrics)
        
        return best_threshold, best_metrics
