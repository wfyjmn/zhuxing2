"""
自动化阈值优化系统
基于历史数据的统计分布与模型表现优化
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Any, Tuple, List
import json
import os


class AutoThresholdOptimizer:
    """自动化阈值优化器
    
    为每个特征自动计算最优阈值，基于统计分布和预测效果
    """
    
    def __init__(self, data: pd.DataFrame, target_variable: str, features: List[str]):
        """
        初始化阈值优化器
        
        Args:
            data: 包含特征和目标变量的数据集
            target_variable: 目标变量名称（0/1分类）
            features: 需要优化阈值的特征列表
        """
        self.data = data
        self.target = target_variable
        self.features = features
        self.thresholds = {}
        
    def calculate_optimal_thresholds(self, method: str = 'ensemble') -> Dict[str, Any]:
        """
        为每个特征自动计算最优阈值
        
        Args:
            method: 优化方法 ['quantile', 'information_gain', 'model_based', 'ensemble']
        
        Returns:
            包含每个特征最优阈值的字典
        """
        thresholds = {}
        
        for feature in self.features:
            if feature not in self.data.columns:
                print(f"Warning: Feature {feature} not found in data")
                continue
                
            if method == 'quantile':
                thresholds[feature] = self._quantile_based_threshold(feature)
            elif method == 'information_gain':
                thresholds[feature] = self._information_gain_threshold(feature)
            elif method == 'model_based':
                thresholds[feature] = self._model_based_threshold(feature)
            elif method == 'ensemble':
                # 综合多种方法
                threshold_stats = self._quantile_based_threshold(feature)
                threshold_ig = self._information_gain_threshold(feature)
                threshold_model = self._model_based_threshold(feature)
                
                # 综合多种方法
                thresholds[feature] = self._ensemble_thresholds(
                    threshold_stats, threshold_ig, threshold_model
                )
            else:
                raise ValueError(f"Unknown method: {method}")
                
        self.thresholds = thresholds
        return thresholds
    
    def _quantile_based_threshold(self, feature: str) -> Dict[str, float]:
        """基于历史数据的分位数统计"""
        positive_data = self.data[self.data[self.target] == 1][feature].dropna()
        negative_data = self.data[self.data[self.target] == 0][feature].dropna()
        
        if len(positive_data) == 0 or len(negative_data) == 0:
            return {'optimal': 0, 'method': 'quantile'}
        
        # 计算正样本的关键分位数
        q75_positive = positive_data.quantile(0.75)
        q90_positive = positive_data.quantile(0.90)
        
        # 计算负样本的关键分位数
        q25_negative = negative_data.quantile(0.25)
        q50_negative = negative_data.quantile(0.50)
        
        # 基于统计差异设置阈值
        conservative = q75_positive  # 保守阈值
        moderate = (q75_positive + q50_negative) / 2  # 适中阈值
        aggressive = q25_negative  # 激进阈值
        
        # 选择最优阈值：适中阈值作为默认
        optimal = moderate
        
        return {
            'conservative': float(conservative),
            'moderate': float(moderate),
            'aggressive': float(aggressive),
            'optimal': float(optimal),
            'method': 'quantile'
        }
    
    def _information_gain_threshold(self, feature: str) -> Dict[str, float]:
        """基于信息增益的方法
        
        使用决策树找到最佳分割点
        """
        X = self.data[[feature]].fillna(0)
        y = self.data[self.target]
        
        # 使用深度为1的决策树（决策桩）找到最佳分割点
        tree = DecisionTreeClassifier(max_depth=1, random_state=42)
        tree.fit(X, y)
        
        threshold = float(tree.tree_.threshold[0])
        
        # 确保阈值在合理范围内
        feature_range = X[feature].max() - X[feature].min()
        if feature_range > 0:
            threshold = np.clip(threshold, X[feature].min(), X[feature].max())
        
        return {
            'optimal': threshold,
            'method': 'information_gain'
        }
    
    def _model_based_threshold(self, feature: str, recall_constraint: float = 0.7) -> Dict[str, float]:
        """基于模型表现的方法
        
        在召回率约束下优化精确率
        """
        feature_data = self.data[feature].fillna(0)
        target_data = self.data[self.target]
        
        # 网格搜索最佳阈值
        unique_values = np.sort(feature_data.unique())
        if len(unique_values) > 100:
            # 如果唯一值太多，采样100个候选阈值
            threshold_candidates = np.percentile(unique_values, np.linspace(0, 100, 100))
        else:
            threshold_candidates = unique_values
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in threshold_candidates:
            # 生成预测
            y_pred = (feature_data > threshold).astype(int)
            
            # 计算指标
            try:
                precision = precision_score(target_data, y_pred, zero_division=0)
                recall = recall_score(target_data, y_pred, zero_division=0)
                f1 = f1_score(target_data, y_pred, zero_division=0)
                
                # 约束优化：召回率≥70%
                if recall >= recall_constraint and f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except:
                continue
        
        return {
            'optimal': float(best_threshold),
            'best_f1': float(best_f1),
            'recall_constraint': recall_constraint,
            'method': 'model_based'
        }
    
    def _ensemble_thresholds(
        self, 
        threshold_stats: Dict, 
        threshold_ig: Dict, 
        threshold_model: Dict
    ) -> Dict[str, Any]:
        """综合多种方法"""
        # 提取各方法的最优阈值
        t_stats = threshold_stats['moderate']
        t_ig = threshold_ig['optimal']
        t_model = threshold_model['optimal']
        
        # 加权平均（可根据实际效果调整权重）
        weights = [0.4, 0.3, 0.3]
        optimal = weights[0] * t_stats + weights[1] * t_ig + weights[2] * t_model
        
        return {
            'quantile_threshold': float(t_stats),
            'ig_threshold': float(t_ig),
            'model_threshold': float(t_model),
            'optimal': float(optimal),
            'method': 'ensemble'
        }
    
    def get_threshold_summary(self) -> pd.DataFrame:
        """获取阈值优化结果汇总"""
        summary_list = []
        
        for feature, threshold_info in self.thresholds.items():
            summary_list.append({
                'feature': feature,
                'optimal_threshold': threshold_info.get('optimal', 0),
                'method': threshold_info.get('method', 'unknown'),
                'details': json.dumps(threshold_info, ensure_ascii=False)
            })
        
        return pd.DataFrame(summary_list)
    
    def save_thresholds(self, filepath: str):
        """保存优化后的阈值"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.thresholds, f, ensure_ascii=False, indent=2)
        print(f"Thresholds saved to {filepath}")
    
    @classmethod
    def load_thresholds(cls, filepath: str) -> Dict[str, Any]:
        """加载保存的阈值"""
        with open(filepath, 'r', encoding='utf-8') as f:
            thresholds = json.load(f)
        return thresholds


def grid_search_feature_threshold(
    data: pd.DataFrame, 
    feature: str, 
    target: str,
    recall_constraint: float = 0.7
) -> Tuple[float, float]:
    """
    对单个特征进行网格搜索，找到最优阈值
    
    Args:
        data: 数据集
        feature: 特征名称
        target: 目标变量名称
        recall_constraint: 召回率约束
    
    Returns:
        (best_threshold, best_score)
    """
    optimizer = AutoThresholdOptimizer(data, target, [feature])
    result = optimizer._model_based_threshold(feature, recall_constraint)
    
    return result['optimal'], result.get('best_f1', 0)
