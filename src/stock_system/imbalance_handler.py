"""
类别不平衡处理模块
提供多种处理类别不平衡的策略：
1. Class Weight（类别权重）
2. Focal Loss（焦点损失）
3. Sample Reweighting（样本重权重）
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict
from sklearn.utils import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')


class ImbalanceHandler:
    """
    类别不平衡处理器
    
    提供多种处理不平衡的策略：
    1. Class Weight: 为不同类别分配不同权重
    2. Focal Loss: 通过降低简单样本的权重来关注困难样本
    3. Sample Reweighting: 基于样本难度或时间进行权重调整
    """
    
    def __init__(self, strategy: str = 'class_weight', **kwargs):
        """
        初始化不平衡处理器
        
        参数:
            strategy: 处理策略
                - 'class_weight': 类别权重
                - 'focal_loss': 焦点损失
                - 'sample_reweight': 样本重权重
                - 'combined': 组合策略
            **kwargs: 策略特定参数
        """
        self.strategy = strategy
        self.params = kwargs
        
        self._validate_strategy()
    
    def _validate_strategy(self):
        """验证策略有效性"""
        valid_strategies = [
            'class_weight',
            'focal_loss',
            'sample_reweight',
            'combined'
        ]
        
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"策略必须是 {valid_strategies} 之一，得到的是 {self.strategy}"
            )
    
    def compute_class_weights(
        self,
        y: np.ndarray,
        method: str = 'balanced'
    ) -> Dict[int, float]:
        """
        计算类别权重
        
        参数:
            y: 目标变量
            method: 权重计算方法
                - 'balanced': 自动计算（n_samples / (n_classes * n_samples_per_class)）
                - 'inverse': 反比例
                - 'sqrt_inv': 平方根反比例
                
        返回:
            类别权重字典 {class_label: weight}
        """
        classes = np.unique(y)
        
        if method == 'balanced':
            # 使用 sklearn 的 balanced 策略
            weights = compute_class_weight('balanced', classes=classes, y=y)
        elif method == 'inverse':
            # 反比例权重
            class_counts = np.array([(y == c).sum() for c in classes])
            weights = 1.0 / class_counts
            weights = weights / weights.sum() * len(classes)
        elif method == 'sqrt_inv':
            # 平方根反比例权重
            class_counts = np.array([(y == c).sum() for c in classes])
            weights = 1.0 / np.sqrt(class_counts)
            weights = weights / weights.sum() * len(classes)
        else:
            raise ValueError(f"未知的权重计算方法: {method}")
        
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
        
        return class_weights
    
    def compute_sample_weights(
        self,
        y: np.ndarray,
        method: str = 'balanced',
        difficulty: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算样本权重
        
        参数:
            y: 目标变量
            method: 权重计算方法
                - 'balanced': 基于类别平衡
                - 'difficulty': 基于样本难度
                - 'temporal': 基于时间衰减（近期样本权重更高）
            difficulty: 样本难度分数（可选）
            
        返回:
            样本权重数组
        """
        if method == 'balanced':
            # 使用 sklearn 的 sample_weight
            sample_weights = compute_sample_weight('balanced', y)
        
        elif method == 'difficulty' and difficulty is not None:
            # 基于样本难度的权重
            # 难度越高，权重越大
            difficulty = np.asarray(difficulty)
            if difficulty.max() > 0:
                difficulty_normalized = difficulty / difficulty.max()
            else:
                difficulty_normalized = np.zeros_like(difficulty)
            
            # 基础权重（类别平衡）
            base_weights = compute_sample_weight('balanced', y)
            
            # 难度调整
            sample_weights = base_weights * (1 + difficulty_normalized)
            sample_weights = sample_weights / sample_weights.mean()
        
        elif method == 'temporal':
            # 时间衰减权重：近期样本权重更高
            n_samples = len(y)
            decay_factor = self.params.get('decay_factor', 0.01)
            
            # 计算时间衰减（从0到1）
            temporal_weights = np.exp(decay_factor * np.arange(n_samples))
            temporal_weights = temporal_weights / temporal_weights.mean()
            
            # 基础权重（类别平衡）
            base_weights = compute_sample_weight('balanced', y)
            
            # 组合权重
            sample_weights = base_weights * temporal_weights
            sample_weights = sample_weights / sample_weights.mean()
        
        else:
            raise ValueError(f"未知的样本权重方法: {method}")
        
        return sample_weights
    
    def compute_focal_loss_weights(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> np.ndarray:
        """
        计算Focal Loss权重
        
        Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
        
        其中:
        - p_t = 预测概率（如果是正类）或 1 - 预测概率（如果是负类）
        - alpha: 平衡因子（正类权重）
        - gamma: 聚焦因子（gamma越大，越关注困难样本）
        
        参数:
            y_true: 真实标签 (0 或 1)
            y_pred: 预测概率
            alpha: 平衡因子
            gamma: 聚焦因子
            
        返回:
            样本权重数组
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # 确保 y_pred 在 (0, 1) 范围内
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # 计算 p_t
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        
        # 计算焦点因子
        focal_factor = (1 - p_t) ** gamma
        
        # 计算权重
        weights = alpha * focal_factor
        
        return weights
    
    def get_model_params(
        self,
        y_train: np.ndarray,
        model_type: str = 'xgboost'
    ) -> Dict[str, any]:
        """
        获取模型的不平衡处理参数
        
        参数:
            y_train: 训练集标签
            model_type: 模型类型 ('xgboost', 'lightgbm', 'sklearn')
            
        返回:
            模型参数字典
        """
        params = {}
        
        if self.strategy == 'class_weight':
            class_weights = self.compute_class_weights(
                y_train,
                method=self.params.get('class_weight_method', 'balanced')
            )
            
            if model_type == 'xgboost':
                # XGBoost 使用 scale_pos_weight
                n_negative = (y_train == 0).sum()
                n_positive = (y_train == 1).sum()
                params['scale_pos_weight'] = n_negative / n_positive
                # 也可以使用 weight 参数
                if self.params.get('use_weight_param', False):
                    params['weight'] = np.array([
                        class_weights[int(y)] for y in y_train
                    ])
            
            elif model_type == 'lightgbm':
                # LightGBM 使用 class_weight
                params['class_weight'] = class_weights
            
            elif model_type == 'sklearn':
                # Scikit-learn 使用 class_weight
                params['class_weight'] = 'balanced'
        
        elif self.strategy == 'focal_loss':
            # Focal Loss 需要在训练过程中动态计算
            # 这里只返回参数，实际计算在训练循环中
            params['alpha'] = self.params.get('alpha', 0.25)
            params['gamma'] = self.params.get('gamma', 2.0)
        
        elif self.strategy == 'sample_reweight':
            sample_weights = self.compute_sample_weights(
                y_train,
                method=self.params.get('sample_method', 'balanced')
            )
            params['sample_weight'] = sample_weights
        
        elif self.strategy == 'combined':
            # 组合策略：类别权重 + 样本权重
            class_weights = self.compute_class_weights(
                y_train,
                method=self.params.get('class_weight_method', 'balanced')
            )
            
            sample_weights = self.compute_sample_weights(
                y_train,
                method=self.params.get('sample_method', 'balanced')
            )
            
            if model_type == 'xgboost':
                n_negative = (y_train == 0).sum()
                n_positive = (y_train == 1).sum()
                params['scale_pos_weight'] = n_negative / n_positive
                params['weight'] = sample_weights
            
            elif model_type == 'lightgbm':
                params['class_weight'] = class_weights
                params['sample_weight'] = sample_weights
            
            elif model_type == 'sklearn':
                params['class_weight'] = 'balanced'
                params['sample_weight'] = sample_weights
        
        return params


class FocalLossXGBoost:
    """
    XGBoost的自定义Focal Loss
    
    注意: XGBoost原生不支持Focal Loss，这里提供近似的实现
    通过调整样本权重和scale_pos_weight来模拟Focal Loss的效果
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        初始化Focal Loss
        
        参数:
            alpha: 平衡因子
            gamma: 聚焦因子
        """
        self.alpha = alpha
        self.gamma = gamma
    
    def compute_weights(
        self,
        y_train: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算Focal Loss权重
        
        参数:
            y_train: 训练标签
            y_pred_proba: 预测概率（可选，如果有则使用动态权重）
            
        返回:
            样本权重数组
        """
        if y_pred_proba is None:
            # 如果没有预测概率，使用静态权重
            # 基础权重：类别平衡
            from sklearn.utils import compute_sample_weight
            weights = compute_sample_weight('balanced', y_train)
            
            # 正类额外权重（alpha）
            weights = np.where(y_train == 1, weights * self.alpha, weights)
        else:
            # 如果有预测概率，使用动态Focal Loss权重
            weights = self._compute_focal_weights(
                y_train,
                y_pred_proba,
                self.alpha,
                self.gamma
            )
        
        return weights
    
    def _compute_focal_weights(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float,
        gamma: float
    ) -> np.ndarray:
        """计算Focal Loss权重"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        focal_factor = (1 - p_t) ** gamma
        
        weights = alpha * focal_factor
        weights = weights / weights.mean()
        
        return weights


class SampleReweighter:
    """
    样本重权重器
    
    基于样本特征进行权重调整
    """
    
    def __init__(
        self,
        base_method: str = 'balanced',
        adjust_factor: float = 1.0
    ):
        """
        初始化样本重权重器
        
        参数:
            base_method: 基础权重方法
            adjust_factor: 调整因子
        """
        self.base_method = base_method
        self.adjust_factor = adjust_factor
    
    def compute_weights(
        self,
        y: np.ndarray,
        features: Optional[np.ndarray] = None,
        custom_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算样本权重
        
        参数:
            y: 目标变量
            features: 特征矩阵（可选，用于计算样本难度）
            custom_weights: 自定义权重（可选）
            
        返回:
            样本权重数组
        """
        from sklearn.utils import compute_sample_weight
        
        # 基础权重
        weights = compute_sample_weight(self.base_method, y)
        
        # 自定义权重调整
        if custom_weights is not None:
            custom_weights = np.asarray(custom_weights)
            weights = weights * custom_weights
        
        # 基于特征难度的调整（如果有特征）
        if features is not None:
            difficulty = self._estimate_sample_difficulty(features, y)
            weights = weights * (1 + difficulty * self.adjust_factor)
        
        # 归一化
        weights = weights / weights.mean()
        
        return weights
    
    def _estimate_sample_difficulty(
        self,
        features: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        估计样本难度
        
        参数:
            features: 特征矩阵
            y: 目标变量
            
        返回:
            样本难度分数（0-1）
        """
        # 使用样本到质心的距离作为难度估计
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 计算每个样本到同类质心的距离
        difficulty = np.zeros(len(y))
        
        for label in np.unique(y):
            mask = y == label
            class_features = features_scaled[mask]
            
            # 计算质心
            centroid = class_features.mean(axis=0)
            
            # 计算到质心的距离
            distances = np.linalg.norm(class_features - centroid, axis=1)
            
            # 归一化到 [0, 1]
            if distances.max() > 0:
                distances_normalized = distances / distances.max()
            else:
                distances_normalized = np.zeros_like(distances)
            
            difficulty[mask] = distances_normalized
        
        return difficulty
