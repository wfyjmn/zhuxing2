"""
资金强度动态阈值学习模块
基于近期资金流数据动态学习阈值
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score


class CapitalIntensityThresholdOptimizer:
    """资金强度阈值优化器"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any] = None):
        """
        初始化资金强度阈值优化器
        
        Args:
            data: 包含资金流数据的数据框
            config: 配置参数
        """
        self.data = data
        self.config = config or {}
        
        # 默认配置
        self.lookback_days = self.config.get('lookback_days', 60)
        self.rolling_quantiles = self.config.get('rolling_quantiles', [0.80, 0.85, 0.90])
        self.zscore_threshold = self.config.get('zscore_threshold', 1.0)
        self.relative_threshold_multiplier = self.config.get('relative_threshold_multiplier', 1.5)
        
    def learn_capital_thresholds(
        self, 
        feature_name: str = 'main_capital_inflow_ratio',
        market_feature_name: str = None
    ) -> Dict[str, Any]:
        """
        基于近期资金数据动态学习阈值
        
        Args:
            feature_name: 资金流特征名称
            market_feature_name: 市场整体资金流特征名称（用于相对阈值）
        
        Returns:
            多种方法计算的阈值
        """
        if feature_name not in self.data.columns:
            raise ValueError(f"Feature {feature_name} not found in data")
        
        capital_inflow = self.data[feature_name].fillna(0)
        thresholds = {}
        
        # 方法1: 移动分位数阈值
        thresholds['rolling_quantiles'] = self._calculate_rolling_quantile_thresholds(
            capital_inflow, feature_name
        )
        
        # 方法2: 基于Z-score的动态阈值
        thresholds['zscore'] = self._calculate_zscore_thresholds(capital_inflow)
        
        # 方法3: 与市场整体对比的相对阈值（如果有市场数据）
        if market_feature_name and market_feature_name in self.data.columns:
            thresholds['relative'] = self._calculate_relative_threshold(
                capital_inflow, 
                self.data[market_feature_name].fillna(0)
            )
        
        # 推荐阈值：综合多种方法
        thresholds['recommended'] = self._select_recommended_threshold(thresholds)
        
        return thresholds
    
    def _calculate_rolling_quantile_thresholds(
        self, 
        capital_inflow: pd.Series,
        feature_name: str
    ) -> Dict[str, pd.Series]:
        """计算滚动分位数阈值"""
        quantile_thresholds = {}
        
        for q in self.rolling_quantiles:
            rolling_q = capital_inflow.rolling(self.lookback_days).quantile(q)
            quantile_thresholds[f'q{int(q*100)}'] = rolling_q
        
        # 默认使用85分位数
        quantile_thresholds['recommended'] = quantile_thresholds['q85']
        
        return quantile_thresholds
    
    def _calculate_zscore_thresholds(
        self, 
        capital_inflow: pd.Series
    ) -> Dict[str, pd.Series]:
        """计算基于Z-score的动态阈值"""
        rolling_mean = capital_inflow.rolling(self.lookback_days).mean()
        rolling_std = capital_inflow.rolling(self.lookback_days).std()
        
        # 均值 + N * 标准差
        zscore_threshold = rolling_mean + self.zscore_threshold * rolling_std
        
        # 处理异常值
        zscore_threshold = zscore_threshold.fillna(0)
        
        return {
            'threshold': zscore_threshold,
            'method': 'zscore',
            'multiplier': self.zscore_threshold
        }
    
    def _calculate_relative_threshold(
        self, 
        capital_inflow: pd.Series,
        market_capital: pd.Series
    ) -> Dict[str, pd.Series]:
        """计算与市场整体对比的相对阈值"""
        # 市场平均值
        market_mean = market_capital.rolling(self.lookback_days).mean()
        
        # 相对阈值：强于市场一定比例
        relative_threshold = market_mean * self.relative_threshold_multiplier
        
        return {
            'threshold': relative_threshold,
            'method': 'relative',
            'multiplier': self.relative_threshold_multiplier,
            'market_mean': market_mean
        }
    
    def _select_recommended_threshold(self, thresholds: Dict[str, Any]) -> float:
        """选择推荐阈值"""
        # 优先使用分位数方法
        if 'rolling_quantiles' in thresholds:
            return float(thresholds['rolling_quantiles']['q85'].iloc[-1])
        elif 'zscore' in thresholds:
            return float(thresholds['zscore']['threshold'].iloc[-1])
        else:
            return 0.05  # 默认阈值
    
    def optimize_capital_features(
        self,
        capital_features: List[str],
        target: pd.Series,
        recall_constraint: float = 0.7
    ) -> Dict[str, Dict[str, Any]]:
        """
        优化多个资金强度特征的阈值
        
        Args:
            capital_features: 资金强度特征列表
            target: 目标变量（0/1）
            recall_constraint: 召回率约束
        
        Returns:
            每个特征的最优阈值和性能指标
        """
        results = {}
        
        for feature in capital_features:
            if feature not in self.data.columns:
                print(f"Warning: Feature {feature} not found, skipping")
                continue
            
            feature_data = self.data[feature].fillna(0)
            
            # 动态学习阈值
            threshold_results = self.learn_capital_thresholds(feature)
            
            # 在历史数据上评估不同阈值的表现
            best_threshold, best_metrics = self._evaluate_threshold_performance(
                feature_data, target, threshold_results, recall_constraint
            )
            
            results[feature] = {
                'optimal_threshold': best_threshold,
                'performance': best_metrics,
                'threshold_methods': threshold_results
            }
        
        return results
    
    def _evaluate_threshold_performance(
        self,
        feature_data: pd.Series,
        target: pd.Series,
        threshold_results: Dict[str, Any],
        recall_constraint: float
    ) -> Tuple[float, Dict[str, float]]:
        """评估不同阈值方法的性能"""
        
        # 收集候选阈值
        candidate_thresholds = []
        
        # 从分位数方法提取
        if 'rolling_quantiles' in threshold_results:
            for q_key, q_series in threshold_results['rolling_quantiles'].items():
                if q_key != 'recommended':
                    candidate_thresholds.append(float(q_series.iloc[-1]))
        
        # 从Z-score方法提取
        if 'zscore' in threshold_results:
            candidate_thresholds.append(float(threshold_results['zscore']['threshold'].iloc[-1]))
        
        # 如果没有候选阈值，使用推荐值
        if not candidate_thresholds:
            candidate_thresholds.append(threshold_results['recommended'])
        
        # 评估每个阈值
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in candidate_thresholds:
            signals = (feature_data > threshold).astype(int)
            
            try:
                precision = precision_score(target, signals, zero_division=0)
                recall = recall_score(target, signals, zero_division=0)
                f1 = f1_score(target, signals, zero_division=0)
                
                if recall >= recall_constraint and f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
            except:
                continue
        
        return best_threshold, best_metrics
    
    def get_dynamic_threshold_series(
        self,
        feature_name: str,
        method: str = 'rolling_q85'
    ) -> pd.Series:
        """
        获取动态阈值序列
        
        Args:
            feature_name: 特征名称
            method: 方法名称 ['rolling_q80', 'rolling_q85', 'rolling_q90', 'zscore', 'relative']
        
        Returns:
            动态阈值时间序列
        """
        threshold_results = self.learn_capital_thresholds(feature_name)
        
        if method == 'rolling_q80':
            return threshold_results['rolling_quantiles']['q80']
        elif method == 'rolling_q85':
            return threshold_results['rolling_quantiles']['q85']
        elif method == 'rolling_q90':
            return threshold_results['rolling_quantiles']['q90']
        elif method == 'zscore':
            return threshold_results['zscore']['threshold']
        elif method == 'relative' and 'relative' in threshold_results:
            return threshold_results['relative']['threshold']
        else:
            # 默认返回推荐阈值（常数序列）
            return pd.Series(
                [threshold_results['recommended']] * len(self.data),
                index=self.data.index
            )


def calculate_capital_persistence(
    capital_inflow: pd.Series,
    window: int = 5
) -> pd.Series:
    """
    计算资金流入持续性
    
    Args:
        capital_inflow: 资金流入序列
        window: 持续性窗口
    
    Returns:
        持续性指标（连续流入的天数比例）
    """
    # 正流入标记
    positive_flow = (capital_inflow > 0).astype(int)
    
    # 滚动窗口内正流入的比例
    persistence = positive_flow.rolling(window).sum() / window
    
    return persistence


def calculate_capital_momentum(
    capital_inflow: pd.Series,
    short_window: int = 5,
    long_window: int = 20
) -> pd.Series:
    """
    计算资金动量
    
    Args:
        capital_inflow: 资金流入序列
        short_window: 短期窗口
        long_window: 长期窗口
    
    Returns:
        资金动量指标
    """
    short_ma = capital_inflow.rolling(short_window).mean()
    long_ma = capital_inflow.rolling(long_window).mean()
    
    momentum = (short_ma - long_ma) / (long_ma.abs() + 1e-8)
    
    return momentum
