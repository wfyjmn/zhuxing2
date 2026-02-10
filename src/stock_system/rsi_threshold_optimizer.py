"""
RSI阈值自动优化模块
支持多周期RSI、动态阈值、背离检测
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI指标
    
    Args:
        data: 价格序列（通常是收盘价）
        period: RSI周期
    
    Returns:
        RSI值序列
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def detect_rsi_divergence(
    price: pd.Series, 
    rsi: pd.Series, 
    window: int = 10
) -> pd.Series:
    """
    检测RSI背离
    
    Args:
        price: 价格序列
        rsi: RSI序列
        window: 检测窗口
    
    Returns:
        背离信号序列（1=看涨背离，-1=看跌背离，0=无背离）
    """
    divergence = pd.Series(0, index=price.index)
    
    for i in range(window, len(price)):
        price_window = price.iloc[i-window:i]
        rsi_window = rsi.iloc[i-window:i]
        
        # 检测看涨背离：价格创新低，RSI未创新低
        if (price_window.iloc[-1] == price_window.min() and 
            rsi_window.iloc[-1] > rsi_window.min()):
            divergence.iloc[i] = 1
        
        # 检测看跌背离：价格创新高，RSI未创新高
        elif (price_window.iloc[-1] == price_window.max() and 
              rsi_window.iloc[-1] < rsi_window.max()):
            divergence.iloc[i] = -1
    
    return divergence


class RSIThresholdOptimizer:
    """RSI阈值优化器"""
    
    def __init__(
        self, 
        price_data: pd.Series, 
        target_returns: pd.Series,
        config: Dict[str, Any] = None
    ):
        """
        初始化RSI阈值优化器
        
        Args:
            price_data: 价格数据
            target_returns: 目标收益率（用于评估阈值效果）
            config: 配置参数
        """
        self.price_data = price_data
        self.target_returns = target_returns
        self.config = config or {}
        
        # 默认配置
        self.rsi_periods = self.config.get('periods', [6, 12, 14, 24])
        self.threshold_grid_start = self.config.get('threshold_grid', {}).get('start', 30)
        self.threshold_grid_end = self.config.get('threshold_grid', {}).get('end', 80)
        self.threshold_grid_step = self.config.get('threshold_grid', {}).get('step', 5)
        self.divergence_detection = self.config.get('divergence_detection', True)
        self.recall_constraint = self.config.get('recall_constraint', 0.7)
        
    def optimize_rsi_thresholds(self) -> Dict[str, Any]:
        """
        自动优化RSI多重周期阈值
        
        Returns:
            各周期RSI的最优阈值和性能指标
        """
        results = {}
        
        for period in self.rsi_periods:
            # 计算RSI
            rsi_values = calculate_rsi(self.price_data, period)
            
            # 网格搜索最佳阈值
            threshold_grid = np.arange(
                self.threshold_grid_start, 
                self.threshold_grid_end + 1, 
                self.threshold_grid_step
            )
            
            best_f1 = 0
            best_threshold = 50
            best_metrics = {}
            
            for threshold in threshold_grid:
                # 生成信号（RSI高于阈值为信号）
                signals = (rsi_values > threshold).astype(int)
                
                # 计算指标
                try:
                    precision = precision_score(
                        self.target_returns, 
                        signals, 
                        zero_division=0
                    )
                    recall = recall_score(
                        self.target_returns, 
                        signals, 
                        zero_division=0
                    )
                    f1 = f1_score(
                        self.target_returns, 
                        signals, 
                        zero_division=0
                    )
                    
                    # 在召回率约束下优化F1
                    if recall >= self.recall_constraint and f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                        best_metrics = {
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        }
                except:
                    continue
            
            results[f'RSI_{period}'] = {
                'optimal_threshold': best_threshold,
                'best_f1': best_f1,
                'precision_at_threshold': best_metrics.get('precision', 0),
                'recall_at_threshold': best_metrics.get('recall', 0),
                'period': period
            }
        
        # 计算多周期组合权重
        if len(results) > 0:
            results['combined'] = self._calculate_combined_weights(results)
        
        return results
    
    def _calculate_combined_weights(self, rsi_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算多周期RSI的组合权重
        基于各周期的F1分数进行加权
        """
        total_f1 = sum([r['best_f1'] for r in rsi_results.values()])
        
        weights = {}
        for period_key, result in rsi_results.items():
            if 'best_f1' in result:
                weights[period_key] = result['best_f1'] / total_f1 if total_f1 > 0 else 0.25
        
        return {
            'weights': weights,
            'method': 'f1_weighted',
            'description': '基于各周期F1分数的加权组合'
        }
    
    def calculate_dynamic_threshold(
        self, 
        rsi_values: pd.Series,
        period: int,
        lookback: int = 60
    ) -> pd.Series:
        """
        计算动态RSI阈值
        基于历史RSI分布的自适应阈值
        
        Args:
            rsi_values: RSI值序列
            period: RSI周期
            lookback: 回望窗口
        
        Returns:
            动态阈值序列
        """
        # 滚动分位数阈值
        rolling_q75 = rsi_values.rolling(lookback).quantile(0.75)
        rolling_q80 = rsi_values.rolling(lookback).quantile(0.80)
        
        # 滚动均值 + 标准差
        rolling_mean = rsi_values.rolling(lookback).mean()
        rolling_std = rsi_values.rolling(lookback).std()
        dynamic_threshold = rolling_mean + rolling_std
        
        # 综合多种方法（取平均）
        combined_threshold = (rolling_q80 + dynamic_threshold) / 2
        
        # 限制阈值在合理范围
        combined_threshold = combined_threshold.clip(30, 80)
        
        return combined_threshold
    
    def optimize_with_divergence(self) -> Dict[str, Any]:
        """
        结合背离检测优化RSI阈值
        
        Returns:
            包含背离检测的优化结果
        """
        results = {}
        
        for period in self.rsi_periods:
            rsi_values = calculate_rsi(self.price_data, period)
            
            # 检测背离
            divergence = detect_rsi_divergence(self.price_data, rsi_values)
            
            # 基础信号：RSI > 阈值
            best_threshold = 50
            best_score = 0
            
            for threshold in range(30, 81, 5):
                base_signals = (rsi_values > threshold).astype(int)
                
                # 结合背离信号
                combined_signals = base_signals.copy()
                combined_signals[divergence == 1] = 1  # 看涨背离增强信号
                
                # 评估
                try:
                    precision = precision_score(
                        self.target_returns, 
                        combined_signals, 
                        zero_division=0
                    )
                    recall = recall_score(
                        self.target_returns, 
                        combined_signals, 
                        zero_division=0
                    )
                    f1 = f1_score(
                        self.target_returns, 
                        combined_signals, 
                        zero_division=0
                    )
                    
                    if recall >= self.recall_constraint and f1 > best_score:
                        best_score = f1
                        best_threshold = threshold
                except:
                    continue
            
            results[f'RSI_{period}_with_divergence'] = {
                'optimal_threshold': best_threshold,
                'best_f1': best_score,
                'divergence_enhanced': True
            }
        
        return results
    
    def generate_rsi_features(self) -> pd.DataFrame:
        """
        生成多周期RSI特征
        
        Returns:
            包含各周期RSI的数据框
        """
        rsi_features = pd.DataFrame(index=self.price_data.index)
        
        for period in self.rsi_periods:
            rsi_values = calculate_rsi(self.price_data, period)
            rsi_features[f'rsi_{period}'] = rsi_values
            
            # 计算动态阈值
            if self.config.get('dynamic_threshold', True):
                rsi_features[f'rsi_{period}_dynamic_threshold'] = self.calculate_dynamic_threshold(
                    rsi_values, period
                )
                # RSI相对于动态阈值的偏差
                rsi_features[f'rsi_{period}_threshold_diff'] = (
                    rsi_values - rsi_features[f'rsi_{period}_dynamic_threshold']
                )
            
            # 检测背离
            if self.divergence_detection:
                divergence = detect_rsi_divergence(self.price_data, rsi_values)
                rsi_features[f'rsi_{period}_divergence'] = divergence
        
        return rsi_features


def optimize_multi_rsi_weights(
    rsi_results: Dict[str, Any],
    target_returns: pd.Series,
    price_data: pd.Series
) -> Dict[str, float]:
    """
    优化多周期RSI的组合权重
    
    Args:
        rsi_results: 各周期RSI的优化结果
        target_returns: 目标收益率
        price_data: 价格数据
    
    Returns:
        最优权重字典
    """
    # 提取各周期RSI值
    rsi_features = pd.DataFrame(index=price_data.index)
    for period_key, result in rsi_results.items():
        if 'period' in result:
            period = result['period']
            rsi_features[period_key] = calculate_rsi(price_data, period)
    
    # 使用网格搜索优化权重
    best_weights = {}
    best_score = 0
    
    # 简化：只测试几组权重组合
    weight_combinations = [
        {k: 1.0/len(rsi_results) for k in rsi_results.keys()},  # 等权重
        {k: 0.5 if '6' in k else 0.2 if '24' in k else 0.15 for k in rsi_results.keys()},  # 短期偏向
        {k: 0.3 if '12' in k else 0.35 if '14' in k else 0.175 for k in rsi_results.keys()},  # 中期偏向
    ]
    
    for weights in weight_combinations:
        # 计算加权组合RSI信号
        weighted_signal = pd.Series(0, index=price_data.index)
        for period_key, weight in weights.items():
            rsi_values = calculate_rsi(price_data, rsi_results[period_key]['period'])
            signal = (rsi_values > rsi_results[period_key]['optimal_threshold']).astype(int)
            weighted_signal += signal * weight
        
        # 评估
        try:
            f1 = f1_score(target_returns, (weighted_signal > 0.5).astype(int), zero_division=0)
            if f1 > best_score:
                best_score = f1
                best_weights = weights
        except:
            continue
    
    return best_weights
