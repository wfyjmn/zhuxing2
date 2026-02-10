"""
动态阈值调整机制
根据市场条件和模型表现动态调整阈值
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime


class DynamicThresholdAdjuster:
    """动态阈值调整器"""
    
    def __init__(
        self, 
        base_threshold: float = 0.5,
        adjustment_factors: Dict[str, float] = None,
        config: Dict[str, Any] = None
    ):
        """
        初始化动态阈值调整器
        
        Args:
            base_threshold: 基础阈值
            adjustment_factors: 调整因子权重 {'volatility_weight': 0.3, 'trend_weight': 0.4, 'performance_weight': 0.3}
            config: 配置参数
        """
        self.base_threshold = base_threshold
        self.adjustment_factors = adjustment_factors or {
            'volatility_weight': 0.3,
            'trend_weight': 0.4,
            'performance_weight': 0.3
        }
        self.config = config or {}
        
        # 配置参数
        self.volatility_thresholds = self.config.get('volatility_thresholds', {
            'high': 0.02,
            'low': 0.005
        })
        self.threshold_bounds = self.config.get('threshold_bounds', [0.3, 0.7])
        
        # 调整历史
        self.adjustment_history = []
        
    def adjust_threshold(
        self, 
        market_conditions: Dict[str, Any],
        model_confidence: float = 0.5,
        timestamp: datetime = None
    ) -> float:
        """
        根据市场条件和模型置信度动态调整阈值
        
        Args:
            market_conditions: 市场条件字典 {'volatility': float, 'trend': str, ...}
            model_confidence: 模型近期置信度
            timestamp: 调整时间戳
        
        Returns:
            调整后的阈值
        """
        # 计算各维度调整因子
        volatility_factor = self._calculate_volatility_factor(market_conditions)
        trend_factor = self._calculate_trend_factor(market_conditions)
        performance_factor = self._calculate_performance_factor(market_conditions, model_confidence)
        
        # 合成调整因子
        adjustment = (
            volatility_factor * self.adjustment_factors.get('volatility_weight', 0.3) +
            trend_factor * self.adjustment_factors.get('trend_weight', 0.4) +
            performance_factor * self.adjustment_factors.get('performance_weight', 0.3)
        )
        
        # 调整阈值
        adjusted_threshold = self.base_threshold * (1 + adjustment)
        
        # 边界限制
        adjusted_threshold = np.clip(adjusted_threshold, self.threshold_bounds[0], self.threshold_bounds[1])
        
        # 记录调整
        adjustment_record = {
            'timestamp': timestamp or datetime.now(),
            'base_threshold': self.base_threshold,
            'adjusted_threshold': adjusted_threshold,
            'adjustment': adjustment,
            'factors': {
                'volatility': volatility_factor,
                'trend': trend_factor,
                'performance': performance_factor
            },
            'market_conditions': market_conditions
        }
        self.adjustment_history.append(adjustment_record)
        
        return adjusted_threshold
    
    def _calculate_volatility_factor(self, market_conditions: Dict[str, Any]) -> float:
        """基于市场波动率调整"""
        volatility = market_conditions.get('volatility', 0.01)
        
        if volatility > self.volatility_thresholds['high']:
            # 高波动：提高阈值，更谨慎
            return 0.1
        elif volatility < self.volatility_thresholds['low']:
            # 低波动：降低阈值，更积极
            return -0.05
        else:
            # 正常波动：不变
            return 0.0
    
    def _calculate_trend_factor(self, market_conditions: Dict[str, Any]) -> float:
        """基于市场趋势调整"""
        trend = market_conditions.get('trend', 'neutral')
        
        if trend == 'bullish':
            # 牛市：降低阈值，更积极
            return -0.08
        elif trend == 'bearish':
            # 熊市：提高阈值，更谨慎
            return 0.12
        elif trend == 'neutral':
            # 震荡市：不变
            return 0.0
        else:
            return 0.0
    
    def _calculate_performance_factor(
        self, 
        market_conditions: Dict[str, Any],
        model_confidence: float
    ) -> float:
        """基于模型近期表现调整"""
        # 如果市场条件中提供了模型表现，使用它
        recent_precision = market_conditions.get('recent_precision', 0.5)
        recent_recall = market_conditions.get('recent_recall', 0.5)
        
        # 基准表现
        target_precision = 0.5
        target_recall = 0.7
        
        # 计算偏差
        precision_diff = recent_precision - target_precision
        recall_diff = recent_recall - target_recall
        
        # 如果精确率太低，提高阈值
        if precision_diff < -0.1:
            return 0.05
        # 如果召回率太低，降低阈值
        elif recall_diff < -0.1:
            return -0.05
        # 如果表现很好，保持当前阈值
        else:
            return 0.0
    
    def adjust_threshold_series(
        self,
        market_conditions_df: pd.DataFrame,
        model_confidence_series: pd.Series = None
    ) -> pd.Series:
        """
        批量调整阈值序列
        
        Args:
            market_conditions_df: 包含市场条件的数据框
            model_confidence_series: 模型置信度序列
        
        Returns:
            调整后的阈值序列
        """
        adjusted_thresholds = []
        
        for i in range(len(market_conditions_df)):
            # 提取当前市场条件
            current_conditions = {
                'volatility': market_conditions_df.iloc[i].get('volatility', 0.01),
                'trend': market_conditions_df.iloc[i].get('trend', 'neutral'),
                'recent_precision': market_conditions_df.iloc[i].get('recent_precision', 0.5),
                'recent_recall': market_conditions_df.iloc[i].get('recent_recall', 0.5)
            }
            
            # 获取当前模型置信度
            current_confidence = model_confidence_series.iloc[i] if model_confidence_series is not None else 0.5
            
            # 调整阈值
            adjusted_threshold = self.adjust_threshold(
                current_conditions,
                current_confidence,
                timestamp=market_conditions_df.index[i] if hasattr(market_conditions_df, 'index') else None
            )
            
            adjusted_thresholds.append(adjusted_threshold)
        
        return pd.Series(adjusted_thresholds, index=market_conditions_df.index)
    
    def get_adjustment_summary(self) -> pd.DataFrame:
        """获取调整历史汇总"""
        if not self.adjustment_history:
            return pd.DataFrame()
        
        summary_data = []
        for record in self.adjustment_history:
            summary_data.append({
                'timestamp': record['timestamp'],
                'base_threshold': record['base_threshold'],
                'adjusted_threshold': record['adjusted_threshold'],
                'adjustment': record['adjustment'],
                'volatility_factor': record['factors']['volatility'],
                'trend_factor': record['factors']['trend'],
                'performance_factor': record['factors']['performance']
            })
        
        return pd.DataFrame(summary_data)
    
    def reset_base_threshold(self, new_base_threshold: float):
        """重置基础阈值"""
        self.base_threshold = new_base_threshold
        print(f"Base threshold updated to {new_base_threshold}")
    
    def update_adjustment_factors(self, new_factors: Dict[str, float]):
        """更新调整因子权重"""
        self.adjustment_factors = new_factors
        print(f"Adjustment factors updated: {new_factors}")


class MarketConditionAnalyzer:
    """市场条件分析器"""
    
    def __init__(self, price_data: pd.Series, window: int = 20):
        """
        初始化市场条件分析器
        
        Args:
            price_data: 价格数据
            window: 分析窗口
        """
        self.price_data = price_data
        self.window = window
        
    def analyze_market_conditions(
        self, 
        start_idx: int = None,
        end_idx: int = None
    ) -> Dict[str, Any]:
        """
        分析当前市场条件
        
        Args:
            start_idx: 开始索引
            end_idx: 结束索引
        
        Returns:
            市场条件字典
        """
        if start_idx is None:
            start_idx = -self.window
        if end_idx is None:
            end_idx = len(self.price_data)
        
        window_data = self.price_data.iloc[start_idx:end_idx]
        
        # 计算波动率（日收益率的标准差）
        returns = window_data.pct_change().dropna()
        volatility = returns.std()
        
        # 计算趋势
        price_change = (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]
        
        # 判断趋势方向
        if price_change > 0.05:
            trend = 'bullish'
        elif price_change < -0.05:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # 计算平均成交额（如果有）
        avg_volume = window_data.mean()  # 简化处理
        
        return {
            'volatility': volatility,
            'trend': trend,
            'price_change': price_change,
            'avg_price': avg_volume,
            'returns_std': returns.std(),
            'returns_mean': returns.mean()
        }
    
    def generate_market_conditions_series(self) -> pd.DataFrame:
        """生成市场条件时间序列"""
        results = []
        
        for i in range(self.window, len(self.price_data)):
            conditions = self.analyze_market_conditions(i-self.window, i)
            results.append(conditions)
        
        return pd.DataFrame(results, index=self.price_data.index[self.window:])


def calculate_market_volatility(price_data: pd.Series, window: int = 20) -> pd.Series:
    """计算市场波动率"""
    returns = price_data.pct_change()
    volatility = returns.rolling(window).std()
    return volatility


def detect_market_trend(
    price_data: pd.Series, 
    short_window: int = 10,
    long_window: int = 30
) -> pd.Series:
    """
    检测市场趋势
    
    Args:
        price_data: 价格数据
        short_window: 短期均线窗口
        long_window: 长期均线窗口
    
    Returns:
        趋势序列（1=上升，0=震荡，-1=下降）
    """
    short_ma = price_data.rolling(short_window).mean()
    long_ma = price_data.rolling(long_window).mean()
    
    # 短期均线相对长期均线的位置
    ma_diff = (short_ma - long_ma) / long_ma
    
    # 定义趋势
    trend = pd.Series(0, index=price_data.index)
    trend[ma_diff > 0.02] = 1  # 上升
    trend[ma_diff < -0.02] = -1  # 下降
    
    return trend
