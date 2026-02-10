"""
阈值优化工具集
为Agent提供阈值优化能力
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from langchain.tools import tool

# 导入阈值优化模块
from stock_system.auto_threshold_optimizer import AutoThresholdOptimizer
from stock_system.rsi_threshold_optimizer import RSIThresholdOptimizer
from stock_system.capital_threshold_optimizer import CapitalIntensityThresholdOptimizer
from stock_system.constrained_optimizer import ConstrainedOptimizer
from stock_system.dynamic_threshold_adjuster import DynamicThresholdAdjuster
from stock_system.multi_objective_optimizer import MultiObjectiveOptimizer


@tool
def optimize_feature_thresholds(
    data_path: str,
    features: str,  # JSON字符串格式的特征列表
    target_variable: str,
    method: str = "ensemble"
) -> str:
    """
    优化单个特征的阈值
    
    Args:
        data_path: 数据文件路径
        features: 特征列表（JSON字符串）
        target_variable: 目标变量名称
        method: 优化方法 [quantile, information_gain, model_based, ensemble]
    
    Returns:
        优化结果摘要
    """
    try:
        # 读取数据
        data = pd.read_csv(data_path)
        feature_list = json.loads(features)
        
        # 创建优化器
        optimizer = AutoThresholdOptimizer(data, target_variable, feature_list)
        thresholds = optimizer.calculate_optimal_thresholds(method)
        
        # 生成摘要
        summary = optimizer.get_threshold_summary()
        
        result = {
            'status': 'success',
            'method': method,
            'thresholds': thresholds,
            'summary': summary.to_dict('records')
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)


@tool
def optimize_rsi_thresholds(
    price_path: str,
    target_path: str,
    periods: str = "[6, 12, 14, 24]",
    recall_constraint: float = 0.7
) -> str:
    """
    优化RSI多重周期阈值
    
    Args:
        price_path: 价格数据文件路径
        target_path: 目标收益率文件路径
        periods: RSI周期列表（JSON字符串）
        recall_constraint: 召回率约束
    
    Returns:
        各周期RSI的最优阈值
    """
    try:
        # 读取数据
        price_data = pd.read_csv(price_path, index_col=0).iloc[:, 0]
        target_returns = pd.read_csv(target_path, index_col=0).iloc[:, 0]
        period_list = json.loads(periods)
        
        # 创建配置
        config = {
            'periods': period_list,
            'recall_constraint': recall_constraint
        }
        
        # 创建优化器
        optimizer = RSIThresholdOptimizer(price_data, target_returns, config)
        results = optimizer.optimize_rsi_thresholds()
        
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)


@tool
def optimize_capital_thresholds(
    data_path: str,
    feature_name: str = "main_capital_inflow_ratio",
    market_feature_name: str = None,
    lookback_days: int = 60
) -> str:
    """
    优化资金强度动态阈值
    
    Args:
        data_path: 数据文件路径
        feature_name: 资金流特征名称
        market_feature_name: 市场整体资金流特征名称
        lookback_days: 回望窗口天数
    
    Returns:
        多种方法计算的阈值
    """
    try:
        # 读取数据
        data = pd.read_csv(data_path)
        
        # 创建配置
        config = {'lookback_days': lookback_days}
        
        # 创建优化器
        optimizer = CapitalIntensityThresholdOptimizer(data, config)
        thresholds = optimizer.learn_capital_thresholds(feature_name, market_feature_name)
        
        # 转换结果为可序列化格式
        result = {}
        for key, value in thresholds.items():
            if isinstance(value, pd.Series):
                result[key] = value.to_dict()
            else:
                result[key] = value
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)


@tool
def optimize_with_constraints(
    features_path: str,
    target_path: str,
    objective: str = "f1",
    recall_constraint: float = 0.7
) -> str:
    """
    在召回率约束下优化阈值
    
    Args:
        features_path: 特征文件路径
        target_path: 目标文件路径
        objective: 优化目标 [f1, precision]
        recall_constraint: 召回率约束
    
    Returns:
        最优阈值和性能指标
    """
    try:
        # 读取数据
        X = pd.read_csv(features_path).values
        y = pd.read_csv(target_path).values.ravel()
        
        # 创建一个简单的模型（这里用随机模型作为示例）
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # 简单划分验证集
        val_size = min(100, len(X) // 4)
        X_val, y_val = X[-val_size:], y[-val_size:]
        
        # 训练模型
        X_train, y_train = X[:-val_size], y[:-val_size]
        model.fit(X_train, y_train)
        
        # 创建约束优化器
        config = {
            'min_recall': recall_constraint,
            'threshold_bounds': [0.3, 0.7]
        }
        optimizer = ConstrainedOptimizer(X_val, y_val, model, config)
        
        # 执行优化
        optimal_threshold, metrics = optimizer.optimize_threshold_for_f1(recall_constraint)
        
        result = {
            'optimal_threshold': float(optimal_threshold),
            'metrics': metrics
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)


@tool
def adjust_threshold_dynamically(
    base_threshold: float,
    volatility: float,
    trend: str,
    recent_precision: float = 0.5,
    recent_recall: float = 0.7
) -> str:
    """
    根据市场条件动态调整阈值
    
    Args:
        base_threshold: 基础阈值
        volatility: 市场波动率
        trend: 市场趋势 [bullish, bearish, neutral]
        recent_precision: 近期精确率
        recent_recall: 近期召回率
    
    Returns:
        调整后的阈值
    """
    try:
        # 创建调整器
        adjuster = DynamicThresholdAdjuster(base_threshold)
        
        # 构建市场条件
        market_conditions = {
            'volatility': volatility,
            'trend': trend,
            'recent_precision': recent_precision,
            'recent_recall': recent_recall
        }
        
        # 调整阈值
        adjusted_threshold = adjuster.adjust_threshold(market_conditions)
        
        # 获取调整摘要
        summary = adjuster.get_adjustment_summary()
        
        result = {
            'base_threshold': base_threshold,
            'adjusted_threshold': float(adjusted_threshold),
            'adjustment': float(adjusted_threshold - base_threshold),
            'factors': {
                'volatility': market_conditions.get('volatility'),
                'trend': market_conditions.get('trend')
            }
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)


@tool
def multi_objective_optimization(
    data_path: str,
    features: str,
    target: str,
    n_trials: int = 50
) -> str:
    """
    多目标优化：平衡精确率、召回率、交易成本
    
    Args:
        data_path: 数据文件路径
        features: 特征列表（JSON字符串）
        target: 目标变量名称
        n_trials: 优化试验次数
    
    Returns:
        最佳参数和得分
    """
    try:
        # 读取数据
        data = pd.read_csv(data_path)
        feature_list = json.loads(features)
        
        # 创建信号生成函数
        def generate_signals(data, features, params):
            # 简单示例：基于模型概率和RSI
            signals = pd.Series(0, index=data.index)
            
            # 模型概率信号
            if 'model_probability' in data.columns:
                model_signal = (data['model_probability'] > params.get('model_threshold', 0.5)).astype(int)
                signals += model_signal
            
            # RSI信号
            if 'rsi_6' in data.columns:
                rsi_signal = (data['rsi_6'] > params.get('rsi_threshold', 50)).astype(int)
                signals += rsi_signal
            
            # 成交量信号
            if 'volume_ratio' in data.columns:
                volume_signal = (data['volume_ratio'] > params.get('volume_threshold', 2.0)).astype(int)
                signals += volume_signal
            
            # 综合信号
            return (signals > 0).astype(int)
        
        # 创建多目标优化器
        config = {
            'n_trials': n_trials,
            'recall_constraint': 0.7,
            'trade_cost_penalty': 0.001
        }
        optimizer = MultiObjectiveOptimizer(data, feature_list, target, config)
        
        # 执行优化
        best_params, best_score = optimizer.optimize(generate_signals)
        
        result = {
            'best_params': best_params,
            'best_score': float(best_score)
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)


@tool
def get_optimization_config(config_path: str = "config/auto_threshold_config.json") -> str:
    """
    获取阈值优化配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置内容
    """
    try:
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        full_path = os.path.join(workspace_path, config_path)
        
        with open(full_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return json.dumps(config, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)
