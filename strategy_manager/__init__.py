"""
strategy_manager - 量化策略管理系统 v3.0

一个功能完整的量化策略管理框架，支持策略评估、回测、参数优化和数据库管理。

主要模块：
- config: 配置管理
- data_manager: 数据管理（三级缓存）
- backtest_engine: 回测引擎
- strategy_database: 策略数据库
- parameter_optimizer: 参数优化器
- strategy_manager: 策略管理器主控制器
- strategies: 策略基类和具体策略实现
- main: 命令行入口

使用示例：
    # 创建策略管理器
    from strategy_manager import StrategyManager, Config, StrategyFactory

    config = Config(tushare_token="your_token")
    manager = StrategyManager(config)

    # 创建并注册策略
    from strategy_manager.strategies import ValueStrategy
    strategy = ValueStrategy("value_strategy", {"pe_ttm_max": 25, "roe_min": 12})
    manager.register_strategy("value", strategy)

    # 评估策略
    results = manager.evaluate_strategy(
        name="value",
        data=stock_data,
        buy_date="20240101"
    )
"""

__version__ = "3.0.0"
__author__ = "DeepQuant Team"

# ==================== 公共接口导出 ====================

from .config import Config
from .data_manager import BatchDataManager
from .backtest_engine import BacktestEngine
from .strategy_database import StrategyDatabase
from .parameter_optimizer import ParameterOptimizer, ParamSpec, ParamType, OptimizationResult
from .strategy_manager import StrategyManager

# 策略相关
from .strategies import (
    BaseStrategy,
    ValueStrategy,
    MomentumStrategy,
    GrowthStrategy,
    MeanReversionStrategy,
    StrategyFactory,
    ColumnMapper,
    StrategyError,
    ParameterError,
    DataError,
)

# ==================== 模块便捷访问 ====================

__all__ = [
    # 配置和核心组件
    "Config",
    "StrategyManager",
    "BatchDataManager",
    "BacktestEngine",
    "StrategyDatabase",
    "ParameterOptimizer",
    # 优化相关
    "ParamSpec",
    "ParamType",
    "OptimizationResult",
    # 策略基类和工厂
    "BaseStrategy",
    "StrategyFactory",
    "ColumnMapper",
    # 具体策略
    "ValueStrategy",
    "MomentumStrategy",
    "GrowthStrategy",
    "MeanReversionStrategy",
    # 异常
    "StrategyError",
    "ParameterError",
    "DataError",
    # 版本信息
    "__version__",
]

# ==================== 日志配置 ====================

import logging

# 创建包级logger
logger = logging.getLogger(__name__)

# 确保包导入时至少有基本的日志配置
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==================== 包初始化信息 ====================

def get_package_info() -> dict:
    """
    获取包信息

    Returns:
        包信息字典
    """
    return {
        "name": "strategy_manager",
        "version": __version__,
        "author": __author__,
        "description": "量化策略管理系统",
        "modules": [
            "config",
            "data_manager",
            "backtest_engine",
            "strategy_database",
            "parameter_optimizer",
            "strategy_manager",
            "strategies",
            "main",
        ],
    }


def list_available_strategies() -> list:
    """
    列出所有可用的策略类型

    Returns:
        策略类型信息列表
    """
    return StrategyFactory.list_available()


__all__.extend(["get_package_info", "list_available_strategies", "logger"])
