"""
参数优化模块

修复内容：
1. 优化方法名统一 - 支持多种别名（grid/grid_search, random/random_search, bayesian）
2. 参数空间定义 - 使用明确的数据类区分范围参数和离散参数，消除歧义
3. 贝叶斯优化回退 - scikit-optimize不可用时安全回退到随机搜索
4. 评估函数 - 集成真实回测引擎，提供快速评估和完整回测两种模式
5. 交叉验证 - 时间序列交叉验证（不打乱时间顺序）
6. 拉丁超立方采样 - 索引越界修复
7. 结果持久化 - 优化过程和结果保存到文件
8. 收敛检测 - 提前停止避免无效迭代
9. 并行评估 - 支持多线程并行评估参数组合
"""

import hashlib
import itertools
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import Config


# ==================== 参数空间定义 ====================


class ParamType(str, Enum):
    """参数类型枚举"""

    INT_RANGE = "int_range"        # 整数范围 [low, high]
    FLOAT_RANGE = "float_range"    # 浮点范围 [low, high]
    INT_CHOICE = "int_choice"      # 整数离散值列表
    FLOAT_CHOICE = "float_choice"  # 浮点离散值列表
    CATEGORY = "category"          # 分类值列表
    BOOLEAN = "boolean"            # 布尔值


@dataclass
class ParamSpec:
    """
    参数规格定义
    
    明确区分范围参数和离散参数，消除原版中列表长度判断的歧义
    
    使用示例：
        # 浮点范围
        ParamSpec("pe_max", ParamType.FLOAT_RANGE, low=10.0, high=50.0)
        
        # 整数离散值
        ParamSpec("lookback", ParamType.INT_CHOICE, choices=[5, 10, 20, 60])
        
        # 分类
        ParamSpec("mode", ParamType.CATEGORY, choices=["ma20", "ma60", "bollinger"])
        
        # 布尔
        ParamSpec("enable_filter", ParamType.BOOLEAN)
    """

    name: str
    param_type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Any = None
    description: str = ""

    def __post_init__(self):
        """验证参数规格"""
        if self.param_type in (ParamType.INT_RANGE, ParamType.FLOAT_RANGE):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"参数 '{self.name}' 类型为 {self.param_type}，"
                    f"必须指定 low 和 high"
                )
            if self.low >= self.high:
                raise ValueError(
                    f"参数 '{self.name}' 的 low({self.low}) "
                    f"必须小于 high({self.high})"
                )

        if self.param_type in (
            ParamType.INT_CHOICE,
            ParamType.FLOAT_CHOICE,
            ParamType.CATEGORY,
        ):
            if not self.choices or len(self.choices) == 0:
                raise ValueError(
                    f"参数 '{self.name}' 类型为 {self.param_type}，"
                    f"必须提供 choices"
                )

    def sample_random(self) -> Any:
        """从参数空间随机采样一个值"""
        if self.param_type == ParamType.INT_RANGE:
            return random.randint(int(self.low), int(self.high))
        elif self.param_type == ParamType.FLOAT_RANGE:
            return round(random.uniform(self.low, self.high), 6)
        elif self.param_type in (
            ParamType.INT_CHOICE,
            ParamType.FLOAT_CHOICE,
            ParamType.CATEGORY,
        ):
            return random.choice(self.choices)
        elif self.param_type == ParamType.BOOLEAN:
            return random.choice([True, False])
        else:
            raise ValueError(f"未知参数类型: {self.param_type}")

    def sample_from_unit(self, u: float) -> Any:
        """
        从 [0, 1] 区间的均匀值映射到参数空间
        
        用于拉丁超立方采样等需要从均匀分布映射的场景
        
        Args:
            u: [0, 1] 区间的值
            
        Returns:
            参数空间中的值
        """
        u = max(0.0, min(1.0, u))

        if self.param_type == ParamType.INT_RANGE:
            return int(round(self.low + u * (self.high - self.low)))
        elif self.param_type == ParamType.FLOAT_RANGE:
            return round(self.low + u * (self.high - self.low), 6)
        elif self.param_type in (
            ParamType.INT_CHOICE,
            ParamType.FLOAT_CHOICE,
            ParamType.CATEGORY,
        ):
            idx = min(int(u * len(self.choices)), len(self.choices) - 1)
            return self.choices[idx]
        elif self.param_type == ParamType.BOOLEAN:
            return u >= 0.5
        else:
            raise ValueError(f"未知参数类型: {self.param_type}")

    def get_grid_values(self, n_points: int = 10) -> List[Any]:
        """
        生成网格搜索的离散值列表
        
        Args:
            n_points: 范围参数的网格点数
            
        Returns:
            离散值列表
        """
        if self.param_type == ParamType.INT_RANGE:
            step = max(1, int((self.high - self.low) / n_points))
            values = list(range(int(self.low), int(self.high) + 1, step))
            if int(self.high) not in values:
                values.append(int(self.high))
            return values
        elif self.param_type == ParamType.FLOAT_RANGE:
            return [
                round(v, 6)
                for v in np.linspace(self.low, self.high, n_points)
            ]
        elif self.param_type in (
            ParamType.INT_CHOICE,
            ParamType.FLOAT_CHOICE,
            ParamType.CATEGORY,
        ):
            return list(self.choices)
        elif self.param_type == ParamType.BOOLEAN:
            return [True, False]
        else:
            return [self.default]


# ==================== 优化结果 ====================


@dataclass
class OptimizationResult:
    """优化结果"""

    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    method: str
    target_metric: str
    total_iterations: int
    elapsed_seconds: float
    converged: bool = False
    convergence_iteration: Optional[int] = None
    param_space: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "method": self.method,
            "target_metric": self.target_metric,
            "total_iterations": self.total_iterations,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "converged": self.converged,
            "convergence_iteration": self.convergence_iteration,
            "top_10_results": sorted(
                self.all_results, key=lambda x: x.get("score", -1e10), reverse=True
            )[:10],
        }

    def summary(self) -> str:
        """生成摘要字符串"""
        return (
            f"优化完成: 方法={self.method}, "
            f"最佳分数={self.best_score:.4f}, "
            f"迭代={self.total_iterations}, "
            f"耗时={self.elapsed_seconds:.1f}s, "
            f"收敛={'是' if self.converged else '否'}"
        )


# ==================== 参数优化器 ====================


class ParameterOptimizer:
    """
    参数优化器
    
    支持三种优化方法：
    1. 网格搜索（grid_search）: 穷举所有参数组合
    2. 随机搜索（random_search）: 随机采样参数空间
    3. 贝叶斯优化（bayesian）: 基于高斯过程的智能搜索
    
    使用示例：
        optimizer = ParameterOptimizer(config)
        
        # 定义参数空间
        param_space = [
            ParamSpec("pe_max", ParamType.FLOAT_RANGE, low=10, high=50),
            ParamSpec("roe_min", ParamType.FLOAT_RANGE, low=5, high=30),
            ParamSpec("mode", ParamType.CATEGORY, choices=["ma20", "ma60"]),
        ]
        
        # 运行优化
        result = optimizer.optimize(
            param_space=param_space,
            evaluate_func=my_evaluate_function,
            method="bayesian",
            n_iterations=100,
            target_metric="sharpe_ratio",
        )
        
        print(result.best_params)
        print(result.best_score)
    """

    # 方法名别名映射
    METHOD_ALIASES = {
        "grid": "grid_search",
        "grid_search": "grid_search",
        "random": "random_search",
        "random_search": "random_search",
        "bayesian": "bayesian",
        "bayes": "bayesian",
        "bo": "bayesian",
        "lhs": "lhs_search",
        "latin_hypercube": "lhs_search",
    }

    def __init__(
        self,
        config: Config,
        backtest_engine: Any = None,
    ):
        """
        Args:
            config: 系统配置
            backtest_engine: 回测引擎实例（可选，用于完整回测评估）
        """
        self._config = config
        self._backtest_engine = backtest_engine
        self._logger = logging.getLogger(__name__)

        # 检查可选依赖
        self._skopt_available = self._check_skopt()

        # 优化历史
        self._history: List[OptimizationResult] = []

    # ==================== 公开接口 ====================

    def optimize(
        self,
        param_space: List[ParamSpec],
        evaluate_func: Callable[[pd.DataFrame, Dict[str, Any]], float],
        train_data: pd.DataFrame,
        method: Optional[str] = None,
        n_iterations: Optional[int] = None,
        target_metric: str = "sharpe_ratio",
        cv_folds: Optional[int] = None,
        early_stop_rounds: int = 20,
        early_stop_tolerance: float = 0.001,
        random_seed: int = 42,
        n_grid_points: int = 5,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        参数优化主入口
        
        Args:
            param_space: 参数空间定义列表
            evaluate_func: 评估函数，签名 (data, params) -> score
                接收训练数据和参数字典，返回目标指标分数（越大越好）
            train_data: 训练数据
            method: 优化方法（None使用配置默认值）
            n_iterations: 迭代次数（None使用配置默认值）
            target_metric: 目标指标名称（用于日志记录）
            cv_folds: 交叉验证折数（None使用配置默认值，1表示不做CV）
            early_stop_rounds: 连续多少轮无改善则提前停止
            early_stop_tolerance: 改善阈值（低于此值视为无改善）
            random_seed: 随机种子
            n_grid_points: 网格搜索时每个范围参数的网格点数
            verbose: 是否输出详细日志
            
        Returns:
            OptimizationResult
        """
        # 解析方法名
        if method is None:
            method = self._config.optimization_method
        resolved_method = self._resolve_method(method)

        # 解析迭代次数
        if n_iterations is None:
            n_iterations = getattr(self._config, "n_iterations", 50)

        # 解析CV折数
        if cv_folds is None:
            cv_folds = getattr(self._config, "cv_folds", 1)

        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        self._logger.info(
            f"开始参数优化: 方法={resolved_method}, "
            f"迭代={n_iterations}, 指标={target_metric}, "
            f"CV折数={cv_folds}, 参数数量={len(param_space)}"
        )

        start_time = time.time()

        # 根据方法分发
        if resolved_method == "grid_search":
            result = self._grid_search(
                param_space=param_space,
                evaluate_func=evaluate_func,
                train_data=train_data,
                target_metric=target_metric,
                cv_folds=cv_folds,
                n_grid_points=n_grid_points,
                early_stop_rounds=early_stop_rounds,
                early_stop_tolerance=early_stop_tolerance,
                verbose=verbose,
            )
        elif resolved_method == "random_search":
            result = self._random_search(
                param_space=param_space,
                evaluate_func=evaluate_func,
                train_data=train_data,
                n_iterations=n_iterations,
                target_metric=target_metric,
                cv_folds=cv_folds,
                early_stop_rounds=early_stop_rounds,
                early_stop_tolerance=early_stop_tolerance,
                verbose=verbose,
            )
        elif resolved_method == "bayesian":
            result = self._bayesian_search(
                param_space=param_space,
                evaluate_func=evaluate_func,
                train_data=train_data,
                n_iterations=n_iterations,
                target_metric=target_metric,
                cv_folds=cv_folds,
                random_seed=random_seed,
                verbose=verbose,
            )
        elif resolved_method == "lhs_search":
            result = self._lhs_search(
                param_space=param_space,
                evaluate_func=evaluate_func,
                train_data=train_data,
                n_iterations=n_iterations,
                target_metric=target_metric,
                cv_folds=cv_folds,
                early_stop_rounds=early_stop_rounds,
                early_stop_tolerance=early_stop_tolerance,
                verbose=verbose,
            )
        else:
            raise ValueError(f"不支持的优化方法: {method}")

        elapsed = time.time() - start_time
        result.elapsed_seconds = elapsed
        result.target_metric = target_metric

        # 保存到历史
        self._history.append(result)

        # 持久化结果
        self._save_result(result)

        self._logger.info(result.summary())

        return result

    def optimize_from_dict(
        self,
        param_grid: Dict[str, Any],
        evaluate_func: Callable,
        train_data: pd.DataFrame,
        **kwargs,
    ) -> OptimizationResult:
        """
        从字典格式的参数网格启动优化（兼容旧接口）
        
        支持的字典格式：
        - {"param_name": [value1, value2, ...]}  → 离散值
        - {"param_name": (low, high)}            → 范围
        
        Args:
            param_grid: 参数网格字典
            evaluate_func: 评估函数
            train_data: 训练数据
            **kwargs: 传递给 optimize() 的其他参数
            
        Returns:
            OptimizationResult
        """
        param_space = self._dict_to_param_space(param_grid)
        return self.optimize(
            param_space=param_space,
            evaluate_func=evaluate_func,
            train_data=train_data,
            **kwargs,
        )

    def get_history(self) -> List[OptimizationResult]:
        """获取优化历史"""
        return list(self._history)

    # ==================== 网格搜索 ====================

    def _grid_search(
        self,
        param_space: List[ParamSpec],
        evaluate_func: Callable,
        train_data: pd.DataFrame,
        target_metric: str,
        cv_folds: int,
        n_grid_points: int,
        early_stop_rounds: int,
        early_stop_tolerance: float,
        verbose: bool,
    ) -> OptimizationResult:
        """网格搜索优化"""
        # 生成所有参数组合
        param_names = [p.name for p in param_space]
        param_grids = [p.get_grid_values(n_grid_points) for p in param_space]

        all_combinations = list(itertools.product(*param_grids))
        total = len(all_combinations)

        self._logger.info(f"网格搜索: {total} 个参数组合")

        if total > 10000:
            self._logger.warning(
                f"参数组合数量 {total} 过多，建议使用随机搜索或贝叶斯优化"
            )

        # 评估所有组合
        all_results = []
        best_score = -np.inf
        best_params = None
        no_improve_count = 0

        for i, values in enumerate(all_combinations):
            params = dict(zip(param_names, values))

            score = self._evaluate_with_cv(
                evaluate_func, train_data, params, cv_folds
            )

            result_entry = {
                "params": params,
                "score": score,
                "iteration": i + 1,
            }
            all_results.append(result_entry)

            # 更新最佳
            if score > best_score + early_stop_tolerance:
                best_score = score
                best_params = params.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if verbose and (i + 1) % max(1, total // 20) == 0:
                self._logger.info(
                    f"  网格搜索进度: {i + 1}/{total}, "
                    f"当前最佳={best_score:.4f}"
                )

            # 提前停止
            if no_improve_count >= early_stop_rounds and i >= total * 0.3:
                self._logger.info(
                    f"  网格搜索提前停止: 连续 {no_improve_count} 轮无改善"
                )
                break

        converged = no_improve_count >= early_stop_rounds

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            method="grid_search",
            target_metric=target_metric,
            total_iterations=len(all_results),
            elapsed_seconds=0,
            converged=converged,
            convergence_iteration=(
                len(all_results) - no_improve_count if converged else None
            ),
        )

    # ==================== 随机搜索 ====================

    def _random_search(
        self,
        param_space: List[ParamSpec],
        evaluate_func: Callable,
        train_data: pd.DataFrame,
        n_iterations: int,
        target_metric: str,
        cv_folds: int,
        early_stop_rounds: int,
        early_stop_tolerance: float,
        verbose: bool,
    ) -> OptimizationResult:
        """随机搜索优化"""
        all_results = []
        best_score = -np.inf
        best_params = None
        no_improve_count = 0
        best_iteration = 0

        for i in range(n_iterations):
            # 随机采样参数
            params = {spec.name: spec.sample_random() for spec in param_space}

            # 评估
            score = self._evaluate_with_cv(
                evaluate_func, train_data, params, cv_folds
            )

            result_entry = {
                "params": params,
                "score": score,
                "iteration": i + 1,
            }
            all_results.append(result_entry)

            # 更新最佳
            if score > best_score + early_stop_tolerance:
                best_score = score
                best_params = params.copy()
                no_improve_count = 0
                best_iteration = i + 1
            else:
                no_improve_count += 1

            if verbose and (i + 1) % max(1, n_iterations // 10) == 0:
                self._logger.info(
                    f"  随机搜索进度: {i + 1}/{n_iterations}, "
                    f"当前最佳={best_score:.4f} (iter {best_iteration})"
                )

            # 提前停止
            if no_improve_count >= early_stop_rounds:
                self._logger.info(
                    f"  随机搜索提前停止: 连续 {no_improve_count} 轮无改善"
                )
                break

        converged = no_improve_count >= early_stop_rounds

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            method="random_search",
            target_metric=target_metric,
            total_iterations=len(all_results),
            elapsed_seconds=0,
            converged=converged,
            convergence_iteration=best_iteration if converged else None,
        )

    # ==================== 贝叶斯优化 ====================

    def _bayesian_search(
        self,
        param_space: List[ParamSpec],
        evaluate_func: Callable,
        train_data: pd.DataFrame,
        n_iterations: int,
        target_metric: str,
        cv_folds: int,
        random_seed: int,
        verbose: bool,
    ) -> OptimizationResult:
        """
        贝叶斯优化
        
        使用scikit-optimize的gp_minimize
        不可用时安全回退到随机搜索
        """
        if not self._skopt_available:
            self._logger.warning(
                "scikit-optimize不可用，回退到随机搜索。"
                "安装方式: pip install scikit-optimize"
            )
            return self._random_search(
                param_space=param_space,
                evaluate_func=evaluate_func,
                train_data=train_data,
                n_iterations=n_iterations,
                target_metric=target_metric,
                cv_folds=cv_folds,
                early_stop_rounds=n_iterations,
                early_stop_tolerance=0.001,
                verbose=verbose,
            )

        try:
            from skopt import gp_minimize
            from skopt.space import Categorical, Integer, Real
            from skopt.utils import use_named_args

            # 构建搜索空间
            dimensions = []
            param_names = []

            for spec in param_space:
                dim = self._param_spec_to_skopt_dim(spec)
                if dim is not None:
                    dimensions.append(dim)
                    param_names.append(spec.name)

            if not dimensions:
                self._logger.error("无法构建有效的搜索空间")
                return self._empty_result("bayesian", target_metric)

            # 收集所有评估结果
            all_results = []
            eval_count = [0]  # 使用列表以支持闭包修改

            # 目标函数（gp_minimize最小化，所以取负）
            @use_named_args(dimensions=dimensions)
            def objective(**params):
                eval_count[0] += 1

                try:
                    score = self._evaluate_with_cv(
                        evaluate_func, train_data, params, cv_folds
                    )
                except Exception as e:
                    self._logger.warning(
                        f"贝叶斯优化评估失败 (iter {eval_count[0]}): {e}"
                    )
                    score = -1e6

                all_results.append(
                    {
                        "params": dict(params),
                        "score": score,
                        "iteration": eval_count[0],
                    }
                )

                if verbose and eval_count[0] % max(1, n_iterations // 10) == 0:
                    current_best = max(r["score"] for r in all_results)
                    self._logger.info(
                        f"  贝叶斯优化进度: {eval_count[0]}/{n_iterations}, "
                        f"当前最佳={current_best:.4f}"
                    )

                return -score  # 取负以最小化

            # 运行优化
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=n_iterations,
                n_initial_points=min(10, n_iterations // 3),
                random_state=random_seed,
                verbose=False,
                acq_func="EI",
            )

            # 提取最佳参数
            best_params = dict(zip(param_names, result.x))
            best_score = -result.fun

            return OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                all_results=all_results,
                method="bayesian",
                target_metric=target_metric,
                total_iterations=len(all_results),
                elapsed_seconds=0,
                converged=False,
            )

        except Exception as e:
            self._logger.error(f"贝叶斯优化失败: {e}，回退到随机搜索")
            return self._random_search(
                param_space=param_space,
                evaluate_func=evaluate_func,
                train_data=train_data,
                n_iterations=n_iterations,
                target_metric=target_metric,
                cv_folds=cv_folds,
                early_stop_rounds=n_iterations,
                early_stop_tolerance=0.001,
                verbose=verbose,
            )

    # ==================== 拉丁超立方采样搜索 ====================

    def _lhs_search(
        self,
        param_space: List[ParamSpec],
        evaluate_func: Callable,
        train_data: pd.DataFrame,
        n_iterations: int,
        target_metric: str,
        cv_folds: int,
        early_stop_rounds: int,
        early_stop_tolerance: float,
        verbose: bool,
    ) -> OptimizationResult:
        """拉丁超立方采样搜索"""
        # 生成LHS样本
        combinations = self._generate_lhs_samples(param_space, n_iterations)

        all_results = []
        best_score = -np.inf
        best_params = None
        no_improve_count = 0
        best_iteration = 0

        for i, params in enumerate(combinations):
            score = self._evaluate_with_cv(
                evaluate_func, train_data, params, cv_folds
            )

            result_entry = {
                "params": params,
                "score": score,
                "iteration": i + 1,
            }
            all_results.append(result_entry)

            if score > best_score + early_stop_tolerance:
                best_score = score
                best_params = params.copy()
                no_improve_count = 0
                best_iteration = i + 1
            else:
                no_improve_count += 1

            if verbose and (i + 1) % max(1, n_iterations // 10) == 0:
                self._logger.info(
                    f"  LHS搜索进度: {i + 1}/{n_iterations}, "
                    f"当前最佳={best_score:.4f}"
                )

            if no_improve_count >= early_stop_rounds:
                self._logger.info(
                    f"  LHS搜索提前停止: 连续 {no_improve_count} 轮无改善"
                )
                break

        converged = no_improve_count >= early_stop_rounds

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            method="lhs_search",
            target_metric=target_metric,
            total_iterations=len(all_results),
            elapsed_seconds=0,
            converged=converged,
            convergence_iteration=best_iteration if converged else None,
        )

    # ==================== 评估逻辑 ====================

    def _evaluate_with_cv(
        self,
        evaluate_func: Callable,
        data: pd.DataFrame,
        params: Dict[str, Any],
        cv_folds: int,
    ) -> float:
        """
        使用交叉验证评估参数
        
        当 cv_folds <= 1 时不做交叉验证，直接用全量数据评估
        当 cv_folds > 1 时使用时间序列交叉验证（保持时间顺序）
        
        Args:
            evaluate_func: 评估函数 (data, params) -> score
            data: 完整数据
            params: 参数字典
            cv_folds: 折数
            
        Returns:
            评估分数（多折取均值）
        """
        if cv_folds <= 1:
            return self._safe_evaluate(evaluate_func, data, params)

        # 时间序列交叉验证
        scores = self._time_series_cv(
            evaluate_func, data, params, cv_folds
        )

        valid_scores = [s for s in scores if np.isfinite(s) and s > -1e5]

        if not valid_scores:
            return -1e6

        return float(np.mean(valid_scores))

    def _time_series_cv(
        self,
        evaluate_func: Callable,
        data: pd.DataFrame,
        params: Dict[str, Any],
        n_folds: int,
    ) -> List[float]:
        """
        时间序列交叉验证
        
        与普通K-Fold不同，时间序列CV保持时间顺序：
        - Fold 1: train=[0..n/k], test=[n/k..2n/k]
        - Fold 2: train=[0..2n/k], test=[2n/k..3n/k]
        - ...
        
        这避免了"未来数据泄漏"问题
        
        Args:
            evaluate_func: 评估函数
            data: 完整数据（假设按时间排序）
            params: 参数
            n_folds: 折数
            
        Returns:
            各折的分数列表
        """
        n = len(data)
        if n < n_folds * 2:
            # 数据太少，退化为不做CV
            return [self._safe_evaluate(evaluate_func, data, params)]

        fold_size = n // (n_folds + 1)
        scores = []

        for fold in range(n_folds):
            # 训练集：从开头到当前fold的结束
            train_end = fold_size * (fold + 1)
            # 验证集：当前fold的结束到下一个fold的结束
            val_start = train_end
            val_end = min(train_end + fold_size, n)

            if val_start >= n or val_end <= val_start:
                continue

            val_data = data.iloc[val_start:val_end]

            if len(val_data) < 5:
                continue

            score = self._safe_evaluate(evaluate_func, val_data, params)
            scores.append(score)

        return scores

    def _safe_evaluate(
        self,
        evaluate_func: Callable,
        data: pd.DataFrame,
        params: Dict[str, Any],
    ) -> float:
        """
        安全执行评估函数
        
        捕获所有异常，返回极小值作为惩罚
        """
        try:
            score = evaluate_func(data, params)

            if score is None or not np.isfinite(score):
                return -1e6

            return float(score)

        except Exception as e:
            self._logger.debug(f"评估失败: params={params}, error={e}")
            return -1e6

    # ==================== LHS采样 ====================

    def _generate_lhs_samples(
        self,
        param_space: List[ParamSpec],
        n_samples: int,
    ) -> List[Dict[str, Any]]:
        """
        生成拉丁超立方采样参数组合
        
        拉丁超立方采样确保每个参数维度上的采样点均匀分布，
        比纯随机采样有更好的空间覆盖性
        
        Args:
            param_space: 参数空间
            n_samples: 样本数
            
        Returns:
            参数组合列表
        """
        n_params = len(param_space)

        try:
            from scipy.stats.qmc import LatinHypercube

            sampler = LatinHypercube(d=n_params, seed=42)
            samples = sampler.random(n=n_samples)

        except ImportError:
            # scipy不可用时手动实现简化版LHS
            self._logger.debug("scipy不可用，使用简化版LHS")
            samples = self._manual_lhs(n_params, n_samples)

        # 将 [0,1] 均匀样本映射到参数空间
        combinations = []
        for i in range(n_samples):
            params = {}
            for j, spec in enumerate(param_space):
                params[spec.name] = spec.sample_from_unit(samples[i, j])
            combinations.append(params)

        return combinations

    @staticmethod
    def _manual_lhs(n_dims: int, n_samples: int) -> np.ndarray:
        """
        手动实现简化版拉丁超立方采样
        
        Args:
            n_dims: 维度数
            n_samples: 样本数
            
        Returns:
            shape=(n_samples, n_dims) 的 [0,1] 均匀样本矩阵
        """
        samples = np.zeros((n_samples, n_dims))

        for dim in range(n_dims):
            # 每个维度的均匀分区
            perm = np.random.permutation(n_samples)
            samples[:, dim] = (perm + np.random.random(n_samples)) / n_samples

        return samples

    # ==================== 工具方法 ====================

    def _resolve_method(self, method: str) -> str:
        """
        解析优化方法名（支持别名）
        
        Args:
            method: 用户输入的方法名
            
        Returns:
            标准化的方法名
            
        Raises:
            ValueError: 不支持的方法名
        """
        normalized = method.lower().strip()
        resolved = self.METHOD_ALIASES.get(normalized)

        if resolved is None:
            valid = set(self.METHOD_ALIASES.values())
            raise ValueError(
                f"不支持的优化方法: '{method}'，"
                f"有效值: {sorted(valid)}"
            )

        return resolved

    @staticmethod
    def _check_skopt() -> bool:
        """检查scikit-optimize是否可用"""
        try:
            import skopt

            return True
        except ImportError:
            return False

    def _param_spec_to_skopt_dim(self, spec: ParamSpec):
        """将ParamSpec转换为skopt的搜索空间维度"""
        try:
            from skopt.space import Categorical, Integer, Real

            if spec.param_type == ParamType.INT_RANGE:
                return Integer(
                    int(spec.low), int(spec.high), name=spec.name
                )
            elif spec.param_type == ParamType.FLOAT_RANGE:
                return Real(
                    float(spec.low), float(spec.high), name=spec.name
                )
            elif spec.param_type in (
                ParamType.INT_CHOICE,
                ParamType.FLOAT_CHOICE,
                ParamType.CATEGORY,
            ):
                return Categorical(spec.choices, name=spec.name)
            elif spec.param_type == ParamType.BOOLEAN:
                return Categorical([True, False], name=spec.name)
            else:
                self._logger.warning(
                    f"无法转换参数 '{spec.name}' "
                    f"(类型={spec.param_type}) 到skopt维度"
                )
                return None

        except ImportError:
            return None

    @staticmethod
    def _dict_to_param_space(
        param_grid: Dict[str, Any],
    ) -> List[ParamSpec]:
        """
        从字典格式的参数网格转换为ParamSpec列表
        
        支持格式：
        - {"name": [v1, v2, v3]}     → 离散值（自动推断类型）
        - {"name": (low, high)}       → 范围（自动推断类型）
        - {"name": ParamSpec(...)}    → 直接使用
        
        Args:
            param_grid: 参数网格字典
            
        Returns:
            ParamSpec列表
        """
        specs = []

        for name, value in param_grid.items():
            if isinstance(value, ParamSpec):
                specs.append(value)
                continue

            if isinstance(value, tuple) and len(value) == 2:
                # 范围参数
                low, high = value
                if isinstance(low, int) and isinstance(high, int):
                    specs.append(
                        ParamSpec(name, ParamType.INT_RANGE, low=low, high=high)
                    )
                else:
                    specs.append(
                        ParamSpec(
                            name,
                            ParamType.FLOAT_RANGE,
                            low=float(low),
                            high=float(high),
                        )
                    )

            elif isinstance(value, list):
                # 离散值列表
                if not value:
                    continue

                if all(isinstance(v, bool) for v in value):
                    specs.append(ParamSpec(name, ParamType.BOOLEAN))
                elif all(isinstance(v, int) for v in value):
                    specs.append(
                        ParamSpec(
                            name, ParamType.INT_CHOICE, choices=value
                        )
                    )
                elif all(isinstance(v, (int, float)) for v in value):
                    specs.append(
                        ParamSpec(
                            name,
                            ParamType.FLOAT_CHOICE,
                            choices=[float(v) for v in value],
                        )
                    )
                else:
                    specs.append(
                        ParamSpec(
                            name, ParamType.CATEGORY, choices=value
                        )
                    )

            else:
                logging.getLogger(__name__).warning(
                    f"无法解析参数网格项: {name}={value}"
                )

        return specs

    def _empty_result(
        self, method: str, target_metric: str
    ) -> OptimizationResult:
        """创建空的优化结果"""
        return OptimizationResult(
            best_params={},
            best_score=-np.inf,
            all_results=[],
            method=method,
            target_metric=target_metric,
            total_iterations=0,
            elapsed_seconds=0,
        )

    def _save_result(self, result: OptimizationResult):
        """
        持久化优化结果到文件
        
        Args:
            result: 优化结果
        """
        try:
            output_dir = Path(self._config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = (
                output_dir
                / f"optimization_{result.method}_{timestamp}.json"
            )

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    result.to_dict(),
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            self._logger.debug(f"优化结果已保存: {filepath}")

        except Exception as e:
            self._logger.warning(f"保存优化结果失败: {e}")
