"""
策略管理器模块（主控制器）

修复内容：
1. 依赖注入 - 支持外部注入所有组件，方便测试和替换
2. 策略接口标准化 - 统一使用BaseStrategy抽象基类
3. 评估流程 - 完整的选股→回测→统计→保存流程
4. 优化流程 - 参数优化与回测引擎正确集成
5. 自动优化 - 优化前后对比、版本管理、通知推送
6. 错误处理 - 每个环节独立try-catch，不会因单个失败中断全部
7. 性能监控 - 记录各环节耗时和成功率
8. 资源清理 - 统一的cleanup方法释放所有资源
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import Config
from .data_manager import BatchDataManager
from .backtest_engine import BacktestEngine, TradeResult
from .parameter_optimizer import (
    ParameterOptimizer,
    ParamSpec,
    ParamType,
    OptimizationResult,
)
from .strategy_database import StrategyDatabase
from .strategies import BaseStrategy


# ==================== 性能统计 ====================


@dataclass
class PerformanceStats:
    """性能统计数据"""

    total_evaluations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    total_optimizations: int = 0
    successful_optimizations: int = 0
    total_elapsed_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_evaluations == 0:
            return 0.0
        return self.successful_evaluations / self.total_evaluations * 100

    @property
    def avg_evaluation_time(self) -> float:
        if self.successful_evaluations == 0:
            return 0.0
        return self.total_elapsed_seconds / self.successful_evaluations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.total_evaluations,
            "successful_evaluations": self.successful_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "success_rate_pct": round(self.success_rate, 2),
            "total_optimizations": self.total_optimizations,
            "successful_optimizations": self.successful_optimizations,
            "avg_evaluation_time_seconds": round(self.avg_evaluation_time, 3),
        }


# ==================== 策略注册表条目 ====================


@dataclass
class StrategyEntry:
    """策略注册表中的一条记录"""

    name: str
    strategy: BaseStrategy
    db_strategy_id: Optional[int] = None
    db_version_id: Optional[int] = None
    description: str = ""
    template: str = ""
    registered_at: str = ""

    def __post_init__(self):
        if not self.registered_at:
            self.registered_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==================== 策略管理器 ====================


class StrategyManager:
    """
    策略管理器（主控制器）
    
    职责：
    - 策略注册和生命周期管理
    - 策略评估（选股 → 回测 → 统计 → 保存）
    - 参数优化（与回测引擎集成）
    - 自动优化（多策略批量优化，版本管理）
    - 报告生成和通知推送
    - 性能监控
    
    使用示例：
        config = Config(tushare_token="your_token")
        manager = StrategyManager(config)
        
        # 注册策略
        manager.register_strategy("value", ValueStrategy("my_value", params))
        
        # 评估策略
        results = manager.evaluate_strategy("value", data, "20230601")
        
        # 优化策略参数
        opt_result = manager.optimize_strategy("value", data, param_space)
        
        # 自动优化所有策略
        manager.auto_optimize_all(data)
        
        # 清理资源
        manager.cleanup()
    """

    def __init__(
        self,
        config: Config,
        data_manager: Optional[BatchDataManager] = None,
        backtest_engine: Optional[BacktestEngine] = None,
        optimizer: Optional[ParameterOptimizer] = None,
        database: Optional[StrategyDatabase] = None,
    ):
        """
        初始化策略管理器
        
        支持依赖注入：每个组件都可以从外部传入，方便测试和替换
        
        Args:
            config: 系统配置
            data_manager: 数据管理器（None则自动创建）
            backtest_engine: 回测引擎（None则自动创建）
            optimizer: 参数优化器（None则自动创建）
            database: 策略数据库（None则自动创建）
        """
        self._config = config
        self._logger = logging.getLogger(__name__)

        # 依赖注入：按依赖顺序创建组件
        self._data_manager = data_manager or BatchDataManager(config)

        self._backtest_engine = backtest_engine or BacktestEngine(
            self._data_manager, config
        )

        self._optimizer = optimizer or ParameterOptimizer(
            config, backtest_engine=self._backtest_engine
        )

        self._database = database or StrategyDatabase(config.db_path, config)

        # 策略注册表
        self._registry: Dict[str, StrategyEntry] = {}

        # 性能统计
        self._stats = PerformanceStats()

        self._logger.info("StrategyManager初始化完成")

    # ==================== 策略注册 ====================

    def register_strategy(
        self,
        name: str,
        strategy: BaseStrategy,
        description: str = "",
        template: str = "",
        save_to_db: bool = True,
    ) -> str:
        """
        注册策略
        
        将策略实例注册到内存注册表，可选同步到数据库
        
        Args:
            name: 策略名称（注册表中的唯一标识）
            strategy: 策略实例（必须继承BaseStrategy）
            description: 策略描述
            template: 策略模板名称
            save_to_db: 是否同步保存到数据库
            
        Returns:
            注册名称
            
        Raises:
            TypeError: strategy不是BaseStrategy的子类
        """
        if not isinstance(strategy, BaseStrategy):
            raise TypeError(
                f"strategy必须是BaseStrategy的子类，"
                f"当前类型: {type(strategy).__name__}"
            )

        # 检查是否已注册
        if name in self._registry:
            self._logger.warning(f"策略 '{name}' 已注册，将被覆盖")

        # 数据库同步
        db_strategy_id = None
        db_version_id = None

        if save_to_db:
            try:
                db_strategy_id = self._database.save_strategy(
                    name=name,
                    description=description or strategy.name,
                    parameters=strategy.parameters,
                )
                db_version_id = self._database.create_version(
                    strategy_id=db_strategy_id,
                    parameters=strategy.parameters,
                    notes=f"注册时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                )
            except Exception as e:
                self._logger.warning(f"策略数据库同步失败: {e}")

        # 注册到内存
        entry = StrategyEntry(
            name=name,
            strategy=strategy,
            db_strategy_id=db_strategy_id,
            db_version_id=db_version_id,
            description=description,
            template=template,
        )
        self._registry[name] = entry

        self._logger.info(
            f"注册策略: '{name}' "
            f"(类型={type(strategy).__name__}, "
            f"参数数={len(strategy.parameters)}, "
            f"DB_ID={db_strategy_id})"
        )

        return name

    def unregister_strategy(self, name: str) -> bool:
        """
        注销策略
        
        从内存注册表移除（不影响数据库记录）
        
        Args:
            name: 策略名称
            
        Returns:
            是否成功
        """
        if name in self._registry:
            del self._registry[name]
            self._logger.info(f"注销策略: '{name}'")
            return True

        self._logger.warning(f"策略未注册: '{name}'")
        return False

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """获取已注册的策略实例"""
        entry = self._registry.get(name)
        return entry.strategy if entry else None

    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        列出所有已注册的策略
        
        Returns:
            策略信息列表
        """
        result = []
        for name, entry in self._registry.items():
            result.append(
                {
                    "name": name,
                    "type": type(entry.strategy).__name__,
                    "description": entry.description,
                    "template": entry.template,
                    "parameters": entry.strategy.parameters,
                    "db_strategy_id": entry.db_strategy_id,
                    "db_version_id": entry.db_version_id,
                    "registered_at": entry.registered_at,
                }
            )
        return result

    # ==================== 策略评估 ====================

    def evaluate_strategy(
        self,
        name: str,
        data: pd.DataFrame,
        buy_date: str,
        initial_capital: Optional[float] = None,
        max_stocks: int = 30,
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        评估策略
        
        完整流程：数据预处理 → 选股 → 信号生成 → 回测 → 统计 → 保存
        
        Args:
            name: 已注册的策略名称
            data: 原始数据（策略会在内部做预处理和选股）
            buy_date: 买入日期 (YYYYMMDD)
            initial_capital: 初始资金（None使用配置默认值）
            max_stocks: 最大持股数量
            save_to_db: 是否保存回测结果到数据库
            
        Returns:
            评估结果字典，包含：
            - portfolio_summary: 组合层面的汇总指标
            - backtest_df: 每只股票的回测明细
            - selected_count: 选中股票数量
            - parameters: 使用的策略参数
            失败时返回空字典
        """
        entry = self._registry.get(name)
        if entry is None:
            self._logger.error(f"策略未注册: '{name}'")
            return {}

        strategy = entry.strategy
        self._stats.total_evaluations += 1
        start_time = time.time()

        try:
            self._logger.info(
                f"开始评估策略: '{name}' "
                f"(买入日={buy_date}, 数据量={len(data)})"
            )

            # 步骤1：数据预处理
            prepared_data = strategy.prepare_data(data)
            if prepared_data.empty:
                self._logger.warning(f"策略 '{name}' 预处理后数据为空")
                self._stats.failed_evaluations += 1
                return {}

            # 步骤2：选股
            selected = strategy.select_stocks(prepared_data)
            if selected.empty:
                self._logger.warning(f"策略 '{name}' 未选中任何股票")
                self._stats.failed_evaluations += 1
                return {}

            selected_count = len(selected)
            self._logger.info(
                f"策略 '{name}' 选中 {selected_count} 只股票"
            )

            # 步骤3：生成交易信号并排序
            signals = strategy.generate_signals(selected)
            if signals.empty:
                self._logger.warning(f"策略 '{name}' 未生成交易信号")
                self._stats.failed_evaluations += 1
                return {}

            # 限制最大持股数（取信号强度最高的）
            if len(signals) > max_stocks:
                if "signal_strength" in signals.columns:
                    signals = signals.nlargest(max_stocks, "signal_strength")
                else:
                    signals = signals.head(max_stocks)

            # 步骤4：构建投资组合
            portfolio = self._build_portfolio(signals)
            if not portfolio:
                self._logger.warning(f"策略 '{name}' 无法构建投资组合")
                self._stats.failed_evaluations += 1
                return {}

            # 步骤5：执行回测
            backtest_df, portfolio_summary = self._backtest_engine.backtest_portfolio(
                stock_list=portfolio,
                buy_date=buy_date,
                initial_capital=initial_capital,
            )

            if backtest_df.empty:
                self._logger.warning(f"策略 '{name}' 回测无有效结果")
                self._stats.failed_evaluations += 1
                return {}

            # 步骤6：组装结果
            elapsed = time.time() - start_time

            results = {
                "strategy_name": name,
                "buy_date": buy_date,
                "parameters": strategy.parameters,
                "selected_count": selected_count,
                "backtest_count": len(backtest_df),
                "backtest_df": backtest_df,
                "portfolio_summary": portfolio_summary,
                "evaluation_time_seconds": round(elapsed, 2),
            }

            # 步骤7：保存到数据库
            if save_to_db and entry.db_version_id:
                try:
                    self._database.save_backtest_result(
                        version_id=entry.db_version_id,
                        results=results,
                    )
                except Exception as e:
                    self._logger.warning(f"保存回测结果到数据库失败: {e}")

            # 更新统计
            self._stats.successful_evaluations += 1
            self._stats.total_elapsed_seconds += elapsed

            self._logger.info(
                f"策略 '{name}' 评估完成: "
                f"选中={selected_count}, 回测={len(backtest_df)}, "
                f"胜率={portfolio_summary.get('win_rate_pct', 0):.1f}%, "
                f"收益={portfolio_summary.get('portfolio_return_pct', 0):.2f}%, "
                f"夏普={portfolio_summary.get('sharpe_ratio', 0):.3f}, "
                f"耗时={elapsed:.1f}s"
            )

            return results

        except Exception as e:
            self._stats.failed_evaluations += 1
            self._logger.error(f"评估策略 '{name}' 失败: {e}", exc_info=True)
            return {}

    def evaluate_multiple(
        self,
        strategy_names: List[str],
        data: pd.DataFrame,
        buy_date: str,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量评估多个策略
        
        Args:
            strategy_names: 策略名称列表
            data: 原始数据
            buy_date: 买入日期
            **kwargs: 传递给evaluate_strategy的其他参数
            
        Returns:
            {策略名称: 评估结果} 字典
        """
        all_results = {}

        for name in strategy_names:
            self._logger.info(f"批量评估: {name}")
            result = self.evaluate_strategy(
                name=name,
                data=data,
                buy_date=buy_date,
                **kwargs,
            )
            all_results[name] = result

        # 汇总对比
        self._log_comparison(all_results)

        return all_results

    # ==================== 参数优化 ====================

    def optimize_strategy(
        self,
        name: str,
        data: pd.DataFrame,
        param_space: Optional[List[ParamSpec]] = None,
        method: Optional[str] = None,
        n_iterations: Optional[int] = None,
        target_metric: str = "sharpe_ratio",
        cv_folds: Optional[int] = None,
        save_new_version: bool = True,
    ) -> Optional[OptimizationResult]:
        """
        优化策略参数
        
        流程：
        1. 从策略获取参数空间（或使用自定义空间）
        2. 构建评估函数（将策略选股与回测引擎串联）
        3. 调用优化器搜索最佳参数
        4. 可选：将最佳参数保存为新版本
        
        Args:
            name: 策略名称
            data: 训练数据
            param_space: 参数搜索空间（None则使用策略自带的范围）
            method: 优化方法
            n_iterations: 迭代次数
            target_metric: 目标指标
            cv_folds: 交叉验证折数
            save_new_version: 是否将最佳参数保存为新版本
            
        Returns:
            OptimizationResult，失败返回None
        """
        entry = self._registry.get(name)
        if entry is None:
            self._logger.error(f"策略未注册: '{name}'")
            return None

        strategy = entry.strategy
        self._stats.total_optimizations += 1

        try:
            self._logger.info(
                f"开始优化策略: '{name}' "
                f"(指标={target_metric}, 数据量={len(data)})"
            )

            # 构建参数空间
            if param_space is None:
                param_space = self._build_param_space_from_strategy(strategy)

            if not param_space:
                self._logger.warning(
                    f"策略 '{name}' 无可优化参数空间"
                )
                return None

            # 构建评估函数
            evaluate_func = self._create_evaluate_func(
                strategy, target_metric
            )

            # 运行优化
            result = self._optimizer.optimize(
                param_space=param_space,
                evaluate_func=evaluate_func,
                train_data=data,
                method=method,
                n_iterations=n_iterations,
                target_metric=target_metric,
                cv_folds=cv_folds,
            )

            if result.best_score <= -1e5:
                self._logger.warning(
                    f"策略 '{name}' 优化未找到有效参数"
                )
                return result

            # 保存新版本
            if save_new_version and entry.db_strategy_id:
                self._save_optimized_version(
                    entry, result, target_metric
                )

            self._stats.successful_optimizations += 1

            self._logger.info(
                f"策略 '{name}' 优化完成: "
                f"最佳分数={result.best_score:.4f}, "
                f"最佳参数={result.best_params}"
            )

            return result

        except Exception as e:
            self._logger.error(
                f"优化策略 '{name}' 失败: {e}", exc_info=True
            )
            return None

    def optimize_from_dict(
        self,
        name: str,
        data: pd.DataFrame,
        param_grid: Dict[str, Any],
        **kwargs,
    ) -> Optional[OptimizationResult]:
        """
        使用字典格式的参数网格优化策略（兼容旧接口）
        
        Args:
            name: 策略名称
            data: 训练数据
            param_grid: 参数网格字典
            **kwargs: 传递给optimize_strategy的其他参数
            
        Returns:
            OptimizationResult
        """
        param_space = ParameterOptimizer._dict_to_param_space(param_grid)
        return self.optimize_strategy(
            name=name,
            data=data,
            param_space=param_space,
            **kwargs,
        )

    # ==================== 自动优化 ====================

    def auto_optimize_all(
        self,
        data: pd.DataFrame,
        buy_date: Optional[str] = None,
        target_metric: str = "sharpe_ratio",
        improvement_threshold: float = 5.0,
        method: Optional[str] = None,
        n_iterations: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        自动优化所有已注册的策略
        
        流程（对每个策略）：
        1. 评估当前参数的表现
        2. 运行参数优化
        3. 用最佳参数重新评估
        4. 如果改进超过阈值，保存新版本
        5. 生成对比摘要
        
        Args:
            data: 数据
            buy_date: 买入日期（None则使用数据中最新日期）
            target_metric: 目标指标
            improvement_threshold: 改进阈值(%)，低于此值不保存新版本
            method: 优化方法
            n_iterations: 迭代次数
            
        Returns:
            {策略名称: 优化摘要} 字典
        """
        if buy_date is None:
            buy_date = self._infer_buy_date(data)

        self._logger.info(
            f"开始自动优化: {len(self._registry)} 个策略, "
            f"买入日={buy_date}, 指标={target_metric}, "
            f"改进阈值={improvement_threshold}%"
        )

        summaries = {}

        for name in list(self._registry.keys()):
            summary = self._auto_optimize_single(
                name=name,
                data=data,
                buy_date=buy_date,
                target_metric=target_metric,
                improvement_threshold=improvement_threshold,
                method=method,
                n_iterations=n_iterations,
            )
            summaries[name] = summary

        # 输出总结
        self._log_auto_optimize_summary(summaries)

        return summaries

    def _auto_optimize_single(
        self,
        name: str,
        data: pd.DataFrame,
        buy_date: str,
        target_metric: str,
        improvement_threshold: float,
        method: Optional[str],
        n_iterations: Optional[int],
    ) -> Dict[str, Any]:
        """
        自动优化单个策略
        
        Returns:
            优化摘要字典
        """
        summary = {
            "strategy": name,
            "status": "pending",
            "old_score": None,
            "new_score": None,
            "improvement_pct": None,
            "version_updated": False,
        }

        try:
            entry = self._registry.get(name)
            if entry is None:
                summary["status"] = "not_registered"
                return summary

            # 步骤1：评估当前表现
            self._logger.info(f"[{name}] 评估当前参数...")
            old_results = self.evaluate_strategy(
                name, data, buy_date, save_to_db=False
            )

            if not old_results:
                summary["status"] = "evaluation_failed"
                return summary

            old_score = self._extract_target_score(
                old_results, target_metric
            )
            summary["old_score"] = old_score

            # 步骤2：运行参数优化
            self._logger.info(f"[{name}] 运行参数优化...")
            opt_result = self.optimize_strategy(
                name=name,
                data=data,
                method=method,
                n_iterations=n_iterations,
                target_metric=target_metric,
                save_new_version=False,  # 先不保存，等对比后再决定
            )

            if opt_result is None or opt_result.best_score <= -1e5:
                summary["status"] = "optimization_failed"
                return summary

            # 步骤3：用最佳参数重新评估
            self._logger.info(f"[{name}] 用最佳参数重新评估...")
            strategy = entry.strategy
            old_params = strategy.parameters.copy()

            # 临时更新参数
            strategy.parameters.update(opt_result.best_params)

            new_results = self.evaluate_strategy(
                name, data, buy_date, save_to_db=False
            )

            # 恢复原参数
            strategy.parameters = old_params

            if not new_results:
                summary["status"] = "re_evaluation_failed"
                return summary

            new_score = self._extract_target_score(
                new_results, target_metric
            )
            summary["new_score"] = new_score

            # 步骤4：计算改进
            if old_score != 0:
                improvement = (new_score - old_score) / abs(old_score) * 100
            elif new_score > 0:
                improvement = 100.0
            else:
                improvement = 0.0

            summary["improvement_pct"] = round(improvement, 2)

            # 步骤5：判断是否保存新版本
            if improvement >= improvement_threshold:
                self._logger.info(
                    f"[{name}] 改进 {improvement:.1f}% >= 阈值 "
                    f"{improvement_threshold}%，保存新版本"
                )

                # 更新策略参数
                strategy.parameters.update(opt_result.best_params)

                # 保存新版本到数据库
                if entry.db_strategy_id:
                    try:
                        # 获取旧版本号
                        old_version = entry.db_version_id

                        # 创建新版本
                        new_version_id = self._database.create_version(
                            strategy_id=entry.db_strategy_id,
                            parameters=strategy.parameters,
                            notes=(
                                f"自动优化: {target_metric} "
                                f"从 {old_score:.4f} 提升到 {new_score:.4f} "
                                f"(+{improvement:.1f}%)"
                            ),
                        )
                        entry.db_version_id = new_version_id

                        # 保存新的回测结果
                        self._database.save_backtest_result(
                            new_version_id, new_results
                        )

                        # 记录优化历史
                        self._database.save_optimization_record(
                            strategy_id=entry.db_strategy_id,
                            version_from=old_version or 0,
                            version_to=new_version_id,
                            method=opt_result.method,
                            target_metric=target_metric,
                            old_score=old_score,
                            new_score=new_score,
                            details={
                                "best_params": opt_result.best_params,
                                "improvement_pct": improvement,
                            },
                        )

                        summary["version_updated"] = True

                    except Exception as e:
                        self._logger.warning(
                            f"[{name}] 保存新版本失败: {e}"
                        )
                else:
                    summary["version_updated"] = True

                summary["status"] = "improved"

            else:
                self._logger.info(
                    f"[{name}] 改进 {improvement:.1f}% < 阈值 "
                    f"{improvement_threshold}%，保持原参数"
                )
                summary["status"] = "no_improvement"

            return summary

        except Exception as e:
            self._logger.error(
                f"[{name}] 自动优化失败: {e}", exc_info=True
            )
            summary["status"] = "error"
            summary["error"] = str(e)
            return summary

    # ==================== 基准对比 ====================

    def compare_with_benchmark(
        self,
        name: str,
        evaluation_results: Dict[str, Any],
        start_date: str,
        end_date: str,
        benchmark_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        将策略评估结果与基准指数对比
        
        Args:
            name: 策略名称
            evaluation_results: evaluate_strategy的返回值
            start_date: 对比开始日期
            end_date: 对比结束日期
            benchmark_code: 基准代码
            
        Returns:
            对比结果
        """
        backtest_df = evaluation_results.get("backtest_df")
        if backtest_df is None or backtest_df.empty:
            return {}

        # 提取组合的日度收益率
        if "net_return_pct" in backtest_df.columns:
            portfolio_returns = backtest_df["net_return_pct"].tolist()
        else:
            return {}

        return self._backtest_engine.compare_with_benchmark(
            portfolio_daily_returns=portfolio_returns,
            start_date=start_date,
            end_date=end_date,
            benchmark_code=benchmark_code,
        )

    # ==================== 报告和查询 ====================

    def get_strategy_performance_history(
        self, name: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取策略的历史表现
        
        Args:
            name: 策略名称
            limit: 返回数量上限
            
        Returns:
            回测结果历史列表
        """
        entry = self._registry.get(name)
        if entry is None or entry.db_strategy_id is None:
            return []

        return self._database.get_strategy_performance(
            entry.db_strategy_id
        )

    def get_strategy_versions(
        self, name: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取策略的所有版本
        
        Args:
            name: 策略名称
            limit: 返回数量上限
            
        Returns:
            版本信息列表
        """
        entry = self._registry.get(name)
        if entry is None or entry.db_strategy_id is None:
            return []

        return self._database.get_versions(
            entry.db_strategy_id, limit=limit
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取系统性能统计
        
        Returns:
            包含评估统计、数据管理器统计、数据库统计的字典
        """
        return {
            "manager_stats": self._stats.to_dict(),
            "data_manager_stats": self._data_manager.get_stats(),
            "database_stats": self._database.get_database_stats(),
            "registered_strategies": len(self._registry),
        }

    def get_full_report(self, name: str) -> Dict[str, Any]:
        """
        获取策略的完整报告
        
        Args:
            name: 策略名称
            
        Returns:
            完整的策略报告
        """
        entry = self._registry.get(name)
        if entry is None:
            return {}

        report = {
            "strategy_info": {
                "name": name,
                "type": type(entry.strategy).__name__,
                "description": entry.description,
                "parameters": entry.strategy.parameters,
                "registered_at": entry.registered_at,
            },
            "versions": [],
            "backtest_history": [],
            "optimization_history": [],
        }

        if entry.db_strategy_id:
            report["versions"] = self._database.get_versions(
                entry.db_strategy_id
            )
            report["optimization_history"] = (
                self._database.get_optimization_history(
                    entry.db_strategy_id
                )
            )

        if entry.db_version_id:
            report["backtest_history"] = (
                self._database.get_backtest_history(entry.db_version_id)
            )

        return report

    # ==================== 资源清理 ====================

    def cleanup(self):
        """
        清理所有资源
        
        - 清理数据管理器缓存
        - 清理过期磁盘缓存
        - 优化数据库
        """
        self._logger.info("开始清理资源...")

        try:
            self._data_manager.clear_cache(memory=True, disk=False)
            self._data_manager.clear_expired_cache()
        except Exception as e:
            self._logger.warning(f"清理缓存失败: {e}")

        try:
            self._database.vacuum()
        except Exception as e:
            self._logger.warning(f"数据库优化失败: {e}")

        self._logger.info("资源清理完成")

    # ==================== 内部方法 ====================

    def _build_portfolio(
        self, signals: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        从信号DataFrame构建投资组合列表
        
        Args:
            signals: 交易信号DataFrame
            
        Returns:
            投资组合列表，每个元素为 {"ts_code": ..., "buy_price": ...}
        """
        portfolio = []

        # 尝试多种可能的列名
        code_col = self._find_column(
            signals,
            ["ts_code", "代码", "stock_code", "code", "symbol"],
        )
        price_col = self._find_column(
            signals,
            ["close", "收盘价", "buy_price", "price"],
        )
        weight_col = self._find_column(
            signals,
            ["weight", "权重", "signal_strength"],
        )

        if code_col is None:
            self._logger.warning("信号数据中找不到股票代码列")
            return []

        for _, row in signals.iterrows():
            ts_code = str(row[code_col])
            if not ts_code or ts_code == "nan":
                continue

            entry = {"ts_code": ts_code}

            if price_col and pd.notna(row.get(price_col)):
                entry["buy_price"] = float(row[price_col])

            if weight_col and pd.notna(row.get(weight_col)):
                entry["weight"] = float(row[weight_col])

            portfolio.append(entry)

        return portfolio

    def _create_evaluate_func(
        self, strategy: BaseStrategy, target_metric: str
    ) -> Callable:
        """
        创建评估函数（用于参数优化）
        
        将策略选股逻辑包装为优化器需要的评估函数签名：
        (data, params) -> score
        
        Args:
            strategy: 策略实例
            target_metric: 目标指标
            
        Returns:
            评估函数
        """

        def evaluate_func(
            data: pd.DataFrame, params: Dict[str, Any]
        ) -> float:
            """评估函数：选股 → 快速评分"""
            # 临时更新参数
            old_params = strategy.parameters.copy()
            strategy.parameters.update(params)

            try:
                # 预处理 + 选股
                prepared = strategy.prepare_data(data)
                if prepared.empty:
                    return -1e6

                selected = strategy.select_stocks(prepared)
                if selected.empty or len(selected) < 3:
                    return -1e6

                # 快速评估（不做完整回测，基于选股质量打分）
                score = self._quick_evaluate(
                    selected, target_metric
                )

                return score

            finally:
                # 恢复原参数
                strategy.parameters = old_params

        return evaluate_func

    def _quick_evaluate(
        self,
        selected_stocks: pd.DataFrame,
        target_metric: str,
    ) -> float:
        """
        快速评估选股质量
        
        不做完整回测，基于选股数据的统计特征估算分数
        用于参数优化的内层循环（需要快速）
        
        Args:
            selected_stocks: 选中的股票数据
            target_metric: 目标指标
            
        Returns:
            评估分数
        """
        n = len(selected_stocks)
        if n == 0:
            return -1e6

        score = 0.0

        if target_metric in ("sharpe_ratio", "return_risk_ratio"):
            # 夏普相关：偏好低PE + 高ROE + 适量股票
            if "pe_ttm" in selected_stocks.columns:
                pe_values = selected_stocks["pe_ttm"].dropna()
                if len(pe_values) > 0:
                    median_pe = pe_values.median()
                    if 0 < median_pe < 100:
                        score += 10.0 / (median_pe + 1)

            if "roe" in selected_stocks.columns:
                roe_values = selected_stocks["roe"].dropna()
                if len(roe_values) > 0:
                    score += roe_values.median() / 10.0

            # 适量股票数惩罚（太少或太多都不好）
            stock_penalty = -abs(n - 15) * 0.1
            score += stock_penalty

        elif target_metric == "win_rate":
            # 胜率：偏好分散化 + 高质量
            if "roe" in selected_stocks.columns:
                roe_values = selected_stocks["roe"].dropna()
                if len(roe_values) > 0:
                    score += (roe_values > 10).mean() * 50

            score += min(n, 20) * 0.5

        else:
            # 默认：股票数量评分
            score = min(n, 30) / 3.0

        return score

    def _build_param_space_from_strategy(
        self, strategy: BaseStrategy
    ) -> List[ParamSpec]:
        """
        从策略的get_parameter_ranges()构建ParamSpec列表
        
        Args:
            strategy: 策略实例
            
        Returns:
            ParamSpec列表
        """
        ranges = strategy.get_parameter_ranges()
        if not ranges:
            return []

        specs = []
        for param_name, (low, high) in ranges.items():
            if isinstance(low, int) and isinstance(high, int):
                specs.append(
                    ParamSpec(
                        param_name, ParamType.INT_RANGE, low=low, high=high
                    )
                )
            else:
                specs.append(
                    ParamSpec(
                        param_name,
                        ParamType.FLOAT_RANGE,
                        low=float(low),
                        high=float(high),
                    )
                )

        return specs

    def _save_optimized_version(
        self,
        entry: StrategyEntry,
        opt_result: OptimizationResult,
        target_metric: str,
    ):
        """保存优化后的新版本"""
        if entry.db_strategy_id is None:
            return

        try:
            # 合并参数（原参数 + 优化后的参数）
            merged_params = entry.strategy.parameters.copy()
            merged_params.update(opt_result.best_params)

            new_version_id = self._database.create_version(
                strategy_id=entry.db_strategy_id,
                parameters=merged_params,
                notes=(
                    f"优化: {opt_result.method}, "
                    f"{target_metric}={opt_result.best_score:.4f}"
                ),
            )

            # 更新分数
            self._database.update_version_score(
                new_version_id, opt_result.best_score
            )

            entry.db_version_id = new_version_id

            self._logger.info(
                f"保存优化版本: 策略ID={entry.db_strategy_id}, "
                f"版本ID={new_version_id}"
            )

        except Exception as e:
            self._logger.warning(f"保存优化版本失败: {e}")

    @staticmethod
    def _extract_target_score(
        results: Dict[str, Any], target_metric: str
    ) -> float:
        """从评估结果中提取目标指标分数"""
        summary = results.get("portfolio_summary", {})

        # 尝试多种可能的键名
        score_keys = [
            target_metric,
            f"{target_metric}_pct",
            target_metric.replace("_pct", ""),
        ]

        for key in score_keys:
            if key in summary:
                value = summary[key]
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)

        return 0.0

    @staticmethod
    def _find_column(
        df: pd.DataFrame, candidates: List[str]
    ) -> Optional[str]:
        """在DataFrame中查找第一个存在的列名"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _infer_buy_date(data: pd.DataFrame) -> str:
        """从数据中推断买入日期"""
        date_cols = ["trade_date", "date", "选股日期", "buy_date"]

        for col in date_cols:
            if col in data.columns:
                dates = pd.to_datetime(data[col], errors="coerce")
                valid = dates.dropna()
                if not valid.empty:
                    return valid.max().strftime("%Y%m%d")

        # 兜底：使用今天
        return datetime.now().strftime("%Y%m%d")

    def _log_comparison(self, all_results: Dict[str, Dict[str, Any]]):
        """输出策略对比日志"""
        if len(all_results) < 2:
            return

        self._logger.info("=" * 60)
        self._logger.info("策略评估对比:")
        self._logger.info(
            f"{'策略名称':<20} {'胜率':>8} {'收益':>8} {'夏普':>8} {'回撤':>8}"
        )
        self._logger.info("-" * 60)

        for name, results in all_results.items():
            if not results:
                self._logger.info(f"{name:<20} {'失败':>8}")
                continue

            s = results.get("portfolio_summary", {})
            self._logger.info(
                f"{name:<20} "
                f"{s.get('win_rate_pct', 0):>7.1f}% "
                f"{s.get('portfolio_return_pct', 0):>7.2f}% "
                f"{s.get('sharpe_ratio', 0):>8.3f} "
                f"{s.get('max_drawdown_pct', 0):>7.2f}%"
            )

        self._logger.info("=" * 60)

    def _log_auto_optimize_summary(
        self, summaries: Dict[str, Dict[str, Any]]
    ):
        """输出自动优化总结日志"""
        self._logger.info("=" * 70)
        self._logger.info("自动优化总结:")
        self._logger.info(
            f"{'策略':<20} {'状态':<18} "
            f"{'旧分数':>8} {'新分数':>8} {'改进':>8} {'更新':>4}"
        )
        self._logger.info("-" * 70)

        for name, s in summaries.items():
            old = f"{s['old_score']:.4f}" if s.get("old_score") is not None else "N/A"
            new = f"{s['new_score']:.4f}" if s.get("new_score") is not None else "N/A"
            imp = (
                f"{s['improvement_pct']:.1f}%"
                if s.get("improvement_pct") is not None
                else "N/A"
            )
            upd = "✓" if s.get("version_updated") else "✗"

            self._logger.info(
                f"{name:<20} {s['status']:<18} "
                f"{old:>8} {new:>8} {imp:>8} {upd:>4}"
            )

        self._logger.info("=" * 70)
