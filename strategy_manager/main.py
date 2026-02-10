#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化策略管理系统 v3.0 - 主程序入口

功能：
- 策略评估：对已注册策略执行选股和回测
- 参数优化：搜索最佳策略参数
- 自动优化：批量优化所有策略并对比
- 系统信息：查看已注册策略、性能统计、数据库状态

使用方式：
    # 评估价值策略
    python -m strategy_manager.main --action evaluate --strategy value --buy-date 20230601

    # 优化动量策略参数
    python -m strategy_manager.main --action optimize --strategy momentum --iterations 50

    # 自动优化所有策略
    python -m strategy_manager.main --action auto_optimize --buy-date 20230601

    # 查看系统信息
    python -m strategy_manager.main --action info

    # 使用自定义配置文件
    python -m strategy_manager.main --config my_config.yaml --action evaluate

    # 使用示例数据运行
    python -m strategy_manager.main --action evaluate --use-sample-data
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data_manager import BatchDataManager
from .backtest_engine import BacktestEngine
from .parameter_optimizer import ParameterOptimizer, ParamSpec, ParamType
from .strategy_database import StrategyDatabase
from .strategy_manager import StrategyManager
from .strategies import (
    BaseStrategy,
    ValueStrategy,
    MomentumStrategy,
    GrowthStrategy,
    MeanReversionStrategy,
    StrategyFactory,
)


# ==================== 示例数据生成 ====================


def generate_sample_data(
    n_stocks: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成模拟股票数据（用于测试和演示）
    
    生成包含基本面和行情数据的横截面数据，
    模拟真实的A股选股场景
    
    Args:
        n_stocks: 股票数量
        seed: 随机种子
        
    Returns:
        模拟数据DataFrame
    """
    np.random.seed(seed)

    # 股票代码
    exchanges = ["SZ"] * (n_stocks // 2) + ["SH"] * (n_stocks - n_stocks // 2)
    ts_codes = [
        f"{i:06d}.{ex}" for i, ex in zip(range(1, n_stocks + 1), exchanges)
    ]

    # 股票名称（部分包含ST标记用于测试过滤）
    names = [f"模拟股票{i}" for i in range(1, n_stocks + 1)]
    for idx in np.random.choice(n_stocks, size=max(1, n_stocks // 20), replace=False):
        names[idx] = f"*ST模拟{idx + 1}"

    # 行业
    industries = [
        "电子", "计算机", "医药生物", "食品饮料", "银行",
        "非银金融", "房地产", "机械设备", "化工", "电气设备",
        "汽车", "传媒", "通信", "农林牧渔", "有色金属",
    ]

    # 生成各项指标
    data = {
        "ts_code": ts_codes,
        "name": names,
        "industry": np.random.choice(industries, n_stocks),
        # 行情数据
        "close": np.random.lognormal(mean=3.0, sigma=0.8, size=n_stocks).round(2),
        "open": np.random.lognormal(mean=3.0, sigma=0.8, size=n_stocks).round(2),
        "high": np.random.lognormal(mean=3.0, sigma=0.8, size=n_stocks).round(2),
        "low": np.random.lognormal(mean=3.0, sigma=0.8, size=n_stocks).round(2),
        "vol": np.random.lognormal(mean=12, sigma=1.5, size=n_stocks).round(0),
        "pct_chg": np.random.normal(0, 3, n_stocks).round(2),
        "turnover_rate": np.random.lognormal(mean=0.5, sigma=0.8, size=n_stocks).round(2),
        "volume_ratio": np.random.lognormal(mean=0.2, sigma=0.5, size=n_stocks).round(2),
        # 基本面数据
        "pe_ttm": np.random.lognormal(mean=2.8, sigma=0.7, size=n_stocks).round(2),
        "pb": np.random.lognormal(mean=0.5, sigma=0.6, size=n_stocks).round(2),
        "roe": np.random.normal(12, 8, n_stocks).round(2),
        "total_mv": np.random.lognormal(mean=4.5, sigma=1.2, size=n_stocks).round(2),
        "dv_ratio": np.abs(np.random.normal(1.5, 1.5, n_stocks)).round(2),
        # 成长指标
        "revenue_yoy": np.random.normal(15, 25, n_stocks).round(2),
        "profit_yoy": np.random.normal(10, 35, n_stocks).round(2),
    }

    df = pd.DataFrame(data)

    # 确保价格关系合理
    for i in range(len(df)):
        prices = sorted([df.loc[i, "open"], df.loc[i, "close"]])
        df.loc[i, "low"] = min(prices[0], df.loc[i, "low"])
        df.loc[i, "high"] = max(prices[1], df.loc[i, "high"])

    # 添加一些超跌股票（供均值回归策略使用）
    oversold_indices = np.random.choice(n_stocks, size=n_stocks // 10, replace=False)
    df.loc[oversold_indices, "pct_chg"] = np.random.uniform(-15, -5, len(oversold_indices)).round(2)

    # 添加一些高动量股票（供动量策略使用）
    momentum_indices = np.random.choice(n_stocks, size=n_stocks // 10, replace=False)
    df.loc[momentum_indices, "pct_chg"] = np.random.uniform(5, 15, len(momentum_indices)).round(2)
    df.loc[momentum_indices, "volume_ratio"] = np.random.uniform(2, 5, len(momentum_indices)).round(2)

    logging.getLogger(__name__).info(
        f"生成模拟数据: {len(df)} 只股票, "
        f"PE范围=[{df['pe_ttm'].min():.1f}, {df['pe_ttm'].max():.1f}], "
        f"ROE范围=[{df['roe'].min():.1f}, {df['roe'].max():.1f}]"
    )

    return df


def load_data_from_files(data_dir: str) -> pd.DataFrame:
    """
    从本地文件加载历史选股数据
    
    支持CSV和Parquet格式
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        合并后的DataFrame
    """
    import glob
    import re

    logger = logging.getLogger(__name__)

    all_data = []
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"数据目录不存在: {data_dir}")
        return pd.DataFrame()

    # 加载CSV文件
    for csv_file in sorted(data_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, encoding="utf_8_sig")

            if df.empty:
                continue

            # 从文件名提取日期
            date_match = re.search(r"(\d{8})", csv_file.name)
            if date_match and "选股日期" not in df.columns:
                df["选股日期"] = date_match.group(1)

            all_data.append(df)
            logger.debug(f"加载文件: {csv_file.name} ({len(df)} 行)")

        except Exception as e:
            logger.warning(f"读取文件失败 {csv_file.name}: {e}")

    # 加载Parquet文件
    for pq_file in sorted(data_path.glob("*.parquet")):
        try:
            df = pd.read_parquet(pq_file)
            if not df.empty:
                all_data.append(df)
                logger.debug(f"加载文件: {pq_file.name} ({len(df)} 行)")
        except Exception as e:
            logger.warning(f"读取文件失败 {pq_file.name}: {e}")

    if not all_data:
        logger.warning(f"目录 {data_dir} 中未找到有效数据文件")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(
        f"从 {len(all_data)} 个文件加载了 {len(combined)} 条记录"
    )

    return combined


# ==================== 各动作的执行函数 ====================


def action_evaluate(
    manager: StrategyManager,
    data: pd.DataFrame,
    args: argparse.Namespace,
):
    """执行策略评估"""
    logger = logging.getLogger(__name__)

    strategy_name = args.strategy
    buy_date = args.buy_date

    if strategy_name == "all":
        # 评估所有策略
        strategy_names = list(manager.list_strategies())
        strategy_names = [s["name"] for s in manager.list_strategies()]

        if not strategy_names:
            logger.error("没有已注册的策略")
            return

        logger.info(f"评估所有策略: {strategy_names}")
        all_results = manager.evaluate_multiple(
            strategy_names=strategy_names,
            data=data,
            buy_date=buy_date,
            max_stocks=args.max_stocks,
        )

        # 输出汇总
        print_evaluation_summary(all_results)

    else:
        # 评估单个策略
        if manager.get_strategy(strategy_name) is None:
            logger.error(
                f"策略 '{strategy_name}' 未注册。"
                f"可用策略: {[s['name'] for s in manager.list_strategies()]}"
            )
            return

        results = manager.evaluate_strategy(
            name=strategy_name,
            data=data,
            buy_date=buy_date,
            max_stocks=args.max_stocks,
        )

        if results:
            print_single_evaluation(strategy_name, results)
        else:
            logger.warning(f"策略 '{strategy_name}' 评估无结果")


def action_optimize(
    manager: StrategyManager,
    data: pd.DataFrame,
    args: argparse.Namespace,
):
    """执行参数优化"""
    logger = logging.getLogger(__name__)

    strategy_name = args.strategy

    if manager.get_strategy(strategy_name) is None:
        logger.error(f"策略 '{strategy_name}' 未注册")
        return

    logger.info(
        f"开始优化策略 '{strategy_name}': "
        f"方法={args.method}, 迭代={args.iterations}"
    )

    result = manager.optimize_strategy(
        name=strategy_name,
        data=data,
        method=args.method,
        n_iterations=args.iterations,
        target_metric=args.metric,
        save_new_version=True,
    )

    if result:
        print_optimization_result(strategy_name, result)
    else:
        logger.warning("参数优化失败")


def action_auto_optimize(
    manager: StrategyManager,
    data: pd.DataFrame,
    args: argparse.Namespace,
):
    """执行自动优化"""
    logger = logging.getLogger(__name__)

    logger.info("开始自动优化所有策略...")

    summaries = manager.auto_optimize_all(
        data=data,
        buy_date=args.buy_date,
        target_metric=args.metric,
        improvement_threshold=args.threshold,
        method=args.method,
        n_iterations=args.iterations,
    )

    print_auto_optimize_summary(summaries)


def action_info(manager: StrategyManager, args: argparse.Namespace):
    """显示系统信息"""
    print("\n" + "=" * 70)
    print("量化策略管理系统 v3.0 - 系统信息")
    print("=" * 70)

    # 已注册策略
    strategies = manager.list_strategies()
    print(f"\n📋 已注册策略 ({len(strategies)} 个):")
    print("-" * 60)

    if strategies:
        for s in strategies:
            print(
                f"  {s['name']:<20} "
                f"类型={s['type']:<20} "
                f"参数数={len(s.get('parameters', {}))}"
            )
    else:
        print("  (无)")

    # 可用策略类型
    available = StrategyFactory.list_available()
    print(f"\n🏭 可用策略类型 ({len(available)} 种):")
    print("-" * 60)

    for a in available:
        print(
            f"  {a['type']:<20} "
            f"{a['class']:<25} "
            f"必要参数={a['required_parameters']}"
        )

    # 性能统计
    stats = manager.get_performance_stats()
    print("\n📊 性能统计:")
    print("-" * 60)

    mgr_stats = stats.get("manager_stats", {})
    for key, value in mgr_stats.items():
        print(f"  {key}: {value}")

    # 数据管理器统计
    dm_stats = stats.get("data_manager_stats", {})
    if dm_stats:
        print("\n📦 数据管理器:")
        for key, value in dm_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    # 数据库统计
    db_stats = stats.get("database_stats", {})
    if db_stats:
        print("\n🗄️  数据库:")
        for key, value in db_stats.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)


def action_list_versions(
    manager: StrategyManager, args: argparse.Namespace
):
    """列出策略版本"""
    strategy_name = args.strategy

    versions = manager.get_strategy_versions(strategy_name)

    if not versions:
        print(f"策略 '{strategy_name}' 无版本记录")
        return

    print(f"\n策略 '{strategy_name}' 的版本历史:")
    print("-" * 70)
    print(
        f"{'版本':>6} {'创建时间':<22} {'当前':>4} {'生产':>4} {'分数':>8}"
    )
    print("-" * 70)

    for v in versions:
        is_current = "✓" if v.get("is_current") else ""
        is_prod = "✓" if v.get("is_production") else ""
        score = v.get("optimization_score")
        score_str = f"{score:.4f}" if score is not None else "N/A"

        print(
            f"{v.get('version', 0):>6} "
            f"{v.get('created_at', ''):<22} "
            f"{is_current:>4} "
            f"{is_prod:>4} "
            f"{score_str:>8}"
        )


def action_report(
    manager: StrategyManager, args: argparse.Namespace
):
    """生成策略报告"""
    strategy_name = args.strategy

    report = manager.get_full_report(strategy_name)

    if not report:
        print(f"策略 '{strategy_name}' 无报告数据")
        return

    # 输出到JSON文件
    output_dir = Path(manager._config.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{strategy_name}_{timestamp}.json"

    # 过滤不可序列化的内容
    serializable_report = _filter_serializable(report)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, ensure_ascii=False, indent=2, default=str)

    print(f"报告已保存到: {report_path}")

    # 控制台摘要
    info = report.get("strategy_info", {})
    print(f"\n策略报告: {info.get('name', strategy_name)}")
    print(f"  类型: {info.get('type', 'N/A')}")
    print(f"  参数: {json.dumps(info.get('parameters', {}), ensure_ascii=False)}")
    print(f"  版本数: {len(report.get('versions', []))}")
    print(f"  回测记录: {len(report.get('backtest_history', []))}")
    print(f"  优化记录: {len(report.get('optimization_history', []))}")


# ==================== 输出格式化 ====================


def print_evaluation_summary(all_results: Dict[str, Dict[str, Any]]):
    """打印多策略评估汇总"""
    print("\n" + "=" * 80)
    print("策略评估汇总")
    print("=" * 80)

    header = (
        f"{'策略名称':<20} {'选中':>4} {'回测':>4} "
        f"{'胜率':>7} {'收益':>8} {'夏普':>8} {'回撤':>8} {'耗时':>6}"
    )
    print(header)
    print("-" * 80)

    for name, results in all_results.items():
        if not results:
            print(f"{name:<20} {'失败':<60}")
            continue

        s = results.get("portfolio_summary", {})
        print(
            f"{name:<20} "
            f"{results.get('selected_count', 0):>4} "
            f"{results.get('backtest_count', 0):>4} "
            f"{s.get('win_rate_pct', 0):>6.1f}% "
            f"{s.get('portfolio_return_pct', 0):>7.2f}% "
            f"{s.get('sharpe_ratio', 0):>8.3f} "
            f"{s.get('max_drawdown_pct', 0):>7.2f}% "
            f"{results.get('evaluation_time_seconds', 0):>5.1f}s"
        )

    print("=" * 80)


def print_single_evaluation(name: str, results: Dict[str, Any]):
    """打印单策略评估结果"""
    s = results.get("portfolio_summary", {})

    print("\n" + "=" * 60)
    print(f"策略评估结果: {name}")
    print("=" * 60)

    print(f"\n📊 基础信息:")
    print(f"  选中股票: {results.get('selected_count', 0)} 只")
    print(f"  成功回测: {results.get('backtest_count', 0)} 只")
    print(f"  评估耗时: {results.get('evaluation_time_seconds', 0):.1f} 秒")

    print(f"\n📈 收益指标:")
    print(f"  组合收益:    {s.get('portfolio_return_pct', 0):>8.2f}%")
    print(f"  平均收益:    {s.get('avg_return_pct', 0):>8.2f}%")
    print(f"  中位数收益:  {s.get('median_return_pct', 0):>8.2f}%")
    print(f"  最佳收益:    {s.get('best_return_pct', 0):>8.2f}%")
    print(f"  最差收益:    {s.get('worst_return_pct', 0):>8.2f}%")
    print(f"  总利润:      {s.get('total_profit', 0):>10.2f}")

    print(f"\n🎯 胜率指标:")
    print(f"  胜率:        {s.get('win_rate_pct', 0):>8.1f}%")
    print(f"  盈利笔数:    {s.get('win_count', 0)}")
    print(f"  亏损笔数:    {s.get('lose_count', 0)}")
    print(f"  平均盈利:    {s.get('avg_win_pct', 0):>8.2f}%")
    print(f"  平均亏损:    {s.get('avg_loss_pct', 0):>8.2f}%")
    print(f"  盈亏比:      {s.get('profit_loss_ratio', 0):>8.2f}")

    print(f"\n⚠️ 风险指标:")
    print(f"  最大回撤:    {s.get('max_drawdown_pct', 0):>8.2f}%")
    print(f"  夏普比率:    {s.get('sharpe_ratio', 0):>8.3f}")
    print(f"  收益风险比:  {s.get('return_risk_ratio', 0):>8.3f}")
    print(f"  收益标准差:  {s.get('std_return_pct', 0):>8.2f}%")

    print(f"\n💰 成本和持有:")
    print(f"  总交易成本:  {s.get('total_cost', 0):>10.2f}")
    print(f"  平均成本率:  {s.get('avg_cost_ratio_pct', 0):>8.3f}%")
    print(f"  平均持有天:  {s.get('avg_holding_days', 0):>8.1f}")

    # 退出原因
    exit_counts = s.get("exit_reason_counts", {})
    if exit_counts:
        print(f"\n🚪 退出原因:")
        for reason, count in exit_counts.items():
            print(f"  {reason}: {count}")

    # 明细（前10条）
    backtest_df = results.get("backtest_df")
    if backtest_df is not None and not backtest_df.empty:
        print(f"\n📋 回测明细（前10条）:")
        display_cols = [
            "ts_code", "buy_date", "sell_date", "buy_price",
            "sell_price", "net_return_pct", "exit_reason",
            "holding_days", "sharpe_ratio",
        ]
        existing_cols = [c for c in display_cols if c in backtest_df.columns]

        if existing_cols:
            display_df = backtest_df[existing_cols].head(10)
            print(display_df.to_string(index=False))

    print("\n" + "=" * 60)


def print_optimization_result(
    name: str, result: "OptimizationResult"
):
    """打印优化结果"""
    print("\n" + "=" * 60)
    print(f"参数优化结果: {name}")
    print("=" * 60)

    print(f"\n  方法:       {result.method}")
    print(f"  目标指标:   {result.target_metric}")
    print(f"  总迭代:     {result.total_iterations}")
    print(f"  耗时:       {result.elapsed_seconds:.1f} 秒")
    print(f"  是否收敛:   {'是' if result.converged else '否'}")
    print(f"  最佳分数:   {result.best_score:.6f}")

    print(f"\n  最佳参数:")
    for param, value in result.best_params.items():
        if isinstance(value, float):
            print(f"    {param}: {value:.4f}")
        else:
            print(f"    {param}: {value}")

    # Top 5 结果
    sorted_results = sorted(
        result.all_results,
        key=lambda x: x.get("score", -1e10),
        reverse=True,
    )[:5]

    if sorted_results:
        print(f"\n  Top 5 参数组合:")
        for i, r in enumerate(sorted_results, 1):
            print(f"    #{i}: score={r['score']:.4f}, params={r['params']}")

    print("\n" + "=" * 60)


def print_auto_optimize_summary(
    summaries: Dict[str, Dict[str, Any]]
):
    """打印自动优化汇总"""
    print("\n" + "=" * 70)
    print("自动优化汇总")
    print("=" * 70)

    header = (
        f"{'策略':<20} {'状态':<18} "
        f"{'旧分数':>8} {'新分数':>8} {'改进':>8} {'更新':>4}"
    )
    print(header)
    print("-" * 70)

    improved_count = 0
    for name, s in summaries.items():
        old = f"{s['old_score']:.4f}" if s.get("old_score") is not None else "N/A"
        new = f"{s['new_score']:.4f}" if s.get("new_score") is not None else "N/A"
        imp = (
            f"{s['improvement_pct']:.1f}%"
            if s.get("improvement_pct") is not None
            else "N/A"
        )
        upd = "✓" if s.get("version_updated") else "✗"

        if s.get("version_updated"):
            improved_count += 1

        print(
            f"{name:<20} {s['status']:<18} "
            f"{old:>8} {new:>8} {imp:>8} {upd:>4}"
        )

    print("-" * 70)
    print(
        f"总计: {len(summaries)} 个策略, "
        f"{improved_count} 个有改进"
    )
    print("=" * 70)


# ==================== 辅助函数 ====================


def _filter_serializable(obj: Any) -> Any:
    """递归过滤不可序列化的对象"""
    if isinstance(obj, dict):
        return {
            k: _filter_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_filter_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return f"<DataFrame: {obj.shape[0]} rows × {obj.shape[1]} cols>"
    elif isinstance(obj, pd.Series):
        return f"<Series: {len(obj)} items>"
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def register_default_strategies(
    manager: StrategyManager,
    custom_params: Optional[Dict[str, Dict]] = None,
):
    """
    注册默认策略集
    
    Args:
        manager: 策略管理器
        custom_params: 自定义参数覆盖 {"策略名": {参数字典}}
    """
    if custom_params is None:
        custom_params = {}

    # 价值策略
    value_params = {
        "pe_ttm_max": 25,
        "roe_min": 12,
        "pb_max": 3.0,
        "market_cap_min": 30,
        "enable_st_filter": True,
    }
    value_params.update(custom_params.get("value", {}))
    manager.register_strategy(
        "value",
        ValueStrategy("价值投资", value_params),
        description="基于PE、ROE、PB的价值选股策略",
    )

    # 动量策略
    momentum_params = {
        "lookback_period": 20,
        "momentum_threshold": 5.0,
        "volume_ratio_min": 2.0,
        "min_price": 5.0,
    }
    momentum_params.update(custom_params.get("momentum", {}))
    manager.register_strategy(
        "momentum",
        MomentumStrategy("动量交易", momentum_params),
        description="基于价格动量和量能的短期策略",
    )

    # 成长策略
    growth_params = {
        "revenue_growth_min": 15.0,
        "profit_growth_min": 20.0,
        "roe_min": 8.0,
        "pe_ttm_max": 50,
    }
    growth_params.update(custom_params.get("growth", {}))
    manager.register_strategy(
        "growth",
        GrowthStrategy("成长投资", growth_params),
        description="关注营收和利润增长的成长股策略",
    )

    # 均值回归策略
    mr_params = {
        "oversold_threshold": -8.0,
        "pe_ttm_max": 40,
        "roe_min": 5.0,
    }
    mr_params.update(custom_params.get("mean_reversion", {}))
    manager.register_strategy(
        "mean_reversion",
        MeanReversionStrategy("均值回归", mr_params),
        description="基于超跌反弹的短期策略",
    )


def resolve_buy_date(args: argparse.Namespace) -> str:
    """解析买入日期参数"""
    if args.buy_date:
        return args.buy_date

    # 默认使用上一个交易日（简化：使用昨天）
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y%m%d")


# ==================== 命令行参数解析 ====================


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="量化策略管理系统 v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --action evaluate --strategy value --buy-date 20230601
  %(prog)s --action optimize --strategy momentum --iterations 100
  %(prog)s --action auto_optimize --threshold 10
  %(prog)s --action info
  %(prog)s --action evaluate --strategy all --use-sample-data
        """,
    )

    # 全局参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (YAML或JSON)",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="evaluate",
        choices=[
            "evaluate",
            "optimize",
            "auto_optimize",
            "info",
            "versions",
            "report",
        ],
        help="执行动作 (default: evaluate)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="value",
        help="策略名称，'all'表示所有策略 (default: value)",
    )

    # 数据参数
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="数据文件目录 (default: data)",
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="使用生成的模拟数据（用于测试）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="模拟数据的股票数量 (default: 200)",
    )

    # 回测参数
    parser.add_argument(
        "--buy-date",
        type=str,
        default=None,
        help="买入日期 YYYYMMDD (default: 昨天)",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=30,
        help="最大持股数量 (default: 30)",
    )

    # 优化参数
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["grid", "random", "bayesian", "lhs"],
        help="优化方法 (default: 使用配置文件)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="优化迭代次数 (default: 使用配置文件)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "win_rate", "return_risk_ratio"],
        help="优化目标指标 (default: sharpe_ratio)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="自动优化改进阈值%% (default: 5.0)",
    )

    # 日志参数
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (default: 使用配置文件)",
    )

    # Tushare Token
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Tushare Token (也可通过环境变量 TUSHARE_TOKEN 设置)",
    )

    return parser


# ==================== 主函数 ====================


def main() -> int:
    """
    主函数
    
    Returns:
        退出码 (0=成功, 1=失败)
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        # 1. 加载配置
        config_kwargs = {}

        # Token优先级：命令行 > 环境变量
        token = args.token or os.environ.get("TUSHARE_TOKEN", "")
        if token:
            config_kwargs["tushare_token"] = token

        if args.log_level:
            config_kwargs["log_level"] = args.log_level

        if args.config:
            config_path = Path(args.config)
            if config_path.suffix in (".yaml", ".yml"):
                config = Config.from_yaml(args.config)
            elif config_path.suffix == ".json":
                config = Config.from_json(args.config)
            else:
                print(f"不支持的配置文件格式: {config_path.suffix}")
                return 1

            # 覆盖命令行指定的参数
            if config_kwargs:
                config.update(**config_kwargs)
        else:
            config = Config(**config_kwargs)

        logger = logging.getLogger(__name__)
        logger.info("配置加载完成")

        # 2. 创建策略管理器
        manager = StrategyManager(config)

        # 3. 注册默认策略
        register_default_strategies(manager)
        logger.info(
            f"注册了 {len(manager.list_strategies())} 个策略"
        )

        # 4. 根据action分发
        if args.action == "info":
            action_info(manager, args)
            return 0

        if args.action == "versions":
            action_list_versions(manager, args)
            return 0

        if args.action == "report":
            action_report(manager, args)
            return 0

        # 以下action需要数据
        if args.use_sample_data:
            logger.info(
                f"使用模拟数据 ({args.sample_size} 只股票)"
            )
            data = generate_sample_data(n_stocks=args.sample_size)
        else:
            logger.info(f"从 {args.data_dir} 加载数据...")
            data = load_data_from_files(args.data_dir)

            if data.empty:
                logger.warning(
                    "未找到本地数据，切换到模拟数据。"
                    "使用 --use-sample-data 跳过此警告"
                )
                data = generate_sample_data(n_stocks=args.sample_size)

        logger.info(f"数据就绪: {len(data)} 条记录")

        # 解析买入日期
        args.buy_date = resolve_buy_date(args)
        logger.info(f"买入日期: {args.buy_date}")

        # 执行对应action
        if args.action == "evaluate":
            action_evaluate(manager, data, args)

        elif args.action == "optimize":
            action_optimize(manager, data, args)

        elif args.action == "auto_optimize":
            action_auto_optimize(manager, data, args)

        # 5. 清理
        manager.cleanup()

        logger.info("程序执行完成")
        return 0

    except KeyboardInterrupt:
        print("\n用户中断")
        return 130

    except Exception as e:
        logging.getLogger(__name__).error(
            f"程序执行失败: {e}", exc_info=True
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
