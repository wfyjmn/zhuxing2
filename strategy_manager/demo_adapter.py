#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示脚本 - 展示如何使用适配后的策略管理系统

功能：
1. 演示市场状态检测
2. 演示选股A/B/C的运行
3. 演示回测和对比
4. 演示错误检测和修正
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_manager.config import Config
from strategy_manager.adapter import MarketStateDetector, ScreenerAdapter
from strategy_manager.simple_backtest import SimpleBacktestEngine


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def generate_demo_data(n_stocks: int = 200) -> pd.DataFrame:
    """
    生成演示数据
    
    模拟真实的选股数据格式
    """
    import numpy as np

    np.random.seed(42)

    # 股票代码
    exchanges = ["SZ"] * (n_stocks // 2) + ["SH"] * (n_stocks - n_stocks // 2)
    ts_codes = [f"{i:06d}.{ex}" for i, ex in zip(range(1, n_stocks + 1), exchanges)]

    # 股票名称
    names = [f"股票{i}" for i in range(1, n_stocks + 1)]
    for idx in np.random.choice(n_stocks, size=max(1, n_stocks // 20), replace=False):
        names[idx] = f"*ST模拟{idx + 1}"

    data = {
        "ts_code": ts_codes,
        "name": names,
        "industry": np.random.choice(["电子", "计算机", "医药", "银行"], n_stocks),
        "close": np.random.lognormal(mean=3.0, sigma=0.8, size=n_stocks).round(2),
        "pct_chg": np.random.normal(5, 5, n_stocks).round(2),
        "turnover_rate": np.random.lognormal(mean=1.0, sigma=0.5, n_stocks).round(2),
        "volume_ratio": np.random.lognormal(mean=0.2, sigma=0.5, n_stocks).round(2),
    }

    return pd.DataFrame(data)


def demo_market_state():
    """演示市场状态检测"""
    print("\n" + "="*70)
    print("【演示 1】市场状态检测")
    print("="*70)

    config = Config()
    detector = MarketStateDetector(config)

    market_info = detector.detect_market_state()

    print(f"\n当前市场状态:")
    print(f"  状态: {market_info['state']}")
    print(f"  描述: {market_info['description']}")
    print(f"  20日均线: {market_info['ma20']}")
    print(f"  当前价格: {market_info['current_price']}")

    # 推荐策略
    recommended = detector.recommend_strategy(market_info['state'])
    print(f"  推荐策略: {recommended}")


def demo_screener():
    """演示选股程序"""
    print("\n" + "="*70)
    print("【演示 2】选股程序运行")
    print("="*70)

    config = Config()
    adapter = ScreenerAdapter(config)

    # 生成演示数据
    print("\n生成演示数据...")
    data = generate_demo_data(200)
    print(f"  原始股票数: {len(data)}")

    # 运行选股A
    print("\n[选股A] 运行中...")
    result_a = adapter.run_screener_a(data)
    print(f"  选中股票数: {len(result_a)}")

    if not result_a.empty:
        print("\n  Top 10 选股结果:")
        print(result_a[["ts_code", "name", "pct_chg", "turnover_rate"]].head(10).to_string(index=False))

    # 运行选股B
    print("\n[选股B] 运行中...")
    result_b = adapter.run_screener_b(data)
    print(f"  选中股票数: {len(result_b)}")

    # 运行选股C
    print("\n[选股C] 运行中...")
    result_c = adapter.run_screener_c(data, enable_industry=True)
    print(f"  选中股票数: {len(result_c)}")

    return result_a, result_b, result_c


def demo_backtest(selected_df: pd.DataFrame):
    """演示回测功能"""
    print("\n" + "="*70)
    print("【演示 3】回测和对比")
    print("="*70)

    if selected_df is None or selected_df.empty:
        print("没有选股结果，跳过回测演示")
        return

    config = Config()
    adapter = ScreenerAdapter(config)

    # 使用昨天的日期作为买入日期
    buy_date = (datetime.now().replace(day=1)).strftime("%Y%m%d")  # 使用本月1日

    print(f"\n买入日期: {buy_date}")
    print(f"选股数量: {len(selected_df)}")

    result = adapter.backtest_and_compare(
        selected_df=selected_df,
        buy_date=buy_date,
        hold_days=5
    )

    if "error" in result:
        print(f"\n回测失败: {result['error']}")
        return

    print("\n回测报告:")
    print(result["report"])


def demo_error_detection(backtest_df: pd.DataFrame):
    """演示错误检测和修正"""
    print("\n" + "="*70)
    print("【演示 4】错误检测和修正")
    print("="*70)

    if backtest_df is None or backtest_df.empty:
        print("没有回测结果，跳过错误检测演示")
        return

    config = Config()
    adapter = ScreenerAdapter(config)

    detection = adapter.detect_and_correct_errors(backtest_df)

    print("\n检测结果:")

    if detection["errors"]:
        print("  ⚠️ 发现问题:")
        for error in detection["errors"]:
            print(f"    - {error}")
    else:
        print("  ✅ 未发现明显问题")

    if detection["suggestions"]:
        print("\n  💡 改进建议:")
        for suggestion in detection["suggestions"]:
            print(f"    - {suggestion}")


def main():
    """主函数"""
    setup_logging()

    print("\n" + "="*70)
    print(" " * 15 + "策略管理系统适配演示")
    print("="*70)

    try:
        # 1. 市场状态检测
        demo_market_state()

        # 2. 选股程序
        result_a, result_b, result_c = demo_screener()

        # 3. 回测和对比
        demo_backtest(result_a)

        # 4. 错误检测和修正
        print("\n注意: 错误检测需要实际回测数据，演示中跳过")

        print("\n" + "="*70)
        print("演示完成！")
        print("="*70)

    except Exception as e:
        print(f"\n演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
