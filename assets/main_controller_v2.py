#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepQuant 主控程序 V2.0 - 增强版
===================================

功能：协调各模块运行，实现完整的闭环系统（整合策略管理器适配器）

工作流程：
1. 市场状态检测（20日均线）
2. 运行选股筛选（选股A/B/C）
3. 回测和对比（简化版，不实际持仓）
4. 错误检测和修正
5. 创建验证跟踪记录
6. 生成验证报告

新增功能（V2.0）：
- 使用新的策略管理器适配器
- 自动回测功能
- 错误检测和修正建议
- 支持选择性运行

作者：DeepQuant Team
版本：2.0
日期：2024
"""

import os
import sys
import subprocess
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==================== 导入适配器 ====================

try:
    # 尝试导入策略管理器适配器
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategy_manager'))
    from strategy_manager import Config, MarketStateDetector, ScreenerAdapter
    ADAPTER_AVAILABLE = True
except ImportError as e:
    print(f"[警告] 策略管理器适配器未找到: {e}")
    print("[信息] 将使用原有模式运行")
    ADAPTER_AVAILABLE = False


def print_banner():
    """打印程序横幅"""
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant Pro V2.0 (增强版)")
    print(" " * 15 + "智能选股 · 回测对比 · 错误检测")
    if ADAPTER_AVAILABLE:
        print(" " * 12 + "[策略管理器适配器已启用]")
    print("="*80)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DeepQuant 主控程序 V2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整流程
  python main_controller_v2.py full

  # 仅运行选股
  python main_controller_v2.py select

  # 使用适配器模式运行选股
  python main_controller_v2.py select --use-adapter --enable-backtest

  # 运行市场状态检测
  python main_controller_v2.py detect-market

  # 运行选股并检测错误
  python main_controller_v2.py select --detect-errors
        """
    )

    # 模式选择
    parser.add_argument(
        'mode',
        nargs='?',
        default='full',
        choices=['full', 'select', 'validate', 'optimize', 'detect-market', 'test'],
        help='运行模式'
    )

    # 适配器选项
    parser.add_argument(
        '--use-adapter',
        action='store_true',
        help='使用策略管理器适配器'
    )

    parser.add_argument(
        '--no-adapter',
        action='store_true',
        help='禁用适配器，使用原有模式'
    )

    # 选股选项
    parser.add_argument(
        '--screeners',
        type=str,
        nargs='+',
        choices=['A', 'B', 'C', 'all'],
        default=['all'],
        help='选择运行的选股程序'
    )

    # 回测选项
    parser.add_argument(
        '--enable-backtest',
        action='store_true',
        help='启用回测功能'
    )

    parser.add_argument(
        '--hold-days',
        type=int,
        default=5,
        help='回测持有天数'
    )

    # 错误检测
    parser.add_argument(
        '--detect-errors',
        action='store_true',
        help='启用错误检测'
    )

    # 数据文件
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='指定选股数据文件'
    )

    # 输出选项
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出'
    )

    return parser.parse_args()


# ==================== 市场状态检测 ====================

def detect_market_state(args):
    """
    市场状态检测（使用适配器或原有系统）

    Args:
        args: 命令行参数

    Returns:
        市场状态信息字典
    """
    print("\n" + "="*80)
    print("【阶段 0】市场状态检测")
    print("="*80)

    use_adapter = args.use_adapter if args.use_adapter else (ADAPTER_AVAILABLE and not args.no_adapter)

    # 使用适配器
    if use_adapter:
        try:
            config = Config()
            detector = MarketStateDetector(config)

            market_info = detector.detect_market_state()

            print(f"\n📊 市场状态检测结果:")
            print(f"  状态: {market_info['state']}")
            print(f"  描述: {market_info['description']}")
            print(f"  20日均线: {market_info['ma20']}")
            print(f"  当前价格: {market_info['current_price']}")
            print(f"  偏离度: {market_info['deviation_pct']:.2f}%")

            # 推荐策略
            recommended = detector.recommend_strategy(market_info['state'])
            print(f"\n💡 推荐策略: {recommended}")

            # 根据市场状态给建议
            if market_info['state'] == 'bull':
                print("   建议: 使用动量策略，适合追涨")
            elif market_info['state'] == 'bear':
                print("   建议: 使用价值策略，谨慎操作")
            else:
                print("   建议: 使用价值策略，观望为主")

            return market_info

        except Exception as e:
            print(f"\n❌ 适配器模式失败: {e}")
            print("[信息] 尝试使用原有系统...")

    # 使用原有系统
    try:
        from market_weather import MarketWeather

        weather = MarketWeather()
        forecast = weather.get_weather_forecast()

        if not forecast['allow_trading']:
            print("\n" + "⚠️"*40)
            print(f"\n[系统提醒] 当前市场天气: {forecast['weather']}")
            print(f"[系统提醒] 系统建议: {forecast['action']}")
            print(f"[系统提醒] 策略调整: {forecast['strategy_adj']}")
            print("\n[决定] 暂停选股，空仓休息")
            print("[提示] '雨天不出门'，保护资金安全比赚钱更重要")
            print("⚠️"*40 + "\n")

            return {'state': 'bear', 'allow_trading': False}

        print(f"\n[系统] 当前市场天气: {forecast['weather']}")
        print(f"[系统] 系统建议: {forecast['action']}")
        print(f"[系统] 阈值调整: {forecast['threshold_adj']:+}分")

        return {'state': forecast.get('market_state', 'neutral'), 'allow_trading': True}

    except Exception as e:
        print(f"\n⚠️ 市场状态检测失败: {e}")
        print("[信息] 继续执行选股流程")

        return {'state': 'neutral', 'allow_trading': True}


# ==================== 选股流程 ====================

def run_stock_selection_adapted(args, market_state=None):
    """
    使用适配器运行选股流程

    Args:
        args: 命令行参数
        market_state: 市场状态信息

    Returns:
        选股结果字典
    """
    print("\n" + "="*80)
    print("【阶段 1】选股筛选（适配器模式）")
    print("="*80)

    try:
        config = Config()
        adapter = ScreenerAdapter(config)

        # 加载数据
        if args.data_file and os.path.exists(args.data_file):
            print(f"\n[步骤 1.0] 加载数据: {args.data_file}")
            data = pd.read_csv(args.data_file, encoding='utf_8_sig')
        else:
            print("\n[步骤 1.0] 生成演示数据（实际应从API获取）")
            import numpy as np
            np.random.seed(42)

            n_stocks = 500
            exchanges = ["SZ"] * (n_stocks // 2) + ["SH"] * (n_stocks - n_stocks // 2)
            ts_codes = [f"{i:06d}.{ex}" for i, ex in zip(range(1, n_stocks + 1), exchanges)]

            data = {
                "ts_code": ts_codes,
                "name": [f"股票{i}" for i in range(1, n_stocks + 1)],
                "industry": np.random.choice(["电子", "计算机", "医药", "银行"], n_stocks),
                "close": np.random.lognormal(mean=3.0, sigma=0.8, size=n_stocks).round(2),
                "pct_chg": np.random.normal(5, 5, n_stocks).round(2),
                "turnover_rate": np.random.lognormal(mean=1.0, sigma=0.5, n_stocks).round(2),
                "volume_ratio": np.random.lognormal(mean=0.2, sigma=0.5, n_stocks).round(2),
            }
            data = pd.DataFrame(data)

        print(f"  数据量: {len(data)} 只股票")

        # 选择运行的选股程序
        if 'all' in args.screeners:
            screeners_to_run = ['A', 'B', 'C']
        else:
            screeners_to_run = args.screeners

        results = {}
        market_state_key = market_state['state'] if market_state else 'neutral'

        for screener in screeners_to_run:
            print(f"\n[步骤 1.{screeners_to_run.index(screener) + 1}] 运行选股{screener}...")

            if screener == 'A':
                result = adapter.run_screener_a(data, market_state=market_state_key)
            elif screener == 'B':
                result = adapter.run_screener_b(data)
            elif screener == 'C':
                result = adapter.run_screener_c(data, enable_industry=True)

            results[f'选股{screener}'] = result

            print(f"  选中数量: {len(result)} 只")

            # 保存结果
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)

            output_file = output_dir / f"selected_stocks_{screener}_{datetime.now().strftime('%Y%m%d')}.csv"
            result.to_csv(output_file, index=False, encoding='utf_8_sig')
            print(f"  已保存: {output_file}")

            # 回测
            if args.enable_backtest and len(result) > 0:
                print(f"\n  [回测] 开始回测选股{screener}...")
                backtest_result = adapter.backtest_and_compare(
                    selected_df=result,
                    buy_date=datetime.now().strftime('%Y%m%d'),
                    hold_days=args.hold_days
                )

                if 'error' not in backtest_result:
                    print(backtest_result['report'])

                    # 错误检测
                    if args.detect_errors:
                        detection = adapter.detect_and_correct_errors(
                            backtest_result['backtest_df']
                        )

                        if detection['errors']:
                            print("\n  ⚠️ 发现问题:")
                            for error in detection['errors']:
                                print(f"    - {error}")

                        if detection['suggestions']:
                            print("\n  💡 改进建议:")
                            for suggestion in detection['suggestions']:
                                print(f"    - {suggestion}")

        print("\n[✅ 完成] 选股筛选流程已完成")

        return results

    except Exception as e:
        print(f"\n❌ 适配器模式选股失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_stock_selection_original(args):
    """
    使用原有模式运行选股流程

    Args:
        args: 命令行参数

    Returns:
        是否成功
    """
    print("\n" + "="*80)
    print("【阶段 1】选股筛选（原有模式）")
    print("="*80)

    print("\n[步骤 1.1] 运行第1轮筛选...")
    try:
        result = subprocess.run(
            [sys.executable, '柱形选股-筛选.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            print(f"[错误] 第1轮筛选失败")
            print(result.stderr)
            return False

        print("[完成] 第1轮筛选成功")

    except Exception as e:
        print(f"[错误] 执行第1轮筛选失败: {e}")
        return False

    print("\n[步骤 1.2] 运行第2轮筛选...")
    try:
        result = subprocess.run(
            [sys.executable, '柱形选股-第2轮.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            print(f"[错误] 第2轮筛选失败")
            print(result.stderr)
            return False

        print("[完成] 第2轮筛选成功")

    except Exception as e:
        print(f"[错误] 执行第2轮筛选失败: {e}")
        return False

    print("\n[✅ 完成] 选股筛选流程已完成")
    return True


def run_stock_selection(args, market_state=None):
    """
    运行选股流程（自动选择模式）

    Args:
        args: 命令行参数
        market_state: 市场状态信息

    Returns:
        选股结果
    """
    use_adapter = args.use_adapter if args.use_adapter else (ADAPTER_AVAILABLE and not args.no_adapter)

    if use_adapter:
        return run_stock_selection_adapted(args, market_state)
    else:
        return run_stock_selection_original(args)


# ==================== 完整流程 ====================

def run_full_pipeline(args):
    """运行完整流程"""
    print_banner()

    # 阶段 0：市场状态检测
    market_state = detect_market_state(args)

    # 如果市场不允许交易，则退出
    if not market_state.get('allow_trading', True):
        print("\n[决定] 根据市场状态，暂停选股")
        return True

    # 阶段 1：选股
    selection_result = run_stock_selection(args, market_state)

    if not selection_result:
        print("\n[❌ 失败] 选股阶段失败，流程终止")
        return False

    print("\n" + "="*80)
    print("【✅ 完成】完整流程执行完毕")
    print("="*80)

    return True


def run_select_mode(args):
    """仅运行选股"""
    print_banner()

    # 市场状态检测
    market_state = detect_market_state(args)

    # 选股
    run_stock_selection(args, market_state)


def run_detect_market_mode(args):
    """仅运行市场状态检测"""
    print_banner()

    detect_market_state(args)


def run_test_mode(args):
    """运行测试模式"""
    print_banner()
    print("\n[测试模式] 验证适配器功能\n")

    if ADAPTER_AVAILABLE:
        print("✅ 策略管理器适配器可用")

        try:
            config = Config()
            print("✅ 配置模块可用")

            detector = MarketStateDetector(config)
            print("✅ 市场状态检测器可用")

            adapter = ScreenerAdapter(config)
            print("✅ 选股适配器可用")

            print("\n所有模块正常！")

        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
    else:
        print("❌ 策略管理器适配器不可用")
        print("[提示] 请检查 strategy_manager 模块是否正确安装")


# ==================== 主函数 ====================

def main():
    """主函数"""
    args = parse_arguments()

    # 根据模式执行
    if args.mode == 'full':
        run_full_pipeline(args)
    elif args.mode == 'select':
        run_select_mode(args)
    elif args.mode == 'detect-market':
        run_detect_market_mode(args)
    elif args.mode == 'test':
        run_test_mode(args)
    else:
        print(f"\n[警告] 未知模式: {args.mode}")
        print("[信息] 可用模式: full, select, validate, optimize, detect-market, test")


if __name__ == '__main__':
    main()
