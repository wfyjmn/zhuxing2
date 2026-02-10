#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
短线集合程序 - 一键运行所有选股程序
==================================

功能：自动依次运行选股A、选股B、选股C三个程序，并分别生成输出文件

运行流程：
1. 可选：市场状态检测
2. 运行选股A（主动选股）→ 输出 selected_stocks_YYYYMMDD.csv
3. 运行选股B（风险过滤）→ 输出 risk_filtered_stocks_YYYYMMDD.csv
4. 运行选股C（组合型）→ 输出 combined_stocks_YYYYMMDD.csv
5. 可选：回测和对比
6. 可选：错误检测
7. 汇总所有结果，生成完整报告

使用时机：盘后15:10分运行（需要完整的盘后数据）

作者：实盘验证
Python版本：3.8+
依赖：tushare==1.4.24, pandas==2.2.2, numpy==2.2.6, python-dotenv==1.2.1

新增功能（v3.0）：
- 市场状态检测
- 自动回测功能
- 错误检测和修正建议
- 使用新的策略管理器适配器
"""

import subprocess
import os
import sys
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

# ==================== 配置区域 ====================
load_dotenv()

# 导入统一配置
from config.screening_config import (
    API_CONFIG,
    FILTER_CONFIG,
    OUTPUT_CONFIG,
    PATH_CONFIG
)

# 获取工作目录
WORKSPACE_PATH = os.getenv('COZE_WORKSPACE_PATH', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== 命令行参数 ====================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="短线集合程序 - 一键运行所有选股程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python scripts/run_all_screeners.py

  # 只运行选股A和选股C
  python scripts/run_all_screeners.py --screeners A C

  # 启用市场状态检测
  python scripts/run_all_screeners.py --detect-market-state

  # 启用回测功能（持有5天）
  python scripts/run_all_screeners.py --enable-backtest --hold-days 5

  # 使用新的适配器模式
  python scripts/run_all_screeners.py --use-adapter

  # 启用错误检测
  python scripts/run_all_screeners.py --detect-errors

  # 完整功能
  python scripts/run_all_screeners.py --screeners A B C --detect-market-state --enable-backtest --hold-days 5 --detect-errors
        """
    )

    # 选股程序选择
    parser.add_argument(
        '--screeners',
        type=str,
        nargs='+',
        choices=['A', 'B', 'C', 'all'],
        default=['all'],
        help='选择要运行的选股程序 (A/B/C/all)，默认运行所有'
    )

    # 市场状态检测
    parser.add_argument(
        '--detect-market-state',
        action='store_true',
        help='启用市场状态检测（20日均线判断牛市/震荡市/熊市）'
    )

    # 回测选项
    parser.add_argument(
        '--enable-backtest',
        action='store_true',
        help='启用回测功能，计算选股后的收益'
    )

    parser.add_argument(
        '--hold-days',
        type=int,
        default=5,
        help='回测持有天数，默认5天'
    )

    # 错误检测
    parser.add_argument(
        '--detect-errors',
        action='store_true',
        help='启用错误检测，自动检测选股结果中的问题'
    )

    # 适配器模式
    parser.add_argument(
        '--use-adapter',
        action='store_true',
        help='使用新的策略管理器适配器模式'
    )

    # 输出选项
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='指定输出目录（覆盖配置文件）'
    )

    # 调试选项
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模拟运行，不实际执行选股程序'
    )

    return parser.parse_args()


# ==================== 市场状态检测 ====================

def detect_market_state(args):
    """
    检测市场状态

    Args:
        args: 命令行参数

    Returns:
        市场状态信息字典
    """
    print("\n" + "=" * 80)
    print("【市场状态检测】")
    print("=" * 80)

    try:
        # 尝试使用新的适配器
        if args.use_adapter:
            try:
                import sys
                sys.path.insert(0, os.path.join(WORKSPACE_PATH, 'strategy_manager'))
                from strategy_manager import Config, MarketStateDetector

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
                print(f"  推荐策略: {recommended}")

                return market_info

            except ImportError as e:
                print(f"\n⚠️  策略管理器模块未找到，使用简单检测: {e}")

        # 简单的市场状态检测（使用Tushare）
        import tushare as ts
        import pandas as pd

        ts.set_token(os.getenv('TUSHARE_TOKEN', ''))
        pro = ts.pro_api()

        from datetime import timedelta

        # 获取上证指数数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')

        df = pro.index_daily(
            ts_code='000001.SH',
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and len(df) >= 20:
            df = df.sort_values('trade_date').tail(60).reset_index(drop=True)

            latest = df.iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            deviation_pct = (latest['close'] - ma20) / ma20 * 100

            if deviation_pct > 3.0:
                state = 'bull'
                description = f"牛市（指数偏离均线+{deviation_pct:.2f}%）"
            elif deviation_pct < -3.0:
                state = 'bear'
                description = f"熊市（指数偏离均线{deviation_pct:.2f}%）"
            else:
                state = 'neutral'
                description = f"震荡市（指数偏离均线{deviation_pct:.2f}%）"

            market_info = {
                'state': state,
                'description': description,
                'ma20': round(ma20, 2),
                'current_price': round(latest['close'], 2),
                'deviation_pct': round(deviation_pct, 2)
            }

            print(f"\n📊 市场状态检测结果:")
            print(f"  状态: {state}")
            print(f"  描述: {description}")
            print(f"  20日均线: {ma20}")
            print(f"  当前价格: {latest['close']}")
            print(f"  偏离度: {deviation_pct:.2f}%")

            return market_info
        else:
            print(f"\n⚠️  数据不足，无法检测市场状态")
            return None

    except Exception as e:
        print(f"\n❌ 市场状态检测失败: {e}")
        return None


# ==================== 选股程序运行 ====================

def run_screener(screener_name, script_path, output_file_pattern, args):
    """
    运行单个选股程序

    Args:
        screener_name: 选股程序名称（选股A/选股B/选股C）
        script_path: 脚本路径
        output_file_pattern: 输出文件名模式
        args: 命令行参数

    Returns:
        success: 是否成功
        output_file: 输出文件路径
        stock_count: 选股数量
    """
    print("=" * 80)
    print(f"[正在运行] {screener_name}")
    print("=" * 80)

    if args.dry_run:
        print(f"\n[模拟运行] 跳过实际执行")
        return True, None, 0

    try:
        # 运行选股程序
        start_time = time.time()
        result = subprocess.run(
            ['python3', script_path],
            capture_output=True,
            text=True,
            cwd=WORKSPACE_PATH
        )
        end_time = time.time()

        # 打印输出
        if args.verbose:
            print(result.stdout)

        if result.returncode != 0:
            print(f"\n❌ {screener_name} 运行失败:")
            print(result.stderr)
            return False, None, 0

        # 查找输出文件
        output_dir = args.output_dir or os.path.join(WORKSPACE_PATH, PATH_CONFIG.get('output_dir', 'output'))

        # 获取最新交易日
        import pandas as pd
        import tushare as ts
        ts.set_token(os.getenv('TUSHARE_TOKEN', ''))
        pro = ts.pro_api()

        from datetime import timedelta
        trade_cal = pro.trade_cal(
            exchange='SSE',
            start_date=(datetime.now() - timedelta(days=API_CONFIG['trade_cal_days'])).strftime('%Y%m%d')
        )
        trade_cal = trade_cal[trade_cal.is_open == 1]
        trade_date = trade_cal.iloc[-1]['cal_date']

        # 尝试不同的文件名格式
        possible_files = [
            os.path.join(output_dir, f'{output_file_pattern}_{trade_date}.csv'),
            os.path.join(output_dir, f'{output_file_pattern}_{datetime.now().strftime("%Y%m%d")}.csv'),
        ]

        output_file = None
        for file in possible_files:
            if os.path.exists(file):
                output_file = file
                break

        if not output_file:
            # 查找最新的匹配文件
            import glob
            files = glob.glob(os.path.join(output_dir, f'{output_file_pattern}_*.csv'))
            if files:
                output_file = max(files, key=os.path.getmtime)
                print(f"\n⚠️  未找到当日结果，使用最新文件: {output_file}")
            else:
                print(f"\n❌ 未找到输出文件")
                return False, None, 0

        # 读取股票数量
        try:
            df = pd.read_csv(output_file, encoding='utf_8_sig')
            stock_count = len(df)
        except Exception as e:
            print(f"\n⚠️  无法读取股票数量: {e}")
            stock_count = 0

        duration = end_time - start_time
        print(f"\n✅ {screener_name} 运行成功！")
        print(f"   执行时间: {duration:.2f} 秒")
        print(f"   输出文件: {output_file}")
        print(f"   选股数量: {stock_count} 只")

        return True, output_file, stock_count

    except Exception as e:
        print(f"\n❌ 运行 {screener_name} 出错: {e}")
        return False, None, 0


# ==================== 回测功能 ====================

def run_backtest(output_file, screener_name, hold_days, args):
    """
    运行回测功能

    Args:
        output_file: 选股结果文件
        screener_name: 选股程序名称
        hold_days: 持有天数
        args: 命令行参数

    Returns:
        回测结果字典
    """
    if not output_file or not os.path.exists(output_file):
        print(f"\n⚠️  {screener_name} 无输出文件，跳过回测")
        return None

    print("\n" + "=" * 80)
    print(f"[回测] {screener_name}")
    print("=" * 80)

    try:
        import pandas as pd

        # 读取选股结果
        selected_df = pd.read_csv(output_file, encoding='utf_8_sig')

        if selected_df.empty:
            print(f"\n⚠️  选股结果为空，跳过回测")
            return None

        # 尝试使用新的适配器
        if args.use_adapter:
            try:
                import sys
                sys.path.insert(0, os.path.join(WORKSPACE_PATH, 'strategy_manager'))
                from strategy_manager import Config, ScreenerAdapter

                config = Config()
                adapter = ScreenerAdapter(config)

                # 获取买入日期（从文件名推断）
                import re
                date_match = re.search(r'(\d{8})', os.path.basename(output_file))
                buy_date = date_match.group(1) if date_match else datetime.now().strftime('%Y%m%d')

                # 回测
                result = adapter.backtest_and_compare(
                    selected_df=selected_df,
                    buy_date=buy_date,
                    hold_days=hold_days
                )

                if 'error' not in result:
                    print("\n回测报告:")
                    print(result['report'])

                    return result
                else:
                    print(f"\n❌ 回测失败: {result['error']}")
                    return None

            except ImportError as e:
                print(f"\n⚠️  策略管理器模块未找到，跳过回测: {e}")
                return None

        # 简单回测（如果无法使用适配器）
        print(f"\n⚠️  简单回测功能暂未实现")
        print(f"   选股数量: {len(selected_df)} 只")
        print(f"   建议使用 --use-adapter 启用完整回测功能")

        return None

    except Exception as e:
        print(f"\n❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 错误检测 ====================

def run_error_detection(backtest_result, screener_name, args):
    """
    运行错误检测

    Args:
        backtest_result: 回测结果
        screener_name: 选股程序名称
        args: 命令行参数

    Returns:
        检测结果字典
    """
    if not backtest_result or 'backtest_df' not in backtest_result:
        return None

    print("\n" + "=" * 80)
    print(f"[错误检测] {screener_name}")
    print("=" * 80)

    try:
        if args.use_adapter:
            import sys
            sys.path.insert(0, os.path.join(WORKSPACE_PATH, 'strategy_manager'))
            from strategy_manager import Config, ScreenerAdapter

            config = Config()
            adapter = ScreenerAdapter(config)

            detection = adapter.detect_and_correct_errors(backtest_result['backtest_df'])

            if detection['errors']:
                print("\n⚠️  发现问题:")
                for error in detection['errors']:
                    print(f"   - {error}")
            else:
                print("\n✅ 未发现明显问题")

            if detection['suggestions']:
                print("\n💡 改进建议:")
                for suggestion in detection['suggestions']:
                    print(f"   - {suggestion}")

            return detection

        return None

    except Exception as e:
        print(f"\n❌ 错误检测失败: {e}")
        return None


# ==================== 汇总报告 ====================

def print_summary(results, market_state=None):
    """
    打印汇总报告
    """
    print("\n" + "=" * 80)
    print("短线集合 - 运行汇总报告")
    print("=" * 80)
    print(f"\n运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 市场状态
    if market_state:
        print(f"\n📊 市场状态: {market_state['description']}")

    total_stocks = 0
    success_count = 0

    for screener, result in results.items():
        if 'success' in result:
            success, output_file, stock_count, backtest_result = result['success'], result['output_file'], result['stock_count'], result.get('backtest_result')
        else:
            success, output_file, stock_count = result
            backtest_result = None

        status = "✅ 成功" if success else "❌ 失败"
        file_info = output_file if output_file else "N/A"

        print(f"\n【{screener}】")
        print(f"   状态: {status}")
        print(f"   选股数量: {stock_count} 只")
        print(f"   输出文件: {file_info}")

        # 回测结果
        if backtest_result and 'stats' in backtest_result:
            stats = backtest_result['stats']
            print(f"   回测结果:")
            print(f"     胜率: {stats['win_rate']}%")
            print(f"     平均收益: {stats['avg_return']}%")
            print(f"     最佳收益: {stats['best_return']}%")

        if success:
            total_stocks += stock_count
            success_count += 1

    print("\n" + "-" * 80)
    print(f"总计: {success_count}/{len(results)} 个程序运行成功")
    print(f"总计选股数量: {total_stocks} 只（包含重复股票）")
    print("=" * 80)

    # 使用建议
    if success_count == 3:
        print("\n💡 使用建议：")
        print("   - 选股A（主动选股）：适合市场明确时，广泛撒网")
        print("   - 选股B（风险过滤）：适合任何市场，风险最低")
        print("   - 选股C（组合型）：推荐使用，双重保障，质量最高")
        print("\n📊 推荐优先级：选股C > 选股A > 选股B")
    elif success_count > 0:
        print("\n💡 部分程序运行成功，请使用成功的程序结果")
    else:
        print("\n❌ 所有程序运行失败，请检查配置和网络连接")


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    print("=" * 80)
    print("短线集合程序 - 一键运行所有选股程序 v3.0")
    print("=" * 80)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 显示配置信息
    if args.verbose:
        print(f"\n运行配置:")
        print(f"  选股程序: {' '.join(args.screeners) if args.screeners != ['all'] else '所有'}")
        print(f"  市场状态检测: {'启用' if args.detect_market_state else '禁用'}")
        print(f"  回测功能: {'启用' if args.enable_backtest else '禁用'}")
        print(f"  回测持有天数: {args.hold_days} 天")
        print(f"  错误检测: {'启用' if args.detect_errors else '禁用'}")
        print(f"  使用适配器: {'是' if args.use_adapter else '否'}")
        print(f"  输出目录: {args.output_dir or '默认'}")
        print(f"  模拟运行: {'是' if args.dry_run else '否'}")

    # 1. 市场状态检测
    market_state = None
    if args.detect_market_state:
        market_state = detect_market_state(args)

    # 2. 定义要运行的选股程序
    screener_configs = {
        'A': {
            'name': '选股A',
            'script': 'scripts/ai_stock_screener.py',
            'output_pattern': 'selected_stocks'
        },
        'B': {
            'name': '选股B',
            'script': 'scripts/ai_stock_screener_v2.py',
            'output_pattern': 'risk_filtered_stocks'
        },
        'C': {
            'name': '选股C',
            'script': 'scripts/ai_stock_screener_v3.py',
            'output_pattern': 'combined_stocks'
        }
    }

    # 选择要运行的程序
    if 'all' in args.screeners:
        screeners_to_run = ['A', 'B', 'C']
    else:
        screeners_to_run = args.screeners

    print(f"\n将依次运行以下程序：")
    for key in screeners_to_run:
        config = screener_configs[key]
        print(f"  {key}. {config['name']} - {config['script']}")
    print("\n请耐心等待，所有程序将依次运行...\n")

    results = {}

    # 3. 依次运行选股程序
    for screener_key in screeners_to_run:
        config = screener_configs[screener_key]

        success, output_file, stock_count = run_screener(
            config['name'],
            config['script'],
            config['output_pattern'],
            args
        )

        # 回测
        backtest_result = None
        if success and args.enable_backtest:
            backtest_result = run_backtest(
                output_file,
                config['name'],
                args.hold_days,
                args
            )

            # 错误检测
            if backtest_result and args.detect_errors:
                run_error_detection(backtest_result, config['name'], args)

        results[config['name']] = {
            'success': success,
            'output_file': output_file,
            'stock_count': stock_count,
            'backtest_result': backtest_result
        }

        # 程序之间添加延时，避免API限流
        if screener_key != screeners_to_run[-1]:
            time.sleep(API_CONFIG['request_delay'])

    # 4. 打印汇总报告
    print_summary(results, market_state)

    print("\n程序运行完成！")


if __name__ == '__main__':
    main()
