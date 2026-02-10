#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
短线集合程序 - 一键运行所有选股程序
==================================

功能：自动依次运行选股A、选股B、选股C三个程序，并分别生成输出文件

运行流程：
1. 运行选股A（主动选股）→ 输出 selected_stocks_YYYYMMDD.csv
2. 运行选股B（风险过滤）→ 输出 risk_filtered_stocks_YYYYMMDD.csv
3. 运行选股C（组合型）→ 输出 combined_stocks_YYYYMMDD.csv
4. 汇总所有结果，生成完整报告

使用时机：盘后15:10分运行（需要完整的盘后数据）

作者：实盘验证
Python版本：3.8+
依赖：tushare==1.4.24, pandas==2.2.2, numpy==2.2.6, python-dotenv==1.2.1
"""

import subprocess
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# ==================== 配置区域 ====================
load_dotenv()

# 获取工作目录
WORKSPACE_PATH = os.getenv('COZE_WORKSPACE_PATH', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_screener(screener_name, script_path, output_file_pattern):
    """
    运行单个选股程序

    Args:
        screener_name: 选股程序名称（选股A/选股B/选股C）
        script_path: 脚本路径
        output_file_pattern: 输出文件名模式

    Returns:
        success: 是否成功
        output_file: 输出文件路径
        stock_count: 选股数量
    """
    print("=" * 80)
    print(f"[正在运行] {screener_name}")
    print("=" * 80)

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
        print(result.stdout)

        if result.returncode != 0:
            print(f"\n❌ {screener_name} 运行失败:")
            print(result.stderr)
            return False, None, 0

        # 查找输出文件
        output_dir = os.path.join(WORKSPACE_PATH, 'assets/data')

        # 获取最新交易日
        import pandas as pd
        import tushare as ts
        ts.set_token(os.getenv('TUSHARE_TOKEN', ''))
        pro = ts.pro_api()

        from datetime import timedelta
        trade_cal = pro.trade_cal(
            exchange='SSE',
            start_date=(datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
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


def print_summary(results):
    """
    打印汇总报告
    """
    print("\n" + "=" * 80)
    print("短线集合 - 运行汇总报告")
    print("=" * 80)
    print(f"\n运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_stocks = 0
    success_count = 0

    for screener, result in results.items():
        success, output_file, stock_count = result
        status = "✅ 成功" if success else "❌ 失败"
        file_info = output_file if output_file else "N/A"

        print(f"\n【{screener}】")
        print(f"   状态: {status}")
        print(f"   选股数量: {stock_count} 只")
        print(f"   输出文件: {file_info}")

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


def main():
    """主函数"""
    print("=" * 80)
    print("短线集合程序 - 一键运行所有选股程序")
    print("=" * 80)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n将依次运行以下程序：")
    print("  1. 选股A - 主动选股（市场状态感知 + 量化策略）")
    print("  2. 选股B - 风险过滤（排除危险股票）")
    print("  3. 选股C - 组合型（选股A + 选股B）")
    print("\n请耐心等待，所有程序将依次运行...\n")

    # 定义要运行的选股程序
    screeners = [
        {
            'name': '选股A',
            'script': 'scripts/ai_stock_screener.py',
            'output_pattern': 'selected_stocks'
        },
        {
            'name': '选股B',
            'script': 'scripts/ai_stock_screener_v2.py',
            'output_pattern': 'risk_filtered_stocks'
        },
        {
            'name': '选股C',
            'script': 'scripts/ai_stock_screener_v3.py',
            'output_pattern': 'combined_stocks'
        }
    ]

    results = {}

    # 依次运行选股程序
    for screener in screeners:
        success, output_file, stock_count = run_screener(
            screener['name'],
            screener['script'],
            screener['output_pattern']
        )

        results[screener['name']] = (success, output_file, stock_count)

        # 程序之间添加2秒延时，避免API限流
        if screener != screeners[-1]:
            time.sleep(2)

    # 打印汇总报告
    print_summary(results)

    print("\n程序运行完成！")


if __name__ == '__main__':
    main()
