#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI辅助短线选股程序（选股A）
基于20日均线判断市场状态

核心逻辑：
1. 行情判断：使用20日均线判断市场状态（牛市/熊市/震荡市）
2. 策略选择：根据市场状态决定交易策略
3. AI筛选：执行严格的量化筛选，找出符合条件的股票池

作者：实盘验证
Python版本：3.8+
依赖：tushare==1.4.24, pandas==2.2.2, numpy==2.2.6, python-dotenv==1.2.1
"""

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# ==================== 配置区域 ====================
load_dotenv()

# 工作空间路径
WORKSPACE_PATH = os.getenv('COZE_WORKSPACE_PATH', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE = os.path.join(WORKSPACE_PATH, 'assets/data/selected_stocks_{}.csv'.format(datetime.now().strftime('%Y%m%d')))

# Tushare Token
TS_TOKEN = os.getenv('TUSHARE_TOKEN', '')

if not TS_TOKEN:
    raise ValueError("❌ 请在.env文件中设置TUSHARE_TOKEN")

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 股票筛选参数
SCREENING_PARAMS = {
    'market_cap_min': 20,       # 最小市值（亿）
    'market_cap_max': 300,      # 最大市值（亿）
    'pe_ttm_min': 0,            # 最小PE(TTM)（允许亏损股）
    'pe_ttm_max': 60,           # 最大PE(TTM)
    'volume_ratio_min': 1.5,    # 成交量倍数（>= 1.5倍5日均量）
    'price_min': 3,             # 最低价格（元）
    'price_max': 50,            # 最高价格（元）
    'turnover_min': 3,          # 最小换手率（%）
    'turnover_max': 20,         # 最大换手率（%）
}

# 排除前缀（科创板、创业板、ST、北交所）
EXCLUDE_PREFIX = ['300', '301', '688', '8', '4', '920']

# ==================== 核心功能函数 ====================

def get_market_status_ma20():
    """
    使用20日均线判断市场状态
    返回: (市场状态, 建议策略, 信号强度)
    """
    print("\n[步骤1/3] 正在判断市场状态（使用20日均线）...")

    try:
        # 获取沪深300指数（代表大盘）
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')

        df_index = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date)
        df_index = df_index.sort_values('trade_date')

        if len(df_index) < 20:
            print("  ❌ 数据不足，无法计算20日均线")
            return "数据不足", "空仓观望", 0

        # 计算20日均线
        df_index['ma_20'] = df_index['close'].rolling(20).mean()
        latest = df_index.iloc[-1]

        current_price = latest['close']
        ma_20 = latest['ma_20']

        # 计算偏离度
        deviation = (current_price - ma_20) / ma_20 * 100

        print(f"  - 沪深300收盘价: {current_price:.2f}")
        print(f"  - 20日均线: {ma_20:.2f}")
        print(f"  - 偏离度: {deviation:+.2f}%")

        # 判断逻辑
        if deviation > 3:  # 高于均线3%以上
            regime = "牛市"
            strategy = "积极做多"
            strength = deviation
        elif deviation < -3:  # 低于均线3%以下
            regime = "熊市"
            strategy = "空仓或防守"
            strength = deviation
        else:  # 在均线3%范围内
            regime = "震荡市"
            strategy = "精选个股"
            strength = 0

        print(f"  - 市场状态: {regime}")
        print(f"  - 建议策略: {strategy}")
        print(f"  - 信号强度: {strength:.2f}")

        return regime, strategy, strength

    except Exception as e:
        print(f"  ❌ 获取市场状态失败: {e}")
        return "数据不足", "空仓观望", 0


def get_trade_cal():
    """获取最近交易日"""
    try:
        trade_cal = pro.trade_cal(exchange='SSE', start_date=(datetime.now() - timedelta(days=10)).strftime('%Y%m%d'))
        trade_cal = trade_cal[trade_cal.is_open == 1]
        latest_date = trade_cal.iloc[-1]['cal_date']
        return latest_date
    except Exception as e:
        print(f"❌ 获取交易日失败: {e}")
        return None


def screen_stocks(market_regime):
    """
    股票筛选函数
    """
    print(f"\n[步骤2/3] 正在进行股票筛选（市场状态：{market_regime}）...")

    # 获取最近交易日
    trade_date = get_trade_cal()
    if not trade_date:
        print("❌ 未能获取交易日，程序退出。")
        return pd.DataFrame()

    print(f"  - 交易日: {trade_date}")

    # ==================== 步骤1：基础过滤 ====================
    print("\n  [2.1] 基础过滤...")

    # 获取股票基本信息
    print("    - 获取股票基本信息...")
    try:
        stock_basic = pro.stock_basic(exchange='', list_status='L',
                                     fields='ts_code,symbol,name,area,industry,list_date,market')
        print(f"    - 获取到 {len(stock_basic)} 只股票")
    except Exception as e:
        print(f"    ❌ 获取股票基本信息失败: {e}")
        return pd.DataFrame()

    # 过滤掉排除前缀的股票
    print("    - 过滤科创板、创业板、ST股、北交所...")
    stock_basic = stock_basic[~stock_basic['ts_code'].str[:3].isin(EXCLUDE_PREFIX)]
    print(f"    - 过滤后剩余 {len(stock_basic)} 只股票")

    # ==================== 步骤2：获取行情数据 ====================
    print("\n  [2.2] 获取行情数据...")

    try:
        df_daily = pro.daily(trade_date=trade_date)
        print(f"    - 获取到 {len(df_daily)} 只股票的行情数据")
    except Exception as e:
        print(f"    ❌ 获取行情数据失败: {e}")
        return pd.DataFrame()

    # 合并数据
    df = df_daily.merge(stock_basic[['ts_code', 'name', 'industry', 'list_date']], on='ts_code', how='left')

    # ==================== 步骤3：获取技术指标 ====================
    print("\n  [2.3] 获取技术指标...")

    try:
        # 获取每日指标（PE、市值等）
        df_daily_basic = pro.daily_basic(trade_date=trade_date,
                                         fields='ts_code,pe_ttm,total_mv,circ_mv')
        df = df.merge(df_daily_basic, on='ts_code', how='left')
        print(f"    - 获取到 {len(df)} 只股票的技术指标")
    except Exception as e:
        print(f"    ❌ 获取技术指标失败: {e}")
        return pd.DataFrame()

    # ==================== 步骤4：应用筛选条件 ====================
    print("\n  [2.4] 应用筛选条件...")

    # 市值过滤（亿）
    df['total_mv'] = df['total_mv'] / 10000  # 转换为亿元
    df = df[(df['total_mv'] >= SCREENING_PARAMS['market_cap_min']) &
            (df['total_mv'] <= SCREENING_PARAMS['market_cap_max'])]
    print(f"    - 市值筛选后: {len(df)} 只")

    # PE过滤
    df = df[(df['pe_ttm'] >= SCREENING_PARAMS['pe_ttm_min']) &
            (df['pe_ttm'] <= SCREENING_PARAMS['pe_ttm_max'])]
    print(f"    - PE筛选后: {len(df)} 只")

    # 价格过滤
    df = df[(df['close'] >= SCREENING_PARAMS['price_min']) &
            (df['close'] <= SCREENING_PARAMS['price_max'])]
    print(f"    - 价格筛选后: {len(df)} 只")

    # 涨幅过滤（只保留上涨的股票）
    df = df[df['pct_chg'] > 0]
    print(f"    - 涨幅筛选后: {len(df)} 只")

    # ==================== 步骤5：计算高级指标 ====================
    print("\n  [2.5] 计算高级指标...")

    try:
        # 获取过去5日数据计算成交量倍数
        start_date_5d = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        df_hist = pro.daily(ts_code=df['ts_code'].tolist(),
                           start_date=start_date_5d, end_date=trade_date)

        if len(df_hist) > 0:
            # 计算5日平均成交量
            df_hist_5d = df_hist.groupby('ts_code')['vol'].rolling(5).mean().reset_index()
            df_hist_5d.columns = ['ts_code', 'vol_5d']
            df_hist_5d = df_hist_5d.dropna().groupby('ts_code').last()

            df = df.merge(df_hist_5d[['vol_5d']], on='ts_code', how='left')

            # 计算成交量倍数
            df['volume_ratio'] = df['vol'] / df['vol_5d']
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

            # 成交量倍数过滤
            df = df[df['volume_ratio'] >= SCREENING_PARAMS['volume_ratio_min']]
            print(f"    - 成交量倍数筛选后: {len(df)} 只")

        # 计算换手率
        df['turnover_rate'] = (df['vol'] * 100 / df['total_mv'] / 10000).round(2)

        # 换手率过滤
        df = df[(df['turnover_rate'] >= SCREENING_PARAMS['turnover_min']) &
                (df['turnover_rate'] <= SCREENING_PARAMS['turnover_max'])]
        print(f"    - 换手率筛选后: {len(df)} 只")

        # 计算上市天数
        df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d')
        df['list_days'] = (datetime.now() - df['list_date']).dt.days

        # 过滤新股（上市60天内）
        df = df[df['list_days'] > 60]
        print(f"    - 新股过滤后: {len(df)} 只")

    except Exception as e:
        print(f"    ⚠️  计算高级指标时出错: {e}")

    # ==================== 步骤6：输出结果 ====================
    print(f"\n  [2.6] 筛选完成，共 {len(df)} 只股票")

    if len(df) == 0:
        return pd.DataFrame()

    # 选择输出字段
    output_cols = ['ts_code', 'name', 'industry', 'close', 'pct_chg',
                   'volume_ratio', 'turnover_rate', 'total_mv', 'pe_ttm', 'list_days']

    df_output = df[output_cols].copy()
    df_output.columns = ['代码', '名称', '行业板块', '收盘价', '涨幅(%)',
                         '成交量倍数', '换手率(%)', '市值(亿)', 'PE(TTM)', '上市天数']

    # 排序：按涨幅降序
    df_output = df_output.sort_values('涨幅(%)', ascending=False)

    return df_output


def output_results(df, market_regime, strategy):
    """
    输出结果
    """
    if len(df) == 0:
        print("\n" + "="*80)
        print("筛选结果：未找到符合条件的股票")
        print("="*80)
        print(f"\n市场状态: {market_regime}")
        print(f"建议策略: {strategy}")
        print("\n当前市场环境不适合买入，建议空仓观望。")
        print("="*80)
        return

    print("\n" + "="*80)
    print("筛选结果")
    print("="*80)
    print(f"\n市场状态: {market_regime}")
    print(f"建议策略: {strategy}")
    print(f"选股数量: {len(df)} 只\n")

    print(df.to_string(index=False))

    # 保存到CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf_8_sig')
    print(f"\n✅ 结果已保存到: {OUTPUT_FILE}")
    print("="*80)


def main():
    """主函数"""
    print("=" * 80)
    print("AI辅助短线选股程序（选股A）")
    print("=" * 80)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 步骤1：判断市场状态
    market_regime, strategy, strength = get_market_status_ma20()

    # 步骤2：筛选股票
    df = screen_stocks(market_regime)

    # 步骤3：输出结果
    output_results(df, market_regime, strategy)

    print("\n程序运行完成！")


if __name__ == '__main__':
    main()
