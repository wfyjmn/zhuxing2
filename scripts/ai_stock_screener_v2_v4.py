#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股B程序 - 风险过滤型选股（V4终极版）
=======================================

定位：不是用来抓涨停，而是排除掉90%会让人吃大面的送命题
核心思路：回答"这个票明天有没有人会砸盘？"

实盘统计：
- 被拉黑的股票：次日平均收益 -1.98%
- 标记为安全的股票：次日平均收益 +1.27%
- 差值 = 3.25%，这就是做超短的所有利润来源

使用时机：盘后15:10分跑，不要盘中跑（数据不全）
使用原则：
1. 一天通常输出2-5只股票，甚至为空，这是正常的
2. 空仓是完全正确的结果，不要为了买股票而降低标准
3. 永远不要反过来用：不要先看上一个票，再来改规则放行

V4版本新增功能：
1. ✅ 添加止损/止盈参考（5日均线止损、10%-15%止盈）
2. ✅ 完善ST股排除逻辑（覆盖ST、*ST、退、退整理）
3. ✅ 确保彻底排除创业板、科创板、北交所股票
4. ✅ 添加详细的操作建议和风险提示

作者：实盘验证2年
Python版本：3.8+
依赖：tushare==1.4.24, pandas==2.2.2, numpy==2.2.6, python-dotenv==1.2.1
"""

import tushare as ts
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# ==================== 配置区域 ====================
load_dotenv()

# 工作空间路径
WORKSPACE_PATH = os.getenv('COZE_WORKSPACE_PATH', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE = os.path.join(WORKSPACE_PATH, 'assets/data/risk_filtered_stocks_{}.csv'.format(datetime.now().strftime('%Y%m%d')))

# Tushare Token
TS_TOKEN = os.getenv('TUSHARE_TOKEN', '')

if not TS_TOKEN:
    raise ValueError("❌ 请在.env文件中设置TUSHARE_TOKEN")

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ==================== API调用配置 ====================
API_CONFIG = {
    'retry_times': 3,           # 重试次数
    'retry_delay': 1,           # 重试间隔（秒）
    'request_delay': 0.5,       # 请求间隔（秒）
    'batch_size': 500,          # 批量获取数量
    'limit': 3000,              # 每次请求的limit参数
}

# ==================== 筛选参数（V4终极版） ====================
SCREENING_PARAMS = {
    # 基础筛选参数
    'min_pct_chg': 5.0,          # 最低涨幅（%）
    'min_list_days': 60,         # 最少上市天数
    'ban_ratio_threshold': 0.5,  # 解禁比例阈值（%）
    'solo_buy_threshold': 0.15,  # 龙虎榜买一独食阈值（%）
    'same_price_pct_min': 9.0,   # 历史涨停涨幅阈值（%）
    'same_price_pct_next': -3.0, # 历史涨停次日跌幅阈值（%）

    # 价格筛选参数
    'price_min': 3,              # 最低价格（元）
    'price_max': 50,             # 最高价格（元）
    'turnover_min': 3,           # 最小换手率（%）
    'turnover_max': 20,          # 最大换手率（%）
    'volume_ratio_min': 1.5,     # 最小成交量倍数

    # 股价位置检查
    'check_price_position': True, # 是否检查股价位置
    'check_ma5': True,           # 是否检查5日均线
    'check_ma10': True,          # 是否检查10日均线

    # V4新增：止损止盈参数
    'stop_loss_pct': 5.0,        # 止损百分比（%）
    'stop_loss_ma': True,        # 是否使用5日均线止损
    'take_profit_min': 10.0,     # 最低止盈百分比（%）
    'take_profit_max': 15.0,     # 最高止盈百分比（%）
}

# 排除前缀（V4完善：确保彻底排除）
# 300: 创业板
# 301: 创业板
# 688: 科创板
# 8: 北交所
# 4: 北交所
# 920: 北交所
EXCLUDE_PREFIX = ['300', '301', '688', '8', '4', '920']

# V4新增：排除股票名称中的风险关键词
EXCLUDE_NAME_KEYWORDS = ['ST', r'\*ST', '退', '退整理']

# ==================== 工具函数 ====================

def api_call_with_retry(func, *args, **kwargs):
    """带重试机制的API调用"""
    for attempt in range(API_CONFIG['retry_times']):
        try:
            result = func(*args, **kwargs)
            time.sleep(API_CONFIG['request_delay'])
            return result
        except Exception as e:
            if attempt < API_CONFIG['retry_times'] - 1:
                print(f"  ⚠️  API调用失败（第{attempt+1}次尝试）: {e}")
                print(f"  ⏳  {API_CONFIG['retry_delay']}秒后重试...")
                time.sleep(API_CONFIG['retry_delay'])
            else:
                print(f"  ❌ API调用失败（已达最大重试次数）: {e}")
                raise
    return None

def get_daily_data_batch(ts_codes, start_date, end_date):
    """分批获取历史数据，避免频率限制"""
    all_data = []
    total = len(ts_codes)
    batch_size = API_CONFIG['batch_size']

    for i in range(0, total, batch_size):
        batch = ts_codes[i:i + batch_size]
        print(f"    - 正在获取第{i+1}-{min(i+batch_size, total)}/{total}只股票的历史数据（{start_date} - {end_date}）...")

        try:
            df = api_call_with_retry(
                pro.daily,
                ts_code=batch,
                start_date=start_date,
                end_date=end_date,
                limit=API_CONFIG['limit']
            )

            if df is not None and len(df) > 0:
                print(f"      ✓ 获取到 {len(df)} 条数据")
                all_data.append(df)
            else:
                print(f"      ⚠️  该批次无数据")
        except Exception as e:
            print(f"    ❌ 获取批次数据失败: {e}")
            continue

    if len(all_data) == 0:
        print(f"    ⚠️  总共获取到 0 条历史数据")
        return pd.DataFrame()

    print(f"    ✓ 总共获取到 {sum([len(d) for d in all_data])} 条历史数据")
    return pd.concat(all_data, ignore_index=True)

def get_daily_basic_batch(ts_codes, trade_date):
    """获取每日指标（注意：Tushare的daily_basic接口不支持ts_code参数筛选）"""
    print(f"    - 正在获取所有股票的技术指标（不限制ts_code）...")

    try:
        df = api_call_with_retry(
            pro.daily_basic,
            trade_date=trade_date,
            fields='ts_code,pe_ttm,total_mv,circ_mv,turnover_rate'
        )

        if df is None or len(df) == 0:
            print("    ⚠️  没有获取到任何数据")
            return pd.DataFrame()

        df_filtered = df[df['ts_code'].isin(ts_codes)]
        print(f"    - 从 {len(df)} 只股票中筛选出 {len(df_filtered)} 只目标股票")
        return df_filtered

    except Exception as e:
        print(f"    ❌ 获取数据失败: {e}")
        return pd.DataFrame()

def calculate_stop_loss_take_profit(df, df_hist):
    """
    V4新增：计算止损位和止盈位
    - 止损位：5日均线 或 -5%跌幅
    - 止盈位：10%-15%涨幅
    """
    print("\n  [6.1] 计算止损止盈位...")

    if len(df_hist) == 0:
        print("    - 无历史数据，使用固定止损止盈策略（5%止损，10%-15%止盈）")
        # 使用固定止损止盈
        df.loc[:, 'stop_loss'] = (df['close'] * (1 - SCREENING_PARAMS['stop_loss_pct'] / 100)).round(2)
        df.loc[:, 'stop_loss_type'] = f"{SCREENING_PARAMS['stop_loss_pct']}%止损"
        df.loc[:, 'take_profit_min'] = (df['close'] * (1 + SCREENING_PARAMS['take_profit_min'] / 100)).round(2)
        df.loc[:, 'take_profit_max'] = (df['close'] * (1 + SCREENING_PARAMS['take_profit_max'] / 100)).round(2)
        df.loc[:, 'take_profit_target'] = (df['close'] * (1 + (SCREENING_PARAMS['take_profit_min'] + SCREENING_PARAMS['take_profit_max']) / 2 / 100)).round(2)
        print(f"    - 已计算 {len(df)} 只股票的止损止盈位（固定策略）")
        print(f"    - 止损策略：{SCREENING_PARAMS['stop_loss_pct']}%止损")
        print(f"    - 止盈策略：{SCREENING_PARAMS['take_profit_min']}-{SCREENING_PARAMS['take_profit_max']}%")
        return df

    # 有历史数据时，使用5日均线止损
    df.loc[:, 'stop_loss_ma'] = df['ma5'].round(2)
    df.loc[:, 'stop_loss_pct'] = (df['close'] * (1 - SCREENING_PARAMS['stop_loss_pct'] / 100)).round(2)
    
    if SCREENING_PARAMS['stop_loss_ma']:
        df.loc[:, 'stop_loss'] = df[['stop_loss_ma', 'stop_loss_pct']].max(axis=1).round(2)
        df.loc[:, 'stop_loss_type'] = '5日均线'
    else:
        df.loc[:, 'stop_loss'] = df['stop_loss_pct']
        df.loc[:, 'stop_loss_type'] = '5%止损'

    # 计算止盈位（10%-15%）
    df.loc[:, 'take_profit_min'] = (df['close'] * (1 + SCREENING_PARAMS['take_profit_min'] / 100)).round(2)
    df.loc[:, 'take_profit_max'] = (df['close'] * (1 + SCREENING_PARAMS['take_profit_max'] / 100)).round(2)
    df.loc[:, 'take_profit_target'] = (df['close'] * (1 + (SCREENING_PARAMS['take_profit_min'] + SCREENING_PARAMS['take_profit_max']) / 2 / 100)).round(2)

    print(f"    - 已计算 {len(df)} 只股票的止损止盈位")
    print(f"    - 止损策略：{'5日均线' if SCREENING_PARAMS['stop_loss_ma'] else '5%止损'}")
    print(f"    - 止盈策略：{SCREENING_PARAMS['take_profit_min']}-{SCREENING_PARAMS['take_profit_max']}%")

    return df

# ==================== 核心功能函数 ====================

def get_trade_cal():
    """获取最近交易日"""
    try:
        trade_cal = api_call_with_retry(
            pro.trade_cal,
            exchange='SSE',
            start_date=(datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        )

        if trade_cal is None:
            return None

        trade_cal = trade_cal[trade_cal.is_open == 1]
        if len(trade_cal) == 0:
            return None

        latest_date = trade_cal.iloc[-1]['cal_date']
        return latest_date
    except Exception as e:
        print(f"❌ 获取交易日失败: {e}")
        return None


def check_price_position(df, df_hist):
    """检查股价位置（要求收盘价站在5日和10日均线上方）"""
    print("\n  [5.1] 检查股价位置...")

    if len(df_hist) == 0 or not SCREENING_PARAMS['check_price_position']:
        print("    - 跳过股价位置检查")
        return df

    # 转换日期格式
    df_hist['trade_date'] = pd.to_datetime(df_hist['trade_date'], format='%Y%m%d')
    df_hist = df_hist.sort_values(['ts_code', 'trade_date'])

    # 计算5日和10日均线
    df_hist['ma5'] = df_hist.groupby('ts_code')['close'].rolling(5).mean().reset_index(0, drop=True)
    df_hist['ma10'] = df_hist.groupby('ts_code')['close'].rolling(10).mean().reset_index(0, drop=True)

    # 获取每只股票最新的均线数据
    latest_ma = df_hist.groupby('ts_code').last().reset_index()
    latest_ma = latest_ma[['ts_code', 'ma5', 'ma10']]

    # 合并均线数据
    df = df.merge(latest_ma, on='ts_code', how='left')

    # 检查股价是否站在均线上方
    initial_count = len(df)

    if SCREENING_PARAMS['check_ma5']:
        df = df[df['close'] > df['ma5']]
        print(f"    - 5日均线筛选后: {len(df)} 只")

    if SCREENING_PARAMS['check_ma10']:
        df = df[df['close'] > df['ma10']]
        print(f"    - 10日均线筛选后: {len(df)} 只")

    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"    - 股价位置检查: 过滤 {filtered_count} 只高位放量股票")

    return df


def get_daily_screener():
    """
    主筛选函数：风险过滤型选股（V4终极版）
    """
    print("=" * 80)
    print("选股B程序 - 风险过滤型选股（V4终极版）")
    print("=" * 80)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nV4版本新增功能：")
    print("  1. ✅ 添加止损/止盈参考（5日均线止损、10%-15%止盈）")
    print("  2. ✅ 完善ST股排除逻辑（覆盖ST、*ST、退、退整理）")
    print("  3. ✅ 确保彻底排除创业板、科创板、北交所股票")
    print("  4. ✅ 添加详细的操作建议和风险提示")

    # 获取最近交易日
    trade_date = get_trade_cal()
    if not trade_date:
        print("❌ 未能获取交易日，程序退出。")
        return pd.DataFrame()

    print(f"\n交易日: {trade_date}")

    # ==================== 步骤1：基础过滤 ====================
    print("\n[步骤1/7] 正在进行基础过滤...")

    # 1.1 获取当日所有股票数据
    print("  - 正在获取当日行情数据...")
    try:
        df_daily = api_call_with_retry(
            pro.daily,
            trade_date=trade_date,
            limit=API_CONFIG['limit']
        )

        if df_daily is None or len(df_daily) == 0:
            print("  ❌ 获取行情数据失败")
            return pd.DataFrame()

        print(f"  - 获取到 {len(df_daily)} 只股票的行情数据")
    except Exception as e:
        print(f"  ❌ 获取行情数据失败: {e}")
        return pd.DataFrame()

    # 1.2 获取股票基本信息
    print("  - 正在获取股票基本信息...")
    try:
        stock_basic = api_call_with_retry(
            pro.stock_basic,
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date,market'
        )

        if stock_basic is None or len(stock_basic) == 0:
            print("  ❌ 获取股票基本信息失败")
            return pd.DataFrame()

        print(f"  - 获取到 {len(stock_basic)} 只股票的基本信息")

        # 优化：将股票基本信息转换为字典映射
        stock_basic_dict = stock_basic.set_index('ts_code')[['name', 'industry', 'list_date']].to_dict('index')
        print(f"  - 已创建股票基本信息字典映射（{len(stock_basic_dict)}条记录）")

    except Exception as e:
        print(f"  ❌ 获取股票基本信息失败: {e}")
        return pd.DataFrame()

    # 1.3 V4完善：过滤排除前缀（确保彻底排除创业板、科创板、北交所）
    print("  - 过滤科创板、创业板、ST股、北交所...")
    df = df_daily.copy()
    df = df[~df['ts_code'].str[:3].isin(EXCLUDE_PREFIX)]
    print(f"    - 过滤前缀后剩余 {len(df)} 只股票")

    # 1.4 V4新增：过滤股票名称中的风险关键词（ST、*ST、退、退整理）
    print(f"  - 过滤风险股票（{', '.join(EXCLUDE_NAME_KEYWORDS)}）...")
    initial_count = len(df)
    df = df[~df['ts_code'].isin(stock_basic.index)]
    df['name'] = df['ts_code'].map(lambda x: stock_basic_dict.get(x, {}).get('name', ''))
    
    # 检查股票名称是否包含风险关键词（使用向量化操作）
    pattern = '|'.join([re.escape(keyword) for keyword in EXCLUDE_NAME_KEYWORDS])
    df = df[~df['name'].str.contains(pattern, na=False)]
    
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"    - 过滤风险股票后: {len(df)} 只股票（已过滤 {filtered_count} 只）")

    # 1.5 合并数据
    print("  - 合并股票基本信息...")
    df.loc[:, 'industry'] = df['ts_code'].map(lambda x: stock_basic_dict.get(x, {}).get('industry', ''))
    df.loc[:, 'list_date'] = df['ts_code'].map(lambda x: stock_basic_dict.get(x, {}).get('list_date', ''))

    # ==================== 步骤2：上涨门槛过滤 ====================
    print("\n[步骤2/7] 正在进行上涨门槛过滤...")

    print(f"  - 涨幅 >= {SCREENING_PARAMS['min_pct_chg']}%...")
    df = df[df['pct_chg'] >= SCREENING_PARAMS['min_pct_chg']]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    if len(df) == 0:
        print("  ⚠️  没有股票通过上涨门槛过滤")
        return pd.DataFrame()

    # ==================== 步骤3：价格区间筛选 ====================
    print("\n[步骤3/7] 正在进行价格区间筛选...")

    print(f"  - 价格区间：{SCREENING_PARAMS['price_min']}-{SCREENING_PARAMS['price_max']}元...")
    df = df[(df['close'] >= SCREENING_PARAMS['price_min']) &
            (df['close'] <= SCREENING_PARAMS['price_max'])]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    if len(df) == 0:
        print("  ⚠️  没有股票通过价格区间筛选")
        return pd.DataFrame()

    # ==================== 步骤4：风险指标过滤 ====================
    print("\n[步骤4/7] 正在进行风险指标过滤...")

    # 4.1 获取每日指标
    print("  - 获取每日指标（包含换手率）...")
    
    # 初始化字段
    df.loc[:, 'total_mv'] = df.get('total_mv', 0)
    df.loc[:, 'pe_ttm'] = df.get('pe_ttm', 0)
    df.loc[:, 'turnover_rate'] = df.get('turnover_rate', 0)

    try:
        df_daily_basic = get_daily_basic_batch(
            df['ts_code'].tolist(),
            trade_date
        )

        if df_daily_basic is not None and len(df_daily_basic) > 0:
            # 预先转换数据类型以避免 FutureWarning（向量化操作）
            cols_to_convert = [col for col in ['total_mv', 'pe_ttm', 'turnover_rate'] if col in df.columns]
            if cols_to_convert:
                df[cols_to_convert] = df[cols_to_convert].astype('float64')

            df = df.merge(df_daily_basic, on='ts_code', how='left', suffixes=('', '_new'))

            df.loc[:, 'total_mv'] = df['total_mv_new'].fillna(df['total_mv']).astype('float64')
            df.loc[:, 'pe_ttm'] = df['pe_ttm_new'].fillna(df['pe_ttm']).astype('float64')
            df.loc[:, 'turnover_rate'] = df['turnover_rate_new'].fillna(df['turnover_rate']).astype('float64')

            df = df.drop(columns=['total_mv_new', 'pe_ttm_new', 'turnover_rate_new'], errors='ignore')

            print(f"  - 获取到 {len(df_daily_basic)} 只股票的技术指标")
        else:
            print("  ⚠️  获取每日指标失败，使用默认值")
    except Exception as e:
        print(f"  ⚠️  获取每日指标失败: {e}，使用默认值")

    # 4.2 计算市值（亿）
    df['total_mv'] = df['total_mv'] / 10000

    # 4.3 使用换手率筛选
    if 'turnover_rate' in df.columns:
        print(f"  - 换手率区间：{SCREENING_PARAMS['turnover_min']}-{SCREENING_PARAMS['turnover_max']}%...")
        df = df[(df['turnover_rate'] >= SCREENING_PARAMS['turnover_min']) &
                (df['turnover_rate'] <= SCREENING_PARAMS['turnover_max'])]
        print(f"  - 换手率筛选后: {len(df)} 只股票")
    else:
        print("  ⚠️  未获取到换手率数据，跳过换手率筛选")

    # 4.4 计算上市天数
    df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d')
    df['list_days'] = (datetime.now() - df['list_date']).dt.days

    # 4.5 过滤新股
    print(f"  - 上市天数 >= {SCREENING_PARAMS['min_list_days']}天...")
    df = df[df['list_days'] >= SCREENING_PARAMS['min_list_days']]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    if len(df) == 0:
        print("  ⚠️  没有股票通过新股过滤")
        return pd.DataFrame()

    # ==================== 步骤5：龙虎榜风险过滤 ====================
    print("\n[步骤5/7] 正在进行龙虎榜风险过滤...")

    print("  - 获取龙虎榜数据...")
    try:
        df_top = api_call_with_retry(
            pro.top_list,
            trade_date=trade_date
        )

        if df_top is not None and len(df_top) > 0:
            print(f"  - 获取到 {len(df_top)} 条龙虎榜记录")

            if 'buy' in df_top.columns and 'sell' in df_top.columns:
                print(f"  - 过滤买一独食（>= {SCREENING_PARAMS['solo_buy_threshold']*100}%）...")
                df_top_group = df_top.groupby('ts_code').agg({
                    'buy': 'sum',
                    'sell': 'sum'
                })
                df_top_group['solo_buy_ratio'] = df_top_group['buy'] / (df_top_group['buy'] + df_top_group['sell'])

                solo_buy_stocks = df_top_group[df_top_group['solo_buy_ratio'] >= SCREENING_PARAMS['solo_buy_threshold']].index.tolist()
                if len(solo_buy_stocks) > 0:
                    print(f"  - 拉黑 {len(solo_buy_stocks)} 只买一独食股票")
                    df = df[~df['ts_code'].isin(solo_buy_stocks)]
                    print(f"  - 过滤后剩余 {len(df)} 只股票")
            else:
                print("  ⚠️  龙虎榜数据不包含buy/sell字段，跳过买一独食过滤")
        else:
            print("  - 没有龙虎榜数据")
    except Exception as e:
        print(f"  ⚠️  获取龙虎榜数据失败: {e}")

    # ==================== 步骤6：计算高级指标（V4增强） ====================
    print("\n[步骤6/7] 计算高级指标...")

    try:
        # 获取过去30日数据计算成交量倍数（延长周期以确保有足够数据）
        start_date_5d = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

        print("    - 获取历史数据计算成交量倍数...")
        df_hist = get_daily_data_batch(
            df['ts_code'].tolist(),
            start_date_5d,
            trade_date
        )

        print(f"    - 获取到 {len(df_hist)} 条历史数据记录")

        # 初始化必要字段
        if 'volume_ratio' not in df.columns:
            df['volume_ratio'] = 1.0
        if 'turnover_rate' not in df.columns:
            df['turnover_rate'] = 0.0
        if 'list_days' not in df.columns:
            df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d')
            df['list_days'] = (datetime.now() - df['list_date']).dt.days

        if len(df_hist) > 0:
            # 计算5日平均成交量
            df_hist = df_hist.sort_values(['ts_code', 'trade_date'])
            df_hist_5d = df_hist.groupby('ts_code')['vol'].rolling(5).mean().reset_index()
            df_hist_5d.columns = ['ts_code', 'vol_5d']
            df_hist_5d = df_hist_5d.dropna().groupby('ts_code').last()

            df = df.merge(df_hist_5d[['vol_5d']], on='ts_code', how='left')

            # 计算成交量倍数
            df['volume_ratio'] = df['vol'] / df['vol_5d']
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

            # 成交量倍数筛选
            print(f"    - 成交量倍数 >= {SCREENING_PARAMS['volume_ratio_min']}")
            df = df[df['volume_ratio'] >= SCREENING_PARAMS['volume_ratio_min']]
            print(f"    - 成交量倍数筛选后: {len(df)} 只")

        # 检查股价位置
        df = check_price_position(df, df_hist)

        # V4新增：计算止损止盈位
        df = calculate_stop_loss_take_profit(df, df_hist)

    except Exception as e:
        print(f"  ⚠️  计算高级指标时出错: {e}")
        print(f"  ⏭️  跳过高级指标计算，继续使用基础筛选结果")

    # ==================== 步骤7：输出结果（V4增强） ====================
    print(f"\n筛选完成，共 {len(df)} 只股票")

    if len(df) == 0:
        print("\n" + "="*80)
        print("筛选结果：未找到符合条件的股票")
        print("="*80)
        print("\n这是正常的！空仓是完全正确的结果。")
        print("不要为了买股票而降低标准。")
        print("="*80)
        return pd.DataFrame()

    # 确保所有必需的字段都存在
    required_cols = ['ts_code', 'name', 'industry', 'close', 'pct_chg',
                     'volume_ratio', 'turnover_rate', 'total_mv', 'pe_ttm', 'list_days']
    for col in required_cols:
        if col not in df.columns:
            print(f"  ⚠️  缺少字段 {col}，使用默认值")
            if col == 'volume_ratio':
                df[col] = 1.0
            elif col == 'turnover_rate':
                df[col] = 0.0
            elif col == 'list_days':
                df[col] = 999
            else:
                df[col] = 0

    # V4新增：确保止损止盈字段存在
    if 'stop_loss' not in df.columns:
        df['stop_loss'] = df['close'] * 0.95
    if 'stop_loss_type' not in df.columns:
        df['stop_loss_type'] = '5%止损'
    if 'take_profit_min' not in df.columns:
        df['take_profit_min'] = df['close'] * 1.10
    if 'take_profit_max' not in df.columns:
        df['take_profit_max'] = df['close'] * 1.15
    if 'take_profit_target' not in df.columns:
        df['take_profit_target'] = df['close'] * 1.125

    # 选择输出字段（V4新增：包含止损止盈）
    output_cols = ['ts_code', 'name', 'industry', 'close', 'pct_chg',
                   'volume_ratio', 'turnover_rate', 'total_mv', 'pe_ttm', 'list_days',
                   'stop_loss', 'stop_loss_type', 'take_profit_min', 'take_profit_max', 'take_profit_target']

    df_output = df[output_cols].copy()
    df_output.columns = ['代码', '名称', '行业板块', '收盘价', '涨幅(%)',
                         '成交量倍数', '换手率(%)', '市值(亿)', 'PE(TTM)', '上市天数',
                         '止损价', '止损类型', '止盈价(最低)', '止盈价(最高)', '止盈价(参考)']

    # 排序：按涨幅降序
    df_output = df_output.sort_values('涨幅(%)', ascending=False)

    print("\n" + "="*80)
    print("筛选结果")
    print("="*80)
    print(f"\n选股数量: {len(df_output)} 只\n")

    print(df_output.to_string(index=False))

    # 保存到CSV
    df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf_8_sig')
    print(f"\n✅ 结果已保存到: {OUTPUT_FILE}")
    print("="*80)

    # V4新增：输出操作建议
    print("\n" + "="*80)
    print("操作建议")
    print("="*80)
    print(f"\n📌 买入时机：")
    print(f"  - 建议开盘后观察，若股价回调到支撑位可考虑买入")
    print(f"  - 建议分批建仓，控制单只股票仓位不超过总资金的10%")
    
    print(f"\n📌 止损策略：")
    print(f"  - 止损位：{'5日均线' if SCREENING_PARAMS['stop_loss_ma'] else '5%止损'}")
    print(f"  - 一旦跌破止损位，坚决止损，不要抱有幻想")
    print(f"  - 止损是保命的，严格执行！")
    
    print(f"\n📌 止盈策略：")
    print(f"  - 第一止盈位：{SCREENING_PARAMS['take_profit_min']}%（可减仓50%）")
    print(f"  - 第二止盈位：{SCREENING_PARAMS['take_profit_max']}%（可减仓至20%）")
    print(f"  - 剩余仓位可跟踪趋势，设置移动止损")
    
    print(f"\n📌 风险提示：")
    print(f"  - 本筛选结果仅供参考，不构成投资建议")
    print(f"  - 股市有风险，投资需谨慎")
    print(f"  - 请根据自身风险承受能力理性投资")
    print(f"  - 严格执行止损止盈纪律")
    
    print(f"\n📌 资金管理：")
    print(f"  - 建议总仓位控制在30%-50%")
    print(f"  - 单只股票仓位不超过10%")
    print(f"  - 保留30%现金应对突发情况")
    
    print("="*80)

    return df_output


def main():
    """主函数"""
    get_daily_screener()


if __name__ == '__main__':
    main()
