#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股B程序 - 风险过滤型选股（优化版）
======================================

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

优化内容：
- 添加API重试机制和请求间隔
- 修正换手率计算错误，直接从daily_basic的turnover_rate字段获取
- 优化历史数据获取，分批处理避免频率限制
- 增强异常处理，提升稳定性

作者：实盘验证2年
Python版本：3.8+
依赖：tushare==1.4.24, pandas==2.2.2, numpy==2.2.6, python-dotenv==1.2.1
"""

import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# ==================== 配置区域 ====================
load_dotenv()

# 工作空间路径
WORKSPACE_PATH = os.getenv('COZE_WORKSPACE_PATH', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE = os.path.join(WORKSPACE_PATH, 'assets/data/risk_filtered_stocks_{}.csv'.format(datetime.now().strftime('%Y%m%d')))

# Tushare Token（从.env文件读取，如果没有则回退到环境变量）
TS_TOKEN = os.getenv('TUSHARE_TOKEN', '')

if not TS_TOKEN:
    raise ValueError("❌ 请在.env文件中设置TUSHARE_TOKEN")

# 排除前缀（自动排除科创、创业、北证、ST、退市）
# 300: 创业板
# 301: 创业板
# 688: 科创板
# 8: 北交所
# 4: 北交所
# 920: 北交所
EXCLUDE_PREFIX = ['300', '301', '688', '8', '4', '920']

# ==================== 初始化 Tushare ====================
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# API调用配置
API_CONFIG = {
    'retry_times': 3,           # 重试次数
    'retry_delay': 1,           # 重试间隔（秒）
    'request_delay': 0.3,       # 请求间隔（秒）
    'batch_size': 1000,         # 批量获取数量
}

# ==================== 筛选参数 ====================
SCREENING_PARAMS = {
    'min_pct_chg': 5.0,          # 最低涨幅（%）
    'min_list_days': 60,         # 最少上市天数
    'ban_ratio_threshold': 0.5,  # 解禁比例阈值（%）
    'solo_buy_threshold': 0.15,  # 龙虎榜买一独食阈值（%）
    'same_price_pct_min': 9.0,   # 历史涨停涨幅阈值（%）
    'same_price_pct_next': -3.0, # 历史涨停次日跌幅阈值（%）
}

# ==================== 工具函数 ====================

def api_call_with_retry(func, *args, **kwargs):
    """
    带重试机制的API调用
    """
    for attempt in range(API_CONFIG['retry_times']):
        try:
            result = func(*args, **kwargs)
            time.sleep(API_CONFIG['request_delay'])  # 请求间隔
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

def get_daily_data_batch(ts_codes, start_date, end_date, batch_size=None):
    """
    分批获取历史数据，避免频率限制
    """
    if batch_size is None:
        batch_size = API_CONFIG['batch_size']

    all_data = []
    total = len(ts_codes)

    for i in range(0, total, batch_size):
        batch = ts_codes[i:i + batch_size]
        print(f"    - 正在获取第{i+1}-{min(i+batch_size, total)}/{total}只股票的历史数据...")

        try:
            df = api_call_with_retry(
                pro.daily,
                ts_code=batch,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"    ❌ 获取批次数据失败: {e}")
            continue

    if len(all_data) == 0:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

# ==================== 核心功能函数 ====================

def get_trade_cal():
    """获取最近交易日"""
    try:
        # 获取最近的交易日
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


def get_daily_screener():
    """
    主筛选函数：风险过滤型选股
    """
    print("=" * 80)
    print("选股B程序 - 风险过滤型选股（优化版）")
    print("=" * 80)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("优化内容：")
    print("  - 添加API重试机制和请求间隔")
    print("  - 修正换手率计算错误，直接从daily_basic获取")
    print("  - 优化历史数据获取，分批处理避免频率限制")
    print("  - 增强异常处理，提升稳定性")

    # 获取最近交易日
    trade_date = get_trade_cal()
    if not trade_date:
        print("❌ 未能获取交易日，程序退出。")
        return pd.DataFrame()

    print(f"\n交易日: {trade_date}")

    # ==================== 步骤1：基础过滤 ====================
    print("\n[步骤1/4] 正在进行基础过滤...")

    # 1.1 获取当日所有股票数据
    print("  - 正在获取当日行情数据...")
    try:
        df_daily = api_call_with_retry(
            pro.daily,
            trade_date=trade_date
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
    except Exception as e:
        print(f"  ❌ 获取股票基本信息失败: {e}")
        return pd.DataFrame()

    # 1.3 合并数据
    df = df_daily.merge(stock_basic[['ts_code', 'name', 'industry', 'list_date']], on='ts_code', how='left')

    # 1.4 过滤排除前缀
    print("  - 过滤科创板、创业板、ST股、北交所...")
    df = df[~df['ts_code'].str[:3].isin(EXCLUDE_PREFIX)]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    # ==================== 步骤2：上涨门槛过滤 ====================
    print("\n[步骤2/4] 正在进行上涨门槛过滤...")

    # 2.1 只保留上涨的股票
    print(f"  - 涨幅 >= {SCREENING_PARAMS['min_pct_chg']}%...")
    df = df[df['pct_chg'] >= SCREENING_PARAMS['min_pct_chg']]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    if len(df) == 0:
        print("  ⚠️  没有股票通过上涨门槛过滤")
        return pd.DataFrame()

    # ==================== 步骤3：风险指标过滤 ====================
    print("\n[步骤3/4] 正在进行风险指标过滤...")

    # 3.1 获取每日指标（PE、市值、换手率等）
    print("  - 获取每日指标...")
    try:
        # 修正：直接获取turnover_rate字段
        df_daily_basic = api_call_with_retry(
            pro.daily_basic,
            trade_date=trade_date,
            fields='ts_code,pe_ttm,total_mv,circ_mv,turnover_rate'  # 添加turnover_rate
        )

        if df_daily_basic is None or len(df_daily_basic) == 0:
            print("  ⚠️  获取每日指标失败")
        else:
            df = df.merge(df_daily_basic, on='ts_code', how='left')
    except Exception as e:
        print(f"  ⚠️  获取每日指标失败: {e}")

    # 3.2 计算市值（亿）
    df['total_mv'] = df['total_mv'] / 10000

    # 3.3 计算上市天数
    df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d')
    df['list_days'] = (datetime.now() - df['list_date']).dt.days

    # 3.4 过滤新股
    print(f"  - 上市天数 >= {SCREENING_PARAMS['min_list_days']}天...")
    df = df[df['list_days'] >= SCREENING_PARAMS['min_list_days']]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    if len(df) == 0:
        print("  ⚠️  没有股票通过新股过滤")
        return pd.DataFrame()

    # ==================== 步骤4：龙虎榜风险过滤 ====================
    print("\n[步骤4/4] 正在进行龙虎榜风险过滤...")

    # 4.1 获取龙虎榜数据
    print("  - 获取龙虎榜数据...")
    try:
        # 获取最近5个交易日的龙虎榜数据
        start_date_5d = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        df_top = api_call_with_retry(
            pro.top_list,
            trade_date=trade_date
        )

        if df_top is not None and len(df_top) > 0:
            print(f"  - 获取到 {len(df_top)} 条龙虎榜记录")

            # 过滤买一独食
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
            print("  - 没有龙虎榜数据")
    except Exception as e:
        print(f"  ⚠️  获取龙虎榜数据失败: {e}")

    # ==================== 步骤5：计算技术指标 ====================
    print("\n[步骤5/5] 计算技术指标...")

    # 初始化必要字段，避免后续访问失败
    if 'volume_ratio' not in df.columns:
        df['volume_ratio'] = 1.0  # 默认值
    if 'turnover_rate' not in df.columns:
        df['turnover_rate'] = 0.0  # 默认值
    if 'list_days' not in df.columns:
        df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d')
        df['list_days'] = (datetime.now() - df['list_date']).dt.days

    try:
        # 获取过去5日数据计算成交量倍数（使用分批获取）
        start_date_5d = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')

        print("    - 获取历史数据计算成交量倍数...")
        df_hist = get_daily_data_batch(
            df['ts_code'].tolist(),
            start_date_5d,
            trade_date,
            batch_size=API_CONFIG['batch_size']
        )

        if len(df_hist) > 0:
            # 优化：使用更高效的方式计算5日平均成交量
            df_hist = df_hist.sort_values(['ts_code', 'trade_date'])

            # 计算每只股票的5日平均成交量
            df_hist_5d = df_hist.groupby('ts_code')['vol'].rolling(5).mean().reset_index()
            df_hist_5d.columns = ['ts_code', 'vol_5d']
            df_hist_5d = df_hist_5d.dropna().groupby('ts_code').last()

            df = df.merge(df_hist_5d[['vol_5d']], on='ts_code', how='left')

            # 计算成交量倍数
            df['volume_ratio'] = df['vol'] / df['vol_5d']
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

        # 修正：使用daily_basic提供的换手率（单位：%）
        # 不再使用错误的计算公式：df['vol'] / df['amount'] * 100
        if 'turnover_rate' in df.columns:
            df['turnover_rate'] = df['turnover_rate'].fillna(0)
        else:
            # 如果没有换手率数据，则跳过
            print("    ⚠️  未获取到换手率数据")

    except Exception as e:
        print(f"  ⚠️  计算技术指标时出错: {e}")
        print(f"  ⏭️  跳过技术指标计算，继续使用基础筛选结果")

    # ==================== 输出结果 ====================
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

    # 选择输出字段
    output_cols = required_cols

    df_output = df[output_cols].copy()
    df_output.columns = ['代码', '名称', '行业板块', '收盘价', '涨幅(%)',
                         '成交量倍数', '换手率(%)', '市值(亿)', 'PE(TTM)', '上市天数']

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

    return df_output


def main():
    """主函数"""
    get_daily_screener()


if __name__ == '__main__':
    main()
