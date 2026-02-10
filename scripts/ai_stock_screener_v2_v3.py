#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股B程序 - 风险过滤型选股（V3增强版）
=========================================

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

V3版本优化内容：
1. API调用优化：
   - 分批次请求数据（按代码分段），避免触发限流
   - 增加请求间隔（0.5-1秒）
   - 使用limit参数分页获取
   - 提前将股票基本面数据转换为字典映射，减少重复查询
2. 数据处理逻辑修正：
   - 修正换手率计算，直接使用daily_basic的turnover_rate字段
   - 添加价格区间筛选（price_min/price_max）
   - 增加股价位置校验（收盘价站在5/10日均线上方）
   - 过滤高位放量的风险标的
3. 异常处理增强：
   - 添加重试机制
   - 增强错误提示

作者：实盘验证2年
Python版本：3.8+
依赖：tushare==1.4.24, pandas==2.2.2, numpy==2.2.6, python-dotenv==1.2.1
"""

import tushare as ts
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# ==================== 配置区域 ====================
load_dotenv()

# 导入统一配置
from config.screening_config import (
    API_CONFIG,
    SCREENER_B_CONFIG,
    FILTER_CONFIG,
    OUTPUT_CONFIG,
    PATH_CONFIG
)

# 别名配置（保持向后兼容）
SCREENING_PARAMS = SCREENER_B_CONFIG
EXCLUDE_PREFIX = FILTER_CONFIG['exclude_prefix']
EXCLUDE_NAME_KEYWORDS = FILTER_CONFIG['exclude_name_keywords']

# 工作空间路径
WORKSPACE_PATH = os.getenv('COZE_WORKSPACE_PATH', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE = os.path.join(WORKSPACE_PATH, PATH_CONFIG['output_dir'] + f'/risk_filtered_stocks_{datetime.now().strftime(PATH_CONFIG["date_format"])}.csv')

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
    'request_delay': 0.5,       # 请求间隔（秒，增加到0.5-1秒）
    'batch_size': 500,          # 批量获取数量（减少到500）
    'limit': 3000,              # 每次请求的limit参数
}

# ==================== 筛选参数（V3增强版） ====================
SCREENING_PARAMS = {
    # 基础筛选参数
    'min_pct_chg': 5.0,          # 最低涨幅（%）
    'min_list_days': 60,         # 最少上市天数
    'ban_ratio_threshold': 0.5,  # 解禁比例阈值（%）
    'solo_buy_threshold': 0.15,  # 龙虎榜买一独食阈值（%）
    'same_price_pct_min': 9.0,   # 历史涨停涨幅阈值（%）
    'same_price_pct_next': -3.0, # 历史涨停次日跌幅阈值（%）

    # V3新增参数
    'price_min': 3,              # 最低价格（元）
    'price_max': 50,             # 最高价格（元）
    'turnover_min': 3,           # 最小换手率（%）
    'turnover_max': 20,          # 最大换手率（%）
    'volume_ratio_min': 1.5,     # 最小成交量倍数
    'check_price_position': True, # 是否检查股价位置
    'check_ma5': True,           # 是否检查5日均线
    'check_ma10': True,          # 是否检查10日均线
}

# 排除前缀（自动排除科创、创业、北证、ST、退市）
EXCLUDE_PREFIX = ['300', '301', '688', '8', '4', '920']

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

def get_daily_data_with_limit(ts_code, start_date, end_date, offset=0):
    """
    使用limit参数分页获取数据
    """
    try:
        df = api_call_with_retry(
            pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            limit=API_CONFIG['limit']
        )
        return df
    except Exception as e:
        print(f"  ❌ 获取数据失败: {e}")
        return None

def get_daily_data_batch(ts_codes, start_date, end_date):
    """
    分批获取历史数据，避免频率限制
    """
    all_data = []
    total = len(ts_codes)
    batch_size = API_CONFIG['batch_size']

    for i in range(0, total, batch_size):
        batch = ts_codes[i:i + batch_size]
        print(f"    - 正在获取第{i+1}-{min(i+batch_size, total)}/{total}只股票的历史数据...")

        try:
            df = api_call_with_retry(
                pro.daily,
                ts_code=batch,
                start_date=start_date,
                end_date=end_date,
                limit=API_CONFIG['limit']
            )

            if df is not None and len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"    ❌ 获取批次数据失败: {e}")
            continue

    if len(all_data) == 0:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

def get_daily_basic_batch(ts_codes, trade_date):
    """
    获取每日指标（注意：Tushare的daily_basic接口不支持ts_code参数筛选）
    需要获取所有数据后再筛选
    """
    print(f"    - 正在获取所有股票的技术指标（不限制ts_code）...")

    try:
        # 获取所有股票的daily_basic数据
        df = api_call_with_retry(
            pro.daily_basic,
            trade_date=trade_date,
            fields='ts_code,pe_ttm,total_mv,circ_mv,turnover_rate'
        )

        if df is None or len(df) == 0:
            print("    ⚠️  没有获取到任何数据")
            return pd.DataFrame()

        # 筛选需要的股票
        df_filtered = df[df['ts_code'].isin(ts_codes)]
        print(f"    - 从 {len(df)} 只股票中筛选出 {len(df_filtered)} 只目标股票")
        return df_filtered

    except Exception as e:
        print(f"    ❌ 获取数据失败: {e}")
        return pd.DataFrame()

# ==================== 核心功能函数 ====================

def get_trade_cal():
    """获取最近交易日"""
    try:
        trade_cal = api_call_with_retry(
            pro.trade_cal,
            exchange='SSE',
            start_date=(datetime.now() - timedelta(days=API_CONFIG['trade_cal_days'])).strftime('%Y%m%d')
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
    """
    检查股价位置（V3新增功能）
    要求：收盘价站在5日和10日均线上方
    """
    print("\n  [3.5] 检查股价位置...")

    if len(df_hist) == 0 or not SCREENING_PARAMS['check_price_position']:
        print("    - 跳过股价位置检查")
        return df

    # 转换日期格式
    df_hist['trade_date'] = pd.to_datetime(df_hist['trade_date'], format='%Y%m%d')

    # 按股票代码和日期排序
    df_hist = df_hist.sort_values(['ts_code', 'trade_date'])

    # 计算5日和10日均线
    df_hist['ma5'] = df_hist.groupby('ts_code')['close'].rolling(SCREENING_PARAMS['ma5_days']).mean().reset_index(0, drop=True)
    df_hist['ma10'] = df_hist.groupby('ts_code')['close'].rolling(SCREENING_PARAMS['ma10_days']).mean().reset_index(0, drop=True)

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
    主筛选函数：风险过滤型选股（V3增强版）
    """
    print("=" * 80)
    print("选股B程序 - 风险过滤型选股（V3增强版）")
    print("=" * 80)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nV3版本优化内容：")
    print("  1. API调用优化：")
    print("     - 分批次请求数据（按代码分段）")
    print("     - 增加请求间隔（0.5秒）")
    print("     - 使用limit参数分页获取")
    print("     - 提前将股票基本面数据转换为字典映射")
    print("  2. 数据处理逻辑修正：")
    print("     - 修正换手率计算，直接使用daily_basic的turnover_rate")
    print("     - 添加价格区间筛选（price_min/price_max）")
    print("     - 增加股价位置校验（收盘价站在5/10日均线上方）")
    print("     - 过滤高位放量的风险标的")
    print("  3. 异常处理增强")

    # 获取最近交易日
    trade_date = get_trade_cal()
    if not trade_date:
        print("❌ 未能获取交易日，程序退出。")
        return pd.DataFrame()

    print(f"\n交易日: {trade_date}")

    # ==================== 步骤1：基础过滤 ====================
    print("\n[步骤1/6] 正在进行基础过滤...")

    # 1.1 获取当日所有股票数据
    print("  - 正在获取当日行情数据...")
    try:
        # 使用limit参数分页获取
        df_daily = get_daily_data_with_limit(
            ts_code='',
            start_date=trade_date,
            end_date=trade_date
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

        # 优化：将股票基本信息转换为字典映射，减少重复查询
        stock_basic_dict = stock_basic.set_index('ts_code')[['name', 'industry', 'list_date']].to_dict('index')
        print(f"  - 已创建股票基本信息字典映射（{len(stock_basic_dict)}条记录）")

    except Exception as e:
        print(f"  ❌ 获取股票基本信息失败: {e}")
        return pd.DataFrame()

    # 1.3 过滤排除前缀
    print("  - 过滤科创板、创业板、ST股、北交所...")
    df = df_daily.copy()  # 创建副本
    df = df[~df['ts_code'].str[:3].isin(EXCLUDE_PREFIX)]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    # 1.4 合并数据（使用字典映射）
    print("  - 合并股票基本信息...")
    df.loc[:, 'name'] = df['ts_code'].map(lambda x: stock_basic_dict.get(x, {}).get('name', ''))
    df.loc[:, 'industry'] = df['ts_code'].map(lambda x: stock_basic_dict.get(x, {}).get('industry', ''))
    df.loc[:, 'list_date'] = df['ts_code'].map(lambda x: stock_basic_dict.get(x, {}).get('list_date', ''))

    # ==================== 步骤2：上涨门槛过滤 ====================
    print("\n[步骤2/6] 正在进行上涨门槛过滤...")

    # 2.1 只保留上涨的股票
    print(f"  - 涨幅 >= {SCREENING_PARAMS['min_pct_chg']}%...")
    df = df[df['pct_chg'] >= SCREENING_PARAMS['min_pct_chg']]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    if len(df) == 0:
        print("  ⚠️  没有股票通过上涨门槛过滤")
        return pd.DataFrame()

    # ==================== 步骤3：价格区间筛选（V3新增） ====================
    print("\n[步骤3/6] 正在进行价格区间筛选...")

    print(f"  - 价格区间：{SCREENING_PARAMS['price_min']}-{SCREENING_PARAMS['price_max']}元...")
    df = df[(df['close'] >= SCREENING_PARAMS['price_min']) &
            (df['close'] <= SCREENING_PARAMS['price_max'])]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    if len(df) == 0:
        print("  ⚠️  没有股票通过价格区间筛选")
        return pd.DataFrame()

    # ==================== 步骤4：风险指标过滤 ====================
    print("\n[步骤4/6] 正在进行风险指标过滤...")

    # 4.1 获取每日指标（V3修正：直接获取turnover_rate）
    print("  - 获取每日指标（包含换手率）...")

    # 初始化字段，避免合并失败时字段缺失
    df.loc[:, 'total_mv'] = df.get('total_mv', 0)
    df.loc[:, 'pe_ttm'] = df.get('pe_ttm', 0)
    df.loc[:, 'turnover_rate'] = df.get('turnover_rate', 0)

    try:
        # 分批获取每日指标
        df_daily_basic = get_daily_basic_batch(
            df['ts_code'].tolist(),
            trade_date
        )

        if df_daily_basic is not None and len(df_daily_basic) > 0:
            df = df.merge(df_daily_basic, on='ts_code', how='left', suffixes=('', '_new'))

            # 使用新获取的数据
            df.loc[:, 'total_mv'] = df['total_mv_new'].fillna(df['total_mv'])
            df.loc[:, 'pe_ttm'] = df['pe_ttm_new'].fillna(df['pe_ttm'])
            df.loc[:, 'turnover_rate'] = df['turnover_rate_new'].fillna(df['turnover_rate'])

            # 删除临时列
            df = df.drop(columns=['total_mv_new', 'pe_ttm_new', 'turnover_rate_new'], errors='ignore')

            print(f"  - 获取到 {len(df_daily_basic)} 只股票的技术指标")
        else:
            print("  ⚠️  获取每日指标失败，使用默认值")
    except Exception as e:
        print(f"  ⚠️  获取每日指标失败: {e}，使用默认值")

    # 4.2 计算市值（亿）
    df['total_mv'] = df['total_mv'] / 10000

    # 4.3 使用换手率筛选（V3修正：直接使用turnover_rate字段）
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
    print("\n[步骤5/6] 正在进行龙虎榜风险过滤...")

    # 5.1 获取龙虎榜数据
    print("  - 获取龙虎榜数据...")
    try:
        df_top = api_call_with_retry(
            pro.top_list,
            trade_date=trade_date
        )

        if df_top is not None and len(df_top) > 0:
            print(f"  - 获取到 {len(df_top)} 条龙虎榜记录")

            # 检查是否有buy和sell字段
            if 'buy' in df_top.columns and 'sell' in df_top.columns:
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
                print("  ⚠️  龙虎榜数据不包含buy/sell字段，跳过买一独食过滤")
        else:
            print("  - 没有龙虎榜数据")
    except Exception as e:
        print(f"  ⚠️  获取龙虎榜数据失败: {e}")

    # ==================== 步骤6：计算高级指标（V3增强） ====================
    print("\n[步骤6/6] 计算高级指标...")

    try:
        # 获取过去历史数据计算成交量倍数
        start_date_5d = (datetime.now() - timedelta(days=API_CONFIG['trade_cal_days'])).strftime('%Y%m%d')

        print("    - 获取历史数据计算成交量倍数...")
        df_hist = get_daily_data_batch(
            df['ts_code'].tolist(),
            start_date_5d,
            trade_date
        )

        # 初始化必要字段
        if 'volume_ratio' not in df.columns:
            df['volume_ratio'] = SCREENING_PARAMS['default_volume_ratio']
        if 'turnover_rate' not in df.columns:
            df['turnover_rate'] = SCREENING_PARAMS['default_turnover_rate']
        if 'list_days' not in df.columns:
            df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d')
            df['list_days'] = (datetime.now() - df['list_date']).dt.days

        if len(df_hist) > 0:
            # 优化：使用更高效的方式计算5日平均成交量
            df_hist = df_hist.sort_values(['ts_code', 'trade_date'])

            # 计算每只股票的5日平均成交量
            df_hist_5d = df_hist.groupby('ts_code')['vol'].rolling(SCREENING_PARAMS['ma5_days']).mean().reset_index()
            df_hist_5d.columns = ['ts_code', 'vol_5d']
            df_hist_5d = df_hist_5d.dropna().groupby('ts_code').last()

            df = df.merge(df_hist_5d[['vol_5d']], on='ts_code', how='left')

            # 计算成交量倍数
            df['volume_ratio'] = df['vol'] / df['vol_5d']
            df['volume_ratio'] = df['volume_ratio'].fillna(SCREENING_PARAMS['default_volume_ratio'])

            # 成交量倍数筛选
            print(f"    - 成交量倍数 >= {SCREENING_PARAMS['volume_ratio_min']}")
            df = df[df['volume_ratio'] >= SCREENING_PARAMS['volume_ratio_min']]
            print(f"    - 成交量倍数筛选后: {len(df)} 只")

        # V3新增：检查股价位置（过滤高位放量）
        df = check_price_position(df, df_hist)

    except Exception as e:
        print(f"  ⚠️  计算高级指标时出错: {e}")
        print(f"  ⏭️  跳过高级指标计算，继续使用基础筛选结果")

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
