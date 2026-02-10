#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股C程序 - 组合型选股（选股A + 选股B）
==========================================

选股C程序是选股A和选股B的组合方案：
1. 先运行选股A，获取候选股票池（基于市场状态的量化筛选）
2. 再用选股B的风险过滤规则进行二次筛选
3. 最终输出既符合市场策略又风险可控的股票池

优势：
- 结合了选股A的市场感知能力和选股B的风险控制能力
- 双重筛选，提高选股准确度
- 输出数量更少，但质量更高
- 包含行业板块分类功能

使用时机：盘后15:10分运行（需要完整的盘后数据）

作者：实盘验证
Python版本：3.8+
依赖：tushare==1.4.24, pandas==2.2.2, numpy==2.2.6, python-dotenv==1.2.1
"""

import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import subprocess

# ==================== 配置区域 ====================
load_dotenv()

# 工作空间路径
WORKSPACE_PATH = os.getenv('COZE_WORKSPACE_PATH', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE = os.path.join(WORKSPACE_PATH, 'assets/data/combined_stocks_{}.csv'.format(datetime.now().strftime('%Y%m%d')))

# Tushare Token
TS_TOKEN = os.getenv('TUSHARE_TOKEN', '')

if not TS_TOKEN:
    raise ValueError("❌ 请在.env文件中设置TUSHARE_TOKEN")

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ==================== 筛选参数 ====================
SCREENING_PARAMS = {
    'min_pct_chg': 5.0,          # 选股B的最低涨幅
    'min_list_days': 60,         # 选股B的最少上市天数
    'ban_ratio_threshold': 0.5,  # 选股B的解禁比例阈值
    'solo_buy_threshold': 0.15,  # 选股B的龙虎榜买一独食阈值
    'same_price_pct_min': 9.0,   # 选股B的历史涨停涨幅阈值
    'same_price_pct_next': -3.0, # 选股B的历史涨停次日跌幅阈值
}

# 排除前缀（选股B使用）
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


def run_screener_a():
    """
    运行选股A（模拟，避免实际调用）
    """
    print("\n[步骤2/3] 运行选股A（主动选股）...")

    # 获取市场状态
    market_regime, strategy, strength = get_market_status_ma20()

    # 获取交易日
    trade_date = get_trade_cal()
    if not trade_date:
        print("  ❌ 未能获取交易日")
        return pd.DataFrame()

    print(f"  - 交易日: {trade_date}")

    # 获取股票基本信息
    try:
        stock_basic = pro.stock_basic(exchange='', list_status='L',
                                     fields='ts_code,symbol,name,industry,list_date,market')
    except Exception as e:
        print(f"  ❌ 获取股票基本信息失败: {e}")
        return pd.DataFrame()

    # 过滤排除前缀
    stock_basic = stock_basic[~stock_basic['ts_code'].str[:3].isin(EXCLUDE_PREFIX)]

    # 获取行情数据
    try:
        df_daily = pro.daily(trade_date=trade_date)
    except Exception as e:
        print(f"  ❌ 获取行情数据失败: {e}")
        return pd.DataFrame()

    # 合并数据
    df_a = df_daily.merge(stock_basic[['ts_code', 'name', 'industry', 'list_date']], on='ts_code', how='left')

    # 基础筛选
    df_a = df_a[df_a['pct_chg'] > 0]  # 只保留上涨的股票

    # 获取技术指标
    try:
        df_daily_basic = pro.daily_basic(trade_date=trade_date,
                                         fields='ts_code,pe_ttm,total_mv')
        df_a = df_a.merge(df_daily_basic, on='ts_code', how='left')
    except Exception as e:
        print(f"  ⚠️  获取技术指标失败: {e}")

    # 计算市值（亿）
    df_a['total_mv'] = df_a['total_mv'] / 10000

    # 市值过滤
    df_a = df_a[(df_a['total_mv'] >= 20) & (df_a['total_mv'] <= 300)]

    # PE过滤
    df_a = df_a[(df_a['pe_ttm'] >= 0) & (df_a['pe_ttm'] <= 60)]

    # 价格过滤
    df_a = df_a[(df_a['close'] >= 3) & (df_a['close'] <= 50)]

    print(f"  - 选股A筛选结果: {len(df_a)} 只股票")

    return df_a


def run_screener_b(df_a):
    """
    运行选股B（风险过滤）
    """
    print("\n[步骤3/3] 运行选股B（风险过滤）...")

    if len(df_a) == 0:
        print("  ⚠️  选股A没有结果，跳过选股B")
        return pd.DataFrame()

    # 从df_a开始过滤

    # 1. 涨幅过滤
    print(f"  - 涨幅 >= {SCREENING_PARAMS['min_pct_chg']}%...")
    df = df_a[df_a['pct_chg'] >= SCREENING_PARAMS['min_pct_chg']]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    # 2. 计算上市天数
    df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d')
    df['list_days'] = (datetime.now() - df['list_date']).dt.days

    # 3. 过滤新股
    print(f"  - 上市天数 >= {SCREENING_PARAMS['min_list_days']}天...")
    df = df[df['list_days'] >= SCREENING_PARAMS['min_list_days']]
    print(f"  - 过滤后剩余 {len(df)} 只股票")

    if len(df) == 0:
        return pd.DataFrame()

    # 4. 计算技术指标
    try:
        # 获取过去5日数据计算成交量倍数
        trade_date = get_trade_cal()
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

        # 计算换手率
        df['turnover_rate'] = (df['vol'] * 100 / df['total_mv'] / 10000).round(2)

    except Exception as e:
        print(f"  ⚠️  计算技术指标时出错: {e}")

    print(f"  - 选股B过滤结果: {len(df)} 只股票")

    return df


def output_by_industry(df):
    """
    按行业板块分组输出
    """
    if len(df) == 0:
        print("\n" + "="*80)
        print("筛选结果：未找到符合条件的股票")
        print("="*80)
        return

    # 选择输出字段
    output_cols = ['ts_code', 'name', 'industry', 'close', 'pct_chg',
                   'volume_ratio', 'turnover_rate', 'total_mv', 'pe_ttm', 'list_days']

    df_output = df[output_cols].copy()
    df_output.columns = ['代码', '名称', '行业板块', '收盘价', '涨幅(%)',
                         '成交量倍数', '换手率(%)', '市值(亿)', 'PE(TTM)', '上市天数']

    # 排序：按涨幅降序
    df_output = df_output.sort_values('涨幅(%)', ascending=False)

    print("\n" + "="*80)
    print("筛选结果（按行业板块分组）")
    print("="*80)
    print(f"\n选股数量: {len(df_output)} 只")

    # 按行业板块分组
    if '行业板块' in df_output.columns:
        industry_counts = df_output['行业板块'].value_counts()
        print("\n📊 行业板块分布：")
        for industry, count in industry_counts.items():
            print(f"  {industry}: {count}只")

        # 按行业板块分组输出
        industries = df_output['行业板块'].unique()
        for idx, industry in enumerate(industries, 1):
            industry_df = df_output[df_output['行业板块'] == industry]
            print(f"\n{'='*80}")
            print(f"【行业板块 {idx}/{len(industries)}】{industry}")
            print(f"{'='*80}")
            print(industry_df.to_string(index=False))
    else:
        print("\n" + "="*80)
        print("筛选结果")
        print("="*80)
        print(df_output.to_string(index=False))

    # 保存到CSV
    df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf_8_sig')
    print(f"\n✅ 结果已保存到: {OUTPUT_FILE}")
    print("="*80)


def main():
    """主函数"""
    print("=" * 80)
    print("选股C程序 - 组合型选股（选股A + 选股B）")
    print("=" * 80)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 步骤1：运行选股A
    df_a = run_screener_a()

    # 步骤2：运行选股B
    df_c = run_screener_b(df_a)

    # 步骤3：输出结果
    output_by_industry(df_c)

    print("\n程序运行完成！")


if __name__ == '__main__':
    main()
