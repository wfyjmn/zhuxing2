#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股C程序测试版本
"""

import tushare as ts
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

# API配置
API_CONFIG = {
    'token': os.getenv('TUSHARE_TOKEN'),
    'retry_times': 3,
    'retry_delay': 1,
    'batch_size': 500,
    'limit': 5000,
}

pro = ts.pro_api(API_CONFIG['token'])

def api_call_with_retry(func, **kwargs):
    """带重试的API调用"""
    for attempt in range(API_CONFIG['retry_times']):
        try:
            result = func(**kwargs)
            if result is None or len(result) == 0:
                return None
            return result
        except Exception as e:
            if attempt < API_CONFIG['retry_times'] - 1:
                print(f"    ⚠️  API调用失败（第{attempt+1}次重试）: {e}")
                time.sleep(API_CONFIG['retry_delay'])
            else:
                print(f"    ❌ API调用失败（已重试{API_CONFIG['retry_times']}次）: {e}")
                return None
    return None

def main():
    """测试主函数"""
    print("=" * 80)
    print("选股C程序 - 测试版本")
    print("=" * 80)
    
    # 获取交易日
    print("\n[步骤1] 获取交易日...")
    trade_cal = api_call_with_retry(
        pro.trade_cal,
        exchange='SSE',
        start_date=(datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
    )
    
    if trade_cal is None:
        print("  ❌ 无法获取交易日")
        return
    
    trade_cal = trade_cal[trade_cal.is_open == 1]
    trade_date = trade_cal.iloc[-1]['cal_date']
    print(f"  ✓ 交易日: {trade_date}")
    
    # 获取股票基本信息
    print("\n[步骤2] 获取股票基本信息...")
    stock_basic = api_call_with_retry(
        pro.stock_basic,
        exchange='',
        list_status='L',
        fields='ts_code,symbol,name,area,industry,list_date,market'
    )
    
    if stock_basic is None:
        print("  ❌ 无法获取股票基本信息")
        return
    
    print(f"  ✓ 获取到 {len(stock_basic)} 只股票")
    
    # 创建字典映射
    stock_basic_dict = stock_basic.set_index('ts_code')[['name', 'industry', 'list_date']].to_dict('index')
    print(f"  ✓ 已创建字典映射")
    
    # 获取当日行情
    print("\n[步骤3] 获取当日行情...")
    df_daily = api_call_with_retry(
        pro.daily,
        trade_date=trade_date,
        limit=API_CONFIG['limit']
    )
    
    if df_daily is None:
        print("  ❌ 无法获取行情数据")
        return
    
    print(f"  ✓ 获取到 {len(df_daily)} 只股票的行情数据")
    
    # 过滤
    print("\n[步骤4] 基础过滤...")
    
    # 排除科创/创业板/北交所
    print("  - 排除科创/创业板/北交所...")
    df = df_daily.copy()
    pattern = '^688|^300|^301|^43|^83|^87|^88|^BJ'
    df = df[~df['ts_code'].str.match(pattern, na=False)]
    print(f"    → 剩余 {len(df)} 只股票")
    
    # 排除ST股
    print("  - 排除ST股...")
    df['name'] = df['ts_code'].map(lambda x: stock_basic_dict.get(x, {}).get('name', ''))
    pattern = 'ST|\\*ST|退|退整理'
    df = df[~df['name'].str.contains(pattern, na=False)]
    print(f"    → 剩余 {len(df)} 只股票")
    
    # 涨幅筛选
    print("  - 涨幅 >= 5%...")
    df = df[df['pct_chg'] >= 5.0]
    print(f"    → 剩余 {len(df)} 只股票")
    
    # 价格筛选
    print("  - 价格 3-50 元...")
    df = df[(df['close'] >= 3) & (df['close'] <= 50)]
    print(f"    → 剩余 {len(df)} 只股票")
    
    # 合并基本信息
    print("\n[步骤5] 合并基本信息...")
    df.loc[:, 'industry'] = df['ts_code'].map(lambda x: stock_basic_dict.get(x, {}).get('industry', ''))
    df.loc[:, 'list_date'] = df['ts_code'].map(lambda x: stock_basic_dict.get(x, {}).get('list_date', ''))
    
    # 计算上市天数
    df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
    df = df[df['list_date'].notna()]
    df['list_days'] = (datetime.now() - df['list_date']).dt.days
    df = df[df['list_days'] >= 60]
    print(f"  → 剩余 {len(df)} 只股票")
    
    if len(df) == 0:
        print("\n  ⚠️  没有股票通过筛选")
        return
    
    # 获取每日指标
    print("\n[步骤6] 获取每日指标...")
    df_daily_basic = api_call_with_retry(
        pro.daily_basic,
        trade_date=trade_date,
        fields='ts_code,pe_ttm,total_mv,turnover_rate'
    )
    
    if df_daily_basic is None:
        print("  ⚠️  无法获取技术指标")
    else:
        df_daily_basic = df_daily_basic[df_daily_basic['ts_code'].isin(df['ts_code'].tolist())]
        print(f"  ✓ 获取到 {len(df_daily_basic)} 只股票的技术指标")
        
        # 合并数据
        cols_to_convert = [col for col in ['total_mv', 'pe_ttm', 'turnover_rate'] if col in df_daily_basic.columns]
        if cols_to_convert:
            df_daily_basic[cols_to_convert] = df_daily_basic[cols_to_convert].astype('float64')
        
        # 只合并不存在的列
        for col in ['total_mv', 'pe_ttm', 'turnover_rate']:
            if col in df_daily_basic.columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df_daily_basic.set_index('ts_code')[col])
                else:
                    df = df.merge(df_daily_basic[['ts_code', col]], on='ts_code', how='left')
        df['total_mv'] = df['total_mv'] / 10000
    
    # 换手率筛选
    print("  - 换手率 3-20%...")
    df = df[(df['turnover_rate'] >= 3) & (df['turnover_rate'] <= 20)]
    print(f"    → 剩余 {len(df)} 只股票")
    
    # 计算综合评分
    print("\n[步骤7] 计算综合评分...")
    df['score_pct_chg'] = (df['pct_chg'] / df['pct_chg'].max() * 100).fillna(0)
    
    if 'turnover_rate' in df.columns:
        df['score_turnover'] = (df['turnover_rate'] / df['turnover_rate'].max() * 100).fillna(0)
    else:
        df['score_turnover'] = 0
    
    df['score_volume'] = 50  # 默认值
    
    df['composite_score'] = (
        df['score_pct_chg'] * 0.4 +
        df['score_turnover'] * 0.3 +
        df['score_volume'] * 0.3
    )
    
    # 格式化输出
    print("\n[步骤8] 格式化输出...")
    output_cols = ['ts_code', 'name', 'industry', 'close', 'pct_chg', 
                   'turnover_rate', 'total_mv', 'pe_ttm', 'list_days', 'composite_score']
    
    available_cols = [col for col in output_cols if col in df.columns]
    df_output = df[available_cols].copy()
    
    col_mapping = {
        'ts_code': '代码',
        'name': '名称',
        'industry': '行业板块',
        'close': '收盘价',
        'pct_chg': '涨幅(%)',
        'turnover_rate': '换手率(%)',
        'total_mv': '市值(亿)',
        'pe_ttm': 'PE(TTM)',
        'list_days': '上市天数',
        'composite_score': '综合评分'
    }
    df_output.columns = [col_mapping.get(col, col) for col in df_output.columns]
    
    df_output = df_output.sort_values('综合评分', ascending=False)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("筛选结果")
    print("=" * 80)
    print(f"\n选股数量: {len(df_output)} 只\n")
    
    if len(df_output) > 0:
        print(df_output.to_string(index=False))
    else:
        print("未找到符合条件的股票")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
