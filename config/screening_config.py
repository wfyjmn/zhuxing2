#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置文件 - 选股系统
======================

说明：
1. 所有选股程序的配置参数统一管理
2. 便于调试、修改和维护
3. 支持不同场景的配置切换

作者：Coze Coding - Agent搭建专家
版本：1.0
日期：2026-02-10
"""

# ==================== API调用配置 ====================
API_CONFIG = {
    # 重试配置
    'retry_times': 3,           # API调用重试次数
    'retry_delay': 1,           # 重试间隔（秒）
    'request_delay': 0.5,       # 请求间隔（秒），避免频率限制
    
    # 批量获取配置
    'batch_size': 500,          # 批量获取股票数量
    'limit': 5000,              # 单次请求的上限数量
}

# ==================== 选股B配置 ====================
SCREENER_B_CONFIG = {
    # 基础筛选参数
    'min_pct_chg': 5.0,          # 最低涨幅（%）
    'min_list_days': 60,         # 最少上市天数
    'ban_ratio_threshold': 0.5,  # 解禁比例阈值（%）
    'solo_buy_threshold': 0.15,  # 龙虎榜买一独食阈值（%）
    
    # 历史涨停相关参数
    'same_price_pct_min': 9.0,   # 历史涨停涨幅阈值（%）
    'same_price_pct_next': -3.0, # 历史涨停次日跌幅阈值（%）
    
    # 价格筛选参数
    'price_min': 3,              # 最低价格（元）
    'price_max': 50,             # 最高价格（元）
    
    # 换手率筛选参数
    'turnover_min': 3,           # 最小换手率（%）
    'turnover_max': 20,          # 最大换手率（%）
    
    # 成交量倍数参数
    'volume_ratio_min': 1.5,     # 最小成交量倍数
    
    # 均线参数
    'ma5_days': 5,               # 5日均线天数
    'ma10_days': 10,             # 10日均线天数
    
    # 止损止盈参数
    'stop_loss_pct': 5.0,        # 止损百分比（%）
    'stop_loss_ma': True,        # 是否使用5日均线止损
    'take_profit_min': 10.0,     # 最低止盈百分比（%）
    'take_profit_max': 15.0,     # 最高止盈百分比（%）
    'take_profit_avg': 12.5,     # 平均止盈百分比（%）
    
    # 股价位置检查
    'check_price_position': True, # 是否检查股价位置
    'check_ma5': True,           # 是否检查5日均线
    'check_ma10': True,          # 是否检查10日均线
    
    # 历史数据获取参数
    'history_days': 30,          # 历史数据获取天数（用于计算均线和成交量倍数）
    
    # 交易日历查询参数
    'trade_cal_days': 10,        # 查询最近多少天的交易日历
    
    # 默认值参数
    'default_volume_ratio': 1.0,    # 默认成交量倍数
    'default_turnover_rate': 0.0,   # 默认换手率
    'default_list_days': 999,      # 默认上市天数
    'default_value': 0,             # 默认值
}

# ==================== 选股C配置 ====================
SCREENER_C_CONFIG = {
    # 市场状态判断参数
    'ma_days': 20,               # 均线天数（判断市场状态）
    'bull_market_ratio': 0.6,    # 牛市阈值（上涨股票比例）
    'bear_market_ratio': 0.3,    # 熊市阈值（下跌股票比例）
    
    # 基础筛选参数
    'min_pct_chg': 5.0,          # 最低涨幅（%）
    'price_min': 3,              # 最低价格（元）
    'price_max': 50,             # 最高价格（元）
    'turnover_min': 3,           # 最小换手率（%）
    'turnover_max': 20,          # 最大换手率（%）
    'min_list_days': 60,         # 最少上市天数
    
    # 风险过滤参数
    'limit_down_window': 30,     # 跌停时间窗口（天）
    'solo_buy_threshold': 0.15,  # 龙虎榜买一独食阈值
    
    # 解禁参数
    'unlift_days': 30,           # 解禁查询周期（天）
    
    # 评分权重
    'weight_pct_chg': 0.4,       # 涨幅权重
    'weight_turnover': 0.3,      # 换手率权重
    'weight_volume': 0.3,       # 成交量倍数权重
    
    # 历史数据获取参数
    'limit_down_history_days': 30,  # 跌停检查历史天数
    
    # 指数数据获取参数
    'index_history_days': 40,    # 指数历史数据获取天数
    
    # 交易日历查询参数
    'trade_cal_days': 10,        # 查询最近多少天的交易日历
    
    # 默认值参数
    'default_volume_ratio': 1.0,    # 默认成交量倍数
    'default_turnover_rate': 0.0,   # 默认换手率
    'default_list_days': 999,      # 默认上市天数
    'default_value': 0,             # 默认值
}

# ==================== 股票过滤配置 ====================
FILTER_CONFIG = {
    # 排除前缀（科创板、创业板、北交所）
    # 300: 创业板
    # 301: 创业板
    # 688: 科创板
    # 8: 北交所
    # 4: 北交所
    # 920: 北交所
    'exclude_prefix': ['300', '301', '688', '8', '4', '920'],
    
    # 排除股票名称中的风险关键词
    'exclude_name_keywords': ['ST', '*ST', '退', '退整理'],
    
    # 跌停判断参数
    'limit_down_threshold': -9.5,  # 跌停阈值（%）
    
    # 涨停判断参数
    'limit_up_threshold': 9.5,     # 涨停阈值（%）
}

# ==================== 输出配置 ====================
OUTPUT_CONFIG = {
    # CSV输出配置
    'encoding': 'utf_8_sig',      # CSV文件编码（Excel兼容）
    'index': False,               # 是否写入行索引
    
    # 评分配置
    'score_max': 100,             # 评分最大值
    
    # 显示配置
    'display_max_rows': 100,      # 最大显示行数
    'display_width': 80,          # 显示宽度
}

# ==================== 市场指数配置 ====================
INDEX_CONFIG = {
    # 主要指数代码
    'sh_index': '000001.SH',      # 上证指数
    'sz_index': '399001.SZ',      # 深证成指
    'cyb_index': '399006.SZ',     # 创业板指
    'kc_index': '000688.SH',      # 科创50
    
    # 默认使用的指数
    'default_index': '000001.SH',
}

# ==================== 文件路径配置 ====================
PATH_CONFIG = {
    # 输出文件路径
    'output_dir': 'assets/data',
    'log_dir': 'logs',
    
    # 文件命名格式
    'date_format': '%Y%m%d',
}

# ==================== 实盘统计配置 ====================
BACKTEST_CONFIG = {
    # 实盘统计数据
    'blacklisted_avg_return': -1.98,    # 被拉黑股票次日平均收益（%）
    'safe_avg_return': 1.27,             # 安全股票次日平均收益（%）
    'profit_diff': 3.25,                 # 差值（%）
    
    # 回测参数
    'backtest_days': 252,               # 回测天数（一年）
    'commission_rate': 0.0003,           # 手续费率（万三）
    'slippage': 0.001,                   # 滑点（0.1%）
}

# ==================== 配置验证函数 ====================
def validate_config():
    """
    验证配置参数的合理性
    """
    errors = []
    
    # 验证API配置
    if API_CONFIG['retry_times'] < 1:
        errors.append("API配置: retry_times 必须大于0")
    if API_CONFIG['batch_size'] < 1:
        errors.append("API配置: batch_size 必须大于0")
    
    # 验证选股B配置
    if SCREENER_B_CONFIG['min_pct_chg'] < 0:
        errors.append("选股B配置: min_pct_chg 必须大于等于0")
    if SCREENER_B_CONFIG['price_min'] >= SCREENER_B_CONFIG['price_max']:
        errors.append("选股B配置: price_min 必须小于 price_max")
    if SCREENER_B_CONFIG['turnover_min'] >= SCREENER_B_CONFIG['turnover_max']:
        errors.append("选股B配置: turnover_min 必须小于 turnover_max")
    
    # 验证选股C配置
    if SCREENER_C_CONFIG['min_pct_chg'] < 0:
        errors.append("选股C配置: min_pct_chg 必须大于等于0")
    if SCREENER_C_CONFIG['price_min'] >= SCREENER_C_CONFIG['price_max']:
        errors.append("选股C配置: price_min 必须小于 price_max")
    
    # 验证权重总和
    total_weight = (SCREENER_C_CONFIG['weight_pct_chg'] + 
                   SCREENER_C_CONFIG['weight_turnover'] + 
                   SCREENER_C_CONFIG['weight_volume'])
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"选股C配置: 权重总和应该为1.0，当前为{total_weight}")
    
    return errors


# ==================== 配置切换函数 ====================
def get_config(screener_type='B'):
    """
    获取指定选股程序的配置
    
    Args:
        screener_type: 选股类型 ('B' 或 'C')
    
    Returns:
        dict: 配置字典
    """
    config = {
        'api': API_CONFIG,
        'filter': FILTER_CONFIG,
        'output': OUTPUT_CONFIG,
        'index': INDEX_CONFIG,
        'path': PATH_CONFIG,
    }
    
    if screener_type == 'B':
        config.update({'screener': SCREENER_B_CONFIG})
    elif screener_type == 'C':
        config.update({'screener': SCREENER_C_CONFIG})
    else:
        raise ValueError(f"不支持的选股类型: {screener_type}")
    
    return config


# ==================== 配置打印函数 ====================
def print_config(screener_type='B'):
    """
    打印指定选股程序的配置
    
    Args:
        screener_type: 选股类型 ('B' 或 'C')
    """
    config = get_config(screener_type)
    
    print("=" * 80)
    print(f"选股{'B' if screener_type == 'B' else 'C'}配置")
    print("=" * 80)
    
    print("\n【API配置】")
    for key, value in config['api'].items():
        print(f"  {key}: {value}")
    
    print(f"\n【选股{'B' if screener_type == 'B' else 'C'}配置】")
    for key, value in config['screener'].items():
        print(f"  {key}: {value}")
    
    print("\n【过滤配置】")
    for key, value in config['filter'].items():
        print(f"  {key}: {value}")
    
    print("=" * 80)


if __name__ == '__main__':
    # 验证配置
    errors = validate_config()
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("配置验证通过")
    
    # 打印配置
    print()
    print_config('B')
    print()
    print_config('C')
