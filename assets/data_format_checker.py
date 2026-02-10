#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式检查工具
================

功能：检查选股数据格式是否符合策略管理器适配器的要求

必需字段：
- ts_code: 股票代码
- name: 股票名称
- close: 收盘价
- pct_chg: 涨跌幅(%)
- turnover_rate: 换手率(%)
- volume_ratio: 量比

可选字段：
- industry: 行业
- pe_ttm: 市盈率
- pb: 市净率
- roe: 净资产收益率
- total_mv: 总市值
"""

import pandas as pd
import sys
from pathlib import Path


def check_data_format(file_path, verbose=False):
    """
    检查数据文件格式

    Args:
        file_path: 数据文件路径
        verbose: 是否显示详细信息

    Returns:
        检查结果字典
    """
    print("\n" + "="*80)
    print("数据格式检查工具")
    print("="*80)
    print(f"\n检查文件: {file_path}")

    # 必需字段
    REQUIRED_FIELDS = [
        'ts_code',
        'name',
        'close',
        'pct_chg',
        'turnover_rate',
        'volume_ratio'
    ]

    # 可选字段
    OPTIONAL_FIELDS = [
        'industry',
        'pe_ttm',
        'pb',
        'roe',
        'total_mv',
        'dv_ratio',
        'revenue_yoy',
        'profit_yoy',
        'trade_date',
        'open',
        'high',
        'low'
    ]

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'missing_fields': [],
        'extra_fields': [],
        'field_types': {},
        'row_count': 0
    }

    # 检查文件是否存在
    if not Path(file_path).exists():
        results['valid'] = False
        results['errors'].append(f"文件不存在: {file_path}")
        return results

    # 读取文件
    try:
        df = pd.read_csv(file_path, encoding='utf_8_sig')
        results['row_count'] = len(df)

        print(f"\n✅ 文件读取成功")
        print(f"   总行数: {results['row_count']}")

        if df.empty:
            results['valid'] = False
            results['errors'].append("数据为空")
            return results

    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"文件读取失败: {e}")
        return results

    # 检查必需字段
    print(f"\n📋 检查必需字段:")
    for field in REQUIRED_FIELDS:
        if field in df.columns:
            print(f"   ✅ {field}")
            # 记录字段类型
            dtype = str(df[field].dtype)
            results['field_types'][field] = dtype

            # 检查是否有缺失值
            missing = df[field].isna().sum()
            if missing > 0:
                results['warnings'].append(f"字段 '{field}' 有 {missing} 个缺失值")
        else:
            print(f"   ❌ {field} (缺失)")
            results['missing_fields'].append(field)
            results['valid'] = False

    # 检查可选字段
    print(f"\n📋 检查可选字段:")
    for field in OPTIONAL_FIELDS:
        if field in df.columns:
            print(f"   ✅ {field}")
            dtype = str(df[field].dtype)
            results['field_types'][field] = dtype

            missing = df[field].isna().sum()
            if missing > 0:
                results['warnings'].append(f"字段 '{field}' 有 {missing} 个缺失值")

    # 检查额外字段
    extra_fields = [col for col in df.columns if col not in REQUIRED_FIELDS + OPTIONAL_FIELDS]
    if extra_fields:
        print(f"\n📋 额外字段:")
        for field in extra_fields:
            print(f"   ℹ️  {field}")
        results['extra_fields'] = extra_fields

    # 数据类型检查
    print(f"\n📊 数据类型检查:")
    numeric_fields = ['close', 'pct_chg', 'turnover_rate', 'volume_ratio']
    for field in numeric_fields:
        if field in df.columns:
            if pd.api.types.is_numeric_dtype(df[field]):
                print(f"   ✅ {field}: {df[field].dtype}")
            else:
                print(f"   ⚠️  {field}: {df[field].dtype} (建议为数值类型)")
                results['warnings'].append(f"字段 '{field}' 不是数值类型")

    # 统计信息
    if verbose and results['valid']:
        print(f"\n📈 数据统计:")
        for field in ['close', 'pct_chg', 'turnover_rate', 'volume_ratio']:
            if field in df.columns:
                print(f"   {field}:")
                print(f"     最小值: {df[field].min():.2f}")
                print(f"     最大值: {df[field].max():.2f}")
                print(f"     平均值: {df[field].mean():.2f}")
                print(f"     中位数: {df[field].median():.2f}")

    # 数据质量检查
    print(f"\n🔍 数据质量检查:")

    # 检查 ST 股
    if 'name' in df.columns:
        st_count = df['name'].str.contains('ST|退', na=False).sum()
        if st_count > 0:
            print(f"   ⚠️  发现 {st_count} 只 ST/退市股")
            results['warnings'].append(f"数据包含 {st_count} 只 ST/退市股")

    # 检查异常值
    if 'pct_chg' in df.columns:
        extreme_count = (abs(df['pct_chg']) > 20).sum()
        if extreme_count > 0:
            print(f"   ℹ️  发现 {extreme_count} 只极端涨跌幅股票(>20%)")

    if 'turnover_rate' in df.columns:
        high_turnover = (df['turnover_rate'] > 50).sum()
        if high_turnover > 0:
            print(f"   ℹ️  发现 {high_turnover} 只超高换手率股票(>50%)")

    # 返回结果
    return results


def print_summary(results):
    """打印检查结果摘要"""
    print("\n" + "="*80)
    print("检查结果摘要")
    print("="*80)

    if results['valid']:
        print("\n✅ 数据格式检查通过")
    else:
        print("\n❌ 数据格式检查失败")

    print(f"\n总行数: {results['row_count']}")

    if results['errors']:
        print(f"\n❌ 错误 ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"   - {error}")

    if results['warnings']:
        print(f"\n⚠️  警告 ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"   - {warning}")

    if results['missing_fields']:
        print(f"\n❌ 缺失字段 ({len(results['missing_fields'])}):")
        for field in results['missing_fields']:
            print(f"   - {field}")

    if results['extra_fields']:
        print(f"\nℹ️  额外字段 ({len(results['extra_fields'])}):")
        for field in results['extra_fields']:
            print(f"   - {field}")

    print("\n" + "="*80)

    # 修复建议
    if not results['valid']:
        print("\n💡 修复建议:")
        for field in results['missing_fields']:
            print(f"   - 添加字段: {field}")
        print("\n示例:")
        print("""
        # 确保数据包含以下字段
        required_columns = ['ts_code', 'name', 'close', 'pct_chg', 'turnover_rate', 'volume_ratio']

        # 如果缺失，添加默认值
        for col in required_columns:
            if col not in df.columns:
                if col in ['close']:
                    df[col] = 0.0
                elif col in ['pct_chg', 'turnover_rate', 'volume_ratio']:
                    df[col] = 0.0
                else:
                    df[col] = ""
        """)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="数据格式检查工具")
    parser.add_argument('file', help='数据文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')

    args = parser.parse_args()

    results = check_data_format(args.file, args.verbose)
    print_summary(results)

    return 0 if results['valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
