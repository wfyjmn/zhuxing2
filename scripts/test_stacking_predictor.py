#!/usr/bin/env python3
"""
集成预测器测试脚本
测试 StackignEnsemblePredictor 的功能
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.data_collector import MarketDataCollector
from stock_system.enhanced_features import EnhancedFeatureEngineer
from stock_system.stacking_predictor import StackingEnsemblePredictor


def main():
    """主函数"""
    print("=" * 70)
    print("集成预测器测试")
    print("=" * 70)
    
    # 加载最新模型
    models_dir = os.path.join(workspace_path, "assets/models")
    model_files = sorted([f for f in os.listdir(models_dir) if f.startswith('stacking_ensemble') and f.endswith('.pkl')])
    
    if not model_files:
        print("❌ 未找到集成模型文件")
        return
    
    model_path = os.path.join(models_dir, model_files[-1])
    print(f"\n加载模型: {model_files[-1]}")
    
    # 加载优化后的阈值
    results_dir = os.path.join(workspace_path, "assets/results")
    opt_files = sorted([f for f in os.listdir(results_dir) if f.startswith('threshold_optimization') and f.endswith('.json')])
    
    if opt_files:
        import json
        with open(os.path.join(results_dir, opt_files[-1]), 'r', encoding='utf-8') as f:
            opt_data = json.load(f)
        threshold = opt_data.get('best_threshold', 0.5)
        metrics = opt_data.get('metrics', {})
        print(f"使用优化后的阈值: {threshold:.4f}")
        print(f"  精确率: {metrics.get('precision', 0):.2%}")
        print(f"  召回率: {metrics.get('recall', 0):.2%}")
    else:
        threshold = 0.5
        print(f"使用默认阈值: {threshold}")
    
    # 初始化预测器
    predictor = StackingEnsemblePredictor(model_path, threshold=threshold)
    
    # 打印模型信息
    print("\n" + "-" * 70)
    print("模型信息:")
    model_info = predictor.get_model_info()
    print(f"  基学习器: {model_info['base_models']}")
    print(f"  元学习器: {model_info['meta_model']}")
    print(f"  特征集大小: {model_info['feature_sets']}")
    
    # 加载测试数据
    print("\n" + "-" * 70)
    print("加载测试数据...")
    collector = MarketDataCollector()
    engineer = EnhancedFeatureEngineer()
    
    # 获取少量股票进行测试
    stock_codes = collector.get_stock_pool_tree(
        pool_size=100,
        exclude_markets=['BJ'],
        exclude_board_types=['688', '300', '301']
    )
    
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    all_data = []
    print(f"⏳ 正在采集数据...")
    print(f"  股票池: {len(stock_codes)} 只股票")
    
    for idx, code in enumerate(stock_codes):
        try:
            df = collector.get_daily_data(code, start_date, end_date)
            
            if df is None or len(df) < 60:
                print(f"  ⚠️ {code}: 数据不足 ({len(df) if df is not None else 0} 条)")
                continue
            
            df_feat = engineer.create_all_features(df)
            
            if df_feat is None or len(df_feat) == 0:
                print(f"  ⚠️ {code}: 特征工程失败")
                continue
            
            # 添加股票代码
            df_feat['ts_code'] = code
            all_data.append(df_feat)
            
            if (idx + 1) % 5 == 0:
                print(f"  进度: {idx + 1}/{len(stock_codes)}, 成功: {len(all_data)}")
            
            # 获取至少5只股票的数据
            if len(all_data) >= 5:
                break
            
        except Exception as e:
            print(f"  ✗ {code}: {e}")
            continue
    
    if len(all_data) == 0:
        print("❌ 未获取到任何数据")
        return
    
    import pandas as pd
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ 数据加载完成: {len(df)} 条记录")
    
    # 测试预测功能
    print("\n" + "-" * 70)
    print("测试预测功能...")
    result_df = predictor.predict_signal(df)
    
    print(f"✓ 预测完成")
    print(f"  总样本数: {len(result_df)}")
    print(f"  平均概率: {result_df['prob'].mean():.4f}")
    print(f"  最大概率: {result_df['prob'].max():.4f}")
    print(f"  最小概率: {result_df['prob'].min():.4f}")
    
    # 测试信号统计
    print("\n" + "-" * 70)
    print("信号统计:")
    stats = predictor.get_signal_stats(df)
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  信号分布:")
    for signal_id, signal_name in stats['signal_names'].items():
        count = stats['signal_distribution'].get(signal_id, 0)
        percentage = count / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
        print(f"    {signal_name} ({signal_id}): {count} ({percentage:.2f}%)")
    
    # 测试选股功能
    print("\n" + "-" * 70)
    print("测试选股功能...")
    selected_stocks = predictor.select_stocks(df, min_signal=2, top_n=5)
    
    print(f"✓ 选股完成: {len(selected_stocks)} 只股票")
    print("\n前 5 只推荐股票:")
    for idx, row in selected_stocks.head(5).iterrows():
        signal_name = stats['signal_names'].get(row['signal'], '未知')
        print(f"  {row['ts_code']}: 概率={row['prob']:.4f}, 信号={signal_name}")
    
    # 测试不同信号等级
    print("\n" + "-" * 70)
    print("测试不同信号等级:")
    for min_signal in [2, 3]:
        selected = predictor.select_stocks(df, min_signal=min_signal)
        signal_name = stats['signal_names'].get(min_signal, '未知')
        print(f"  {signal_name}及以上: {len(selected)} 只股票")
    
    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
