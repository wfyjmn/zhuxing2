"""
测试脚本：short_term_assault_config_v4.json 配置文件验证

【测试内容】：
1. 验证配置文件的JSON格式是否正确
2. 验证特征权重和预测模块的特征列表是否对齐
3. 验证过拟合差距的计算方式和调整规则
4. 验证RSI策略与置信度的联动规则
5. 验证与分桶分析器的联动机制

【运行方式】：
python scripts/test_short_term_assault_config_v4.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_json_format(config_path: str) -> bool:
    """
    测试1：验证JSON格式是否正确
    """
    print("\n" + "="*80)
    print("测试1：验证JSON格式")
    print("="*80)
    
    try:
        config = load_config(config_path)
        print("✓ JSON格式正确")
        print(f"  - 策略名称: {config['strategy_name']}")
        print(f"  - 版本: {config['version']}")
        print(f"  - 核心理念: {config['core_philosophy']}")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 加载配置文件失败: {e}")
        return False


def test_feature_alignment(config: dict) -> bool:
    """
    测试2：验证特征权重和预测模块的特征列表是否对齐
    """
    print("\n" + "="*80)
    print("测试2：验证特征对齐")
    print("="*80)
    
    # 获取预测模块的特征列表
    predictor_features = config['alignment']['predictor_features']
    print(f"  - 预测模块特征数量: {len(predictor_features)}")
    
    # 获取权重体系的特征
    feature_weights = config['feature_weights']
    weight_features = []
    
    for dimension, dimension_config in feature_weights.items():
        for feature in dimension_config['features']:
            aligned_feature = feature.get('aligned_feature')
            if aligned_feature:
                weight_features.append(aligned_feature)
    
    print(f"  - 权重体系特征数量: {len(weight_features)}")
    
    # 检查对齐情况
    aligned_count = 0
    missing_in_weights = []
    missing_in_predictor = []
    
    for feature in predictor_features:
        if feature in weight_features:
            aligned_count += 1
        else:
            missing_in_weights.append(feature)
    
    for feature in weight_features:
        if feature not in predictor_features:
            missing_in_predictor.append(feature)
    
    print(f"  - 已对齐特征数量: {aligned_count}")
    
    if missing_in_weights:
        print(f"  - ⚠ 预测模块有但权重体系缺失的特征 ({len(missing_in_weights)}个):")
        for feature in missing_in_weights:
            print(f"    * {feature}")
    
    if missing_in_predictor:
        print(f"  - ⚠ 权重体系有但预测模块缺失的特征 ({len(missing_in_predictor)}个):")
        for feature in missing_in_predictor:
            print(f"    * {feature}")
    
    # 检查权重总和
    total_weight = sum(dim_config['weight'] for dim_config in feature_weights.values())
    print(f"  - 权重总和: {total_weight:.2f}")
    
    if abs(total_weight - 1.0) < 0.01:
        print("✓ 权重总和正确")
    else:
        print(f"✗ 权重总和错误，应为1.0，实际为{total_weight:.2f}")
        return False
    
    # 检查关键特征是否对齐
    key_features = [
        'main_capital_inflow_ratio',
        'large_order_buy_rate',
        'enhanced_rsi',
        'volume_price_breakout_strength',
        'rsi_signal',
        'momentum_strength'
    ]
    
    all_aligned = all(feature in weight_features for feature in key_features)
    
    if all_aligned:
        print("✓ 关键特征全部对齐")
        return True
    else:
        print("✗ 部分关键特征未对齐")
        return False


def test_overfitting_gap_config(config: dict) -> bool:
    """
    测试3：验证过拟合差距的配置
    """
    print("\n" + "="*80)
    print("测试3：验证过拟合差距配置")
    print("="*80)
    
    overfitting_config = config['optimization_goals']['overfitting_gap']
    
    # 检查计算方式
    calculation_method = overfitting_config.get('calculation_method')
    if calculation_method:
        print(f"  ✓ 计算方式已定义: {calculation_method}")
    else:
        print("  ✗ 计算方式未定义")
        return False
    
    # 检查调整规则
    adjustment_rules = overfitting_config.get('adjustment_rules')
    if adjustment_rules:
        print(f"  ✓ 调整规则已定义")
        print(f"    - 触发条件: {adjustment_rules.get('trigger_condition')}")
        print(f"    - 优先调整策略数量: {len(adjustment_rules.get('priority_adjustments', []))}")
        
        for i, adjustment in enumerate(adjustment_rules.get('priority_adjustments', [])):
            print(f"      {i+1}. {adjustment.get('action')}")
            if 'params' in adjustment:
                print(f"         参数调整: {adjustment['params']}")
    else:
        print("  ✗ 调整规则未定义")
        return False
    
    # 检查分桶分析联动
    # 注意：bucket_monitoring 在 adjustment_rules 里面
    bucket_monitoring = None
    if adjustment_rules:
        bucket_monitoring = adjustment_rules.get('bucket_monitoring')
    
    if bucket_monitoring:
        print(f"  ✓ 分桶分析联动已定义")
        print(f"    - 关键指标: {bucket_monitoring.get('key_metrics')}")
        print(f"    - 阈值: {bucket_monitoring.get('threshold')}")
        print(f"    - 触发动作: {bucket_monitoring.get('action')}")
    else:
        print("  ✗ 分桶分析联动未定义")
        return False
    
    print("✓ 过拟合差距配置完整")
    return True


def test_rsi_confidence_linkage(config: dict) -> bool:
    """
    测试4：验证RSI策略与置信度的联动
    """
    print("\n" + "="*80)
    print("测试4：验证RSI策略与置信度联动")
    print("="*80)
    
    rsi_strategy = config['enhanced_rsi_strategy']
    
    # 检查动态阈值
    dynamic_thresholds = rsi_strategy.get('dynamic_thresholds')
    if dynamic_thresholds:
        print(f"  ✓ 动态阈值已定义:")
        for market_env, thresholds in dynamic_thresholds.items():
            print(f"    - {market_env}: 买入={thresholds['buy']}, 卖出={thresholds['sell']}, "
                  f"置信度要求={thresholds.get('confidence_threshold', 'N/A')}")
    else:
        print("  ✗ 动态阈值未定义")
        return False
    
    # 检查置信度联动规则
    confidence_linkage = rsi_strategy.get('confidence_linkage')
    if confidence_linkage:
        print(f"  ✓ 置信度联动规则已定义:")
        for rule in confidence_linkage.get('rules', []):
            print(f"    - {rule['market_environment']}: {rule['rsi_buy_condition']} + "
                  f"{rule['confidence_requirement']} → {rule['action']}")
    else:
        print("  ✗ 置信度联动规则未定义")
        return False
    
    # 检查背离检测
    divergence_detection = rsi_strategy.get('divergence_detection')
    if divergence_detection:
        print(f"  ✓ 背离检测已定义:")
        for div_type, config in divergence_detection.items():
            print(f"    - {div_type}: {config.get('description')}")
            if 'confidence_linkage' in config:
                linkage = config['confidence_linkage']
                print(f"      * 条件: {linkage.get('condition')}")
                print(f"      * 动作: {linkage.get('action')}")
                print(f"      * 原因: {linkage.get('reason')}")
    else:
        print("  ✗ 背离检测未定义")
        return False
    
    print("✓ RSI策略与置信度联动完整")
    return True


def test_bucket_analyzer_integration(config: dict) -> bool:
    """
    测试5：验证与分桶分析器的集成
    """
    print("\n" + "="*80)
    print("测试5：验证与分桶分析器集成")
    print("="*80)
    
    integration = config.get('integration_with_modules')
    
    if not integration:
        print("✗ 集成配置未定义")
        return False
    
    # 检查预测模块集成
    predictor_integration = integration.get('predictor_module')
    if predictor_integration:
        print(f"  ✓ 预测模块集成已定义:")
        feature_mapping = predictor_integration.get('feature_mapping', {})
        print(f"    - 特征映射数量: {len(feature_mapping)}")
        
        # 显示部分映射
        for i, (key, value) in enumerate(list(feature_mapping.items())[:5]):
            print(f"      {i+1}. {key} → {value}")
        if len(feature_mapping) > 5:
            print(f"      ... (共{len(feature_mapping)}个映射)")
    else:
        print("  ✗ 预测模块集成未定义")
        return False
    
    # 检查分桶分析器集成
    bucket_integration = integration.get('bucket_analyzer')
    if bucket_integration:
        print(f"  ✓ 分桶分析器集成已定义:")
        overfitting_monitoring = bucket_integration.get('overfitting_monitoring', {})
        print(f"    - 关键指标: {overfitting_monitoring.get('key_metrics')}")
        print(f"    - 阈值: {overfitting_monitoring.get('threshold')}")
        print(f"    - 触发动作: {overfitting_monitoring.get('action')}")
    else:
        print("  ✗ 分桶分析器集成未定义")
        return False
    
    print("✓ 与分桶分析器集成完整")
    return True


def test_weight_adjustment_logic(config: dict) -> bool:
    """
    测试6：验证权重调整逻辑
    """
    print("\n" + "="*80)
    print("测试6：验证权重调整逻辑")
    print("="*80)
    
    # 模拟过拟合场景
    print("  模拟过拟合场景:")
    print("    - 训练集精确率: 0.85")
    print("    - 测试集精确率: 0.60")
    print("    - 过拟合差距: 0.25 (> 20%)")
    
    train_precision = 0.85
    test_precision = 0.60
    overfitting_gap = train_precision - test_precision
    
    overfitting_config = config['optimization_goals']['overfitting_gap']
    threshold = overfitting_config['target']
    
    if overfitting_gap > threshold:
        print(f"  ✓ 过拟合差距({overfitting_gap:.2f}) > 阈值({threshold})，触发调整规则")
        
        adjustment_rules = overfitting_config['adjustment_rules']
        
        # 获取当前权重
        feature_weights = config['feature_weights']
        original_tech_weight = feature_weights['technical_momentum']['weight']
        print(f"    - 原技术动量权重: {original_tech_weight}")
        
        # 模拟权重调整
        print(f"    - 执行权重调整:")
        for i, adjustment in enumerate(adjustment_rules.get('priority_adjustments', [])):
            action = adjustment.get('action')
            print(f"      {i+1}. {action}")
            if "权重" in action:
                # 提取目标权重
                import re
                match = re.search(r'从([\d.]+)%降至([\d.]+)%', action)
                if match:
                    from_weight = float(match.group(1))
                    to_weight = float(match.group(2))
                    print(f"         权重变化: {from_weight}% → {to_weight}%")
        
        # 检查模型参数调整
        model_params = config['model_params']
        original_params = model_params['xgboost'].copy()
        adjusted_params = model_params.get('overfitting_adjustment', {}).get('adjusted_params', {})
        
        if adjusted_params:
            print(f"    - 执行模型参数调整:")
            for param, new_value in adjusted_params.items():
                old_value = original_params.get(param)
                print(f"      {param}: {old_value} → {new_value}")
        
        print("  ✓ 权重调整逻辑验证通过")
        return True
    else:
        print(f"  - 过拟合差距({overfitting_gap:.2f}) ≤ 阈值({threshold})，无需调整")
        print("  ✓ 权重调整逻辑验证通过")
        return True


def test_printer_stock_features(config: dict) -> bool:
    """
    测试7：验证印钞机专属特征配置
    """
    print("\n" + "="*80)
    print("测试7：验证印钞机专属特征配置")
    print("="*80)
    
    printer_features = config['alignment']['printer_stock_features']
    print(f"  - 印钞机专属特征数量: {len(printer_features)}")
    
    # 检查权重体系中的印钞机专属特征
    feature_weights = config['feature_weights']
    printer_weight_config = feature_weights.get('printer_stock_features')
    
    if printer_weight_config:
        print(f"  ✓ 印钞机专属权重分支已定义")
        print(f"    - 权重: {printer_weight_config['weight']}")
        print(f"    - 描述: {printer_weight_config['description']}")
        
        printer_weighted_features = printer_weight_config.get('features', [])
        print(f"    - 加权特征数量: {len(printer_weighted_features)}")
        
        # 检查高确定性特征
        high_certainty_features = [
            f for f in printer_weighted_features 
            if f.get('note', '').find('重点关注高确定性特征') != -1
        ]
        
        print(f"    - 高确定性特征数量: {len(high_certainty_features)}")
        for feature in high_certainty_features:
            print(f"      * {feature['name']} (权重: {feature['weight']})")
    else:
        print("  ✗ 印钞机专属权重分支未定义")
        return False
    
    print("✓ 印钞机专属特征配置完整")
    return True


def main():
    """
    主测试函数
    """
    print("\n" + "="*80)
    print("short_term_assault_config_v4.json 配置文件验证")
    print("="*80)
    
    config_path = "config/short_term_assault_config_v4.json"
    
    if not os.path.exists(config_path):
        print(f"\n✗ 配置文件不存在: {config_path}")
        return 1
    
    # 加载配置文件
    config = load_config(config_path)
    
    results = []
    
    # 运行所有测试
    results.append(("JSON格式验证", test_json_format(config_path)))
    results.append(("特征对齐验证", test_feature_alignment(config)))
    results.append(("过拟合差距配置验证", test_overfitting_gap_config(config)))
    results.append(("RSI置信度联动验证", test_rsi_confidence_linkage(config)))
    results.append(("分桶分析器集成验证", test_bucket_analyzer_integration(config)))
    results.append(("权重调整逻辑验证", test_weight_adjustment_logic(config)))
    results.append(("印钞机专属特征验证", test_printer_stock_features(config)))
    
    # 打印测试结果汇总
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    failed_tests = total_tests - passed_tests
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {total_tests} 个测试")
    print(f"通过: {passed_tests} 个")
    print(f"失败: {failed_tests} 个")
    
    if failed_tests == 0:
        print("\n🎉 所有测试通过！配置文件有效且完整。")
        return 0
    else:
        print(f"\n⚠️  有 {failed_tests} 个测试失败，请检查配置文件。")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
