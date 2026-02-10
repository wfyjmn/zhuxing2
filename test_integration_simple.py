#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单集成测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("简单集成测试")
print("="*80)
print(f"\n项目根目录: {project_root}")

# ==================== 测试 1: 导入检查 ====================
print("\n【测试 1】导入检查")

try:
    from strategy_manager import Config, MarketStateDetector, ScreenerAdapter, SimpleBacktestEngine
    print("✅ 策略管理器模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ==================== 测试 2: 配置初始化 ====================
print("\n【测试 2】配置初始化")

try:
    config = Config()
    print(f"✅ 配置初始化成功")
    print(f"   数据目录: {config.data_dir}")
    print(f"   输出目录: {config.output_dir}")
except Exception as e:
    print(f"❌ 配置初始化失败: {e}")
    sys.exit(1)

# ==================== 测试 3: 市场状态检测 ====================
print("\n【测试 3】市场状态检测")

try:
    detector = MarketStateDetector(config)
    print("✅ 市场状态检测器初始化成功")

    # 尝试检测
    print("\n尝试检测市场状态...")
    market_info = detector.detect_market_state()
    print(f"✅ 市场状态检测成功")
    print(f"   状态: {market_info['state']}")
    print(f"   描述: {market_info['description']}")
except Exception as e:
    print(f"⚠️  市场状态检测失败（可能需要Token）: {e}")

# ==================== 测试 4: 选股适配器 ====================
print("\n【测试 4】选股适配器")

try:
    adapter = ScreenerAdapter(config)
    print("✅ 选股适配器初始化成功")
except Exception as e:
    print(f"❌ 选股适配器初始化失败: {e}")
    sys.exit(1)

# ==================== 测试 5: 回测引擎 ====================
print("\n【测试 5】回测引擎")

try:
    engine = SimpleBacktestEngine(config)
    print("✅ 回测引擎初始化成功")
except Exception as e:
    print(f"❌ 回测引擎初始化失败: {e}")
    sys.exit(1)

# ==================== 测试总结 ====================
print("\n" + "="*80)
print("测试总结")
print("="*80)
print("\n✅ 所有核心模块初始化成功！")
print("\n建议:")
print("  1. 配置 Tushare Token（如果还没配置）")
print("  2. 运行完整的测试: python strategy_manager/demo_adapter.py")
print("  3. 查看集成指南: assets/INTEGRATION_GUIDE.md")
print("="*80)
