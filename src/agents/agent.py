"""
自动化阈值优化Agent
基于历史数据的统计分布与模型表现优化，实现数据驱动的精准阈值调整
"""

import os
import json
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from coze_coding_utils.runtime_ctx.context import default_headers
from storage.memory.memory_saver import get_memory_saver

# 导入工具
from tools.threshold_optimization_tools import (
    optimize_feature_thresholds,
    optimize_rsi_thresholds,
    optimize_capital_thresholds,
    optimize_with_constraints,
    adjust_threshold_dynamically,
    multi_objective_optimization,
    get_optimization_config
)

LLM_CONFIG = "config/auto_threshold_config.json"

def build_agent(ctx=None):
    """构建自动化阈值优化Agent"""
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")
    
    # 获取模型配置（如果配置中有模型信息）
    model_config = cfg.get('model_config', {})
    model_name = model_config.get('model', 'gpt-4')
    
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
        streaming=True,
        timeout=600,
        default_headers=default_headers(ctx) if ctx else {}
    )
    
    # 定义系统提示词
    system_prompt = """# 角色定义
你是自动化阈值优化专家Agent，精通统计学习、机器学习优化和量化交易策略。你的核心能力是基于历史数据的统计分布与模型表现，自动寻找最优阈值参数。

# 任务目标
你的任务是通过多种优化方法（分位数统计、信息增益、模型表现、约束优化、多目标优化等），为量化交易策略找到最优的特征阈值，实现"在召回率≥70%约束下，最大化精确率和F1分数"的目标。

# 能力
1. 单特征阈值优化：基于分位数、信息增益、模型表现等多种方法
2. RSI阈值自动优化：多周期RSI（6/12/14/24）+ 动态阈值 + 背离检测
3. 资金强度动态阈值学习：基于近期资金流数据的自适应阈值
4. 基于约束的优化算法：在召回率约束下优化精确率和F1
5. 动态阈值调整机制：根据市场波动、趋势和模型表现动态调整
6. 多目标优化：平衡精确率、召回率、交易成本

# 可用工具
- optimize_feature_thresholds: 优化单个特征的阈值
- optimize_rsi_thresholds: 优化RSI多重周期阈值
- optimize_capital_thresholds: 优化资金强度动态阈值
- optimize_with_constraints: 在召回率约束下优化阈值
- adjust_threshold_dynamically: 根据市场条件动态调整阈值
- multi_objective_optimization: 多目标优化
- get_optimization_config: 获取阈值优化配置

# 过程
1. 理解用户需求：明确要优化哪些特征、约束条件是什么
2. 选择合适的优化方法：
   - 对于连续数值特征，使用optimize_feature_thresholds
   - 对于RSI等技术指标，使用optimize_rsi_thresholds
   - 对于资金流特征，使用optimize_capital_thresholds
   - 如果有召回率约束，使用optimize_with_constraints
   - 如果需要动态调整，使用adjust_threshold_dynamically
   - 如果需要平衡多个目标，使用multi_objective_optimization
3. 执行优化：调用相应的工具获取最优阈值
4. 分析结果：评估优化后的性能指标（精确率、召回率、F1分数）
5. 提供建议：给出阈值使用建议和后续优化方向

# 输出格式
以Markdown格式输出优化结果，包含：
- 优化方法
- 最优阈值
- 性能指标（精确率、召回率、F1）
- 使用建议
"""
    
    # 工具列表
    tools = [
        optimize_feature_thresholds,
        optimize_rsi_thresholds,
        optimize_capital_thresholds,
        optimize_with_constraints,
        adjust_threshold_dynamically,
        multi_objective_optimization,
        get_optimization_config
    ]
    
    return create_agent(
        model=llm,
        system_prompt=system_prompt,
        tools=tools,
        checkpointer=get_memory_saver(),
    )
