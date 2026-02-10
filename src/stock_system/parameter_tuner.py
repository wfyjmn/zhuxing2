"""
参数调整模块
功能：根据误差分析结果自适应调整模型参数
"""
import os
import json
import logging
import copy
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParameterTuner:
    """参数调整器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化参数调整器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            config_path = os.path.join(workspace_path, "config/model_config.json")
        
        self.config = self._load_config(config_path)
        self.adjustment_config = self.config['adjustment']
        self.current_params = self.config['xgboost']['params'].copy()
        self.current_threshold = self.config['xgboost']['threshold']
        self.adjustment_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载配置成功")
            return config
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return {}
    
    def determine_adjustment_strategy(self, error_analysis: Dict, metrics: Dict) -> Dict:
        """
        根据误差分析确定调整策略
        
        Args:
            error_analysis: 误差分析结果
            metrics: 性能指标
            
        Returns:
            调整策略字典
        """
        strategy = {
            'adjust_threshold': False,
            'adjust_params': False,
            'adjustments': {}
        }
        
        fp_rate = error_analysis.get('false_positive_rate', 0)
        fn_rate = error_analysis.get('false_negative_rate', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        # 1. 判断是否需要调整阈值
        if fp_rate > 0.30 and precision < 0.25:
            # 假正例过多，提高阈值
            strategy['adjust_threshold'] = True
            new_threshold = min(
                self.current_threshold + self.adjustment_config['adjust_step']['threshold'],
                self.adjustment_config['threshold_range'][1]
            )
            strategy['adjustments']['threshold'] = new_threshold
            logger.info(f"策略：提高阈值以减少假正例，从 {self.current_threshold} 调整到 {new_threshold}")
        
        elif fn_rate > 0.20 and recall < 0.70:
            # 假负例过多，降低阈值
            strategy['adjust_threshold'] = True
            new_threshold = max(
                self.current_threshold - self.adjustment_config['adjust_step']['threshold'],
                self.adjustment_config['threshold_range'][0]
            )
            strategy['adjustments']['threshold'] = new_threshold
            logger.info(f"策略：降低阈值以减少假负例，从 {self.current_threshold} 调整到 {new_threshold}")
        
        # 2. 判断是否需要调整模型参数
        if recall < 0.70:
            # 召回率过低，调整scale_pos_weight增加对正类的关注
            strategy['adjust_params'] = True
            current_weight = self.current_params.get('scale_pos_weight', 1)
            new_weight = min(current_weight + 0.5, 5.0)
            strategy['adjustments']['scale_pos_weight'] = new_weight
            logger.info(f"策略：增加scale_pos_weight以提升召回率，从 {current_weight} 调整到 {new_weight}")
        
        if precision < 0.25:
            # 精确率过低，调整max_depth或learning_rate
            strategy['adjust_params'] = True
            
            # 小幅调整学习率
            current_lr = self.current_params.get('learning_rate', 0.05)
            lr_step = self.adjustment_config['adjust_step']['learning_rate']
            lr_min, lr_max = self.adjustment_config['learning_rate_range']
            
            # 根据模型表现决定增减
            if precision < 0.20:
                # 精确率很低，尝试降低学习率让模型学习更细致
                new_lr = max(current_lr - lr_step, lr_min)
            else:
                # 精确率一般，尝试提高学习率加快收敛
                new_lr = min(current_lr + lr_step, lr_max)
            
            strategy['adjustments']['learning_rate'] = new_lr
            logger.info(f"策略：调整学习率，从 {current_lr} 调整到 {new_lr}")
        
        # 如果没有需要调整的，进行微调
        if not strategy['adjust_threshold'] and not strategy['adjust_params']:
            logger.info("指标良好，进行小幅度微调")
            strategy['adjust_params'] = True
            strategy['adjustments']['gamma'] = self.current_params.get('gamma', 0) + 0.1
        
        return strategy
    
    def apply_adjustments(self, strategy: Dict) -> Tuple[Dict, float]:
        """
        应用参数调整
        
        Args:
            strategy: 调整策略
            
        Returns:
            (新参数, 新阈值)
        """
        new_params = copy.deepcopy(self.current_params)
        new_threshold = self.current_threshold
        
        # 应用调整
        adjustments = strategy.get('adjustments', {})
        
        if 'threshold' in adjustments:
            new_threshold = adjustments['threshold']
        
        for param_name, param_value in adjustments.items():
            if param_name != 'threshold':
                new_params[param_name] = param_value
        
        # 记录调整历史
        adjustment_record = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'old_params': copy.deepcopy(self.current_params),
            'old_threshold': self.current_threshold,
            'new_params': copy.deepcopy(new_params),
            'new_threshold': new_threshold,
            'strategy': strategy
        }
        self.adjustment_history.append(adjustment_record)
        
        # 更新当前参数
        self.current_params = new_params
        self.current_threshold = new_threshold
        
        logger.info(f"参数调整完成")
        logger.info(f"新阈值: {new_threshold}")
        logger.info(f"新参数: {new_params}")
        
        return new_params, new_threshold
    
    def rollback_parameters(self, steps: int = 1) -> Tuple[Dict, float]:
        """
        回滚参数到历史版本
        
        Args:
            steps: 回滚步数
            
        Returns:
            (回滚后的参数, 回滚后的阈值)
        """
        if len(self.adjustment_history) < steps:
            logger.warning(f"调整历史不足 {steps} 步，无法回滚")
            return self.current_params, self.current_threshold
        
        # 获取目标版本的参数
        target_record = self.adjustment_history[-steps]
        
        self.current_params = copy.deepcopy(target_record['old_params'])
        self.current_threshold = target_record['old_threshold']
        
        logger.info(f"参数回滚 {steps} 步完成")
        logger.info(f"回滚后阈值: {self.current_threshold}")
        logger.info(f"回滚后参数: {self.current_params}")
        
        return self.current_params, self.current_threshold
    
    def validate_adjustments(self, new_params: Dict, new_threshold: float) -> bool:
        """
        验证参数调整是否合理
        
        Args:
            new_params: 新参数
            new_threshold: 新阈值
            
        Returns:
            是否合理
        """
        # 验证阈值范围
        threshold_min, threshold_max = self.adjustment_config['threshold_range']
        if not (threshold_min <= new_threshold <= threshold_max):
            logger.warning(f"阈值 {new_threshold} 超出范围 [{threshold_min}, {threshold_max}]")
            return False
        
        # 验证学习率范围
        lr_min, lr_max = self.adjustment_config['learning_rate_range']
        learning_rate = new_params.get('learning_rate', 0.05)
        if not (lr_min <= learning_rate <= lr_max):
            logger.warning(f"学习率 {learning_rate} 超出范围 [{lr_min}, {lr_max}]")
            return False
        
        # 验证max_depth范围
        depth_min, depth_max = self.adjustment_config['max_depth_range']
        max_depth = new_params.get('max_depth', 6)
        if not (depth_min <= max_depth <= depth_max):
            logger.warning(f"max_depth {max_depth} 超出范围 [{depth_min}, {depth_max}]")
            return False
        
        logger.info("参数调整验证通过")
        return True
    
    def save_adjustment_history(self, filename: str = None):
        """
        保存调整历史
        
        Args:
            filename: 文件名
        """
        try:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            if filename is None:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"adjustment_history_{timestamp}.json"
            
            save_path = os.path.join(workspace_path, "assets/logs", filename)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.adjustment_history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存调整历史成功: {save_path}")
        except Exception as e:
            logger.error(f"保存调整历史失败: {e}")
    
    def load_adjustment_history(self, filename: str):
        """
        加载调整历史
        
        Args:
            filename: 文件名
        """
        try:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            load_path = os.path.join(workspace_path, "assets/logs", filename)
            
            with open(load_path, 'r', encoding='utf-8') as f:
                self.adjustment_history = json.load(f)
            
            logger.info(f"加载调整历史成功: {load_path}")
        except Exception as e:
            logger.error(f"加载调整历史失败: {e}")
    
    def auto_adjust(self, error_analysis: Dict, metrics: Dict) -> Tuple[Dict, float, Dict]:
        """
        自动调整参数（完整流程）
        
        Args:
            error_analysis: 误差分析
            metrics: 性能指标
            
        Returns:
            (新参数, 新阈值, 调整策略)
        """
        # 1. 确定调整策略
        strategy = self.determine_adjustment_strategy(error_analysis, metrics)
        
        # 2. 应用调整
        new_params, new_threshold = self.apply_adjustments(strategy)
        
        # 3. 验证调整
        if not self.validate_adjustments(new_params, new_threshold):
            logger.warning("参数调整验证失败，回滚到调整前")
            self.rollback_parameters(steps=1)
            return self.current_params, self.current_threshold, {'adjusted': False}
        
        logger.info("自动参数调整完成")
        return new_params, new_threshold, strategy


def test_parameter_tuner():
    """测试参数调整器"""
    tuner = ParameterTuner()
    
    # 创建测试数据
    print("\n=== 测试参数调整 ===")
    
    # 模拟误差分析
    error_analysis = {
        'total_samples': 100,
        'error_count': 30,
        'false_positive_count': 20,
        'false_negative_count': 10,
        'false_positive_rate': 0.20,
        'false_negative_rate': 0.10
    }
    
    # 模拟性能指标
    metrics = {
        'precision': 0.20,
        'recall': 0.75,
        'f1': 0.31,
        'auc': 0.62
    }
    
    # 自动调整
    new_params, new_threshold, strategy = tuner.auto_adjust(error_analysis, metrics)
    
    print(f"\n调整策略: {strategy}")
    print(f"新阈值: {new_threshold}")
    print(f"新参数: {new_params}")
    
    # 测试回滚
    print("\n=== 测试参数回滚 ===")
    old_params, old_threshold = tuner.rollback_parameters(steps=1)
    print(f"回滚后阈值: {old_threshold}")
    print(f"回滚后参数: {old_params}")


if __name__ == '__main__':
    import pandas as pd
    test_parameter_tuner()
