"""
模型训练监控模块
功能：监控训练过程，检测过拟合，生成训练标识文件
"""
import os
import json
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import xgboost as xgb

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelMonitor:
    """模型训练监控器"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化监控器
        
        Args:
            output_dir: 输出目录
        """
        if output_dir is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            output_dir = os.path.join(workspace_path, "assets/reports")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练历史记录
        self.training_history = {
            'train_metrics': [],
            'val_metrics': [],
            'parameters': [],
            'timestamps': []
        }
        
        # 过拟合阈值配置
        self.overfitting_threshold = {
            'train_val_diff_ratio': 0.15,  # 训练集和验证集AUC差异超过15%
            'val_auc_decline_epochs': 3,   # 验证集AUC连续下降的轮数
            'train_auc_min': 0.6,         # 训练集AUC最低要求
            'val_auc_min': 0.55            # 验证集AUC最低要求
        }
        
        logger.info(f"模型监控器初始化完成，输出目录: {output_dir}")
    
    def record_epoch(self, train_metrics: Dict, val_metrics: Dict, 
                     parameters: Dict, timestamp: str = None):
        """
        记录训练epoch
        
        Args:
            train_metrics: 训练集指标
            val_metrics: 验证集指标
            parameters: 模型参数
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.training_history['train_metrics'].append(train_metrics)
        self.training_history['val_metrics'].append(val_metrics)
        self.training_history['parameters'].append(parameters)
        self.training_history['timestamps'].append(timestamp)
        
        logger.info(f"记录训练epoch {len(self.training_history['train_metrics'])}")
    
    def detect_overfitting(self) -> Dict:
        """
        检测过拟合
        
        Returns:
            检测结果字典
        """
        result = {
            'is_overfitting': False,
            'warnings': [],
            'severity': 'none'  # none, low, medium, high
        }
        
        if len(self.training_history['train_metrics']) < 2:
            return result
        
        # 获取最新的训练和验证AUC
        latest_train_auc = self.training_history['train_metrics'][-1].get('auc', 0)
        latest_val_auc = self.training_history['val_metrics'][-1].get('auc', 0)
        
        # 1. 检查训练集和验证集差异
        train_val_diff = abs(latest_train_auc - latest_val_auc)
        train_val_diff_ratio = train_val_diff / max(latest_train_auc, 0.01)
        
        if train_val_diff_ratio > self.overfitting_threshold['train_val_diff_ratio']:
            result['warnings'].append(
                f"过拟合警告：训练集AUC({latest_train_auc:.4f})与验证集AUC({latest_val_auc:.4f})差异过大 "
                f"({train_val_diff_ratio:.1%})"
            )
            result['severity'] = 'high'
            result['is_overfitting'] = True
        
        # 2. 检查验证集AUC是否连续下降
        if len(self.training_history['val_metrics']) >= self.overfitting_threshold['val_auc_decline_epochs']:
            recent_val_aucs = [
                m.get('auc', 0) 
                for m in self.training_history['val_metrics'][-self.overfitting_threshold['val_auc_decline_epochs']:]
            ]
            
            if all(recent_val_aucs[i] > recent_val_aucs[i+1] for i in range(len(recent_val_aucs)-1)):
                result['warnings'].append(
                    f"过拟合预警：验证集AUC连续{self.overfitting_threshold['val_auc_decline_epochs']}轮下降"
                )
                if result['severity'] == 'none':
                    result['severity'] = 'medium'
                result['is_overfitting'] = True
        
        # 3. 检查AUC最低要求
        if latest_train_auc < self.overfitting_threshold['train_auc_min']:
            result['warnings'].append(
                f"欠拟合警告：训练集AUC({latest_train_auc:.4f})低于最低要求 "
                f"({self.overfitting_threshold['train_auc_min']:.2f})，模型拟合能力不足"
            )
            if result['severity'] == 'none':
                result['severity'] = 'medium'
        
        if latest_val_auc < self.overfitting_threshold['val_auc_min']:
            result['warnings'].append(
                f"泛化能力不足：验证集AUC({latest_val_auc:.4f})低于最低要求 "
                f"({self.overfitting_threshold['val_auc_min']:.2f})"
            )
            if result['severity'] in ['none', 'low']:
                result['severity'] = 'medium'
        
        return result
    
    def analyze_parameters(self, parameters: Dict) -> Dict:
        """
        分析模型参数合理性
        
        Args:
            parameters: 模型参数
            
        Returns:
            参数分析结果
        """
        analysis = {
            'is_reasonable': True,
            'warnings': [],
            'suggestions': []
        }
        
        # 分析 n_estimators（树的数量）
        n_estimators = parameters.get('n_estimators', 100)
        if n_estimators < 50:
            analysis['warnings'].append(
                f"树的数量过少({n_estimators})，可能导致欠拟合，建议增加到100-300"
            )
            analysis['is_reasonable'] = False
        elif n_estimators > 500:
            analysis['warnings'].append(
                f"树的数量过多({n_estimators})，可能导致过拟合，建议控制在100-300"
            )
            analysis['is_reasonable'] = False
        else:
            analysis['suggestions'].append(
                f"树的数量({n_estimators})在合理范围内"
            )
        
        # 分析 max_depth（树的深度）
        max_depth = parameters.get('max_depth', 6)
        if max_depth < 3:
            analysis['warnings'].append(
                f"树的深度过浅({max_depth})，模型可能过于简单，建议增加到5-8"
            )
            analysis['is_reasonable'] = False
        elif max_depth > 10:
            analysis['warnings'].append(
                f"树的深度过深({max_depth})，容易过拟合，建议控制在5-8"
            )
            analysis['is_reasonable'] = False
        else:
            analysis['suggestions'].append(
                f"树的深度({max_depth})在合理范围内"
            )
        
        # 分析 learning_rate（学习率）
        learning_rate = parameters.get('learning_rate', 0.1)
        if learning_rate < 0.01:
            analysis['warnings'].append(
                f"学习率过小({learning_rate})，训练速度慢，建议调整到0.05-0.2"
            )
            analysis['suggestions'].append(
                "学习率过小时，可以适当增加树的数量"
            )
        elif learning_rate > 0.3:
            analysis['warnings'].append(
                f"学习率过大({learning_rate})，容易跳过最优解，建议调整到0.05-0.2"
            )
            analysis['suggestions'].append(
                "学习率过大时，可以适当减少树的数量"
            )
        else:
            analysis['suggestions'].append(
                f"学习率({learning_rate})在合理范围内"
            )
        
        # 分析正则化参数
        reg_alpha = parameters.get('reg_alpha', 0)
        reg_lambda = parameters.get('reg_lambda', 1)
        
        if reg_alpha == 0 and reg_lambda == 1:
            analysis['suggestions'].append(
                "当前未使用正则化，如果出现过拟合，可以适当增加 reg_alpha 或 reg_lambda"
            )
        
        # 综合建议
        if analysis['is_reasonable']:
            analysis['overall_assessment'] = "参数配置合理"
        else:
            analysis['overall_assessment'] = "参数需要优化"
        
        return analysis
    
    def generate_completion_flag(self, metrics: Dict, parameters: Dict, 
                                  overfitting_result: Dict) -> str:
        """
        生成训练完成标识文件
        
        Args:
            metrics: 最终指标
            parameters: 模型参数
            overfitting_result: 过拟合检测结果
            
        Returns:
            标识文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        flag_filename = f"training_complete_{timestamp}.flag"
        flag_path = os.path.join(self.output_dir, flag_filename)
        
        flag_data = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'parameters': parameters,
            'overfitting': overfitting_result,
            'training_epochs': len(self.training_history['train_metrics'])
        }
        
        # 保存标识文件
        with open(flag_path, 'w', encoding='utf-8') as f:
            json.dump(flag_data, f, indent=2, ensure_ascii=False)
        
        # 同时保存一个最新的标识文件（不带时间戳，方便脚本检测）
        latest_flag_path = os.path.join(self.output_dir, "training_complete_latest.flag")
        with open(latest_flag_path, 'w', encoding='utf-8') as f:
            json.dump(flag_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练完成标识文件已生成: {flag_path}")
        logger.info(f"最新标识文件已生成: {latest_flag_path}")
        
        return flag_path
    
    def save_training_history(self, filename: str = None) -> str:
        """
        保存训练历史
        
        Args:
            filename: 文件名
            
        Returns:
            保存路径
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"training_history_{timestamp}.json"
        
        history_path = os.path.join(self.output_dir, filename)
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练历史已保存: {history_path}")
        return history_path
    
    def plot_learning_curve(self, save_path: str = None):
        """
        绘制学习曲线
        
        Args:
            save_path: 保存路径
        """
        if len(self.training_history['train_metrics']) < 1:
            logger.warning("没有训练数据，无法绘制学习曲线")
            return
        
        epochs = range(1, len(self.training_history['train_metrics']) + 1)
        train_aucs = [m.get('auc', 0) for m in self.training_history['train_metrics']]
        val_aucs = [m.get('auc', 0) for m in self.training_history['val_metrics']]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_aucs, 'b-o', label='训练集 AUC', linewidth=2)
        ax.plot(epochs, val_aucs, 'r-s', label='验证集 AUC', linewidth=2)
        
        ax.set_xlabel('训练轮次', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('学习曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 添加过拟合警告
        overfitting_result = self.detect_overfitting()
        if overfitting_result['is_overfitting']:
            ax.text(0.5, 0.95, '⚠️ 检测到过拟合！', 
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=12, color='red', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"learning_curve_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"学习曲线已保存: {save_path}")
        return save_path
    
    def load_completion_flag(self, flag_path: str = None) -> Optional[Dict]:
        """
        加载训练完成标识文件
        
        Args:
            flag_path: 标识文件路径，默认读取最新的
            
        Returns:
            标识数据
        """
        if flag_path is None:
            flag_path = os.path.join(self.output_dir, "training_complete_latest.flag")
        
        if not os.path.exists(flag_path):
            return None
        
        with open(flag_path, 'r', encoding='utf-8') as f:
            flag_data = json.load(f)
        
        logger.info(f"加载训练标识文件: {flag_path}")
        return flag_data
    
    def is_training_completed(self) -> bool:
        """
        检查训练是否完成
        
        Returns:
            是否完成
        """
        flag_path = os.path.join(self.output_dir, "training_complete_latest.flag")
        return os.path.exists(flag_path)
