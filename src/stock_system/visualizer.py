"""
模型可视化模块
功能：生成模型性能可视化图表
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             confusion_matrix, classification_report)
import xgboost as xgb

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """模型可视化器"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        if output_dir is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            output_dir = os.path.join(workspace_path, "assets/reports")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"模型可视化器初始化完成，输出目录: {output_dir}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       save_path: str = None) -> str:
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='随机猜测 (AUC = 0.5000)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        ax.set_title('ROC 曲线', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 添加性能评估
        if roc_auc >= 0.85:
            performance = "优秀"
            color = "green"
        elif roc_auc >= 0.75:
            performance = "良好"
            color = "blue"
        elif roc_auc >= 0.65:
            performance = "中等"
            color = "orange"
        else:
            performance = "较差"
            color = "red"
        
        ax.text(0.5, 0.5, f"模型性能: {performance}", 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=14, color=color, weight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"roc_curve_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC曲线已保存: {save_path}")
        return save_path
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               threshold: float = 0.5, save_path: str = None) -> str:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            threshold: 阈值
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['跌', '涨'], 
                   yticklabels=['跌', '涨'])
        
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title(f'混淆矩阵 (阈值={threshold})', fontsize=14, fontweight='bold')
        
        # 添加统计信息
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        stats_text = f'准确率: {accuracy:.4f}\n精确率: {precision:.4f}\n召回率: {recall:.4f}'
        ax.text(2.5, 0.5, stats_text, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"confusion_matrix_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩阵已保存: {save_path}")
        return save_path
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                                 top_n: int = 20, save_path: str = None) -> str:
        """
        绘制特征重要性
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            top_n: 显示前N个特征
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        importance = model.feature_importances_
        
        # 排序
        indices = np.argsort(importance)[::-1][:top_n]
        sorted_importance = importance[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(sorted_importance)), sorted_importance, 
                       color='steelblue')
        ax.set_yticks(range(len(sorted_importance)))
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.invert_yaxis()  # 最重要的在上面
        ax.set_xlabel('重要性得分', fontsize=12)
        ax.set_title(f'特征重要性 Top {top_n}', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"feature_importance_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"特征重要性图已保存: {save_path}")
        return save_path
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    save_path: str = None) -> str:
        """
        绘制精确率-召回率曲线
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR曲线 (AUC = {pr_auc:.4f})')
        
        # 计算随机猜测的基准线
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                  label=f'随机猜测 (AUC = {baseline:.4f})')
        
        ax.set_xlabel('召回率 (Recall)', fontsize=12)
        ax.set_ylabel('精确率 (Precision)', fontsize=12)
        ax.set_title('精确率-召回率曲线', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标注最佳阈值附近的点
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        ax.scatter(recall[best_idx], precision[best_idx], c='red', s=100, 
                  marker='*', zorder=5, label='最佳F1点')
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"pr_curve_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"精确率-召回率曲线已保存: {save_path}")
        return save_path
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    save_path: str = None) -> str:
        """
        绘制预测概率分布
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        # 分别绘制涨和跌的预测概率分布
        pred_up = y_pred_proba[y_true == 1]
        pred_down = y_pred_proba[y_true == 0]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 涨的预测分布
        axes[0].hist(pred_up, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0].axvline(0.5, color='blue', linestyle='--', linewidth=2, label='决策边界')
        axes[0].set_xlabel('预测概率（涨）', fontsize=12)
        axes[0].set_ylabel('频数', fontsize=12)
        axes[0].set_title('真实标签为"涨"的预测概率分布', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 跌的预测分布
        axes[1].hist(pred_down, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(0.5, color='blue', linestyle='--', linewidth=2, label='决策边界')
        axes[1].set_xlabel('预测概率（跌）', fontsize=12)
        axes[1].set_ylabel('频数', fontsize=12)
        axes[1].set_title('真实标签为"跌"的预测概率分布', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"prediction_distribution_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"预测分布图已保存: {save_path}")
        return save_path
    
    def plot_industry_sampling(self, stock_pool: List[str], industries: List[str],
                               save_path: str = None) -> str:
        """
        绘制行业采样分布
        
        Args:
            stock_pool: 股票池
            industries: 对应的行业列表
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        # 统计各行业股票数量
        industry_counts = pd.Series(industries).value_counts()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制条形图
        bars = ax.bar(range(len(industry_counts)), industry_counts.values, 
                     color='steelblue', edgecolor='black')
        
        ax.set_xticks(range(len(industry_counts)))
        ax.set_xticklabels(industry_counts.index, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('股票数量', fontsize=12)
        ax.set_title('行业采样分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, industry_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{int(val)}', ha='center', va='bottom', fontsize=9)
        
        # 添加统计信息
        total_stocks = len(stock_pool)
        unique_industries = len(industry_counts)
        avg_per_industry = total_stocks / unique_industries
        
        stats_text = (f'总股票数: {total_stocks}\n'
                     f'行业数: {unique_industries}\n'
                     f'平均每行业: {avg_per_industry:.1f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               va='top', ha='left', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"industry_sampling_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"行业采样分布图已保存: {save_path}")
        return save_path
    
    def create_summary_dashboard(self, metrics: Dict, parameters: Dict,
                                  overfitting_result: Dict, save_path: str = None) -> str:
        """
        创建总结仪表盘
        
        Args:
            metrics: 模型指标
            parameters: 模型参数
            overfitting_result: 过拟合检测结果
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        fig = plt.figure(figsize=(16, 10))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 模型性能指标
        ax1 = fig.add_subplot(gs[0, 0])
        metric_names = ['AUC', '准确率', '精确率', '召回率', 'F1']
        metric_values = [
            metrics.get('auc', 0),
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0)
        ]
        colors = ['green' if v > 0.7 else 'orange' if v > 0.6 else 'red' for v in metric_values]
        ax1.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('得分', fontsize=10)
        ax1.set_title('模型性能指标', fontsize=12, fontweight='bold')
        for i, v in enumerate(metric_values):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        # 2. 模型参数配置
        ax2 = fig.add_subplot(gs[0, 1])
        param_names = ['n_estimators', 'max_depth', 'learning_rate', 
                      'subsample', 'colsample_bytree']
        param_values = [
            parameters.get('n_estimators', 0),
            parameters.get('max_depth', 0),
            parameters.get('learning_rate', 0),
            parameters.get('subsample', 0),
            parameters.get('colsample_bytree', 0)
        ]
        ax2.barh(param_names, param_values, color='steelblue')
        ax2.set_xlabel('参数值', fontsize=10)
        ax2.set_title('模型参数配置', fontsize=12, fontweight='bold')
        
        # 3. 过拟合检测结果
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        overfitting_status = "✅ 无过拟合" if not overfitting_result['is_overfitting'] else "⚠️ 检测到过拟合"
        severity_map = {'none': '绿色', 'low': '黄色', 'medium': '橙色', 'high': '红色'}
        severity_color = {'none': 'green', 'low': 'yellow', 'medium': 'orange', 'high': 'red'}
        severity_text = f"严重程度: {severity_map[overfitting_result['severity']]}"
        
        info_text = (f"模型状态: {overfitting_status}\n\n"
                    f"{severity_text}\n\n")
        
        if overfitting_result['warnings']:
            info_text += "警告信息:\n"
            for warning in overfitting_result['warnings'][:3]:
                info_text += f"• {warning}\n"
        
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax3.set_title('模型健康状态', fontsize=12, fontweight='bold')
        
        # 4. 混淆矩阵（简化版）
        ax4 = fig.add_subplot(gs[1, :2])
        tn = metrics.get('true_negative', 0)
        fp = metrics.get('false_positive', 0)
        fn = metrics.get('false_negative', 0)
        tp = metrics.get('true_positive', 0)
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=['跌', '涨'], yticklabels=['跌', '涨'])
        ax4.set_xlabel('预测标签', fontsize=10)
        ax4.set_ylabel('真实标签', fontsize=10)
        ax4.set_title('混淆矩阵', fontsize=12, fontweight='bold')
        
        # 5. 训练建议
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        suggestions = []
        
        auc_value = metrics.get('auc', 0)
        if auc_value < 0.6:
            suggestions.append("• 欠拟合：增加模型复杂度")
        elif auc_value > 0.85:
            suggestions.append("• 模型表现优秀，保持当前配置")
        
        train_val_diff = abs(metrics.get('train_auc', 0) - metrics.get('val_auc', 0))
        if train_val_diff > 0.15:
            suggestions.append("• 过拟合：增加正则化或减少树的数量")
            suggestions.append("• 过拟合：降低树的深度")
        
        if not suggestions:
            suggestions.append("• 模型状态良好，无需调整")
        
        suggestion_text = "优化建议:\n\n" + "\n".join(suggestions)
        ax5.text(0.1, 0.9, suggestion_text, transform=ax5.transAxes,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax5.set_title('优化建议', fontsize=12, fontweight='bold')
        
        # 6. 整体评估
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        overall_score = (metrics.get('auc', 0) * 0.4 + 
                        metrics.get('accuracy', 0) * 0.2 +
                        metrics.get('f1', 0) * 0.4)
        
        if overall_score >= 0.8:
            grade = "A (优秀)"
            grade_color = "green"
            comment = "模型表现优秀，可以投入实盘使用"
        elif overall_score >= 0.7:
            grade = "B (良好)"
            grade_color = "blue"
            comment = "模型表现良好，建议持续监控"
        elif overall_score >= 0.6:
            grade = "C (中等)"
            grade_color = "orange"
            comment = "模型表现一般，需要进一步优化"
        else:
            grade = "D (较差)"
            grade_color = "red"
            comment = "模型表现较差，建议重新训练"
        
        overall_text = (
            f"{'='*60}\n"
            f"                    模型综合评估报告\n"
            f"{'='*60}\n\n"
            f"综合得分: {overall_score:.4f}\n"
            f"评级: {grade}\n\n"
            f"评价: {comment}\n"
            f"{'='*60}\n"
            f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        ax6.text(0.5, 0.5, overall_text, transform=ax6.transAxes,
                ha='center', va='center', fontsize=12, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=grade_color, alpha=0.2))
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"summary_dashboard_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"总结仪表盘已保存: {save_path}")
        return save_path
