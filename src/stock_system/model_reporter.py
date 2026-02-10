"""
模型报告生成器（集成版）
功能：集成监控、可视化、HTML报告生成，一键生成完整报告
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .monitor import ModelMonitor
from .visualizer import ModelVisualizer
from .report_generator import HTMLReportGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelReporter:
    """模型报告生成器（集成版）"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        if output_dir is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            output_dir = os.path.join(workspace_path, "assets/reports")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化子模块
        self.monitor = ModelMonitor(output_dir)
        self.visualizer = ModelVisualizer(output_dir)
        self.report_generator = HTMLReportGenerator(output_dir)
        
        logger.info(f"模型报告生成器初始化完成，输出目录: {output_dir}")
    
    def generate_full_report(self, model, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              feature_names: List[str], parameters: Dict,
                              stock_pool: List[str] = None,
                              industries: List[str] = None,
                              save_flag: bool = True) -> Dict:
        """
        生成完整报告
        
        Args:
            model: 训练好的模型
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            feature_names: 特征名称
            parameters: 模型参数
            stock_pool: 股票池
            industries: 行业列表
            save_flag: 是否保存标识文件
            
        Returns:
            报告信息字典
        """
        logger.info("=" * 60)
        logger.info("开始生成完整模型报告")
        logger.info("=" * 60)
        
        report_info = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'parameters': parameters,
            'overfitting': {},
            'param_analysis': {},
            'image_paths': {},
            'flag_path': None,
            'html_report': None
        }
        
        try:
            # 1. 模型预测
            logger.info("步骤1/8: 生成模型预测...")
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 计算指标
            logger.info("步骤2/8: 计算模型指标...")
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            train_metrics = {
                'auc': roc_auc_score(y_train, y_train_proba) if y_train_proba is not None else 0,
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred, zero_division=0),
                'recall': recall_score(y_train, y_train_pred, zero_division=0),
                'f1': f1_score(y_train, y_train_pred, zero_division=0)
            }
            
            val_metrics = {
                'auc': roc_auc_score(y_val, y_val_proba) if y_val_proba is not None else 0,
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred, zero_division=0),
                'recall': recall_score(y_val, y_val_pred, zero_division=0),
                'f1': f1_score(y_val, y_val_pred, zero_division=0),
                'train_auc': train_metrics['auc'],
                'val_auc': 0  # 临时值，后面会更新
            }
            
            # 更新val_auc
            val_metrics['val_auc'] = val_metrics['auc']
            
            # 添加混淆矩阵统计
            tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
            val_metrics.update({
                'true_positive': int(tp),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_negative': int(tn)
            })
            
            # 计算投资组合指标
            val_metrics.update(self._calculate_portfolio_metrics(y_val, y_val_pred, y_val_proba))
            
            report_info['metrics'] = val_metrics
            logger.info(f"验证集AUC: {val_metrics['auc']:.4f}")
            logger.info(f"验证集准确率: {val_metrics['accuracy']:.4f}")
            logger.info(f"验证集F1: {val_metrics['f1']:.4f}")
            
            # 3. 记录训练历史
            logger.info("步骤3/8: 记录训练历史...")
            self.monitor.record_epoch(train_metrics, val_metrics, parameters)
            
            # 4. 检测过拟合
            logger.info("步骤4/8: 检测过拟合...")
            overfitting_result = self.monitor.detect_overfitting()
            report_info['overfitting'] = overfitting_result
            
            if overfitting_result['is_overfitting']:
                logger.warning(f"检测到过拟合: {overfitting_result['warnings']}")
            else:
                logger.info("✓ 未检测到过拟合")
            
            # 5. 分析参数
            logger.info("步骤5/8: 分析参数合理性...")
            param_analysis = self.monitor.analyze_parameters(parameters)
            report_info['param_analysis'] = param_analysis
            
            if param_analysis['is_reasonable']:
                logger.info("✓ 参数配置合理")
            else:
                logger.warning(f"参数需要优化: {param_analysis['warnings']}")
            
            # 6. 生成可视化图表
            logger.info("步骤6/8: 生成可视化图表...")
            if y_val_proba is not None:
                # ROC曲线
                roc_path = self.visualizer.plot_roc_curve(y_val, y_val_proba)
                report_info['image_paths']['roc'] = roc_path
                
                # 混淆矩阵
                confusion_path = self.visualizer.plot_confusion_matrix(y_val, y_val_pred)
                report_info['image_paths']['confusion_matrix'] = confusion_path
                
                # 精确率-召回率曲线
                pr_path = self.visualizer.plot_precision_recall_curve(y_val, y_val_proba)
                report_info['image_paths']['pr_curve'] = pr_path
                
                # 预测分布
                dist_path = self.visualizer.plot_prediction_distribution(y_val, y_val_proba)
                report_info['image_paths']['prediction_distribution'] = dist_path
            
            # 特征重要性
            try:
                feat_imp_path = self.visualizer.plot_feature_importance(model, feature_names)
                report_info['image_paths']['feature_importance'] = feat_imp_path
            except Exception as e:
                logger.warning(f"生成特征重要性失败: {e}")
            
            # 学习曲线
            learning_curve_path = self.monitor.plot_learning_curve()
            report_info['image_paths']['learning_curve'] = learning_curve_path
            
            # 总结仪表盘
            dashboard_path = self.visualizer.create_summary_dashboard(
                val_metrics, parameters, overfitting_result
            )
            report_info['image_paths']['summary_dashboard'] = dashboard_path
            
            # 行业采样分布
            if stock_pool and industries:
                industry_path = self.visualizer.plot_industry_sampling(stock_pool, industries)
                report_info['image_paths']['industry_sampling'] = industry_path
            
            logger.info(f"✓ 共生成 {len(report_info['image_paths'])} 个图表")
            
            # 7. 生成HTML报告
            logger.info("步骤7/8: 生成HTML报告...")
            html_report_path = self.report_generator.generate_report(
                metrics=val_metrics,
                parameters=parameters,
                overfitting_result=overfitting_result,
                param_analysis=param_analysis,
                image_paths=report_info['image_paths']
            )
            report_info['html_report'] = html_report_path
            logger.info(f"✓ HTML报告已生成: {html_report_path}")
            
            # 8. 生成标识文件
            if save_flag:
                logger.info("步骤8/8: 生成训练完成标识文件...")
                flag_path = self.monitor.generate_completion_flag(
                    val_metrics, parameters, overfitting_result
                )
                report_info['flag_path'] = flag_path
                logger.info(f"✓ 标识文件已生成: {flag_path}")
            
            logger.info("=" * 60)
            logger.info("完整报告生成成功！")
            logger.info("=" * 60)
            logger.info(f"报告摘要:")
            logger.info(f"  - 综合得分: {(val_metrics['auc']*0.4 + val_metrics['accuracy']*0.2 + val_metrics['f1']*0.4):.4f}")
            logger.info(f"  - 验证集AUC: {val_metrics['auc']:.4f}")
            logger.info(f"  - 过拟合状态: {'是' if overfitting_result['is_overfitting'] else '否'}")
            logger.info(f"  - 图表数量: {len(report_info['image_paths'])}")
            logger.info(f"  - HTML报告: {report_info['html_report']}")
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return report_info
    
    def _calculate_portfolio_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      y_pred_proba: np.ndarray = None) -> Dict:
        """
        计算投资组合指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率
            
        Returns:
            投资组合指标
        """
        metrics = {}
        
        try:
            # 累计收益率（假设每只股票投入1元，涨赚+1%，跌亏-1%）
            returns = np.where(y_pred == 1, np.where(y_true == 1, 0.01, -0.01), 0)
            cumulative_return = np.sum(returns)
            metrics['cumulative_return'] = cumulative_return
            
            # 年化收益率（假设252个交易日）
            n = len(y_true)
            if n > 0:
                annual_return = cumulative_return * (252 / n) if n > 0 else 0
                metrics['annual_return'] = annual_return
            
            # 夏普比率（假设无风险利率为3%）
            if len(returns) > 1:
                excess_returns = returns - 0.03/252  # 日化无风险利率
                sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)
                metrics['sharpe_ratio'] = sharpe_ratio
            else:
                metrics['sharpe_ratio'] = 0
            
            # 最大回撤
            cumulative = np.cumsum(returns)
            if len(cumulative) > 0:
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max)
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                metrics['max_drawdown'] = max_drawdown
            else:
                metrics['max_drawdown'] = 0
            
            # 胜率
            if len(returns) > 0:
                win_rate = np.sum(returns > 0) / np.sum(returns != 0) if np.sum(returns != 0) > 0 else 0
                metrics['win_rate'] = win_rate
            else:
                metrics['win_rate'] = 0
            
            # 假阳性率和假阴性率
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            total_negative = tn + fp
            total_positive = fn + tp
            
            metrics['false_positive_rate'] = fp / total_negative if total_negative > 0 else 0
            metrics['false_negative_rate'] = fn / total_positive if total_positive > 0 else 0
            
        except Exception as e:
            logger.warning(f"计算投资组合指标失败: {e}")
            metrics = {
                'cumulative_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'false_positive_rate': 0,
                'false_negative_rate': 0
            }
        
        return metrics
    
    def load_latest_report(self) -> Optional[Dict]:
        """
        加载最新的训练报告
        
        Returns:
            报告信息字典
        """
        flag_path = self.monitor.load_completion_flag()
        if flag_path is None:
            logger.warning("未找到训练报告")
            return None
        
        return flag_path
    
    def is_training_completed(self) -> bool:
        """
        检查训练是否完成
        
        Returns:
            是否完成
        """
        return self.monitor.is_training_completed()
    
    def print_summary(self, report_info: Dict):
        """
        打印报告摘要
        
        Args:
            report_info: 报告信息
        """
        metrics = report_info['metrics']
        overfitting = report_info['overfitting']
        
        print("\n" + "=" * 60)
        print("模型训练报告摘要")
        print("=" * 60)
        
        # 综合得分
        overall_score = (metrics['auc'] * 0.4 + metrics['accuracy'] * 0.2 + metrics['f1'] * 0.4)
        print(f"\n综合得分: {overall_score:.4f}")
        
        # 性能指标
        print(f"\n模型性能指标:")
        print(f"  AUC:        {metrics['auc']:.4f}")
        print(f"  准确率:      {metrics['accuracy']:.4f}")
        print(f"  精确率:      {metrics['precision']:.4f}")
        print(f"  召回率:      {metrics['recall']:.4f}")
        print(f"  F1分数:     {metrics['f1']:.4f}")
        
        # 投资组合指标
        print(f"\n投资组合指标:")
        print(f"  累计收益率:   {metrics['cumulative_return']:.2%}")
        print(f"  年化收益率:   {metrics['annual_return']:.2%}")
        print(f"  夏普比率:    {metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤:    {metrics['max_drawdown']:.2%}")
        print(f"  胜率:       {metrics['win_rate']:.2%}")
        
        # 过拟合状态
        print(f"\n过拟合检测:")
        if overfitting['is_overfitting']:
            print(f"  状态: ⚠️ 检测到过拟合")
            print(f"  严重程度: {overfitting['severity'].upper()}")
            for warning in overfitting['warnings']:
                print(f"  - {warning}")
        else:
            print(f"  状态: ✅ 无过拟合")
        
        # HTML报告路径
        print(f"\n详细报告: {report_info['html_report']}")
        print("=" * 60 + "\n")
