"""
对比分析模块
功能：对比预测结果与实盘行情，计算关键指标
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化分析器
        
        Args:
            config_path: 模型配置文件路径
        """
        if config_path is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            config_path = os.path.join(workspace_path, "config/model_config.json")
        
        self.config = self._load_config(config_path)
        self.targets = self.config['performance']['targets']
        self.trigger_thresholds = self.config['performance']['trigger_thresholds']
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载配置成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise
    
    def align_predictions_with_actuals(self, 
                                       predictions: Dict[str, pd.DataFrame],
                                       actual_data: Dict[str, pd.DataFrame],
                                       days_ahead: int = 5) -> pd.DataFrame:
        """
        对齐预测结果与实际行情
        
        Args:
            predictions: 预测结果字典 {ts_code: DataFrame}
            actual_data: 实际行情数据字典 {ts_code: DataFrame}
            days_ahead: 向前预测天数
            
        Returns:
            对齐后的DataFrame
        """
        aligned_data = []
        
        for ts_code, pred_df in predictions.items():
            try:
                if ts_code not in actual_data:
                    logger.warning(f"股票 {ts_code} 没有实盘数据")
                    continue
                
                actual_df = actual_data[ts_code]
                
                if actual_df.empty:
                    logger.warning(f"股票 {ts_code} 实盘数据为空")
                    continue
                
                # 取预测时的信息
                pred_row = pred_df.iloc[0]
                
                # 查找对应天后的实际涨跌
                if len(actual_df) > days_ahead:
                    actual_future = actual_df.iloc[days_ahead - 1]  # 第N天的数据
                    actual_current = actual_df.iloc[0]  # 当天的数据
                    
                    # 计算实际涨跌幅
                    actual_change = (actual_future['close'] - actual_current['close']) / actual_current['close']
                    
                    # 确定实际标签 (上涨=1, 下跌=0)
                    actual_label = 1 if actual_change > 0 else 0
                    
                    # 构建对齐记录
                    aligned_row = {
                        'ts_code': ts_code,
                        'predict_date': pred_row.get('trade_date', ''),
                        'actual_date': actual_future.get('trade_date', ''),
                        'predicted_label': int(pred_row['predicted_label']),
                        'predicted_prob': float(pred_row['predicted_prob']),
                        'actual_label': actual_label,
                        'actual_change': actual_change,
                        'actual_close': float(actual_future['close']),
                        'predict_correct': int(pred_row['predicted_label'] == actual_label)
                    }
                    
                    aligned_data.append(aligned_row)
                
            except Exception as e:
                logger.error(f"对齐股票 {ts_code} 数据失败: {e}")
                continue
        
        aligned_df = pd.DataFrame(aligned_data)
        logger.info(f"数据对齐完成，共 {len(aligned_df)} 条记录")
        
        return aligned_df
    
    def calculate_metrics(self, aligned_df: pd.DataFrame) -> Dict:
        """
        计算核心指标
        
        Args:
            aligned_df: 对齐后的DataFrame
            
        Returns:
            指标字典
        """
        try:
            if aligned_df.empty:
                logger.warning("对齐数据为空，无法计算指标")
                return {}
            
            y_true = aligned_df['actual_label'].values
            y_pred = aligned_df['predicted_label'].values
            y_prob = aligned_df['predicted_prob'].values
            
            metrics = {}
            
            # 分类指标
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            
            # AUC
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.5  # 默认值
            
            # 混淆矩阵元素
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            metrics['true_positive'] = int(tp)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['true_negative'] = int(tn)
            
            # 假正率和假负率
            total_positive = tp + fn
            total_negative = tn + fp
            metrics['false_positive_rate'] = fp / total_negative if total_negative > 0 else 0
            metrics['false_negative_rate'] = fn / total_positive if total_positive > 0 else 0
            
            # 投资组合指标
            metrics = self._calculate_portfolio_metrics(aligned_df, metrics)
            
            logger.info(f"指标计算完成: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"计算指标失败: {e}")
            return {}
    
    def _calculate_portfolio_metrics(self, aligned_df: pd.DataFrame, metrics: Dict) -> Dict:
        """
        计算投资组合指标
        
        Args:
            aligned_df: 对齐数据
            metrics: 现有指标字典
            
        Returns:
            包含投资组合指标的字典
        """
        try:
            # 假设平均持仓，等权重投资
            # 计算每次预测正确时的收益
            correct_predictions = aligned_df[aligned_df['predict_correct'] == 1]
            incorrect_predictions = aligned_df[aligned_df['predict_correct'] == 0]
            
            # 计算累计收益
            aligned_df['portfolio_return'] = aligned_df.apply(
                lambda row: row['actual_change'] if row['predicted_label'] == 1 else 0,
                axis=1
            )
            
            cumulative_return = aligned_df['portfolio_return'].sum()
            metrics['cumulative_return'] = float(cumulative_return)
            
            # 年化收益率 (假设每个周期5天，每年约50个周期)
            periods = len(aligned_df)
            if periods > 0:
                metrics['annual_return'] = float(cumulative_return / periods * 50)
            else:
                metrics['annual_return'] = 0.0
            
            # 夏普比率 (简化计算：年化收益 / 标准差)
            if aligned_df['portfolio_return'].std() > 0:
                metrics['sharpe_ratio'] = float(
                    metrics['annual_return'] / (aligned_df['portfolio_return'].std() * np.sqrt(50))
                )
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # 最大回撤
            cumulative_returns = (1 + aligned_df['portfolio_return']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = float(drawdown.min())
            
            # 胜率
            metrics['win_rate'] = float(len(correct_predictions) / len(aligned_df)) if len(aligned_df) > 0 else 0.0
            
            logger.info(f"投资组合指标计算完成")
            return metrics
            
        except Exception as e:
            logger.error(f"计算投资组合指标失败: {e}")
            return metrics
    
    def generate_confusion_matrix(self, aligned_df: pd.DataFrame) -> Dict:
        """
        生成混淆矩阵
        
        Args:
            aligned_df: 对齐数据
            
        Returns:
            混淆矩阵字典
        """
        try:
            if aligned_df.empty:
                return {}
            
            y_true = aligned_df['actual_label'].values
            y_pred = aligned_df['predicted_label'].values
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            result = {
                'matrix': cm.tolist(),
                'labels': ['下跌(0)', '上涨(1)'],
                'true_positive': int(cm[1, 1]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_negative': int(cm[0, 0])
            }
            
            logger.info(f"混淆矩阵生成成功")
            return result
            
        except Exception as e:
            logger.error(f"生成混淆矩阵失败: {e}")
            return {}
    
    def check_performance_targets(self, metrics: Dict) -> Tuple[bool, Dict[str, bool]]:
        """
        检查性能目标是否达成
        
        Args:
            metrics: 指标字典
            
        Returns:
            (是否所有目标都达成, 各目标达成情况)
        """
        target_status = {}
        all_met = True
        
        # 检查各项目标
        for target_name, target_value in self.targets.items():
            actual_value = metrics.get(target_name, 0)
            
            # 对于某些指标（如max_drawdown）是越小越好，其他是越大越好
            if target_name in ['max_drawdown']:
                met = actual_value >= target_value  # 回撤要小于等于目标值
            else:
                met = actual_value >= target_value
            
            target_status[target_name] = met
            
            if not met:
                all_met = False
                logger.warning(f"目标未达标: {target_name} 实际值={actual_value:.4f}, 目标值={target_value:.4f}")
        
        logger.info(f"性能目标检查完成，全部达标={all_met}")
        return all_met, target_status
    
    def should_trigger_adjustment(self, metrics: Dict) -> Tuple[bool, str]:
        """
        判断是否应该触发参数调整
        
        Args:
            metrics: 当前指标
            
        Returns:
            (是否触发, 触发原因)
        """
        recall = metrics.get('recall', 0)
        precision = metrics.get('precision', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        min_recall = self.trigger_thresholds['min_recall']
        min_precision = self.trigger_thresholds['min_precision']
        min_sharpe_ratio = self.trigger_thresholds['min_sharpe_ratio']
        
        reasons = []
        
        if recall < min_recall:
            reasons.append(f"召回率过低: {recall:.4f} < {min_recall}")
        
        if precision < min_precision:
            reasons.append(f"精确率过低: {precision:.4f} < {min_precision}")
        
        if sharpe_ratio < min_sharpe_ratio:
            reasons.append(f"夏普比率过低: {sharpe_ratio:.4f} < {min_sharpe_ratio}")
        
        should_adjust = len(reasons) > 0
        reason_str = '; '.join(reasons) if reasons else "指标均达标"
        
        if should_adjust:
            logger.warning(f"需要触发参数调整: {reason_str}")
        else:
            logger.info("指标均达标，无需调整")
        
        return should_adjust, reason_str
    
    def generate_report(self, metrics: Dict, confusion_matrix: Dict = None) -> str:
        """
        生成分析报告
        
        Args:
            metrics: 指标字典
            confusion_matrix: 混淆矩阵
            
        Returns:
            Markdown格式的报告
        """
        report = []
        report.append("# 股票预测模型性能分析报告\n")
        report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. 核心指标
        report.append("## 1. 核心性能指标\n")
        report.append("| 指标 | 数值 | 目标 | 达标 |\n")
        report.append("|------|------|------|------|\n")
        
        for metric_name, target_value in self.targets.items():
            actual_value = metrics.get(metric_name, 0)
            met = actual_value >= target_value
            met_str = "✓" if met else "✗"
            report.append(f"| {metric_name} | {actual_value:.4f} | {target_value:.4f} | {met_str} |\n")
        
        # 2. 混淆矩阵
        if confusion_matrix:
            report.append("\n## 2. 混淆矩阵\n")
            cm = confusion_matrix['matrix']
            report.append(f"```\n")
            report.append(f"                实际下跌  实际上涨\n")
            report.append(f"预测下跌        {cm[0][0]:6d}   {cm[0][1]:6d}\n")
            report.append(f"预测上涨        {cm[1][0]:6d}   {cm[1][1]:6d}\n")
            report.append(f"```\n")
            
            report.append(f"- 假正例(False Positive): {confusion_matrix['false_positive']} (误判为上涨)\n")
            report.append(f"- 假负例(False Negative): {confusion_matrix['false_negative']} (误判为下跌)\n")
        
        # 3. 误差分析
        report.append("\n## 3. 误差分析\n")
        fp_rate = metrics.get('false_positive_rate', 0) * 100
        fn_rate = metrics.get('false_negative_rate', 0) * 100
        report.append(f"- 假正率: {fp_rate:.2f}% (误判为上涨的比例)\n")
        report.append(f"- 假负率: {fn_rate:.2f}% (误判为下跌的比例)\n")
        
        # 4. 投资组合表现
        report.append("\n## 4. 投资组合表现\n")
        report.append(f"- 累计收益率: {metrics.get('cumulative_return', 0)*100:.2f}%\n")
        report.append(f"- 年化收益率: {metrics.get('annual_return', 0)*100:.2f}%\n")
        report.append(f"- 夏普比率: {metrics.get('sharpe_ratio', 0):.4f}\n")
        report.append(f"- 最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%\n")
        report.append(f"- 胜率: {metrics.get('win_rate', 0)*100:.2f}%\n")
        
        return ''.join(report)
    
    def save_report(self, report: str, filename: str = None):
        """
        保存报告
        
        Args:
            report: 报告内容
            filename: 文件名
        """
        try:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            if filename is None:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"performance_report_{timestamp}.md"
            
            save_path = os.path.join(workspace_path, "assets/logs", filename)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"保存报告成功: {save_path}")
        except Exception as e:
            logger.error(f"保存报告失败: {e}")


def test_analyzer():
    """测试分析器"""
    analyzer = PerformanceAnalyzer()
    
    # 创建测试数据
    print("\n=== 测试对比分析 ===")
    np.random.seed(42)
    
    # 模拟预测结果
    predictions = {}
    for i in range(10):
        pred_df = pd.DataFrame({
            'ts_code': [f'60000{i}.SH'],
            'trade_date': ['20241201'],
            'predicted_label': [np.random.randint(0, 2)],
            'predicted_prob': [np.random.random()]
        })
        predictions[f'60000{i}.SH'] = pred_df
    
    # 模拟实际行情
    actual_data = {}
    for i in range(10):
        dates = pd.date_range('20241201', periods=10)
        actual_df = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': np.random.randn(10) * 0.1 + 10,
            'vol': np.random.randn(10) * 1000 + 100000,
            'amount': np.random.randn(10) * 1000000 + 10000000
        })
        actual_data[f'60000{i}.SH'] = actual_df
    
    # 对齐并分析
    aligned_df = analyzer.align_predictions_with_actuals(predictions, actual_data)
    print(f"\n对齐数据前5行:\n{aligned_df.head()}")
    
    metrics = analyzer.calculate_metrics(aligned_df)
    print(f"\n核心指标:\n{metrics}")
    
    confusion_mat = analyzer.generate_confusion_matrix(aligned_df)
    print(f"\n混淆矩阵:\n{confusion_mat}")
    
    should_adjust, reason = analyzer.should_trigger_adjustment(metrics)
    print(f"\n是否需要调整: {should_adjust}, 原因: {reason}")
    
    report = analyzer.generate_report(metrics, confusion_mat)
    print(f"\n分析报告:\n{report}")


if __name__ == '__main__':
    test_analyzer()
