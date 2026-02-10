"""
误差溯源模块
功能：分析预测误差来源，定位问题
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import Counter
import xgboost as xgb

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ErrorTracker:
    """误差追踪器"""
    
    def __init__(self, predictor=None, config_path: str = None):
        """
        初始化误差追踪器
        
        Args:
            predictor: 股票预测器实例
            config_path: 配置文件路径
        """
        if config_path is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            config_path = os.path.join(workspace_path, "config/model_config.json")
        
        self.config = self._load_config(config_path)
        self.predictor = predictor
        self.features = self.config['data']['train_features']
        
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
    
    def analyze_errors(self, aligned_df: pd.DataFrame) -> Dict:
        """
        分析误差分布
        
        Args:
            aligned_df: 对齐后的预测数据
            
        Returns:
            误差分析结果
        """
        try:
            if aligned_df.empty:
                logger.warning("对齐数据为空，无法分析误差")
                return {}
            
            error_analysis = {}
            
            # 1. 总体误差统计
            total = len(aligned_df)
            errors = aligned_df[aligned_df['predict_correct'] == 0]
            correct = aligned_df[aligned_df['predict_correct'] == 1]
            
            error_analysis['total_samples'] = total
            error_analysis['error_count'] = len(errors)
            error_analysis['correct_count'] = len(correct)
            error_analysis['error_rate'] = len(errors) / total if total > 0 else 0
            
            # 2. 假正例分析（预测为上涨，实际为下跌）
            false_positives = aligned_df[
                (aligned_df['predicted_label'] == 1) & 
                (aligned_df['actual_label'] == 0)
            ]
            error_analysis['false_positive_count'] = len(false_positives)
            error_analysis['false_positive_rate'] = len(false_positives) / total if total > 0 else 0
            
            # 3. 假负例分析（预测为下跌，实际为上涨）
            false_negatives = aligned_df[
                (aligned_df['predicted_label'] == 0) & 
                (aligned_df['actual_label'] == 1)
            ]
            error_analysis['false_negative_count'] = len(false_negatives)
            error_analysis['false_negative_rate'] = len(false_negatives) / total if total > 0 else 0
            
            # 4. 概率分布分析
            error_analysis['probability_distribution'] = self._analyze_probability_distribution(aligned_df)
            
            # 5. 涨跌幅分析
            error_analysis['price_change_analysis'] = self._analyze_price_change_distribution(aligned_df)
            
            logger.info(f"误差分析完成")
            return error_analysis
            
        except Exception as e:
            logger.error(f"分析误差失败: {e}")
            return {}
    
    def _analyze_probability_distribution(self, aligned_df: pd.DataFrame) -> Dict:
        """
        分析预测概率分布
        
        Args:
            aligned_df: 对齐数据
            
        Returns:
            概率分布统计
        """
        try:
            # 按预测正确性分组
            correct = aligned_df[aligned_df['predict_correct'] == 1]
            errors = aligned_df[aligned_df['predict_correct'] == 0]
            
            distribution = {
                'correct': {
                    'mean_prob': float(correct['predicted_prob'].mean()),
                    'std_prob': float(correct['predicted_prob'].std()),
                    'min_prob': float(correct['predicted_prob'].min()),
                    'max_prob': float(correct['predicted_prob'].max())
                },
                'errors': {
                    'mean_prob': float(errors['predicted_prob'].mean()),
                    'std_prob': float(errors['predicted_prob'].std()),
                    'min_prob': float(errors['predicted_prob'].min()),
                    'max_prob': float(errors['predicted_prob'].max())
                }
            }
            
            # 分析概率区间
            prob_bins = [0.0, 0.3, 0.5, 0.7, 1.0]
            aligned_df['prob_bin'] = pd.cut(aligned_df['predicted_prob'], bins=prob_bins)
            
            bin_analysis = {}
            for bin_name, group in aligned_df.groupby('prob_bin'):
                bin_analysis[str(bin_name)] = {
                    'total': len(group),
                    'errors': len(group[group['predict_correct'] == 0]),
                    'error_rate': len(group[group['predict_correct'] == 0]) / len(group) if len(group) > 0 else 0
                }
            
            distribution['by_bin'] = bin_analysis
            
            return distribution
        except Exception as e:
            logger.error(f"分析概率分布失败: {e}")
            return {}
    
    def _analyze_price_change_distribution(self, aligned_df: pd.DataFrame) -> Dict:
        """
        分析涨跌幅分布
        
        Args:
            aligned_df: 对齐数据
            
        Returns:
            涨跌幅统计
        """
        try:
            # 按预测正确性分组
            correct = aligned_df[aligned_df['predict_correct'] == 1]
            errors = aligned_df[aligned_df['predict_correct'] == 0]
            
            analysis = {
                'correct': {
                    'mean_change': float(correct['actual_change'].mean()),
                    'std_change': float(correct['actual_change'].std()),
                    'abs_mean_change': float(abs(correct['actual_change']).mean())
                },
                'errors': {
                    'mean_change': float(errors['actual_change'].mean()),
                    'std_change': float(errors['actual_change'].std()),
                    'abs_mean_change': float(abs(errors['actual_change']).mean())
                }
            }
            
            return analysis
        except Exception as e:
            logger.error(f"分析涨跌幅分布失败: {e}")
            return {}
    
    def identify_error_stocks(self, aligned_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        识别误差最大的股票
        
        Args:
            aligned_df: 对齐数据
            top_n: 返回前N只股票
            
        Returns:
            误差股票DataFrame
        """
        try:
            if aligned_df.empty:
                return pd.DataFrame()
            
            # 添加误差幅度
            aligned_df['error_magnitude'] = abs(aligned_df['actual_change'])
            
            # 按误差幅度排序
            error_stocks = aligned_df[
                aligned_df['predict_correct'] == 0
            ].sort_values('error_magnitude', ascending=False).head(top_n)
            
            return error_stocks
        except Exception as e:
            logger.error(f"识别误差股票失败: {e}")
            return pd.DataFrame()
    
    def get_feature_importance(self) -> Dict:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        try:
            if self.predictor is None or self.predictor.model is None:
                logger.warning("预测器或模型未加载，无法获取特征重要性")
                return {}
            
            # 获取特征重要性
            importance = self.predictor.model.get_score(importance_type='gain')
            
            # 转换为DataFrame
            importance_df = pd.DataFrame([
                {'feature': feat, 'importance': importance.get(f'f{i}', 0)}
                for i, feat in enumerate(self.features)
            ]).sort_values('importance', ascending=False)
            
            result = {
                'feature_importance': importance_df.to_dict('records'),
                'top_features': importance_df.head(10)['feature'].tolist(),
                'low_importance_features': importance_df.tail(5)['feature'].tolist()
            }
            
            logger.info(f"获取特征重要性成功，Top特征: {result['top_features'][:5]}")
            return result
            
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return {}
    
    def analyze_error_by_threshold(self, aligned_df: pd.DataFrame) -> Dict:
        """
        分析不同阈值下的误差
        
        Args:
            aligned_df: 对齐数据
            
        Returns:
            不同阈值下的误差分析
        """
        try:
            if aligned_df.empty:
                return {}
            
            thresholds = np.arange(0.3, 0.6, 0.05)
            threshold_analysis = []
            
            for threshold in thresholds:
                # 使用新阈值重新预测
                new_predictions = (aligned_df['predicted_prob'] >= threshold).astype(int)
                new_labels = aligned_df['actual_label'].values
                
                # 计算指标
                tp = ((new_predictions == 1) & (new_labels == 1)).sum()
                fp = ((new_predictions == 1) & (new_labels == 0)).sum()
                fn = ((new_predictions == 0) & (new_labels == 1)).sum()
                tn = ((new_predictions == 0) & (new_labels == 0)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                threshold_analysis.append({
                    'threshold': float(threshold),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'fp_count': int(fp),
                    'fn_count': int(fn)
                })
            
            result = {
                'threshold_analysis': threshold_analysis,
                'best_threshold_for_precision': min(threshold_analysis, key=lambda x: x['precision']),
                'best_threshold_for_recall': min(threshold_analysis, key=lambda x: -x['recall']),
                'best_threshold_for_f1': min(threshold_analysis, key=lambda x: -x['f1'])
            }
            
            return result
        except Exception as e:
            logger.error(f"分析阈值误差失败: {e}")
            return {}
    
    def generate_error_report(self, aligned_df: pd.DataFrame, error_analysis: Dict = None) -> str:
        """
        生成误差分析报告
        
        Args:
            aligned_df: 对齐数据
            error_analysis: 误差分析结果
            
        Returns:
            Markdown格式的报告
        """
        if error_analysis is None:
            error_analysis = self.analyze_errors(aligned_df)
        
        report = []
        report.append("# 误差溯源分析报告\n")
        report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. 总体误差统计
        report.append("## 1. 总体误差统计\n")
        report.append(f"- 总样本数: {error_analysis.get('total_samples', 0)}\n")
        report.append(f"- 错误数量: {error_analysis.get('error_count', 0)}\n")
        report.append(f"- 正确数量: {error_analysis.get('correct_count', 0)}\n")
        report.append(f"- 误差率: {error_analysis.get('error_rate', 0)*100:.2f}%\n")
        
        # 2. 误差类型分析
        report.append("\n## 2. 误差类型分析\n")
        report.append(f"- 假正例(预测上涨实际下跌): {error_analysis.get('false_positive_count', 0)} ({error_analysis.get('false_positive_rate', 0)*100:.2f}%)\n")
        report.append(f"- 假负例(预测下跌实际上涨): {error_analysis.get('false_negative_count', 0)} ({error_analysis.get('false_negative_rate', 0)*100:.2f}%)\n")
        
        # 3. 概率分布
        if 'probability_distribution' in error_analysis:
            report.append("\n## 3. 预测概率分布\n")
            prob_dist = error_analysis['probability_distribution']
            report.append("### 正确预测\n")
            report.append(f"- 平均概率: {prob_dist['correct']['mean_prob']:.4f}\n")
            report.append(f"- 标准差: {prob_dist['correct']['std_prob']:.4f}\n")
            report.append("### 错误预测\n")
            report.append(f"- 平均概率: {prob_dist['errors']['mean_prob']:.4f}\n")
            report.append(f"- 标准差: {prob_dist['errors']['std_prob']:.4f}\n")
        
        # 4. 误差最大的股票
        report.append("\n## 4. 误差最大的股票\n")
        error_stocks = self.identify_error_stocks(aligned_df, top_n=5)
        if not error_stocks.empty:
            for _, row in error_stocks.iterrows():
                report.append(f"- {row['ts_code']}: 预测={'上涨' if row['predicted_label']==1 else '下跌'}, "
                            f"实际={'上涨' if row['actual_label']==1 else '下跌'}, "
                            f"涨跌幅={row['actual_change']*100:.2f}%\n")
        else:
            report.append("无误差数据\n")
        
        # 5. 调整建议
        report.append("\n## 5. 调整建议\n")
        
        fp_rate = error_analysis.get('false_positive_rate', 0)
        fn_rate = error_analysis.get('false_negative_rate', 0)
        
        if fp_rate > 0.3:
            report.append("- ⚠️ 假正例过多，建议提高分类阈值或调整scale_pos_weight\n")
        if fn_rate > 0.2:
            report.append("- ⚠️ 假负例过多，建议降低分类阈值或优化特征\n")
        
        if self.predictor:
            importance = self.get_feature_importance()
            if 'low_importance_features' in importance:
                report.append(f"- 💡 考虑移除重要性较低的特征: {', '.join(importance['low_importance_features'])}\n")
        
        return ''.join(report)
    
    def save_error_report(self, report: str, filename: str = None):
        """
        保存误差报告
        
        Args:
            report: 报告内容
            filename: 文件名
        """
        try:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            if filename is None:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"error_report_{timestamp}.md"
            
            save_path = os.path.join(workspace_path, "assets/logs", filename)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"保存误差报告成功: {save_path}")
        except Exception as e:
            logger.error(f"保存误差报告失败: {e}")


def test_error_tracker():
    """测试误差追踪器"""
    tracker = ErrorTracker()
    
    # 创建测试数据
    print("\n=== 测试误差分析 ===")
    np.random.seed(42)
    
    aligned_df = pd.DataFrame({
        'ts_code': [f'60000{i}.SH' for i in range(20)],
        'predict_date': ['20241201'] * 20,
        'actual_date': ['20241206'] * 20,
        'predicted_label': np.random.randint(0, 2, 20),
        'predicted_prob': np.random.random(20),
        'actual_label': np.random.randint(0, 2, 20),
        'actual_change': np.random.randn(20) * 0.05,
        'predict_correct': [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    })
    
    # 误差分析
    error_analysis = tracker.analyze_errors(aligned_df)
    print(f"\n误差分析:\n{error_analysis}")
    
    # 识别误差股票
    error_stocks = tracker.identify_error_stocks(aligned_df, top_n=5)
    print(f"\n误差最大的股票:\n{error_stocks}")
    
    # 阈值分析
    threshold_analysis = tracker.analyze_error_by_threshold(aligned_df)
    print(f"\n最优阈值分析:\n{threshold_analysis}")
    
    # 生成报告
    report = tracker.generate_error_report(aligned_df, error_analysis)
    print(f"\n误差报告:\n{report}")


if __name__ == '__main__':
    test_error_tracker()
