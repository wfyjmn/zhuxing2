"""
置信度分桶分析器
分析不同置信度区间的精确率，支持基于置信度的决策策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')


class ConfidenceBucketAnalyzer:
    """
    置信度分桶分析器
    
    功能：
    1. 将预测概率分桶（如 [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]）
    2. 计算每个桶的精确率、召回率等指标
    3. 生成置信度-精确率曲线
    4. 支持自定义分桶策略
    """
    
    def __init__(
        self,
        n_buckets: int = 10,
        bucket_type: str = 'uniform',
        custom_boundaries: Optional[List[float]] = None
    ):
        """
        初始化置信度分桶分析器
        
        参数:
            n_buckets: 分桶数量（仅适用于uniform类型）
            bucket_type: 分桶类型
                - 'uniform': 均匀分桶 [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
                - 'quantile': 分位数分桶（每个桶样本数相同）
                - 'custom': 自定义边界
            custom_boundaries: 自定义边界列表（仅适用于custom类型）
                              例如 [0.3, 0.5, 0.7, 0.9]
        """
        self.n_buckets = n_buckets
        self.bucket_type = bucket_type
        self.custom_boundaries = custom_boundaries
        
        self.buckets = []
        self.bucket_results = {}
        self.confidence_precision_curve = None
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """验证参数有效性"""
        if self.bucket_type not in ['uniform', 'quantile', 'custom']:
            raise ValueError(
                f"bucket_type 必须是 'uniform', 'quantile' 或 'custom'，"
                f"得到的是 {self.bucket_type}"
            )
        
        if self.bucket_type == 'custom' and self.custom_boundaries is None:
            raise ValueError("custom类型需要提供 custom_boundaries")
        
        if self.custom_boundaries is not None:
            if not all(0 < b < 1 for b in self.custom_boundaries):
                raise ValueError("边界必须在 (0, 1) 范围内")
            if sorted(self.custom_boundaries) != self.custom_boundaries:
                raise ValueError("边界必须按升序排列")
    
    def _create_uniform_buckets(self) -> List[Tuple[float, float]]:
        """创建均匀分桶"""
        buckets = []
        step = 1.0 / self.n_buckets
        
        for i in range(self.n_buckets):
            lower = i * step
            upper = (i + 1) * step
            buckets.append((lower, upper))
        
        return buckets
    
    def _create_quantile_buckets(
        self,
        y_proba: np.ndarray
    ) -> List[Tuple[float, float]]:
        """创建分位数分桶"""
        quantiles = np.linspace(0, 1, self.n_buckets + 1)
        boundaries = np.quantile(y_proba, quantiles)
        
        buckets = []
        for i in range(self.n_buckets):
            lower = boundaries[i]
            upper = boundaries[i + 1]
            buckets.append((lower, upper))
        
        return buckets
    
    def _create_custom_buckets(self) -> List[Tuple[float, float]]:
        """创建自定义分桶"""
        boundaries = [0.0] + self.custom_boundaries + [1.0]
        buckets = []
        
        for i in range(len(boundaries) - 1):
            lower = boundaries[i]
            upper = boundaries[i + 1]
            buckets.append((lower, upper))
        
        return buckets
    
    def analyze(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        分析置信度分桶
        
        参数:
            y_true: 真实标签
            y_proba: 预测概率
            threshold: 分类阈值
            
        返回:
            分析结果字典
        """
        # 验证输入
        if len(y_true) != len(y_proba):
            raise ValueError("y_true 和 y_proba 长度不一致")
        
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        
        # 创建分桶
        if self.bucket_type == 'uniform':
            self.buckets = self._create_uniform_buckets()
        elif self.bucket_type == 'quantile':
            self.buckets = self._create_quantile_buckets(y_proba)
        else:
            self.buckets = self._create_custom_buckets()
        
        # 分析每个桶
        self.bucket_results = {}
        
        for i, (lower, upper) in enumerate(self.buckets):
            # 找到落在该桶内的样本
            mask = (y_proba >= lower) & (y_proba < upper)
            if i == len(self.buckets) - 1:  # 最后一个桶包含上界
                mask = (y_proba >= lower) & (y_proba <= upper)
            
            bucket_y_true = y_true[mask]
            bucket_y_proba = y_proba[mask]
            bucket_y_pred = (bucket_y_proba >= threshold).astype(int)
            
            # 计算指标
            n_samples = len(bucket_y_true)
            n_positive = bucket_y_true.sum()
            positive_ratio = n_positive / n_samples if n_samples > 0 else 0
            
            if n_samples > 0 and n_positive > 0:
                precision = precision_score(
                    bucket_y_true, bucket_y_pred, zero_division=0
                )
                recall = recall_score(
                    bucket_y_true, bucket_y_pred, zero_division=0
                )
            else:
                precision = 0.0
                recall = 0.0
            
            # 平均置信度
            avg_confidence = bucket_y_proba.mean() if n_samples > 0 else 0.0
            
            # 混淆矩阵
            if n_samples > 0 and np.sum(bucket_y_pred) > 0:
                cm = confusion_matrix(bucket_y_true, bucket_y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            self.bucket_results[f'bucket_{i}'] = {
                'bucket_id': i,
                'lower_bound': lower,
                'upper_bound': upper,
                'n_samples': n_samples,
                'n_positive': n_positive,
                'positive_ratio': positive_ratio,
                'precision': precision,
                'recall': recall,
                'avg_confidence': avg_confidence,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
        
        # 生成置信度-精确率曲线
        self.confidence_precision_curve = self._generate_confidence_precision_curve(
            y_true, y_proba
        )
        
        # 计算整体统计
        overall_stats = self._calculate_overall_stats(y_true, y_proba, threshold)
        
        return {
            'bucket_results': self.bucket_results,
            'confidence_precision_curve': self.confidence_precision_curve,
            'overall_stats': overall_stats,
            'bucket_type': self.bucket_type,
            'n_buckets': len(self.buckets)
        }
    
    def _generate_confidence_precision_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        生成置信度-精确率曲线
        
        参数:
            y_true: 真实标签
            y_proba: 预测概率
            n_points: 曲线点数
            
        返回:
            包含置信度和对应精确率的DataFrame
        """
        thresholds = np.linspace(0, 1, n_points)
        curve_data = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if y_pred.sum() > 0:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                n_selected = y_pred.sum()
            else:
                precision = 0.0
                recall = 0.0
                n_selected = 0
            
            curve_data.append({
                'threshold': threshold,
                'confidence': threshold,
                'precision': precision,
                'recall': recall,
                'n_selected': n_selected
            })
        
        return pd.DataFrame(curve_data)
    
    def _calculate_overall_stats(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """计算整体统计"""
        y_pred = (y_proba >= threshold).astype(int)
        
        overall_precision = precision_score(y_true, y_pred, zero_division=0)
        overall_recall = recall_score(y_true, y_pred, zero_division=0)
        
        # 高置信度样本的精确率（>= 0.8）
        high_conf_mask = y_proba >= 0.8
        if high_conf_mask.sum() > 0:
            high_conf_precision = precision_score(
                y_true[high_conf_mask],
                y_pred[high_conf_mask],
                zero_division=0
            )
        else:
            high_conf_precision = 0.0
        
        # 低置信度样本的精确率（< 0.5）
        low_conf_mask = y_proba < 0.5
        if low_conf_mask.sum() > 0:
            low_conf_precision = precision_score(
                y_true[low_conf_mask],
                y_pred[low_conf_mask],
                zero_division=0
            )
        else:
            low_conf_precision = 0.0
        
        # 平均置信度
        avg_confidence = y_proba.mean()
        
        return {
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'high_confidence_precision': high_conf_precision,
            'low_confidence_precision': low_conf_precision,
            'avg_confidence': avg_confidence,
            'n_high_confidence': int(high_conf_mask.sum()),
            'n_low_confidence': int(low_conf_mask.sum())
        }
    
    def get_high_precision_buckets(
        self,
        min_precision: float = 0.7,
        min_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取高精确率的桶
        
        参数:
            min_precision: 最小精确率
            min_samples: 最小样本数
            
        返回:
            高精确率桶列表
        """
        high_precision_buckets = []
        
        for bucket_name, bucket_data in self.bucket_results.items():
            if (bucket_data['precision'] >= min_precision and
                bucket_data['n_samples'] >= min_samples):
                high_precision_buckets.append(bucket_data)
        
        return sorted(
            high_precision_buckets,
            key=lambda x: x['avg_confidence'],
            reverse=True
        )
    
    def get_optimal_threshold(
        self,
        metric: str = 'precision',
        target_value: float = 0.7,
        min_samples: int = 5
    ) -> Tuple[float, float]:
        """
        获取最优阈值
        
        参数:
            metric: 优化指标 ('precision' 或 'recall')
            target_value: 目标值
            min_samples: 最小样本数
            
        返回:
            (最优阈值, 对应指标值)
        """
        if self.confidence_precision_curve is None:
            raise ValueError("请先调用 analyze 方法")
        
        curve = self.confidence_precision_curve
        
        # 过滤样本数不足的点
        valid_curve = curve[curve['n_selected'] >= min_samples]
        
        if len(valid_curve) == 0:
            return 0.5, 0.0
        
        if metric == 'precision':
            # 找到满足精确率目标的最小阈值
            valid_points = valid_curve[valid_curve['precision'] >= target_value]
            
            if len(valid_points) > 0:
                # 选择最小阈值（最大化召回率）
                best_idx = valid_points['threshold'].idxmin()
                optimal_threshold = valid_points.loc[best_idx, 'threshold']
                optimal_precision = valid_points.loc[best_idx, 'precision']
            else:
                # 如果没有满足目标的，选择精确率最高的
                best_idx = valid_points['precision'].idxmax()
                optimal_threshold = valid_points.loc[best_idx, 'threshold']
                optimal_precision = valid_points.loc[best_idx, 'precision']
        
        elif metric == 'recall':
            # 找到满足召回率目标的最大阈值
            valid_points = valid_curve[valid_curve['recall'] >= target_value]
            
            if len(valid_points) > 0:
                # 选择最大阈值（最大化精确率）
                best_idx = valid_points['threshold'].idxmax()
                optimal_threshold = valid_points.loc[best_idx, 'threshold']
                optimal_recall = valid_points.loc[best_idx, 'recall']
            else:
                # 如果没有满足目标的，选择召回率最高的
                best_idx = valid_curve['recall'].idxmax()
                optimal_threshold = valid_points.loc[best_idx, 'threshold']
                optimal_recall = valid_points.loc[best_idx, 'recall']
        else:
            raise ValueError(f"未知的指标: {metric}")
        
        metric_value = valid_points.loc[best_idx, metric] if metric in valid_points.columns else 0.0
        
        return optimal_threshold, metric_value
    
    def generate_report(
        self,
        save_path: Optional[str] = None
    ) -> str:
        """
        生成分桶分析报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        if not self.bucket_results:
            raise ValueError("请先调用 analyze 方法")
        
        report_lines = []
        
        report_lines.append("# 置信度分桶分析报告")
        report_lines.append("\n## 一、分桶配置")
        report_lines.append(f"- 分桶类型: {self.bucket_type}")
        report_lines.append(f"- 分桶数量: {len(self.buckets)}")
        
        report_lines.append("\n## 二、分桶详情")
        
        for bucket_name, bucket_data in self.bucket_results.items():
            report_lines.append(f"\n### {bucket_name}")
            report_lines.append(f"- 置信度区间: [{bucket_data['lower_bound']:.2f}, {bucket_data['upper_bound']:.2f})")
            report_lines.append(f"- 样本数: {bucket_data['n_samples']}")
            report_lines.append(f"- 正样本数: {bucket_data['n_positive']}")
            report_lines.append(f"- 正样本比例: {bucket_data['positive_ratio']:.2%}")
            report_lines.append(f"- 平均置信度: {bucket_data['avg_confidence']:.4f}")
            report_lines.append(f"- 精确率: {bucket_data['precision']:.4f}")
            report_lines.append(f"- 召回率: {bucket_data['recall']:.4f}")
            report_lines.append(f"- 真阳性: {bucket_data['true_positives']}")
            report_lines.append(f"- 假阳性: {bucket_data['false_positives']}")
        
        # 高精确率桶
        high_precision_buckets = self.get_high_precision_buckets(min_precision=0.7, min_samples=10)
        
        report_lines.append("\n## 三、高精确率桶（精确率 >= 70%）")
        
        if high_precision_buckets:
            for bucket in high_precision_buckets:
                report_lines.append(f"\n- 置信度区间: [{bucket['lower_bound']:.2f}, {bucket['upper_bound']:.2f})")
                report_lines.append(f"  精确率: {bucket['precision']:.4f}")
                report_lines.append(f"  样本数: {bucket['n_samples']}")
        else:
            report_lines.append("  无高精确率桶")
        
        # 整体统计
        if self.bucket_results:
            # 从第一个桶的结果中获取整体统计（需要重新计算）
            # 这里简化处理，只显示分桶结果
            pass
        
        report_lines.append("\n## 四、建议")
        
        # 分析建议
        if high_precision_buckets:
            best_bucket = high_precision_buckets[0]
            report_lines.append(f"\n- 建议使用高置信度区间 [{best_bucket['lower_bound']:.2f}, {best_bucket['upper_bound']:.2f})")
            report_lines.append(f"  该区间精确率为 {best_bucket['precision']:.4f}")
            report_lines.append(f"  预期可以减少误报")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class ConfidenceBasedFilter:
    """
    基于置信度的过滤器
    
    功能：
    1. 根据置信度对预测进行分级
    2. 高置信度预测自动放行
    3. 低置信度预测标记为人工复核
    4. 极低置信度预测拒绝
    """
    
    def __init__(
        self,
        auto_trade_threshold: float = 0.8,
        manual_review_threshold: float = 0.5,
        reject_threshold: float = 0.3
    ):
        """
        初始化基于置信度的过滤器
        
        参数:
            auto_trade_threshold: 自动交易阈值（>=此值自动执行）
            manual_review_threshold: 人工复核阈值（>=此值人工复核）
            reject_threshold: 拒绝阈值（<此值直接拒绝）
        """
        if not (0 < reject_threshold < manual_review_threshold < auto_trade_threshold <= 1):
            raise ValueError("阈值必须满足: 0 < reject < manual_review < auto_trade <= 1")
        
        self.auto_trade_threshold = auto_trade_threshold
        self.manual_review_threshold = manual_review_threshold
        self.reject_threshold = reject_threshold
        
        self.filter_stats = {
            'auto_trade': 0,
            'manual_review': 0,
            'reject': 0,
            'total': 0
        }
    
    def filter(
        self,
        y_proba: np.ndarray,
        return_labels: bool = True
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        对预测进行过滤
        
        参数:
            y_proba: 预测概率
            return_labels: 是否返回标签
            
        返回:
            (决策标签, 统计信息)
            决策标签: 0=拒绝, 1=人工复核, 2=自动交易
        """
        y_proba = np.asarray(y_proba)
        
        # 初始化决策标签
        if return_labels:
            decisions = np.zeros(len(y_proba), dtype=int)
        
        # 分类统计
        stats = {
            'auto_trade': 0,
            'manual_review': 0,
            'reject': 0,
            'total': len(y_proba)
        }
        
        # 应用过滤器
        for i, proba in enumerate(y_proba):
            if proba >= self.auto_trade_threshold:
                if return_labels:
                    decisions[i] = 2  # 自动交易
                stats['auto_trade'] += 1
            elif proba >= self.manual_review_threshold:
                if return_labels:
                    decisions[i] = 1  # 人工复核
                stats['manual_review'] += 1
            elif proba >= self.reject_threshold:
                if return_labels:
                    decisions[i] = 1  # 人工复核（低置信度）
                stats['manual_review'] += 1
            else:
                if return_labels:
                    decisions[i] = 0  # 拒绝
                stats['reject'] += 1
        
        self.filter_stats = stats
        
        if return_labels:
            return decisions, stats
        else:
            return stats
    
    def evaluate_filter_effectiveness(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        评估过滤器的有效性
        
        参数:
            y_true: 真实标签
            y_proba: 预测概率
            threshold: 基础分类阈值
            
        返回:
            评估结果
        """
        # 应用过滤器
        decisions, stats = self.filter(y_proba, return_labels=True)
        
        # 计算每个决策类别的精确率
        y_pred = (y_proba >= threshold).astype(int)
        
        results = {}
        
        # 自动交易的精确率
        auto_trade_mask = decisions == 2
        if auto_trade_mask.sum() > 0:
            results['auto_trade_precision'] = precision_score(
                y_true[auto_trade_mask],
                y_pred[auto_trade_mask],
                zero_division=0
            )
            results['auto_trade_count'] = int(auto_trade_mask.sum())
        else:
            results['auto_trade_precision'] = 0.0
            results['auto_trade_count'] = 0
        
        # 人工复核的精确率
        manual_review_mask = decisions == 1
        if manual_review_mask.sum() > 0:
            results['manual_review_precision'] = precision_score(
                y_true[manual_review_mask],
                y_pred[manual_review_mask],
                zero_division=0
            )
            results['manual_review_count'] = int(manual_review_mask.sum())
        else:
            results['manual_review_precision'] = 0.0
            results['manual_review_count'] = 0
        
        # 拒绝的比例
        reject_mask = decisions == 0
        results['reject_rate'] = reject_mask.sum() / len(y_true)
        results['reject_count'] = int(reject_mask.sum())
        
        # 整体精确率（不使用过滤器）
        results['overall_precision'] = precision_score(y_true, y_pred, zero_division=0)
        
        # 精确率提升
        results['precision_improvement'] = results['auto_trade_precision'] - results['overall_precision']
        
        # 自动交易覆盖率
        results['auto_trade_coverage'] = stats['auto_trade'] / stats['total']
        
        return results
    
    def generate_filter_report(
        self,
        evaluation: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        生成过滤器报告
        
        参数:
            evaluation: 评估结果
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        report_lines = []
        
        report_lines.append("# 基于置信度的过滤器报告")
        report_lines.append("\n## 一、过滤器配置")
        report_lines.append(f"- 自动交易阈值: >= {self.auto_trade_threshold}")
        report_lines.append(f"- 人工复核阈值: >= {self.manual_review_threshold}")
        report_lines.append(f"- 拒绝阈值: < {self.reject_threshold}")
        
        report_lines.append("\n## 二、过滤效果")
        report_lines.append(f"- 自动交易数量: {evaluation.get('auto_trade_count', 0)}")
        report_lines.append(f"- 人工复核数量: {evaluation.get('manual_review_count', 0)}")
        report_lines.append(f"- 拒绝数量: {evaluation.get('reject_count', 0)}")
        report_lines.append(f"- 自动交易覆盖率: {evaluation.get('auto_trade_coverage', 0):.2%}")
        
        report_lines.append("\n## 三、精确率分析")
        report_lines.append(f"- 整体精确率: {evaluation.get('overall_precision', 0):.4f}")
        report_lines.append(f"- 自动交易精确率: {evaluation.get('auto_trade_precision', 0):.4f}")
        report_lines.append(f"- 人工复核精确率: {evaluation.get('manual_review_precision', 0):.4f}")
        report_lines.append(f"- 精确率提升: {evaluation.get('precision_improvement', 0):.4f}")
        
        report_lines.append("\n## 四、建议")
        
        if evaluation.get('precision_improvement', 0) > 0.1:
            report_lines.append("\n✅ 过滤器有效，显著提升了精确率")
            report_lines.append(f"   精确率提升了 {evaluation.get('precision_improvement', 0):.2%}")
        elif evaluation.get('precision_improvement', 0) > 0:
            report_lines.append("\n⚠️ 过滤器有效，但提升幅度较小")
            report_lines.append(f"   精确率提升了 {evaluation.get('precision_improvement', 0):.2%}")
        else:
            report_lines.append("\n❌ 过滤器未提升精确率，建议调整阈值")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
