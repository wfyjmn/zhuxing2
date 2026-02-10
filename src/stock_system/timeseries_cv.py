"""
时间序列交叉验证模块
实现 Forward-Chaining（扩展窗口）和 Rolling（滑动窗口）交叉验证
专门用于处理时间序列数据，避免数据泄露
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Optional, Any
from sklearn.metrics import precision_score, recall_score, average_precision_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesSplit:
    """
    时间序列交叉验证基类
    
    支持两种策略：
    - Forward-Chaining（扩展窗口）：每个fold的训练集不断扩大
    - Rolling Window（滑动窗口）：每个fold的训练集大小固定
    
    时间顺序：
    Train_1 → Val_1
    Train_2 (= Train_1 + Val_1) → Val_2
    Train_3 (= Train_2 + Val_2) → Val_3
    ...
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        min_train_size: int = 100,
        gap: int = 0,
        strategy: str = 'forward'
    ):
        """
        初始化时间序列交叉验证器
        
        参数:
            n_splits: 交叉验证折数
            test_size: 每个fold的测试集大小（如果为None，则均匀划分）
            min_train_size: 最小训练集大小
            gap: 训练集和测试集之间的间隔（避免数据泄露）
            strategy: 'forward'（扩展窗口）或 'rolling'（滑动窗口）
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = gap
        self.strategy = strategy
        
        if strategy not in ['forward', 'rolling']:
            raise ValueError(f"策略必须是 'forward' 或 'rolling'，得到的是 {strategy}")
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练集和测试集的索引
        
        参数:
            X: 特征矩阵
            y: 目标变量（可选）
            groups: 分组变量（可选）
            
        返回:
            生成器，产生 (train_indices, test_indices) 元组
        """
        n_samples = X.shape[0]
        n_splits = self.n_splits
        
        # 计算每个fold的测试集大小
        if self.test_size is None:
            test_size = (n_samples - self.min_train_size) // (n_splits + 1)
        else:
            test_size = self.test_size
        
        # 生成fold索引
        folds = []
        
        for i in range(n_splits):
            # 测试集的起始和结束索引
            test_start = self.min_train_size + i * test_size
            test_end = test_start + test_size
            
            # 训练集的起始和结束索引
            if self.strategy == 'forward':
                # 扩展窗口：从开始到测试集之前
                train_start = 0
                train_end = test_start - self.gap
            else:
                # 滑动窗口：固定大小的训练集
                train_size = self.min_train_size
                train_start = max(0, test_start - train_size - self.gap)
                train_end = test_start - self.gap
            
            # 确保索引有效
            if test_end > n_samples:
                break
            
            if train_end <= train_start:
                warnings.warn(f"Fold {i+1}: 训练集太小，跳过")
                continue
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            folds.append((train_indices, test_indices))
        
        return folds
    
    def get_n_splits(self) -> int:
        """返回交叉验证的折数"""
        return self.n_splits


class PrecisionTimeSeriesCV:
    """
    面向精确率的时间序列交叉验证
    
    功能:
    1. 执行时间序列交叉验证
    2. 计算多个fold的精确率、召回率、AP等指标
    3. 计算统计显著性（mean±CI）
    4. 生成详细的交叉验证报告
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        min_train_size: int = 100,
        gap: int = 0,
        strategy: str = 'forward',
        confidence_level: float = 0.95
    ):
        """
        初始化精确率时间序列交叉验证
        
        参数:
            n_splits: 交叉验证折数
            test_size: 每个fold的测试集大小
            min_train_size: 最小训练集大小
            gap: 训练集和测试集之间的间隔
            strategy: 'forward'（扩展窗口）或 'rolling'（滑动窗口）
            confidence_level: 置信区间水平（0-1）
        """
        self.cv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=min_train_size,
            gap=gap,
            strategy=strategy
        )
        self.confidence_level = confidence_level
        
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        计算各种评估指标
        
        参数:
            y_true: 真实标签
            y_pred_proba: 预测概率
            threshold: 分类阈值
            
        返回:
            指标字典
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {}
        
        # 基础指标
        metrics['precision'] = precision_score(
            y_true, y_pred, zero_division=0
        )
        metrics['recall'] = recall_score(
            y_true, y_pred, zero_division=0
        )
        
        # 平均精度（AP）
        metrics['ap'] = average_precision_score(
            y_true, y_pred_proba
        )
        
        # Precision@k (前k个预测的精确率)
        k = min(20, len(y_pred_proba))
        top_k_indices = np.argsort(y_pred_proba)[-k:]
        top_k_precision = y_true[top_k_indices].mean()
        metrics['precision_at_k'] = top_k_precision
        
        # 预测的正样本数
        metrics['n_positive_pred'] = y_pred.sum()
        
        # 真实的正样本数
        metrics['n_positive_true'] = y_true.sum()
        
        # 总样本数
        metrics['n_samples'] = len(y_true)
        
        return metrics
    
    def _calculate_confidence_interval(
        self,
        values: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        计算均值和置信区间
        
        参数:
            values: 多个fold的指标值
            
        返回:
            (mean, lower_ci, upper_ci)
        """
        mean = np.mean(values)
        
        # 使用t分布计算置信区间
        n = len(values)
        if n < 2:
            return mean, mean, mean
        
        se = stats.sem(values)  # 标准误差
        t_score = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        
        margin = t_score * se
        
        lower_ci = mean - margin
        upper_ci = mean + margin
        
        return mean, lower_ci, upper_ci
    
    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Dict] = None,
        threshold: float = 0.5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        执行时间序列交叉验证
        
        参数:
            model: 可训练的模型对象（必须有fit和predict_proba方法）
            X: 特征矩阵
            y: 目标变量
            fit_params: fit方法的额外参数
            threshold: 分类阈值
            verbose: 是否打印详细信息
            
        返回:
            交叉验证结果字典
        """
        if fit_params is None:
            fit_params = {}
        
        folds = self.cv.split(X, y)
        
        # 存储每个fold的结果
        fold_results = []
        
        if verbose:
            print("=" * 70)
            print("时间序列交叉验证开始")
            print("=" * 70)
            print(f"策略: {self.cv.strategy}")
            print(f"折数: {len(folds)}")
            print(f"阈值: {threshold}")
            print(f"置信水平: {self.confidence_level * 100:.0f}%")
            print("=" * 70)
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            if verbose:
                print(f"\n【Fold {fold_idx + 1}】")
                print("-" * 70)
            
            # 划分数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 处理样本权重
            fold_fit_params = {}
            if fit_params and 'sample_weight' in fit_params:
                sample_weights = fit_params['sample_weight']
                fold_fit_params['sample_weight'] = sample_weights[train_idx]
            elif fit_params:
                fold_fit_params = fit_params.copy()
            
            if verbose:
                print(f"训练集: {len(X_train)} 样本")
                print(f"测试集: {len(X_test)} 样本")
                print(f"训练集正样本比例: {y_train.mean():.2%}")
                print(f"测试集正样本比例: {y_test.mean():.2%}")
            
            # 训练模型
            try:
                model.fit(X_train, y_train, **fold_fit_params)
            except Exception as e:
                if verbose:
                    print(f"训练失败: {str(e)}")
                continue
            
            # 预测
            try:
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.ndim > 1:
                    y_pred_proba = y_pred_proba[:, 1]  # 取正类的概率
            except Exception as e:
                if verbose:
                    print(f"预测失败: {str(e)}")
                continue
            
            # 计算指标
            metrics = self._calculate_metrics(y_test, y_pred_proba, threshold)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_positive_ratio': y_train.mean(),
                'test_positive_ratio': y_test.mean(),
                **metrics
            })
            
            if verbose:
                print(f"精确率: {metrics['precision']:.4f}")
                print(f"召回率: {metrics['recall']:.4f}")
                print(f"平均精度(AP): {metrics['ap']:.4f}")
                print(f"Precision@{min(20, len(y_test))}: {metrics['precision_at_k']:.4f}")
        
        if not fold_results:
            raise ValueError("交叉验证失败，没有任何fold成功")
        
        # 汇总结果
        summary = self._summarize_results(fold_results)
        
        if verbose:
            self._print_summary(summary)
        
        return {
            'fold_results': fold_results,
            'summary': summary
        }
    
    def _summarize_results(
        self,
        fold_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        汇总交叉验证结果，计算统计显著性
        
        参数:
            fold_results: 每个fold的结果列表
            
        返回:
            汇总结果字典
        """
        summary = {}
        
        # 提取关键指标
        metrics_keys = ['precision', 'recall', 'ap', 'precision_at_k']
        
        for metric in metrics_keys:
            values = [fold[metric] for fold in fold_results]
            
            # 计算均值和置信区间
            mean, lower_ci, upper_ci = self._calculate_confidence_interval(
                np.array(values)
            )
            
            # 计算标准差
            std = np.std(values)
            
            summary[metric] = {
                'mean': mean,
                'std': std,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'values': values
            }
        
        # 计算稳定性（变异系数 CV = std/mean）
        summary['stability'] = {}
        for metric in metrics_keys:
            mean = summary[metric]['mean']
            std = summary[metric]['std']
            if mean > 0:
                cv = std / mean
                summary['stability'][metric] = cv
            else:
                summary['stability'][metric] = np.inf
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """打印汇总结果"""
        print("\n" + "=" * 70)
        print("交叉验证汇总（Mean ± 95% CI）")
        print("=" * 70)
        
        metrics_display = {
            'precision': '精确率（Precision）',
            'recall': '召回率（Recall）',
            'ap': '平均精度（AP）',
            'precision_at_k': 'Precision@20'
        }
        
        for metric_key, display_name in metrics_display.items():
            if metric_key in summary:
                result = summary[metric_key]
                mean = result['mean']
                lower_ci = result['lower_ci']
                upper_ci = result['upper_ci']
                cv = summary['stability'][metric_key]
                
                print(f"\n{display_name}:")
                print(f"  均值: {mean:.4f}")
                print(f"  95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
                print(f"  标准差: {result['std']:.4f}")
                print(f"  变异系数(CV): {cv:.4f} {'(稳定)' if cv < 0.1 else '(较不稳定)' if cv < 0.2 else '(不稳定)'}")
        
        print("\n" + "=" * 70)
    
    def generate_report(
        self,
        cv_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        生成交叉验证报告
        
        参数:
            cv_results: 交叉验证结果
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        fold_results = cv_results['fold_results']
        summary = cv_results['summary']
        
        report_lines = []
        
        report_lines.append("# 时间序列交叉验证报告")
        report_lines.append("\n## 一、交叉验证配置")
        report_lines.append(f"- 策略: {self.cv.strategy}")
        report_lines.append(f"- 折数: {len(fold_results)}")
        report_lines.append(f"- 最小训练集大小: {self.cv.min_train_size}")
        report_lines.append(f"- 间隔: {self.cv.gap}")
        report_lines.append(f"- 置信水平: {self.confidence_level * 100:.0f}%")
        
        report_lines.append("\n## 二、Fold详情")
        
        for fold in fold_results:
            report_lines.append(f"\n### Fold {fold['fold']}")
            report_lines.append(f"- 训练集大小: {fold['train_size']}")
            report_lines.append(f"- 测试集大小: {fold['test_size']}")
            report_lines.append(f"- 训练集正样本比例: {fold['train_positive_ratio']:.2%}")
            report_lines.append(f"- 测试集正样本比例: {fold['test_positive_ratio']:.2%}")
            report_lines.append(f"- 精确率: {fold['precision']:.4f}")
            report_lines.append(f"- 召回率: {fold['recall']:.4f}")
            report_lines.append(f"- 平均精度(AP): {fold['ap']:.4f}")
            report_lines.append(f"- Precision@20: {fold['precision_at_k']:.4f}")
        
        report_lines.append("\n## 三、统计汇总（Mean ± 95% CI）")
        
        metrics_display = {
            'precision': '精确率（Precision）',
            'recall': '召回率（Recall）',
            'ap': '平均精度（AP）',
            'precision_at_k': 'Precision@20'
        }
        
        for metric_key, display_name in metrics_display.items():
            if metric_key in summary:
                result = summary[metric_key]
                cv = summary['stability'][metric_key]
                
                report_lines.append(f"\n### {display_name}")
                report_lines.append(f"- 均值: {result['mean']:.4f}")
                report_lines.append(f"- 95% CI: [{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]")
                report_lines.append(f"- 标准差: {result['std']:.4f}")
                report_lines.append(f"- 变异系数(CV): {cv:.4f}")
                report_lines.append(f"- 稳定性: {'稳定' if cv < 0.1 else '较不稳定' if cv < 0.2 else '不稳定'}")
        
        report_lines.append("\n## 四、Fold间波动")
        
        for metric_key, display_name in metrics_display.items():
            if metric_key in summary:
                values = summary[metric_key]['values']
                report_lines.append(f"\n### {display_name}各Fold值")
                for i, val in enumerate(values):
                    report_lines.append(f"- Fold {i+1}: {val:.4f}")
        
        report_lines.append("\n## 五、结论")
        
        # 判断稳定性
        avg_cv = np.mean(list(summary['stability'].values()))
        if avg_cv < 0.1:
            stability_status = "✅ 稳定"
        elif avg_cv < 0.2:
            stability_status = "⚠️ 较不稳定"
        else:
            stability_status = "❌ 不稳定"
        
        report_lines.append(f"- 平均稳定性: {stability_status}")
        report_lines.append(f"- 平均精确率: {summary['precision']['mean']:.4f}")
        report_lines.append(f"- 平均召回率: {summary['recall']['mean']:.4f}")
        report_lines.append(f"- 平均AP: {summary['ap']['mean']:.4f}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
