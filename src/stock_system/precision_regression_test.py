"""
精确率回归测试器
使用置信区间（CI）防止模型性能回归
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import precision_score, recall_score, average_precision_score
from scipy import stats
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PrecisionRegressionTest:
    """
    精确率回归测试器
    
    功能：
    1. 加载历史基准数据
    2. 计算当前模型的精确率
    3. 判断是否在置信区间内
    4. 如果低于CI下限，拒绝部署
    """
    
    def __init__(
        self,
        baseline_path: Optional[str] = None,
        confidence_level: float = 0.95,
        min_samples: int = 50,
        tolerance: float = 0.02
    ):
        """
        初始化精确率回归测试器
        
        参数:
            baseline_path: 基准数据路径
            confidence_level: 置信水平（0-1）
            min_samples: 最小样本数
            tolerance: 容忍度（允许的性能下降）
        """
        self.baseline_path = baseline_path
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        self.tolerance = tolerance
        
        self.baseline_metrics = None
        self.current_metrics = None
        self.test_results = None
    
    def load_baseline(self, baseline_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载基准数据
        
        参数:
            baseline_path: 基准数据路径
            
        返回:
            基准指标字典
        """
        if baseline_path is None:
            baseline_path = self.baseline_path
        
        if baseline_path is None or not os.path.exists(baseline_path):
            raise FileNotFoundError(f"基准数据文件不存在: {baseline_path}")
        
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
        
        self.baseline_metrics = baseline_data
        
        return self.baseline_metrics
    
    def save_baseline(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        保存基准数据
        
        参数:
            y_true: 真实标签
            y_proba: 预测概率
            threshold: 分类阈值
            metadata: 元数据（模型版本、训练时间等）
            save_path: 保存路径
            
        返回:
            保存路径
        """
        # 计算指标
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'ap': float(average_precision_score(y_true, y_proba)),
            'n_samples': int(len(y_true)),
            'n_positive': int(y_true.sum()),
            'threshold': float(threshold),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # 如果有历史数据，计算统计信息
        if self.baseline_metrics is not None and 'history' in self.baseline_metrics:
            history = self.baseline_metrics['history']
        else:
            history = []
        
        # 添加当前指标到历史
        history.append(metrics)
        
        # 计算历史统计（均值、标准差、置信区间）
        history_df = pd.DataFrame(history)
        
        baseline_data = {
            'history': history,
            'statistics': self._calculate_statistics(history_df),
            'metadata': {
                'confidence_level': self.confidence_level,
                'min_samples': self.min_samples,
                'tolerance': self.tolerance,
                'n_records': len(history)
            }
        }
        
        # 保存
        if save_path is None:
            save_path = 'assets/precision_baseline.json'
        
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2)
        
        self.baseline_metrics = baseline_data
        
        return save_path
    
    def _calculate_statistics(
        self,
        history_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        计算历史统计信息
        
        参数:
            history_df: 历史数据DataFrame
            
        返回:
            统计信息字典
        """
        stats = {}
        
        for metric in ['precision', 'recall', 'ap']:
            values = history_df[metric].values
            
            if len(values) >= 2:
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                sem = stats.sem(values)
                
                # 计算置信区间
                t_score = stats.t.ppf((1 + self.confidence_level) / 2, len(values) - 1)
                ci_width = t_score * sem
                
                stats[metric] = {
                    'mean': float(mean),
                    'std': float(std),
                    'sem': float(sem),
                    'lower_ci': float(mean - ci_width),
                    'upper_ci': float(mean + ci_width),
                    'n_values': len(values)
                }
            else:
                # 样本不足，使用单点估计
                stats[metric] = {
                    'mean': float(values[0]),
                    'std': 0.0,
                    'sem': 0.0,
                    'lower_ci': float(values[0] - self.tolerance),
                    'upper_ci': float(values[0] + self.tolerance),
                    'n_values': len(values)
                }
        
        return stats
    
    def run_test(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        metric: str = 'precision',
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        运行回归测试
        
        参数:
            y_true: 真实标签
            y_proba: 预测概率
            threshold: 分类阈值
            metric: 测试指标 ('precision', 'recall', 'ap')
            strict_mode: 严格模式（True=必须 >= CI下限，False=可以低于CI下限但需说明理由）
            
        返回:
            测试结果字典
        """
        if self.baseline_metrics is None:
            raise ValueError("请先加载或创建基准数据")
        
        # 验证输入
        if len(y_true) < self.min_samples:
            raise ValueError(
                f"样本数不足: {len(y_true)} < {self.min_samples}"
            )
        
        # 计算当前指标
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'precision':
            current_value = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            current_value = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'ap':
            current_value = average_precision_score(y_true, y_proba)
        else:
            raise ValueError(f"未知的指标: {metric}")
        
        self.current_metrics = {
            metric: current_value,
            'n_samples': len(y_true),
            'n_positive': y_true.sum(),
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        # 获取基准统计
        baseline_stats = self.baseline_metrics['statistics'].get(metric, {})
        
        if not baseline_stats:
            raise ValueError(f"基准数据中没有指标: {metric}")
        
        # 判断是否通过测试
        lower_ci = baseline_stats['lower_ci']
        mean_baseline = baseline_stats['mean']
        
        if strict_mode:
            # 严格模式：必须 >= CI下限
            passed = current_value >= lower_ci
        else:
            # 宽松模式：可以低于CI下限，但不能低于 mean - tolerance
            passed = current_value >= (mean_baseline - self.tolerance)
        
        # 计算性能变化
        performance_change = current_value - mean_baseline
        relative_change = performance_change / mean_baseline if mean_baseline > 0 else 0
        
        # 判断回归程度
        if current_value < lower_ci:
            regression_level = 'severe'
        elif current_value < mean_baseline:
            regression_level = 'moderate'
        elif current_value < mean_baseline + self.tolerance:
            regression_level = 'none'
        else:
            regression_level = 'improved'
        
        self.test_results = {
            'metric': metric,
            'current_value': float(current_value),
            'baseline_mean': float(mean_baseline),
            'baseline_lower_ci': float(lower_ci),
            'baseline_upper_ci': float(baseline_stats['upper_ci']),
            'passed': passed,
            'strict_mode': strict_mode,
            'performance_change': float(performance_change),
            'relative_change': float(relative_change),
            'regression_level': regression_level,
            'confidence_level': self.confidence_level,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.test_results
    
    def run_comprehensive_test(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        运行综合测试（测试多个指标）
        
        参数:
            y_true: 真实标签
            y_proba: 预测概率
            threshold: 分类阈值
            strict_mode: 严格模式
            
        返回:
            综合测试结果
        """
        results = {}
        
        for metric in ['precision', 'recall', 'ap']:
            try:
                result = self.run_test(
                    y_true, y_proba, threshold,
                    metric=metric,
                    strict_mode=strict_mode
                )
                results[metric] = result
            except Exception as e:
                results[metric] = {
                    'error': str(e),
                    'passed': False
                }
        
        # 综合判断
        all_passed = all(r.get('passed', False) for r in results.values())
        precision_passed = results.get('precision', {}).get('passed', False)
        
        comprehensive_result = {
            'individual_tests': results,
            'all_passed': all_passed,
            'precision_passed': precision_passed,
            'overall_status': 'passed' if all_passed else 'failed',
            'timestamp': datetime.now().isoformat()
        }
        
        return comprehensive_result
    
    def generate_test_report(
        self,
        save_path: Optional[str] = None
    ) -> str:
        """
        生成测试报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        if self.test_results is None:
            raise ValueError("请先运行测试")
        
        result = self.test_results
        
        report_lines = []
        
        report_lines.append("# 精确率回归测试报告")
        report_lines.append("\n## 一、测试配置")
        report_lines.append(f"- 测试指标: {result['metric']}")
        report_lines.append(f"- 置信水平: {result['confidence_level'] * 100:.0f}%")
        report_lines.append(f"- 严格模式: {'是' if result['strict_mode'] else '否'}")
        report_lines.append(f"- 测试时间: {result['timestamp']}")
        
        report_lines.append("\n## 二、基准数据")
        baseline_stats = self.baseline_metrics['statistics'].get(result['metric'], {})
        report_lines.append(f"- 基准均值: {baseline_stats.get('mean', 0):.4f}")
        report_lines.append(f"- 基准标准差: {baseline_stats.get('std', 0):.4f}")
        report_lines.append(f"- 95% CI下限: {baseline_stats.get('lower_ci', 0):.4f}")
        report_lines.append(f"- 95% CI上限: {baseline_stats.get('upper_ci', 0):.4f}")
        
        report_lines.append("\n## 三、测试结果")
        report_lines.append(f"- 当前值: {result['current_value']:.4f}")
        report_lines.append(f"- 性能变化: {result['performance_change']:+.4f} ({result['relative_change']:+.2%})")
        
        if result['passed']:
            report_lines.append("\n## 四、测试结论")
            report_lines.append("✅ **测试通过**")
            report_lines.append("\n当前模型性能在可接受范围内，可以部署。")
        else:
            report_lines.append("\n## 四、测试结论")
            report_lines.append("❌ **测试失败**")
            report_lines.append("\n当前模型性能低于基准，不能部署。")
            report_lines.append(f"\n**回归级别**: {result['regression_level']}")
            
            if result['regression_level'] == 'severe':
                report_lines.append("\n**建议**:")
                report_lines.append("- 严重性能回归，必须重新训练模型")
                report_lines.append("- 检查数据质量")
                report_lines.append("- 检查模型超参数")
            elif result['regression_level'] == 'moderate':
                report_lines.append("\n**建议**:")
                report_lines.append("- 中等性能回归，建议优化模型")
                report_lines.append("- 分析失败样本")
                report_lines.append("- 考虑调整特征工程")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class RegressionTestSuite:
    """
    回归测试套件
    
    支持多个指标的回归测试和阈值优化
    """
    
    def __init__(
        self,
        baseline_path: str = 'assets/precision_baseline.json',
        confidence_level: float = 0.95
    ):
        """
        初始化回归测试套件
        
        参数:
            baseline_path: 基准数据路径
            confidence_level: 置信水平
        """
        self.baseline_path = baseline_path
        self.confidence_level = confidence_level
        
        self.tests = {}
    
    def add_test(
        self,
        test_name: str,
        metric: str,
        min_value: Optional[float] = None,
        max_degradation: float = 0.05
    ):
        """
        添加测试
        
        参数:
            test_name: 测试名称
            metric: 测试指标
            min_value: 最小值（可选）
            max_degradation: 最大允许的性能下降
        """
        self.tests[test_name] = {
            'metric': metric,
            'min_value': min_value,
            'max_degradation': max_degradation
        }
    
    def run_suite(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        运行测试套件
        
        参数:
            y_true: 真实标签
            y_proba: 预测概率
            threshold: 分类阈值
            
        返回:
            测试套件结果
        """
        if not self.tests:
            raise ValueError("请先添加测试")
        
        results = {}
        
        # 初始化回归测试器
        regression_test = PrecisionRegressionTest(
            baseline_path=self.baseline_path,
            confidence_level=self.confidence_level
        )
        
        # 尝试加载基准
        baseline_exists = os.path.exists(self.baseline_path)
        
        if not baseline_exists:
            # 如果没有基准，创建一个
            regression_test.save_baseline(
                y_true, y_proba, threshold,
                metadata={'initial_baseline': True},
                save_path=self.baseline_path  # 传入自定义路径
            )
            return {
                'status': 'baseline_created',
                'message': '已创建初始基准数据',
                'results': {}
            }
        
        # 加载基准
        regression_test.load_baseline()
        
        # 运行每个测试
        for test_name, test_config in self.tests.items():
            try:
                result = regression_test.run_test(
                    y_true, y_proba, threshold,
                    metric=test_config['metric'],
                    strict_mode=False
                )
                
                # 检查最小值约束
                if test_config['min_value'] is not None:
                    min_passed = result['current_value'] >= test_config['min_value']
                else:
                    min_passed = True
                
                # 检查最大下降约束
                max_degradation_allowed = test_config['max_degradation']
                actual_degradation = max(0, -result['performance_change'])
                degradation_passed = actual_degradation <= max_degradation_allowed
                
                # 综合判断
                passed = result['passed'] and min_passed and degradation_passed
                
                results[test_name] = {
                    **result,
                    'min_value_constraint': test_config['min_value'],
                    'min_value_passed': min_passed,
                    'max_degradation_allowed': max_degradation_allowed,
                    'actual_degradation': actual_degradation,
                    'degradation_passed': degradation_passed,
                    'overall_passed': passed
                }
                
            except Exception as e:
                results[test_name] = {
                    'error': str(e),
                    'passed': False
                }
        
        # 汇总结果
        all_passed = all(r.get('overall_passed', False) for r in results.values())
        
        suite_result = {
            'test_results': results,
            'all_passed': all_passed,
            'overall_status': 'passed' if all_passed else 'failed',
            'timestamp': datetime.now().isoformat()
        }
        
        return suite_result
    
    def generate_suite_report(
        self,
        suite_result: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        生成测试套件报告
        
        参数:
            suite_result: 测试套件结果
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        report_lines = []
        
        report_lines.append("# 回归测试套件报告")
        
        if suite_result.get('status') == 'baseline_created':
            report_lines.append(f"\n**状态**: {suite_result.get('message', '已创建初始基准数据')}")
            report_lines.append("\n请运行测试以评估模型性能。")
            report = "\n".join(report_lines)
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
            return report
        
        report_lines.append(f"\n**测试时间**: {suite_result.get('timestamp', 'N/A')}")
        report_lines.append(f"\n**总体状态**: {'✅ 通过' if suite_result['all_passed'] else '❌ 失败'}")
        
        report_lines.append("\n## 测试详情")
        
        for test_name, result in suite_result['test_results'].items():
            if 'error' in result:
                report_lines.append(f"\n### {test_name}")
                report_lines.append(f"**状态**: ❌ 错误")
                report_lines.append(f"**错误信息**: {result['error']}")
            else:
                report_lines.append(f"\n### {test_name}")
                report_lines.append(f"**指标**: {result['metric']}")
                report_lines.append(f"**当前值**: {result['current_value']:.4f}")
                report_lines.append(f"**基准值**: {result['baseline_mean']:.4f}")
                report_lines.append(f"**性能变化**: {result['performance_change']:+.4f} ({result['relative_change']:+.2%})")
                report_lines.append(f"**状态**: {'✅ 通过' if result['overall_passed'] else '❌ 失败'}")
                
                if not result['overall_passed']:
                    report_lines.append("\n**失败原因**:")
                    if not result['passed']:
                        report_lines.append("- 低于置信区间下限")
                    if not result.get('min_value_passed', True):
                        report_lines.append(f"- 低于最小值 {result.get('min_value_constraint', 0):.4f}")
                    if not result.get('degradation_passed', True):
                        report_lines.append(f"- 性能下降超过最大允许值 {result.get('max_degradation_allowed', 0):.2%}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
