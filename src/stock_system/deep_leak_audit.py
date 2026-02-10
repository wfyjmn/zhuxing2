"""
深入的数据泄露审计
包括前向/后向标签一致性检查、lookahead特征检测、feature drift计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')


class LabelConsistencyChecker:
    """
    标签一致性检查器
    
    检查标签在时间序列上的一致性，检测异常突变
    """
    
    def __init__(
        self,
        window_size: int = 5,
        zscore_threshold: float = 3.0,
        max_change_ratio: float = 0.5
    ):
        """
        初始化标签一致性检查器
        
        参数:
            window_size: 滑动窗口大小
            zscore_threshold: Z分数阈值
            max_change_ratio: 最大变化比例
        """
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold
        self.max_change_ratio = max_change_ratio
        
        self.results = None
    
    def check_forward_consistency(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """
        前向一致性检查
        
        检查标签在未来时间窗口内的稳定性
        
        参数:
            y: 标签数组
            dates: 日期索引（可选）
            
        返回:
            检查结果字典
        """
        results = {
            'type': 'forward',
            'anomalies': [],
            'statistics': {}
        }
        
        # 计算滚动平均和标准差
        rolling_mean = pd.Series(y).rolling(
            window=self.window_size,
            min_periods=1
        ).mean()
        
        rolling_std = pd.Series(y).rolling(
            window=self.window_size,
            min_periods=1
        ).std()
        
        # 检测异常点
        for i in range(self.window_size, len(y)):
            current_value = y[i]
            mean_val = rolling_mean[i - 1]
            std_val = rolling_std[i - 1] if rolling_std[i - 1] > 0 else 0.01
            
            # Z分数检测
            z_score = abs((current_value - mean_val) / std_val)
            
            if z_score > self.zscore_threshold:
                date_str = str(dates[i]) if dates is not None else str(i)
                results['anomalies'].append({
                    'index': i,
                    'date': date_str,
                    'value': float(current_value),
                    'expected_range': [
                        float(mean_val - self.zscore_threshold * std_val),
                        float(mean_val + self.zscore_threshold * std_val)
                    ],
                    'z_score': float(z_score),
                    'type': 'zscore'
                })
        
        # 统计信息
        results['statistics'] = {
            'total_samples': len(y),
            'anomaly_count': len(results['anomalies']),
            'anomaly_ratio': len(results['anomalies']) / len(y) if len(y) > 0 else 0,
            'rolling_mean_avg': float(rolling_mean.mean()),
            'rolling_std_avg': float(rolling_std.mean())
        }
        
        # 添加便捷字段
        results['anomaly_count'] = results['statistics']['anomaly_count']
        results['anomaly_ratio'] = results['statistics']['anomaly_ratio']
        
        return results
    
    def check_backward_consistency(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """
        后向一致性检查
        
        检查标签在过去时间窗口内的稳定性
        
        参数:
            y: 标签数组
            dates: 日期索引（可选）
            
        返回:
            检查结果字典
        """
        # 后向检查等同于前向检查的反向
        y_reversed = y[::-1]
        dates_reversed = dates[::-1] if dates is not None else None
        
        results = self.check_forward_consistency(
            y_reversed, dates_reversed
        )
        
        # 转换索引回原始位置
        total_len = len(y)
        for anomaly in results['anomalies']:
            original_index = total_len - 1 - anomaly['index']
            anomaly['index'] = original_index
            if dates is not None:
                anomaly['date'] = str(dates[original_index])
            anomaly['type'] = 'backward'
        
        results['type'] = 'backward'
        
        return results
    
    def check_label_transition(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """
        检查标签转换模式
        
        检测频繁的标签切换（可能暗示标签不稳定）
        
        参数:
            y: 标签数组
            dates: 日期索引（可选）
            
        返回:
            检查结果字典
        """
        results = {
            'transitions': [],
            'rapid_switches': [],
            'statistics': {}
        }
        
        # 检测标签转换
        for i in range(1, len(y)):
            if y[i] != y[i - 1]:
                date_str = str(dates[i]) if dates is not None else str(i)
                results['transitions'].append({
                    'index': i,
                    'date': date_str,
                    'from': int(y[i - 1]),
                    'to': int(y[i])
                })
        
        # 检测快速切换（短时间内多次转换）
        window = self.window_size
        for i in range(window, len(y)):
            window_transitions = sum(
                1 for j in range(i - window, i)
                if y[j] != y[j - 1]
            )
            
            if window_transitions >= window - 1:
                date_str = str(dates[i]) if dates is not None else str(i)
                results['rapid_switches'].append({
                    'index': i,
                    'date': date_str,
                    'window_transitions': window_transitions
                })
        
        # 统计信息
        results['statistics'] = {
            'total_transitions': len(results['transitions']),
            'transition_rate': len(results['transitions']) / len(y),
            'rapid_switch_count': len(results['rapid_switches']),
            'avg_gap_between_transitions': float(len(y) / (len(results['transitions']) + 1))
        }
        
        return results
    
    def run_all_checks(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """
        运行所有一致性检查
        
        参数:
            y: 标签数组
            dates: 日期索引（可选）
            
        返回:
            综合检查结果
        """
        results = {
            'forward': self.check_forward_consistency(y, dates),
            'backward': self.check_backward_consistency(y, dates),
            'transition': self.check_label_transition(y, dates)
        }
        
        # 综合判断
        has_anomalies = (
            results['forward']['anomaly_count'] > 0 or
            results['backward']['anomaly_count'] > 0 or
            results['transition']['rapid_switch_count'] > 0
        )
        
        results['overall_passed'] = not has_anomalies
        results['overall_status'] = 'passed' if not has_anomalies else 'failed'
        
        self.results = results
        
        return results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成标签一致性检查报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        if self.results is None:
            raise ValueError("请先执行检查")
        
        results = self.results
        
        report_lines = []
        
        report_lines.append("# 标签一致性检查报告")
        report_lines.append("\n## 一、总体结果")
        report_lines.append(f"- **状态**: {'✅ 通过' if results['overall_passed'] else '❌ 失败'}")
        
        # 前向一致性
        report_lines.append("\n## 二、前向一致性检查")
        forward = results['forward']
        report_lines.append(f"- 异常点数: {forward['anomaly_count']}")
        report_lines.append(f"- 异常比例: {forward['anomaly_ratio']:.2%}")
        
        if forward['anomaly_count'] > 0:
            report_lines.append("\n**异常点详情**（前10个）:")
            for anomaly in forward['anomalies'][:10]:
                report_lines.append(
                    f"- 索引{anomaly['index']} ({anomaly['date']}): "
                    f"值={anomaly['value']:.4f}, Z分数={anomaly['z_score']:.2f}"
                )
        
        # 后向一致性
        report_lines.append("\n## 三、后向一致性检查")
        backward = results['backward']
        report_lines.append(f"- 异常点数: {backward['anomaly_count']}")
        report_lines.append(f"- 异常比例: {backward['anomaly_ratio']:.2%}")
        
        if backward['anomaly_count'] > 0:
            report_lines.append("\n**异常点详情**（前10个）:")
            for anomaly in backward['anomalies'][:10]:
                report_lines.append(
                    f"- 索引{anomaly['index']} ({anomaly['date']}): "
                    f"值={anomaly['value']:.4f}, Z分数={anomaly['z_score']:.2f}"
                )
        
        # 标签转换
        report_lines.append("\n## 四、标签转换模式")
        transition = results['transition']
        report_lines.append(f"- 总转换数: {transition['statistics']['total_transitions']}")
        report_lines.append(f"- 转换率: {transition['statistics']['transition_rate']:.2%}")
        report_lines.append(f"- 快速切换数: {transition['statistics']['rapid_switch_count']}")
        
        if transition['statistics']['rapid_switch_count'] > 0:
            report_lines.append("\n**快速切换详情**（前10个）:")
            for switch in transition['rapid_switches'][:10]:
                report_lines.append(
                    f"- 索引{switch['index']} ({switch['date']}): "
                    f"窗口内转换{switch['window_transitions']}次"
                )
        
        # 建议
        if not results['overall_passed']:
            report_lines.append("\n## 五、建议")
            if forward['anomaly_count'] > 0:
                report_lines.append("1. 检查前向一致性异常点，确认标签生成逻辑")
            if backward['anomaly_count'] > 0:
                report_lines.append("2. 检查后向一致性异常点，考虑标签平滑处理")
            if transition['statistics']['rapid_switch_count'] > 0:
                report_lines.append("3. 检查快速切换，可能需要调整标签定义")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class LookaheadFeatureDetector:
    """
    Lookahead特征检测器
    
    检测特征是否包含未来信息
    """
    
    def __init__(
        self,
        lag_range: Tuple[int, int] = (1, 5),
        correlation_threshold: float = 0.3,
        significance_level: float = 0.05
    ):
        """
        初始化Lookahead特征检测器
        
        参数:
            lag_range: 滞后范围（min, max）
            correlation_threshold: 相关性阈值
            significance_level: 显著性水平
        """
        self.lag_range = lag_range
        self.correlation_threshold = correlation_threshold
        self.significance_level = significance_level
        
        self.results = None
    
    def detect_lookahead(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        检测lookahead特征
        
        通过检查特征与未来标签的相关性来检测lookahead
        
        参数:
            X: 特征DataFrame
            y: 标签数组
            
        返回:
            检测结果字典
        """
        results = {
            'lookahead_features': [],
            'lag_correlations': {},
            'feature_analysis': {}
        }
        
        # 对每个特征检查与未来标签的相关性
        for feature in X.columns:
            feature_data = X[feature].values
            correlations = []
            
            for lag in range(self.lag_range[0], self.lag_range[1] + 1):
                # 计算特征与滞后标签的相关性
                if lag < len(y):
                    future_y = y[lag:]
                    current_feature = feature_data[:-lag]
                    
                    if len(current_feature) > 10:
                        corr, p_value = stats.pearsonr(current_feature, future_y)
                        correlations.append({
                            'lag': lag,
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': p_value < self.significance_level
                        })
            
            results['lag_correlations'][feature] = correlations
            
            # 判断是否为lookahead特征
            max_corr = max([abs(c['correlation']) for c in correlations], default=0)
            significant_lags = [
                c for c in correlations
                if c['significant'] and abs(c['correlation']) >= self.correlation_threshold
            ]
            
            is_lookahead = len(significant_lags) > 0
            
            results['feature_analysis'][feature] = {
                'max_correlation': max_corr,
                'max_lag': max([c['lag'] for c in correlations], default=0),
                'significant_lags': significant_lags,
                'is_lookahead': is_lookahead
            }
            
            if is_lookahead:
                results['lookahead_features'].append({
                    'feature': feature,
                    'max_correlation': max_corr,
                    'significant_lags': significant_lags
                })
        
        # 统计信息
        results['statistics'] = {
            'total_features': len(X.columns),
            'lookahead_count': len(results['lookahead_features']),
            'lookahead_ratio': len(results['lookahead_features']) / len(X.columns)
        }
        
        self.results = results
        
        return results
    
    def check_time_alignment(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """
        检查特征和标签的时间对齐
        
        参数:
            X: 特征DataFrame
            y: 标签数组
            dates: 日期索引
            
        返回:
            时间对齐检查结果
        """
        results = {
            'alignment_status': 'unknown',
            'issues': []
        }
        
        # 检查特征中是否包含明显的未来日期信息
        for feature in X.columns:
            feature_data = X[feature].values
            
            # 检查是否有序列相关性
            autocorr = pd.Series(feature_data).autocorr(lag=1)
            
            # 检查是否与标签有极强的负相关（可能是未来标签的逆序）
            corr, p_value = stats.pearsonr(feature_data, y)
            
            if abs(corr) > 0.9:
                results['issues'].append({
                    'feature': feature,
                    'type': 'extreme_correlation',
                    'correlation': corr,
                    'p_value': p_value
                })
        
        results['alignment_status'] = 'aligned' if not results['issues'] else 'misaligned'
        
        return results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成Lookahead特征检测报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        if self.results is None:
            raise ValueError("请先执行检测")
        
        results = self.results
        
        report_lines = []
        
        report_lines.append("# Lookahead特征检测报告")
        report_lines.append("\n## 一、检测结果")
        report_lines.append(f"- 总特征数: {results['statistics']['total_features']}")
        report_lines.append(f"- Lookahead特征数: {results['statistics']['lookahead_count']}")
        report_lines.append(f"- Lookahead比例: {results['statistics']['lookahead_ratio']:.2%}")
        
        if results['lookahead_features']:
            report_lines.append("\n## 二、Lookahead特征列表")
            
            for idx, feature_info in enumerate(results['lookahead_features'], 1):
                report_lines.append(f"\n### {idx}. {feature_info['feature']}")
                report_lines.append(f"- 最大相关性: {feature_info['max_correlation']:.4f}")
                report_lines.append(f"- 显著滞后期: {[lag['lag'] for lag in feature_info['significant_lags']]}")
                report_lines.append("\n详细相关性:")
                for lag_info in feature_info['significant_lags']:
                    report_lines.append(
                        f"  - 滞后{lag_info['lag']}: "
                        f"corr={lag_info['correlation']:.4f}, "
                        f"p={lag_info['p_value']:.4f}"
                    )
            
            report_lines.append("\n## 三、建议")
            report_lines.append("1. 移除或重新设计Lookahead特征")
            report_lines.append("2. 确保特征只使用历史数据")
            report_lines.append("3. 重新检查特征工程逻辑")
        else:
            report_lines.append("\n## 二、结论")
            report_lines.append("✅ 未检测到Lookahead特征，特征工程符合要求。")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class FeatureDriftCalculator:
    """
    特征漂移计算器
    
    检测训练集和测试集之间的特征分布差异
    """
    
    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1,
        js_threshold: float = 0.1
    ):
        """
        初始化特征漂移计算器
        
        参数:
            psi_threshold: PSI阈值
            ks_threshold: KS统计量阈值
            js_threshold: Jensen-Shannon距离阈值
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.js_threshold = js_threshold
        
        self.results = None
    
    def calculate_psi(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        计算PSI（Population Stability Index）
        
        参数:
            train_data: 训练数据
            test_data: 测试数据
            bins: 分箱数
            
        返回:
            PSI值
        """
        # 合并数据计算分位数
        combined = np.concatenate([train_data, test_data])
        _, bin_edges = np.histogram(combined, bins=bins)
        
        # 计算训练集和测试集的分布
        train_hist, _ = np.histogram(train_data, bins=bin_edges)
        test_hist, _ = np.histogram(test_data, bins=bin_edges)
        
        # 归一化
        train_hist = train_hist / train_hist.sum()
        test_hist = test_hist / test_hist.sum()
        
        # 避免除零
        train_hist = np.maximum(train_hist, 0.0001)
        test_hist = np.maximum(test_hist, 0.0001)
        
        # 计算PSI
        psi = np.sum((train_hist - test_hist) * np.log(train_hist / test_hist))
        
        return psi
    
    def calculate_ks_statistic(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算KS统计量
        
        参数:
            train_data: 训练数据
            test_data: 测试数据
            
        返回:
            (KS统计量, p值)
        """
        ks_statistic, p_value = stats.ks_2samp(train_data, test_data)
        return ks_statistic, p_value
    
    def calculate_js_distance(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        计算Jensen-Shannon距离
        
        参数:
            train_data: 训练数据
            test_data: 测试数据
            bins: 分箱数
            
        返回:
            JS距离
        """
        # 合并数据计算分位数
        combined = np.concatenate([train_data, test_data])
        _, bin_edges = np.histogram(combined, bins=bins)
        
        # 计算训练集和测试集的分布
        train_hist, _ = np.histogram(train_data, bins=bin_edges)
        test_hist, _ = np.histogram(test_data, bins=bin_edges)
        
        # 归一化
        train_dist = train_hist / train_hist.sum()
        test_dist = test_hist / test_hist.sum()
        
        # 计算JS距离
        js_distance = jensenshannon(train_dist, test_dist)
        
        return js_distance
    
    def calculate_drift(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        计算所有特征的漂移
        
        参数:
            X_train: 训练集特征
            X_test: 测试集特征
            
        返回:
            漂移计算结果
        """
        results = {
            'feature_drift': {},
            'drift_summary': {}
        }
        
        # 确保特征一致
        common_features = list(set(X_train.columns) & set(X_test.columns))
        
        total_drift_features = 0
        high_drift_features = []
        
        for feature in common_features:
            train_data = X_train[feature].values
            test_data = X_test[feature].values
            
            # 计算各种漂移指标
            psi = self.calculate_psi(train_data, test_data)
            ks_stat, ks_p = self.calculate_ks_statistic(train_data, test_data)
            js_dist = self.calculate_js_distance(train_data, test_data)
            
            # 判断漂移程度
            drift_level = 'none'
            if psi >= self.psi_threshold * 2:
                drift_level = 'high'
                high_drift_features.append(feature)
                total_drift_features += 1
            elif psi >= self.psi_threshold:
                drift_level = 'moderate'
                total_drift_features += 1
            
            results['feature_drift'][feature] = {
                'psi': float(psi),
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'js_distance': float(js_dist),
                'drift_level': drift_level,
                'has_drift': drift_level in ['moderate', 'high']
            }
        
        # 汇总统计
        results['drift_summary'] = {
            'total_features': len(common_features),
            'drift_count': total_drift_features,
            'drift_ratio': total_drift_features / len(common_features),
            'high_drift_count': len(high_drift_features),
            'high_drift_features': high_drift_features
        }
        
        # 综合判断
        has_high_drift = len(high_drift_features) > 0
        results['overall_passed'] = not has_high_drift
        results['overall_status'] = 'passed' if not has_high_drift else 'failed'
        
        self.results = results
        
        return results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成特征漂移报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        if self.results is None:
            raise ValueError("请先执行计算")
        
        results = self.results
        
        report_lines = []
        
        report_lines.append("# 特征漂移报告")
        report_lines.append("\n## 一、漂移汇总")
        report_lines.append(f"- 总特征数: {results['drift_summary']['total_features']}")
        report_lines.append(f"- 漂移特征数: {results['drift_summary']['drift_count']}")
        report_lines.append(f"- 漂移比例: {results['drift_summary']['drift_ratio']:.2%}")
        report_lines.append(f"- 高漂移特征数: {results['drift_summary']['high_drift_count']}")
        report_lines.append(f"\n**总体状态**: {'✅ 通过' if results['overall_passed'] else '❌ 失败'}")
        
        # 高漂移特征
        if results['drift_summary']['high_drift_features']:
            report_lines.append("\n## 二、高漂移特征")
            
            for feature in results['drift_summary']['high_drift_features']:
                drift_info = results['feature_drift'][feature]
                report_lines.append(f"\n### {feature}")
                report_lines.append(f"- PSI: {drift_info['psi']:.4f}")
                report_lines.append(f"- KS统计量: {drift_info['ks_statistic']:.4f}")
                report_lines.append(f"- JS距离: {drift_info['js_distance']:.4f}")
                report_lines.append(f"- 漂移等级: {drift_info['drift_level']}")
        
        # 所有特征的详细漂移
        report_lines.append("\n## 三、所有特征漂移详情")
        
        drift_features = [
            (feature, info['psi'])
            for feature, info in results['feature_drift'].items()
            if info['has_drift']
        ]
        drift_features.sort(key=lambda x: x[1], reverse=True)
        
        for feature, psi in drift_features[:20]:
            drift_info = results['feature_drift'][feature]
            report_lines.append(
                f"- {feature}: PSI={psi:.4f}, "
                f"KS={drift_info['ks_statistic']:.4f}, "
                f"JS={drift_info['js_distance']:.4f}, "
                f"等级={drift_info['drift_level']}"
            )
        
        # 建议
        if not results['overall_passed']:
            report_lines.append("\n## 四、建议")
            report_lines.append("1. 重新训练模型以适应新的特征分布")
            report_lines.append("2. 考虑移除或替换高漂移特征")
            report_lines.append("3. 实施在线监控以持续跟踪特征漂移")
            report_lines.append("4. 使用增量学习方法处理概念漂移")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
