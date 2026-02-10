"""
在线监控和自动回撤系统
包括滚动窗口精确率监控、自动回撤阈值检测、告警机制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')


class RollingPrecisionMonitor:
    """
    滚动窗口精确率监控器
    
    实时监控模型性能，检测精确率下降
    """
    
    def __init__(
        self,
        window_size: int = 50,
        min_samples: int = 10,
        alert_threshold: float = 0.1,
        confidence_interval: float = 0.95
    ):
        """
        初始化滚动精确率监控器
        
        参数:
            window_size: 滚动窗口大小
            min_samples: 最小样本数
            alert_threshold: 告警阈值（精确率下降比例）
            confidence_interval: 置信区间
        """
        self.window_size = window_size
        self.min_samples = min_samples
        self.alert_threshold = alert_threshold
        self.confidence_interval = confidence_interval
        
        self.history = []
        self.baseline_precision = None
        self.alerts = []
    
    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        更新监控数据
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            timestamp: 时间戳（可选）
            
        返回:
            更新结果
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # 计算精确率
        precision = self._calculate_precision(y_true, y_pred)
        
        # 添加到历史记录
        record = {
            'timestamp': timestamp,
            'precision': precision,
            'n_samples': len(y_true)
        }
        self.history.append(record)
        
        # 计算滚动统计
        rolling_stats = self._calculate_rolling_stats()
        
        # 检查告警
        alert = self._check_alert(rolling_stats)
        if alert:
            self.alerts.append(alert)
        
        # 设置基准
        if self.baseline_precision is None and len(self.history) >= self.min_samples:
            self.baseline_precision = np.mean([
                h['precision'] for h in self.history[:self.min_samples]
            ])
        
        return {
            'record': record,
            'rolling_stats': rolling_stats,
            'alert': alert,
            'baseline': self.baseline_precision
        }
    
    def _calculate_precision(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算精确率"""
        from sklearn.metrics import precision_score
        
        # 确保有正样本预测
        if y_pred.sum() == 0:
            return 0.0
        
        return precision_score(y_true, y_pred, zero_division=0)
    
    def _calculate_rolling_stats(self) -> Dict[str, float]:
        """计算滚动统计"""
        if len(self.history) < self.min_samples:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': len(self.history)
            }
        
        # 取最近的窗口数据
        window_data = self.history[-self.window_size:]
        precisions = [h['precision'] for h in window_data]
        
        return {
            'mean': float(np.mean(precisions)),
            'std': float(np.std(precisions)),
            'min': float(np.min(precisions)),
            'max': float(np.max(precisions)),
            'count': len(precisions)
        }
    
    def _check_alert(
        self,
        rolling_stats: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """检查是否需要告警"""
        if self.baseline_precision is None or rolling_stats['count'] < self.min_samples:
            return None
        
        current_precision = rolling_stats['mean']
        precision_drop = (self.baseline_precision - current_precision) / self.baseline_precision
        
        # 检查是否超过阈值
        if precision_drop > self.alert_threshold:
            return {
                'timestamp': datetime.now().isoformat(),
                'type': 'precision_drop',
                'baseline': self.baseline_precision,
                'current': current_precision,
                'drop_ratio': precision_drop,
                'threshold': self.alert_threshold,
                'severity': 'critical' if precision_drop > self.alert_threshold * 2 else 'warning'
            }
        
        return None
    
    def get_performance_trend(
        self,
        n_periods: int = 10
    ) -> pd.DataFrame:
        """
        获取性能趋势
        
        参数:
            n_periods: 时间段数
            
        返回:
            性能趋势DataFrame
        """
        if len(self.history) < n_periods:
            n_periods = len(self.history)
        
        # 将历史数据分片
        chunk_size = len(self.history) // n_periods
        chunks = [
            self.history[i:i + chunk_size]
            for i in range(0, len(self.history), chunk_size)
        ]
        
        # 计算每个时间段的平均精确率
        trend_data = []
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            
            avg_precision = np.mean([h['precision'] for h in chunk])
            trend_data.append({
                'period': i,
                'avg_precision': avg_precision,
                'n_records': len(chunk)
            })
        
        return pd.DataFrame(trend_data)
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成监控报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        report_lines = []
        
        report_lines.append("# 滚动精确率监控报告")
        report_lines.append("\n## 一、基本信息")
        report_lines.append(f"- 总记录数: {len(self.history)}")
        report_lines.append(f"- 窗口大小: {self.window_size}")
        report_lines.append(f"- 基准精确率: {self.baseline_precision:.4f}" if self.baseline_precision else "- 基准精确率: 未设置")
        
        # 当前统计
        current_stats = self._calculate_rolling_stats()
        report_lines.append(f"\n## 二、当前性能")
        report_lines.append(f"- 平均精确率: {current_stats['mean']:.4f}")
        report_lines.append(f"- 标准差: {current_stats['std']:.4f}")
        report_lines.append(f"- 最小值: {current_stats['min']:.4f}")
        report_lines.append(f"- 最大值: {current_stats['max']:.4f}")
        
        # 告警信息
        report_lines.append(f"\n## 三、告警信息")
        report_lines.append(f"- 总告警数: {len(self.alerts)}")
        
        if self.alerts:
            report_lines.append("\n**最近的告警**（最多显示5条）:")
            for alert in self.alerts[-5:]:
                report_lines.append(
                    f"- [{alert['timestamp']}] {alert['type'].upper()}: "
                    f"从{alert['baseline']:.4f}下降到{alert['current']:.4f} "
                    f"(降幅{alert['drop_ratio']:.2%}), 严重程度: {alert['severity']}"
                )
        
        # 性能趋势
        if len(self.history) >= 10:
            report_lines.append("\n## 四、性能趋势")
            trend_df = self.get_performance_trend(n_periods=min(10, len(self.history) // 10))
            
            for _, row in trend_df.iterrows():
                report_lines.append(
                    f"- 阶段{row['period']}: 精确率={row['avg_precision']:.4f}, "
                    f"记录数={row['n_records']}"
                )
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class AutoRollbackThreshold:
    """
    自动回撤阈值管理器
    
    自动管理模型性能阈值，触发回撤机制
    """
    
    def __init__(
        self,
        precision_threshold: float = 0.70,
        min_samples: int = 50,
        rolling_window: int = 100,
        confidence_level: float = 0.95,
        degradation_tolerance: float = 0.05
    ):
        """
        初始化自动回撤阈值管理器
        
        参数:
            precision_threshold: 精确率阈值
            min_samples: 最小样本数
            rolling_window: 滚动窗口大小
            confidence_level: 置信水平
            degradation_tolerance: 退化容忍度
        """
        self.precision_threshold = precision_threshold
        self.min_samples = min_samples
        self.rolling_window = rolling_window
        self.confidence_level = confidence_level
        self.degradation_tolerance = degradation_tolerance
        
        self.predictions = []
        self.baseline_stats = None
        self.rollback_triggered = False
        self.rollback_history = []
    
    def add_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        添加预测结果
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率（可选）
            
        返回:
            添加结果
        """
        from sklearn.metrics import precision_score, recall_score
        
        record = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'timestamp': datetime.now().isoformat()
        }
        
        self.predictions.append(record)
        
        # 检查是否需要触发回撤
        result = self._check_rollback()
        
        return {
            'record': record,
            'rollback_triggered': result['triggered'],
            'reason': result['reason'],
            'current_stats': result['current_stats']
        }
    
    def _check_rollback(self) -> Dict[str, Any]:
        """检查是否需要回撤"""
        if len(self.predictions) < self.min_samples:
            return {
                'triggered': False,
                'reason': '样本数不足',
                'current_stats': None
            }
        
        # 计算当前统计
        window_data = self.predictions[-self.rolling_window:]
        current_precision = np.mean([p['precision'] for p in window_data])
        current_std = np.std([p['precision'] for p in window_data])
        
        current_stats = {
            'mean': current_precision,
            'std': current_std,
            'count': len(window_data)
        }
        
        # 设置基准（如果未设置）
        if self.baseline_stats is None:
            self.baseline_stats = current_stats
            return {
                'triggered': False,
                'reason': '基准已设置',
                'current_stats': current_stats
            }
        
        # 检查是否低于阈值
        if current_precision < self.precision_threshold:
            return {
                'triggered': True,
                'reason': f'精确率{current_precision:.4f}低于阈值{self.precision_threshold:.4f}',
                'current_stats': current_stats
            }
        
        # 检查是否显著退化
        baseline_precision = self.baseline_stats['mean']
        degradation = (baseline_precision - current_precision) / baseline_precision
        
        if degradation > self.degradation_tolerance:
            return {
                'triggered': True,
                'reason': f'精确率退化{degradation:.2%}超过容忍度{self.degradation_tolerance:.2%}',
                'current_stats': current_stats
            }
        
        return {
            'triggered': False,
            'reason': '性能正常',
            'current_stats': current_stats
        }
    
    def trigger_rollback(self, reason: str) -> bool:
        """
        触发回撤
        
        参数:
            reason: 回撤原因
            
        返回:
            是否成功触发
        """
        self.rollback_triggered = True
        
        rollback_record = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'baseline_stats': self.baseline_stats,
            'current_predictions_count': len(self.predictions)
        }
        
        self.rollback_history.append(rollback_record)
        
        # 清空预测历史（可选）
        # self.predictions = []
        # self.baseline_stats = None
        
        return True
    
    def reset(self):
        """重置状态"""
        self.predictions = []
        self.baseline_stats = None
        self.rollback_triggered = False
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        status = {
            'total_predictions': len(self.predictions),
            'baseline_precision': self.baseline_stats['mean'] if self.baseline_stats else None,
            'rollback_triggered': self.rollback_triggered,
            'rollback_count': len(self.rollback_history),
            'threshold': self.precision_threshold
        }
        
        if len(self.predictions) >= self.min_samples:
            window_data = self.predictions[-self.rolling_window:]
            status['current_precision'] = float(np.mean([p['precision'] for p in window_data]))
        
        return status
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成回撤报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        status = self.get_status()
        
        report_lines = []
        
        report_lines.append("# 自动回撤阈值报告")
        report_lines.append("\n## 一、基本信息")
        report_lines.append(f"- 总预测数: {status['total_predictions']}")
        report_lines.append(f"- 精确率阈值: {status['threshold']:.4f}")
        report_lines.append(f"- 滚动窗口: {self.rolling_window}")
        report_lines.append(f"- 最小样本数: {self.min_samples}")
        
        # 当前状态
        report_lines.append("\n## 二、当前状态")
        report_lines.append(f"- 回撤触发: {'是' if status['rollback_triggered'] else '否'}")
        report_lines.append(f"- 基准精确率: {status['baseline_precision']:.4f}" if status['baseline_precision'] else "- 基准精确率: 未设置")
        
        if 'current_precision' in status:
            report_lines.append(f"- 当前精确率: {status['current_precision']:.4f}")
        
        # 回撤历史
        report_lines.append(f"\n## 三、回撤历史")
        report_lines.append(f"- 总回撤次数: {status['rollback_count']}")
        
        if self.rollback_history:
            report_lines.append("\n**回撤记录**:")
            for idx, rollback in enumerate(self.rollback_history, 1):
                report_lines.append(
                    f"\n### {idx}. [{rollback['timestamp']}]"
                )
                report_lines.append(f"- 原因: {rollback['reason']}")
                report_lines.append(f"- 基准精确率: {rollback['baseline_stats']['mean']:.4f}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class OnlineMonitoringSystem:
    """
    在线监控系统
    
    整合滚动精确率监控和自动回撤阈值
    """
    
    def __init__(
        self,
        precision_monitor: Optional[RollingPrecisionMonitor] = None,
        rollback_threshold: Optional[AutoRollbackThreshold] = None,
        alert_callback: Optional[Callable[[Dict], None]] = None
    ):
        """
        初始化在线监控系统
        
        参数:
            precision_monitor: 精确率监控器
            rollback_threshold: 回撤阈值管理器
            alert_callback: 告警回调函数
        """
        self.precision_monitor = precision_monitor or RollingPrecisionMonitor()
        self.rollback_threshold = rollback_threshold or AutoRollbackThreshold()
        self.alert_callback = alert_callback
        
        self.events = []
    
    def process_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        处理预测结果
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率（可选）
            
        返回:
            处理结果
        """
        # 更新精确率监控
        monitor_result = self.precision_monitor.update(y_true, y_pred)
        
        # 添加预测到回撤阈值
        rollback_result = self.rollback_threshold.add_predictions(
            y_true, y_pred, y_pred_proba
        )
        
        # 记录事件
        event = {
            'timestamp': datetime.now().isoformat(),
            'monitor_result': monitor_result,
            'rollback_result': rollback_result
        }
        self.events.append(event)
        
        # 检查告警
        if monitor_result['alert']:
            if self.alert_callback:
                self.alert_callback(monitor_result['alert'])
        
        # 检查回撤
        if rollback_result['rollback_triggered']:
            self.rollback_threshold.trigger_rollback(rollback_result['reason'])
            if self.alert_callback:
                self.alert_callback({
                    'type': 'rollback',
                    'reason': rollback_result['reason']
                })
        
        return event
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'monitor_status': self.precision_monitor.get_performance_trend(),
            'rollback_status': self.rollback_threshold.get_status(),
            'total_events': len(self.events)
        }
    
    def generate_comprehensive_report(
        self,
        save_path: Optional[str] = None
    ) -> str:
        """
        生成综合报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        monitor_report = self.precision_monitor.generate_report()
        rollback_report = self.rollback_threshold.generate_report()
        
        comprehensive_report = """# 在线监控系统综合报告

## 一、精确率监控

{monitor_report}

## 二、回撤阈值

{rollback_report}

## 三、系统状态

- 总事件数: {total_events}

""".format(
            monitor_report=monitor_report,
            rollback_report=rollback_report,
            total_events=len(self.events)
        )
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(comprehensive_report)
        
        return comprehensive_report
