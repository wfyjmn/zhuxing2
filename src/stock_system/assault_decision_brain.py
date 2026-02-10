"""
突击选股智能决策系统（Brain）
整合数据审计、在线监控和选股决策，形成智能的大脑和决策机构
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

from .deep_leak_audit import (
    LabelConsistencyChecker,
    LookaheadFeatureDetector,
    FeatureDriftCalculator
)
from .online_monitoring import (
    RollingPrecisionMonitor,
    AutoRollbackThreshold,
    OnlineMonitoringSystem
)
from .predictor import StockPredictor
from .triple_confirmation import TripleConfirmation
from .confidence_bucket import ConfidenceBasedFilter


class AssaultDecisionBrain:
    """
    突击选股智能决策系统（大脑）
    
    整合数据审计、在线监控和选股决策，形成智能的决策机构
    """
    
    def __init__(
        self,
        config_path: str = "config/short_term_assault_config.json",
        enable_deep_audit: bool = True,
        enable_online_monitoring: bool = True
    ):
        """
        初始化智能决策系统
        
        参数:
            config_path: 配置文件路径
            enable_deep_audit: 是否启用深度审计
            enable_online_monitoring: 是否启用在线监控
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        self.enable_deep_audit = enable_deep_audit
        self.enable_online_monitoring = enable_online_monitoring
        
        # 初始化组件
        self.predictor = StockPredictor(config_path)
        self.triple_confirmation = TripleConfirmation(config_path)
        self.confidence_filter = ConfidenceBasedFilter()
        
        # 初始化审计组件
        if enable_deep_audit:
            self.label_checker = LabelConsistencyChecker()
            self.lookahead_detector = LookaheadFeatureDetector()
            self.drift_calculator = FeatureDriftCalculator()
        
        # 初始化监控组件
        if enable_online_monitoring:
            self.monitoring_system = OnlineMonitoringSystem(
                alert_callback=self._on_alert
            )
        
        # 决策状态
        self.decision_state = {
            'system_status': 'ready',
            'last_audit_time': None,
            'last_audit_result': None,
            'performance_status': 'normal',
            'decision_count': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0
        }
        
        # 审计缓存
        self.audit_cache = {
            'label_consistency': None,
            'lookahead_features': None,
            'feature_drift': None
        }
        
        # 加载历史审计结果
        self._load_audit_history()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def _on_alert(self, alert: Dict[str, Any]):
        """告警回调"""
        print(f"⚠️ 系统告警: {alert}")
        
        # 更新决策状态
        if alert.get('type') == 'precision_drop':
            self.decision_state['performance_status'] = 'degraded'
        elif alert.get('type') == 'rollback':
            self.decision_state['system_status'] = 'rollback'
    
    def _load_audit_history(self):
        """加载审计历史"""
        try:
            history_path = 'assets/audit_history.json'
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    self.decision_state['last_audit_time'] = history.get('last_audit_time')
                    self.decision_state['last_audit_result'] = history.get('last_audit_result')
        except Exception as e:
            print(f"加载审计历史失败: {e}")
    
    def _save_audit_history(self):
        """保存审计历史"""
        try:
            os.makedirs('assets', exist_ok=True)
            history_path = 'assets/audit_history.json'
            history = {
                'last_audit_time': self.decision_state['last_audit_time'],
                'last_audit_result': self.decision_state['last_audit_result']
            }
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"保存审计历史失败: {e}")
    
    # ============ 数据审计模块 ============
    
    def run_pre_training_audit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """
        训练前数据审计
        
        执行完整的数据质量审计，确保数据可靠
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征（可选）
            y_test: 测试集标签（可选）
            dates: 日期索引（可选）
            
        返回:
            审计结果
        """
        if not self.enable_deep_audit:
            return {
                'status': 'skipped',
                'message': '深度审计未启用'
            }
        
        print("=" * 70)
        print("【训练前数据审计】")
        print("=" * 70)
        
        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_passed': True,
            'modules': {}
        }
        
        # 1. 标签一致性检查
        print("\n[1/3] 标签一致性检查...")
        try:
            label_results = self.label_checker.run_all_checks(y_train, dates)
            audit_results['modules']['label_consistency'] = label_results
            
            if not label_results['overall_passed']:
                audit_results['overall_passed'] = False
                print("❌ 标签一致性检查失败")
            else:
                print("✓ 标签一致性检查通过")
        except Exception as e:
            print(f"❌ 标签一致性检查出错: {e}")
            audit_results['overall_passed'] = False
        
        # 2. Lookahead特征检测
        print("\n[2/3] Lookahead特征检测...")
        try:
            lookahead_results = self.lookahead_detector.detect_lookahead(X_train, y_train)
            audit_results['modules']['lookahead_features'] = lookahead_results
            
            if lookahead_results['statistics']['lookahead_count'] > 0:
                print(f"⚠️ 发现{lookahead_results['statistics']['lookahead_count']}个Lookahead特征")
                print(f"  特征: {[f['feature'] for f in lookahead_results['lookahead_features']]}")
            else:
                print("✓ 未检测到Lookahead特征")
        except Exception as e:
            print(f"❌ Lookahead检测出错: {e}")
        
        # 3. 特征漂移检查
        if X_test is not None and y_test is not None:
            print("\n[3/3] 特征漂移检查...")
            try:
                drift_results = self.drift_calculator.calculate_drift(X_train, X_test)
                audit_results['modules']['feature_drift'] = drift_results
                
                if not drift_results['overall_passed']:
                    audit_results['overall_passed'] = False
                    print(f"❌ 特征漂移检查失败（{drift_results['drift_summary']['high_drift_count']}个高漂移特征）")
                else:
                    print("✓ 特征漂移检查通过")
            except Exception as e:
                print(f"❌ 特征漂移检查出错: {e}")
        else:
            print("\n[3/3] 跳过特征漂移检查（无测试集）")
        
        # 更新审计缓存
        self.audit_cache = {
            'label_consistency': audit_results['modules'].get('label_consistency'),
            'lookahead_features': audit_results['modules'].get('lookahead_features'),
            'feature_drift': audit_results['modules'].get('feature_drift')
        }
        
        # 更新决策状态
        self.decision_state['last_audit_time'] = audit_results['timestamp']
        self.decision_state['last_audit_result'] = audit_results
        self._save_audit_history()
        
        # 总结
        print("\n" + "=" * 70)
        if audit_results['overall_passed']:
            print("✅ 数据审计通过，可以开始训练")
        else:
            print("❌ 数据审计失败，请修复数据问题")
        print("=" * 70)
        
        return audit_results
    
    def run_quick_audit(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> bool:
        """
        快速审计（用于实时决策前的快速检查）
        
        参数:
            X: 特征数据
            y: 标签数据
            
        返回:
            是否通过审计
        """
        if not self.enable_deep_audit:
            return True
        
        try:
            # 快速检查数据基本质量
            if len(X) == 0 or len(y) == 0:
                return False
            
            if len(X) != len(y):
                return False
            
            # 检查特征是否有NaN
            if X.isnull().any().any():
                return False
            
            # 检查标签是否只有0和1
            unique_labels = set(y)
            if unique_labels - {0, 1}:
                return False
            
            return True
        except Exception as e:
            print(f"快速审计失败: {e}")
            return False
    
    # ============ 智能决策模块 ============
    
    def make_decision(
        self,
        stock_data: pd.DataFrame,
        current_index: int,
        symbol: str = "STOCK"
    ) -> Dict[str, Any]:
        """
        智能决策
        
        整合所有信息，做出买入/卖出/持有决策
        
        参数:
            stock_data: 股票数据
            current_index: 当前索引
            symbol: 股票代码
            
        返回:
            决策结果
        """
        decision_start_time = datetime.now()
        
        # 1. 检查系统状态
        if self.decision_state['system_status'] == 'rollback':
            return {
                'decision': 'hold',
                'confidence': 0.0,
                'reason': '系统处于回撤状态，暂停决策',
                'timestamp': decision_start_time.isoformat()
            }
        
        if self.decision_state['performance_status'] == 'degraded':
            return {
                'decision': 'hold',
                'confidence': 0.0,
                'reason': '系统性能下降，暂停决策',
                'timestamp': decision_start_time.isoformat()
            }
        
        # 2. 快速审计
        if not self.run_quick_audit(
            stock_data.iloc[:current_index+1],
            stock_data['target'].iloc[:current_index+1].values
        ):
            return {
                'decision': 'hold',
                'confidence': 0.0,
                'reason': '快速审计失败',
                'timestamp': decision_start_time.isoformat()
            }
        
        # 3. 获取预测
        try:
            prediction = self.predictor.predict(
                stock_data.iloc[current_index:current_index+1]
            )
            prediction_proba = self.predictor.predict_proba(
                stock_data.iloc[current_index:current_index+1]
            )
        except Exception as e:
            return {
                'decision': 'hold',
                'confidence': 0.0,
                'reason': f'预测失败: {e}',
                'timestamp': decision_start_time.isoformat()
            }
        
        pred_label = prediction[0]
        pred_proba = prediction_proba[0][1]  # 正类概率
        
        # 4. 置信度过滤
        confidence_result = self.confidence_filter.filter(
            pred_proba,
            pred_label
        )
        
        # 5. 三重确认
        if confidence_result['decision'] == 'approve':
            triple_result = self.triple_confirmation.validate_all(
                stock_data,
                current_index
            )
        else:
            triple_result = {
                'confirmed': False,
                'score': 0,
                'reason': confidence_result['reason']
            }
        
        # 6. 综合决策
        if pred_label == 1 and triple_result['confirmed']:
            decision = 'buy'
            self.decision_state['buy_signals'] += 1
            reason = f"买入: 预测概率={pred_proba:.4f}, 三重确认得分={triple_result['score']:.4f}"
        elif pred_label == 1 and not triple_result['confirmed']:
            decision = 'hold'
            reason = f"持有: 预测为正但未通过三重确认（{triple_result['reason']}）"
        elif pred_label == 0:
            decision = 'sell' if triple_result.get('exit_signal', False) else 'hold'
            if decision == 'sell':
                self.decision_state['sell_signals'] += 1
                reason = "卖出: 预测为负且触发退出信号"
            else:
                reason = "持有: 预测为负但未触发退出信号"
        else:
            decision = 'hold'
            self.decision_state['hold_signals'] += 1
            reason = "持有: 无明确信号"
        
        self.decision_state['decision_count'] += 1
        
        # 7. 如果启用了在线监控，更新监控
        if self.enable_online_monitoring and 'target' in stock_data.columns:
            actual_label = stock_data['target'].iloc[current_index]
            self.monitoring_system.process_predictions(
                np.array([actual_label]),
                np.array([pred_label])
            )
        
        decision_time = datetime.now()
        
        return {
            'decision': decision,
            'confidence': pred_proba,
            'triple_score': triple_result.get('score', 0),
            'reason': reason,
            'timestamp': decision_start_time.isoformat(),
            'processing_time': (decision_time - decision_start_time).total_seconds(),
            'prediction': pred_label,
            'triple_confirmation': triple_result
        }
    
    # ============ 监控和报告模块 ============
    
    def update_performance_monitor(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamp: Optional[str] = None
    ):
        """
        更新性能监控
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            timestamp: 时间戳（可选）
        """
        if not self.enable_online_monitoring:
            return
        
        self.monitoring_system.process_predictions(y_true, y_pred)
    
    def get_decision_status(self) -> Dict[str, Any]:
        """
        获取决策状态
        
        返回:
            决策状态字典
        """
        status = self.decision_state.copy()
        
        # 添加监控状态
        if self.enable_online_monitoring:
            status['monitoring_status'] = self.monitoring_system.get_system_status()
        
        # 添加审计状态
        status['audit_status'] = {
            'last_audit_time': self.decision_state['last_audit_time'],
            'last_audit_passed': (
                self.decision_state['last_audit_result']['overall_passed']
                if self.decision_state['last_audit_result']
                else None
            )
        }
        
        return status
    
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
        report_lines = []
        
        report_lines.append("# 突击选股智能决策系统综合报告")
        report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 系统状态
        report_lines.append("\n## 一、系统状态")
        report_lines.append(f"- 系统状态: {self.decision_state['system_status']}")
        report_lines.append(f"- 性能状态: {self.decision_state['performance_status']}")
        report_lines.append(f"- 决策总数: {self.decision_state['decision_count']}")
        report_lines.append(f"- 买入信号: {self.decision_state['buy_signals']}")
        report_lines.append(f"- 卖出信号: {self.decision_state['sell_signals']}")
        report_lines.append(f"- 持有信号: {self.decision_state['hold_signals']}")
        
        # 2. 审计状态
        report_lines.append("\n## 二、数据审计状态")
        if self.decision_state['last_audit_time']:
            report_lines.append(f"- 最后审计时间: {self.decision_state['last_audit_time']}")
            if self.decision_state['last_audit_result']:
                report_lines.append(f"- 审计结果: {'✅ 通过' if self.decision_state['last_audit_result']['overall_passed'] else '❌ 失败'}")
                
                # 详细审计结果
                audit_modules = self.decision_state['last_audit_result'].get('modules', {})
                for module_name, module_result in audit_modules.items():
                    report_lines.append(f"\n### {module_name}")
                    if module_name == 'label_consistency':
                        report_lines.append(f"- 前向异常: {module_result['forward']['anomaly_count']}")
                        report_lines.append(f"- 后向异常: {module_result['backward']['anomaly_count']}")
                    elif module_name == 'lookahead_features':
                        report_lines.append(f"- Lookahead特征数: {module_result['statistics']['lookahead_count']}")
                    elif module_name == 'feature_drift':
                        report_lines.append(f"- 漂移特征数: {module_result['drift_summary']['drift_count']}")
        else:
            report_lines.append("- 未执行审计")
        
        # 3. 监控状态
        if self.enable_online_monitoring:
            report_lines.append("\n## 三、在线监控状态")
            monitoring_status = self.monitoring_system.get_system_status()
            report_lines.append(f"- 总事件数: {monitoring_status['total_events']}")
            
            rollback_status = monitoring_status['rollback_status']
            report_lines.append(f"- 基准精确率: {rollback_status.get('baseline_precision', 'N/A')}")
            report_lines.append(f"- 当前精确率: {rollback_status.get('current_precision', 'N/A')}")
            report_lines.append(f"- 回撤触发: {'是' if rollback_status['rollback_triggered'] else '否'}")
            report_lines.append(f"- 回撤次数: {rollback_status['rollback_count']}")
        
        # 4. 建议
        report_lines.append("\n## 四、建议")
        
        if self.decision_state['system_status'] != 'ready':
            report_lines.append("- 系统状态异常，请检查系统配置和日志")
        
        if self.decision_state['performance_status'] == 'degraded':
            report_lines.append("- 系统性能下降，建议重新训练模型或调整参数")
        
        if self.decision_state['last_audit_result'] and not self.decision_state['last_audit_result']['overall_passed']:
            report_lines.append("- 数据审计失败，请修复数据问题后重新训练")
        
        if self.enable_online_monitoring:
            rollback_status = self.monitoring_system.get_system_status()['rollback_status']
            if rollback_status['rollback_triggered']:
                report_lines.append("- 检测到性能退化，建议触发回撤机制")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class AssaultTradingSystemIntegrated:
    """
    集成的突击交易系统
    
    整合智能决策系统和交易执行系统
    """
    
    def __init__(
        self,
        config_path: str = "config/short_term_assault_config.json"
    ):
        """
        初始化集成交易系统
        
        参数:
            config_path: 配置文件路径
        """
        self.brain = AssaultDecisionBrain(
            config_path=config_path,
            enable_deep_audit=True,
            enable_online_monitoring=True
        )
        
        self.positions = {}
        self.trade_history = []
    
    def pre_training_check(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[np.ndarray] = None
    ) -> bool:
        """
        训练前检查
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征（可选）
            y_test: 测试集标签（可选）
            
        返回:
            是否可以开始训练
        """
        audit_results = self.brain.run_pre_training_audit(
            X_train, y_train, X_test, y_test
        )
        
        return audit_results['overall_passed']
    
    def execute_trading(
        self,
        stock_data: pd.DataFrame,
        symbol: str = "STOCK"
    ) -> List[Dict[str, Any]]:
        """
        执行交易
        
        参数:
            stock_data: 股票数据
            symbol: 股票代码
            
        返回:
            交易历史
        """
        trades = []
        
        for i in range(10, len(stock_data)):  # 从第10行开始，确保有足够历史数据
            decision = self.brain.make_decision(
                stock_data,
                i,
                symbol
            )
            
            if decision['decision'] in ['buy', 'sell']:
                trade = {
                    'symbol': symbol,
                    'date': stock_data.index[i] if hasattr(stock_data.index[i], 'strftime') else i,
                    'decision': decision['decision'],
                    'confidence': decision['confidence'],
                    'price': stock_data['close'].iloc[i] if 'close' in stock_data.columns else None,
                    'reason': decision['reason']
                }
                
                trades.append(trade)
                self.trade_history.append(trade)
                
                print(f"[{trade['date']}] {decision['decision'].upper()}: {decision['reason']}")
        
        return trades
    
    def generate_final_report(self) -> str:
        """
        生成最终报告
        
        返回:
            报告字符串
        """
        report_lines = []
        
        report_lines.append("# 突击交易系统综合报告")
        report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 决策系统报告
        report_lines.append("\n## 一、决策系统报告")
        brain_report = self.brain.generate_comprehensive_report()
        report_lines.append(brain_report)
        
        # 2. 交易统计
        report_lines.append("\n## 二、交易统计")
        report_lines.append(f"- 总交易数: {len(self.trade_history)}")
        
        if self.trade_history:
            buy_trades = [t for t in self.trade_history if t['decision'] == 'buy']
            sell_trades = [t for t in self.trade_history if t['decision'] == 'sell']
            
            report_lines.append(f"- 买入次数: {len(buy_trades)}")
            report_lines.append(f"- 卖出次数: {len(sell_trades)}")
            
            if buy_trades:
                avg_confidence = np.mean([t['confidence'] for t in buy_trades])
                report_lines.append(f"- 平均买入置信度: {avg_confidence:.4f}")
        
        # 3. 仓位状态
        report_lines.append("\n## 三、仓位状态")
        report_lines.append(f"- 持仓数量: {len(self.positions)}")
        
        for symbol, position in self.positions.items():
            report_lines.append(f"- {symbol}: {position}")
        
        return "\n".join(report_lines)
