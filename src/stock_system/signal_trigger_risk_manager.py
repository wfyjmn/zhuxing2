"""
信号触发与风控管理器 v3.1 - 印钞机股票实时监控

【v3.1 功能】：
- 实现信号触发提醒机制（主动捕捉高确定性标的）
- 优化风控规则（适配印钞机股票）
- 实现动态监控（精确率维护+主动调整）
- 支持多渠道消息推送（邮件、微信、短信）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import warnings
warnings.filterwarnings('ignore')


class SignalTriggerRiskManager:
    """信号触发与风控管理器"""
    
    def __init__(self, config_path: str = "config/short_term_assault_config_v31.json"):
        """
        初始化管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.signal_config = self.config.get('signal_trigger', {})
        self.risk_config = self.config.get('risk_management', {})
        self.monitor_config = self.config.get('dynamic_monitoring', {})
        
        self.triggered_signals = []
        self.active_positions = {}
        self.performance_history = []
        
        print("=" * 70)
        print("信号触发与风控管理器 v3.1 - 印钞机股票实时监控")
        print("=" * 70)
        print(f"✓ 核心功能：主动捕捉高确定性标的 + 严苛风控")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        import json
        from pathlib import Path
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def check_signal_trigger(
        self,
        df: pd.DataFrame,
        model_prob: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        检查信号触发条件
        
        Args:
            df: 包含所有特征的DataFrame
            model_prob: 模型预测概率（如果有）
        
        Returns:
            包含触发标记的DataFrame
        """
        df = df.copy()
        
        # 如果有模型预测概率，添加到DataFrame
        if model_prob is not None:
            df['prediction_prob'] = model_prob
        
        # 获取触发条件
        trigger_conditions = self.signal_config.get('trigger_conditions', {})
        must_satisfy_triple = trigger_conditions.get('must_satisfy_triple_confirmation', True)
        precision_threshold = trigger_conditions.get('precision_threshold', 0.85)
        sharpe_min = trigger_conditions.get('sharpe_ratio_min', 15.0)
        
        # 条件1：必须满足三重确认
        if must_satisfy_triple and 'a_level_signal' in df.columns:
            condition1 = (df['a_level_signal'] == 1)
        else:
            condition1 = pd.Series([True] * len(df))
        
        # 条件2：精确率阈值
        if 'prediction_prob' in df.columns:
            condition2 = (df['prediction_prob'] >= precision_threshold)
        else:
            condition2 = pd.Series([True] * len(df))
        
        # 条件3：夏普比率阈值（如果有）
        if 'sharpe_ratio' in df.columns:
            condition3 = (df['sharpe_ratio'] >= sharpe_min)
        else:
            condition3 = pd.Series([True] * len(df))
        
        # 条件4：必须是印钞机股票候选
        if 'is_money_machine' in df.columns:
            condition4 = (df['is_money_machine'] == 1)
        else:
            condition4 = pd.Series([True] * len(df))
        
        # 生成触发信号
        df['signal_alert'] = (condition1 & condition2 & condition3 & condition4).astype(int)
        
        # 记录触发信号
        triggered_df = df[df['signal_alert'] == 1].copy()
        
        if len(triggered_df) > 0:
            self.triggered_signals.append({
                'datetime': datetime.now(),
                'signals': triggered_df,
                'count': len(triggered_df)
            })
            
            print(f"\n【信号触发】")
            print(f"  - 触发信号数: {len(triggered_df)}")
            print(f"  - 触发条件: 三重确认 + 精确率≥{precision_threshold*100:.0f}% + 印钞机候选")
            
            # 输出触发信号详情
            for idx, row in triggered_df.head(10).iterrows():
                if 'ts_code' in row:
                    print(f"    - {row['ts_code']}: 预测概率={row.get('prediction_prob', 0):.3f}, 印钞机评分={row.get('money_machine_score', 0)}")
        
        return df
    
    def apply_risk_controls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用风控规则
        
        Args:
            df: 包含信号的DataFrame
        
        Returns:
            包含风控标记的DataFrame
        """
        df = df.copy()
        
        # 获取A级信号的风控规则
        for_a_signals = self.risk_config.get('for_a_signals', {})
        stop_loss = for_a_signals.get('stop_loss', 0.03)
        take_profit_initial = for_a_signals.get('take_profit_initial', 0.20)
        trailing_stop_loss = for_a_signals.get('trailing_stop_loss', 0.05)
        
        # 应用流动性筛选
        liquidity_filters = self.risk_config.get('filters', {}).get('liquidity', {})
        min_market_cap = liquidity_filters.get('min_market_cap', 5000000000)
        min_daily_turnover = liquidity_filters.get('min_daily_turnover', 10000000)
        
        if 'market_cap' in df.columns:
            df['liquidity_qualified'] = (
                (df['market_cap'] >= min_market_cap) &
                (df['amount'] >= min_daily_turnover)
            ).astype(int)
        else:
            df['liquidity_qualified'] = (df['amount'] >= min_daily_turnover).astype(int)
        
        # 应用持续性筛选
        sustainability_filters = self.risk_config.get('filters', {}).get('sustainability', {})
        min_continuous_rise_days = sustainability_filters.get('min_continuous_rise_days', 2)
        
        if 'continuous_rise_days' in df.columns:
            df['sustainability_qualified'] = (
                df['continuous_rise_days'] >= min_continuous_rise_days
            ).astype(int)
        else:
            df['sustainability_qualified'] = 1
        
        # 应用风险筛选
        risk_filters = self.risk_config.get('filters', {}).get('risk', {})
        exclude_st = risk_filters.get('exclude_st', True)
        
        if 'is_st' in df.columns:
            df['risk_qualified'] = ((df['is_st'] == 0) if exclude_st else True).astype(int)
        else:
            df['risk_qualified'] = 1
        
        # 综合风控标记
        df['risk_qualified_all'] = (
            df['liquidity_qualified'] &
            df['sustainability_qualified'] &
            df['risk_qualified']
        ).astype(int)
        
        # 设置止损和止盈线
        df['stop_loss_price'] = df['close'] * (1 - stop_loss)
        df['take_profit_price'] = df['close'] * (1 + take_profit_initial)
        
        print(f"\n【风控规则应用】")
        print(f"  - 止损线: {stop_loss*100:.1f}%")
        print(f"  - 止盈线: {take_profit_initial*100:.1f}%")
        print(f"  - 移动止损: {trailing_stop_loss*100:.1f}%")
        print(f"  - 流动性合格率: {df['liquidity_qualified'].mean()*100:.1f}%")
        print(f"  - 持续性合格率: {df['sustainability_qualified'].mean()*100:.1f}%")
        print(f"  - 风险合格率: {df['risk_qualified'].mean()*100:.1f}%")
        
        return df
    
    def send_alert(
        self,
        df: pd.DataFrame,
        alert_method: List[str] = None
    ):
        """
        发送信号提醒
        
        Args:
            df: 包含触发信号的DataFrame
            alert_method: 提醒方式列表 ['email', 'wechat', 'sms']
        """
        if alert_method is None:
            alert_method = self.signal_config.get('alert_mechanism', {}).get('methods', ['email'])
        
        # 获取触发信号
        triggered_df = df[df['signal_alert'] == 1]
        
        if len(triggered_df) == 0:
            return
        
        # 构建提醒消息
        message = f"""
【印钞机股票捕捉提醒】
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
触发数量: {len(triggered_df)}只

触发条件:
- ✓ 三重确认（资金+情绪+技术）
- ✓ 精确率≥{self.signal_config.get('trigger_conditions', {}).get('precision_threshold', 0.85)*100:.0f}%
- ✓ 印钞机候选

候选标的:
"""
        
        for idx, row in triggered_df.head(10).iterrows():
            if 'ts_code' in row:
                message += f"\n  {row['ts_code']}: 预测概率={row.get('prediction_prob', 0):.3f}, 印钞机评分={row.get('money_machine_score', 0)}\n"
        
        message += f"\n风控提醒:\n"
        message += f"  - 止损线: {self.risk_config.get('for_a_signals', {}).get('stop_loss', 0.03)*100:.1f}%\n"
        message += f"  - 止盈线: {self.risk_config.get('for_a_signals', {}).get('take_profit_initial', 0.20)*100:.1f}%\n"
        message += f"  - 移动止损: {self.risk_config.get('for_a_signals', {}).get('trailing_stop_loss', 0.05)*100:.1f}%\n"
        
        # 发送提醒
        if 'email' in alert_method:
            self._send_email_alert(message)
        
        if 'wechat' in alert_method:
            self._send_wechat_alert(message)
        
        if 'sms' in alert_method:
            self._send_sms_alert(message)
    
    def _send_email_alert(self, message: str):
        """发送邮件提醒（需配置SMTP）"""
        print(f"\n【邮件提醒】")
        print(f"  ⚠ 邮件提醒功能需配置SMTP服务器")
        print(f"  消息预览:\n{message[:200]}...")
    
    def _send_wechat_alert(self, message: str):
        """发送微信提醒（需配置企业微信）"""
        print(f"\n【微信提醒】")
        print(f"  ⚠ 微信提醒功能需配置企业微信webhook")
        print(f"  消息预览:\n{message[:200]}...")
    
    def _send_sms_alert(self, message: str):
        """发送短信提醒（需配置短信服务商）"""
        print(f"\n【短信提醒】")
        print(f"  ⚠ 短信提醒功能需配置短信服务商")
        print(f"  消息预览:\n{message[:200]}...")
    
    def monitor_performance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamp: datetime = None
    ) -> Dict[str, float]:
        """
        监控性能指标
        
        Args:
            predictions: 预测结果
            actuals: 实际结果
            timestamp: 时间戳
        
        Returns:
            性能指标字典
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # 计算指标
        pred_class = (predictions >= 0.5).astype(int)
        precision = precision_score(actuals, pred_class)
        recall = recall_score(actuals, pred_class)
        f1 = f1_score(actuals, pred_class)
        
        # 计算盈亏比（模拟）
        positive_returns = predictions[actuals == 1]
        negative_returns = predictions[actuals == 0]
        profit_loss_ratio = np.abs(positive_returns.mean()) / (np.abs(negative_returns.mean()) + 1e-9) if len(negative_returns) > 0 else 0
        
        performance = {
            'timestamp': timestamp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'profit_loss_ratio': profit_loss_ratio,
            'signal_count': (pred_class == 1).sum()
        }
        
        self.performance_history.append(performance)
        
        # 检查是否需要调整
        self._check_adjustment_needed(performance)
        
        return performance
    
    def _check_adjustment_needed(self, performance: Dict[str, float]):
        """
        检查是否需要调整
        
        Args:
            performance: 性能指标
        """
        daily_monitoring = self.monitor_config.get('daily_monitoring', {})
        precision_min = daily_monitoring.get('precision_min', 0.85)
        profit_loss_ratio_min = daily_monitoring.get('profit_loss_ratio_min', 3.0)
        
        print(f"\n【性能监控】")
        print(f"  - 精确率: {performance['precision']*100:.2f}% (目标: ≥{precision_min*100:.0f}%)")
        print(f"  - 召回率: {performance['recall']*100:.2f}%")
        print(f"  - 盈亏比: {performance['profit_loss_ratio']:.2f} (目标: ≥{profit_loss_ratio_min:.1f})")
        print(f"  - 触发信号数: {performance['signal_count']}")
        
        # 检查精确率是否连续3天低于目标
        if len(self.performance_history) >= 3:
            recent_precision = [p['precision'] for p in self.performance_history[-3:]]
            if all(p < precision_min for p in recent_precision):
                print(f"  ⚠ 警告: 精确率连续3天低于目标，建议进行样本提纯、特征重选、参数微调")
        
        # 检查是否长期无触发信号
        if len(self.performance_history) >= 7:
            recent_signals = [p['signal_count'] for p in self.performance_history[-7:]]
            if sum(recent_signals) == 0:
                print(f"  ⚠ 警告: 连续7天无触发信号，建议适当微调筛选条件（不降低精确率底线）")
    
    def generate_daily_report(self) -> pd.DataFrame:
        """
        生成每日报告
        
        Returns:
            每日报告DataFrame
        """
        if len(self.performance_history) == 0:
            print("\n【每日报告】")
            print("  ⚠ 暂无性能数据")
            return pd.DataFrame()
        
        report_df = pd.DataFrame(self.performance_history)
        
        print(f"\n【每日报告】")
        print(f"  - 统计周期: {len(report_df)}天")
        print(f"  - 平均精确率: {report_df['precision'].mean()*100:.2f}%")
        print(f"  - 平均召回率: {report_df['recall'].mean()*100:.2f}%")
        print(f"  - 平均盈亏比: {report_df['profit_loss_ratio'].mean():.2f}")
        print(f"  - 总触发信号数: {report_df['signal_count'].sum()}")
        print(f"  - 每日平均触发: {report_df['signal_count'].mean():.1f}个")
        
        return report_df
    
    def save_performance_history(self, filepath: str):
        """
        保存性能历史记录
        
        Args:
            filepath: 保存路径
        """
        if len(self.performance_history) == 0:
            print("\n【保存性能历史】")
            print("  ⚠ 暂无性能数据")
            return
        
        report_df = pd.DataFrame(self.performance_history)
        report_df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"\n【保存性能历史】")
        print(f"  ✓ 性能历史已保存到: {filepath}")
