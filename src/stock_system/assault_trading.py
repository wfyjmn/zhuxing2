"""
短期突击交易系统
实现精准出击交易规则和风险控制系统
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AssaultTradingSystem:
    """短期突击交易系统"""
    
    def __init__(self, config_path: str = "config/short_term_assault_config.json"):
        """
        初始化交易系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.signal_grading = self.config['signal_grading']
        self.trading_execution = self.config['trading_execution']
        self.risk_management = self.config['risk_management']
        self.positions = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        import json
        from pathlib import Path
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_position_size(self, signal_grade: str, 
                         base_position: float = 0.05) -> float:
        """
        根据信号等级确定仓位大小
        
        Args:
            signal_grade: 信号等级 ('A', 'B', 'C', 'D')
            base_position: 基础仓位（默认5%）
        
        Returns:
            仓位比例
        """
        grade_config = {
            'A': self.signal_grading['A_grade'],
            'B': self.signal_grading['B_grade'],
            'C': self.signal_grading['C_grade'],
            'D': {'position_ratio': 0.0}
        }
        
        config = grade_config.get(signal_grade, {'position_ratio': 0.0})
        
        return base_position * config['position_ratio']
    
    def get_stop_loss(self, signal_grade: str) -> float:
        """
        根据信号等级确定止损比例
        
        Args:
            signal_grade: 信号等级
        
        Returns:
            止损比例（负数）
        """
        grade_config = {
            'A': self.signal_grading['A_grade'],
            'B': self.signal_grading['B_grade'],
            'C': self.signal_grading['C_grade']
        }
        
        config = grade_config.get(signal_grade, {'stop_loss': 0.10})
        
        return -config['stop_loss']
    
    def execute_trade(self, signal_grade: str, 
                     entry_price: float,
                     current_date: str,
                     symbol: str = "STOCK") -> Dict:
        """
        执行交易
        
        Args:
            signal_grade: 信号等级
            entry_price: 入场价格
            current_date: 当前日期
            symbol: 股票代码
        
        Returns:
            交易执行结果
        """
        if signal_grade == 'D':
            return {
                'action': 'none',
                'reason': 'D级信号，不满足确认条件'
            }
        
        # 确定仓位大小
        base_position = 0.05
        position_size = self.get_position_size(signal_grade, base_position)
        
        # 确定止损
        stop_loss = self.get_stop_loss(signal_grade)
        
        # 记录仓位
        position_id = f"{current_date}_{symbol}"
        self.positions[position_id] = {
            'symbol': symbol,
            'entry_price': entry_price,
            'position_size': position_size,
            'entry_date': current_date,
            'highest_price': entry_price,
            'highest_profit': 0.0,
            'signal_grade': signal_grade,
            'stop_loss': stop_loss,
            'trailing_stop': self.signal_grading[f'{signal_grade}_grade'].get('trailing_stop', False),
            'expected_holding_days': self.signal_grading[f'{signal_grade}_grade'].get('expected_holding_days', '3-5')
        }
        
        return {
            'action': 'buy',
            'position_id': position_id,
            'symbol': symbol,
            'entry_price': entry_price,
            'position_size': position_size,
            'signal_grade': signal_grade,
            'stop_loss': stop_loss,
            'reason': f"{signal_grade}级信号，{position_size*100:.1f}%仓位"
        }
    
    def check_exit_conditions(self, position_id: str, 
                             current_price: float,
                             current_date: str) -> Dict:
        """
        检查退出条件（止损/止盈）
        
        Args:
            position_id: 仓位ID
            current_price: 当前价格
            current_date: 当前日期
        
        Returns:
            退出信号
        """
        if position_id not in self.positions:
            return {'action': 'none', 'reason': '仓位不存在'}
        
        position = self.positions[position_id]
        entry_price = position['entry_price']
        
        # 计算当前收益
        profit = (current_price - entry_price) / entry_price
        
        # 更新最高价和最高收益
        if current_price > position['highest_price']:
            position['highest_price'] = current_price
            position['highest_profit'] = profit
        
        # 检查止损条件
        stop_loss_signal = self._check_stop_loss(position, current_price, profit)
        
        # 检查止盈条件
        take_profit_signal = self._check_take_profit(position, current_price, profit)
        
        # 优先执行止损
        if stop_loss_signal['action'] == 'sell':
            result = stop_loss_signal
            del self.positions[position_id]
            return result
        
        # 其次执行止盈
        if take_profit_signal['action'] == 'sell':
            result = take_profit_signal
            if take_profit_signal.get('partial_sell', False):
                position['position_size'] *= (1 - take_profit_signal['sell_ratio'])
            else:
                del self.positions[position_id]
            return result
        
        return {
            'action': 'hold',
            'reason': '无退出信号',
            'current_profit': profit,
            'highest_profit': position['highest_profit']
        }
    
    def _check_stop_loss(self, position: Dict, 
                         current_price: float,
                         profit: float) -> Dict:
        """检查止损条件"""
        stop_loss = position['stop_loss']
        signal_grade = position['signal_grade']
        
        # 固定止损
        if profit <= stop_loss:
            return {
                'action': 'sell',
                'reason': f"固定止损触发（收益{profit*100:.2f}% <= {stop_loss*100:.0f}%）",
                'exit_type': 'fixed_stop_loss',
                'signal_grade': signal_grade
            }
        
        # 移动止损（仅A级信号启用）
        if position.get('trailing_stop', False) and profit > 0.05:
            trailing_profit = (current_price - position['highest_price']) / position['highest_price']
            if trailing_profit <= -0.08:  # 最高点回落8%
                return {
                    'action': 'sell',
                    'reason': f"移动止损触发（最高点回落{abs(trailing_profit)*100:.2f}%）",
                    'exit_type': 'trailing_stop_loss',
                    'highest_profit': position['highest_profit'],
                    'signal_grade': signal_grade
                }
        
        return {'action': 'hold'}
    
    def _check_take_profit(self, position: Dict, 
                          current_price: float,
                          profit: float) -> Dict:
        """检查止盈条件"""
        # 阶梯止盈
        # 盈利10%减仓1/3
        if profit >= 0.10 and profit < 0.15:
            return {
                'action': 'sell',
                'reason': f"阶梯止盈1（盈利{profit*100:.2f}% >= 10%）",
                'exit_type': 'ladder_profit_1',
                'sell_ratio': 0.33,
                'partial_sell': True
            }
        
        # 盈利20%再减仓1/3
        elif profit >= 0.20 and profit < 0.30:
            return {
                'action': 'sell',
                'reason': f"阶梯止盈2（盈利{profit*100:.2f}% >= 20%）",
                'exit_type': 'ladder_profit_2',
                'sell_ratio': 0.33,
                'partial_sell': True
            }
        
        # 盈利30%清仓
        elif profit >= 0.30:
            return {
                'action': 'sell',
                'reason': f"阶梯止盈3（盈利{profit*100:.2f}% >= 30%）",
                'exit_type': 'ladder_profit_3',
                'sell_ratio': 1.0
            }
        
        # 移动止盈（盈利超过10%后启用）
        if profit > 0.10:
            trailing_profit = (current_price - position['highest_price']) / position['highest_price']
            if trailing_profit <= -0.08:
                return {
                    'action': 'sell',
                    'reason': f"移动止盈触发（最高点回落{abs(trailing_profit)*100:.2f}%）",
                    'exit_type': 'trailing_profit',
                    'highest_profit': position['highest_profit']
                }
        
        return {'action': 'hold'}
    
    def get_positions_summary(self) -> List[Dict]:
        """获取当前仓位摘要"""
        summary = []
        for position_id, position in self.positions.items():
            summary.append({
                'position_id': position_id,
                'symbol': position['symbol'],
                'entry_price': position['entry_price'],
                'position_size': position['position_size'],
                'entry_date': position['entry_date'],
                'highest_price': position['highest_price'],
                'highest_profit': position['highest_profit'],
                'signal_grade': position['signal_grade']
            })
        return summary
    
    def simulate_trading(self, df: pd.DataFrame, 
                        signals: List[Dict],
                        initial_capital: float = 100000) -> Dict:
        """
        模拟交易
        
        Args:
            df: 价格数据
            signals: 信号列表
            initial_capital: 初始资金
        
        Returns:
            模拟交易结果
        """
        capital = initial_capital
        trades = []
        position = None
        
        for signal_info in signals:
            date = signal_info['date']
            price = signal_info['price']
            signal_grade = signal_info['signal_grade']
            
            # 如果没有持仓且信号为A/B/C级
            if position is None and signal_grade in ['A', 'B', 'C']:
                # 确定仓位大小
                position_size = self.get_position_size(signal_grade, 0.05)
                position_value = capital * position_size
                
                position = {
                    'entry_date': date,
                    'entry_price': price,
                    'position_value': position_value,
                    'highest_price': price,
                    'highest_profit': 0.0,
                    'signal_grade': signal_grade
                }
                
                trades.append({
                    'date': date,
                    'action': 'buy',
                    'price': price,
                    'value': position_value,
                    'signal_grade': signal_grade,
                    'reason': f"{signal_grade}级信号"
                })
                
            # 如果有持仓，检查是否退出
            elif position is not None:
                profit = (price - position['entry_price']) / position['entry_price']
                
                # 更新最高价
                if price > position['highest_price']:
                    position['highest_price'] = price
                    position['highest_profit'] = profit
                
                # 检查止损止盈
                stop_loss = self.get_stop_loss(position['signal_grade'])
                
                # 止损
                if profit <= stop_loss:
                    sell_value = position['position_value'] * (1 + profit)
                    capital = capital - position['position_value'] + sell_value
                    
                    trades.append({
                        'date': date,
                        'action': 'sell',
                        'price': price,
                        'value': sell_value,
                        'profit': profit,
                        'reason': '止损'
                    })
                    
                    position = None
                
                # 阶梯止盈
                elif profit >= 0.30:
                    sell_value = position['position_value'] * (1 + profit)
                    capital = capital - position['position_value'] + sell_value
                    
                    trades.append({
                        'date': date,
                        'action': 'sell',
                        'price': price,
                        'value': sell_value,
                        'profit': profit,
                        'reason': '止盈30%'
                    })
                    
                    position = None
        
        # 计算最终结果
        total_return = (capital - initial_capital) / initial_capital
        
        # 统计胜率
        sell_trades = [t for t in trades if t['action'] == 'sell']
        if sell_trades:
            win_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
            win_rate = len(win_trades) / len(sell_trades)
            
            avg_profit = np.mean([t['profit'] for t in sell_trades])
            
            # 按信号等级统计
            grade_stats = {}
            for grade in ['A', 'B', 'C']:
                grade_trades = [t for t in sell_trades if t.get('signal_grade') == grade]
                if grade_trades:
                    grade_wins = [t for t in grade_trades if t.get('profit', 0) > 0]
                    grade_stats[grade] = {
                        'count': len(grade_trades),
                        'win_rate': len(grade_wins) / len(grade_trades),
                        'avg_profit': np.mean([t['profit'] for t in grade_trades])
                    }
        else:
            win_rate = 0
            avg_profit = 0
            grade_stats = {}
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': len(sell_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'grade_stats': grade_stats,
            'trades': trades
        }
    
    def get_market_position_limit(self, market_state: str) -> float:
        """
        根据市场状态获取总仓位上限
        
        Args:
            market_state: 市场状态 ('bull', 'range', 'bear')
        
        Returns:
            总仓位上限
        """
        position_limits = self.risk_management['position_sizing']
        return position_limits.get(market_state, 0.60)
