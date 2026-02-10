"""
三重确认机制 - 资金/情绪/技术
确保"出击必中"，减少无效交易
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TripleConfirmation:
    """三重确认机制"""
    
    def __init__(self, config_path: str = "config/short_term_assault_config.json"):
        """
        初始化三重确认机制
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.triple_confirmation = self.config['triple_confirmation']
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        import json
        from pathlib import Path
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def validate_capital_confirmation(self, df: pd.DataFrame, index: int) -> Dict:
        """
        第一重：资金确认
        
        必须条件：
        - 主力资金连续2日净流入
        - 大单买入占比>25%
        - 资金流入速度加快(今日>昨日20%)
        
        Args:
            df: 包含特征的DataFrame
            index: 要验证的索引
        
        Returns:
            资金确认结果
        """
        if index < 5:
            return {
                'confirmed': False,
                'score': 0,
                'details': '数据不足'
            }
        
        must_have_count = 0
        details = []
        
        # 1. 主力资金连续2日净流入
        capital_inflow_1 = df['main_capital_inflow_ratio'].iloc[index]
        capital_inflow_2 = df['main_capital_inflow_ratio'].iloc[index-1]
        
        if capital_inflow_1 > 0.05 and capital_inflow_2 > 0.05:
            must_have_count += 1
            details.append(f"主力资金连续净流入（今日{capital_inflow_1:.2%}，昨日{capital_inflow_2:.2%}）")
        
        # 2. 大单买入占比>25%
        large_order_rate = df['large_order_buy_rate'].iloc[index]
        if large_order_rate > 0.25:
            must_have_count += 1
            details.append(f"大单买入占比{large_order_rate:.2%}")
        
        # 3. 资金流入速度加快(今日>昨日20%)
        inflow_speed = (
            (capital_inflow_1 - capital_inflow_2) / abs(capital_inflow_2)
            if capital_inflow_2 != 0 else 0
        )
        if inflow_speed > 0.2:
            must_have_count += 1
            details.append(f"资金流入速度加快{inflow_speed:.1%}")
        
        # 检查排除条件
        exclusion = []
        if capital_inflow_1 < -0.05:
            exclusion.append("主力资金大幅流出")
        if large_order_rate < 0.3 and df['price_change'].iloc[index] > 0.05:
            # 涨幅大但大单买入少，可能是拉升出货
            exclusion.append("大单卖出占比高")
        
        confirmed = must_have_count >= 2 and len(exclusion) == 0
        
        return {
            'confirmed': confirmed,
            'score': must_have_count / 3,
            'must_have_count': must_have_count,
            'exclusion': exclusion,
            'details': '; '.join(details) if details else '无明显资金特征'
        }
    
    def validate_sentiment_confirmation(self, df: pd.DataFrame, index: int) -> Dict:
        """
        第二重：情绪确认
        
        市场层面：
        - 指数环境：上证指数在20日线上方
        - 涨跌家数：上涨家数>下跌家数
        - 涨停效应：涨停家数>跌停家数2倍以上
        
        板块层面：
        - 板块排名：所属板块涨幅排名前30%
        - 板块强度：板块指数突破关键位置
        - 龙头效应：板块内有涨停龙头
        
        个股层面：
        - 股吧热度：讨论量排名前20%
        - 新闻面：有正向催化剂
        - 技术形态：突破形态完整
        
        Args:
            df: 包含特征的DataFrame
            index: 要验证的索引
        
        Returns:
            情绪确认结果
        """
        if index < 20:
            return {
                'confirmed': False,
                'score': 0,
                'details': '数据不足'
            }
        
        market_score = 0
        sector_score = 0
        individual_score = 0
        details = []
        
        # 市场层面
        # 1. 指数环境（用价格在20日均线上方代理）
        ma_20 = df['ma_20'].iloc[index]
        if df['close'].iloc[index] > ma_20:
            market_score += 1
            details.append("指数在20日均线上方")
        
        # 2. 涨跌家数（用5日涨跌天数比代理）
        up_days = df['up_days_ratio'].iloc[index]
        if up_days > 0.5:
            market_score += 1
            details.append(f"上涨家数占比{up_days:.1%}")
        
        # 3. 涨停效应（用板块热度指数代理）
        sector_heat = df['sector_heat_index'].iloc[index]
        if sector_heat > 0.1:  # 10%以上
            market_score += 1
            details.append(f"板块热度{sector_heat:.1%}")
        
        # 板块层面（简化版：用个股强度代理）
        # 4. 板块排名（用价格涨幅排名代理）
        price_change_5 = df['close'].iloc[index] / df['close'].iloc[index-5] - 1
        if price_change_5 > 0:
            sector_score += 1
            details.append(f"5日涨幅{price_change_5:.1%}")
        
        # 5. 板块强度（用价格突破20日高点代理）
        if df['price_breakout_20'].iloc[index]:
            sector_score += 1
            details.append("价格突破20日高点")
        
        # 6. 龙头效应（用涨停强度代理）
        if df['price_change'].iloc[index] > 0.09:
            sector_score += 1
            details.append("涨停")
        
        # 个股层面
        # 7. 股吧热度（用情绪得分代理）
        sentiment_score = df['stock_sentiment_score'].iloc[index]
        if sentiment_score > 70:
            individual_score += 1
            details.append(f"情绪得分{sentiment_score:.0f}")
        
        # 8. 新闻面（用动量代理）
        momentum_5 = df['momentum_5'].iloc[index]
        if momentum_5 > 0.05:
            individual_score += 1
            details.append(f"5日动量{momentum_5:.1%}")
        
        # 9. 技术形态（用均线多头排列代理）
        if df['ma_bullish_arrangement'].iloc[index]:
            individual_score += 1
            details.append("均线多头排列")
        
        # 综合评分
        total_score = market_score / 3 * 0.4 + sector_score / 3 * 0.3 + individual_score / 3 * 0.3
        
        confirmed = total_score >= 0.6  # 总体得分>60%
        
        return {
            'confirmed': confirmed,
            'score': total_score,
            'market_score': market_score,
            'sector_score': sector_score,
            'individual_score': individual_score,
            'details': '; '.join(details[:5]) if details else '无明显情绪特征'
        }
    
    def validate_technical_confirmation(self, df: pd.DataFrame, index: int) -> Dict:
        """
        第三重：技术确认
        
        动量指标：
        - RSI(6)>60
        - MACD金叉且红柱放大
        - KDJ金叉且J值>50
        
        量价关系：
        - 成交量较20日均量放大>50%
        - 价格突破20日高点
        - 量比>1.5
        
        时间周期：
        - 日线、60分钟线同步看多
        - 调整时间充分(至少3天以上)
        - 突破发生在早盘或尾盘关键时段
        
        Args:
            df: 包含特征的DataFrame
            index: 要验证的索引
        
        Returns:
            技术确认结果
        """
        if index < 20:
            return {
                'confirmed': False,
                'score': 0,
                'details': '数据不足'
            }
        
        momentum_score = 0
        volume_price_score = 0
        time_cycle_score = 0
        details = []
        
        # 动量指标
        # 1. RSI(6)>60
        enhanced_rsi = df['enhanced_rsi'].iloc[index]
        if enhanced_rsi > 60:
            momentum_score += 1
            details.append(f"RSI强化值{enhanced_rsi:.1f}")
        
        # 2. MACD金叉且红柱放大
        macd_hist = df['macd_hist'].iloc[index]
        macd_hist_prev = df['macd_hist'].iloc[index-1]
        if macd_hist > 0 and macd_hist > macd_hist_prev:
            momentum_score += 1
            details.append("MACD金叉且红柱放大")
        
        # 3. KDJ金叉且J值>50
        j_value = df['j_value'].iloc[index]
        if df['kdj_golden_cross'].iloc[index] and j_value > 50:
            momentum_score += 1
            details.append(f"KDJ金叉且J值{j_value:.1f}")
        
        # 量价关系
        # 4. 成交量较20日均量放大>50%
        volume_ratio = df['volume_ratio_5'].iloc[index]
        if volume_ratio > 1.5:
            volume_price_score += 1
            details.append(f"量比{volume_ratio:.1f}")
        
        # 5. 价格突破20日高点
        if df['price_breakout_20'].iloc[index]:
            volume_price_score += 1
            details.append("价格突破20日高点")
        
        # 6. 量价突破强度
        breakout_strength = df['volume_price_breakout_strength'].iloc[index]
        if breakout_strength > 2:
            volume_price_score += 1
            details.append(f"量价突破强度{breakout_strength:.1f}")
        
        # 时间周期
        # 7. 日线、60分钟线同步看多（简化版：用日线趋势和短期动量）
        if df['ma_bullish_arrangement'].iloc[index] and df['momentum_5'].iloc[index] > 0:
            time_cycle_score += 1
            details.append("日线和短期动量同步看多")
        
        # 8. 调整时间充分（用回撤天数代理）
        # 简化版：检查是否在支撑位附近
        price_position = (df['close'].iloc[index] - df['low'].rolling(5).min().iloc[index]) / \
                        (df['high'].rolling(5).max().iloc[index] - df['low'].rolling(5).min().iloc[index])
        if 0.3 < price_position < 0.7:
            time_cycle_score += 1
            details.append("价格在合理区间")
        
        # 9. 攻击形态
        if df['intraday_attack_pattern'].iloc[index] >= 2:
            time_cycle_score += 1
            details.append("存在攻击波")
        
        # 综合评分
        total_score = momentum_score / 3 * 0.4 + volume_price_score / 3 * 0.4 + time_cycle_score / 3 * 0.2
        
        confirmed = total_score >= 0.6
        
        return {
            'confirmed': confirmed,
            'score': total_score,
            'momentum_score': momentum_score,
            'volume_price_score': volume_price_score,
            'time_cycle_score': time_cycle_score,
            'details': '; '.join(details[:5]) if details else '无明显技术特征'
        }
    
    def validate_all_confirmations(self, df: pd.DataFrame, index: int) -> Dict:
        """
        执行三重确认
        
        Args:
            df: 包含特征的DataFrame
            index: 要验证的索引
        
        Returns:
            三重确认结果
        """
        # 资金确认
        capital_result = self.validate_capital_confirmation(df, index)
        
        # 情绪确认
        sentiment_result = self.validate_sentiment_confirmation(df, index)
        
        # 技术确认
        technical_result = self.validate_technical_confirmation(df, index)
        
        # 统计确认数量
        confirmed_count = sum([
            capital_result['confirmed'],
            sentiment_result['confirmed'],
            technical_result['confirmed']
        ])
        
        # 综合评分
        overall_score = (
            capital_result['score'] * 0.4 +
            sentiment_result['score'] * 0.35 +
            technical_result['score'] * 0.25
        )
        
        return {
            'capital': capital_result,
            'sentiment': sentiment_result,
            'technical': technical_result,
            'confirmed_count': confirmed_count,
            'overall_score': overall_score,
            'final_confirmed': confirmed_count >= 2 or overall_score >= 0.6
        }
    
    def get_signal_grade(self, confirmation_result: Dict) -> str:
        """
        根据三重确认结果确定信号等级
        
        Args:
            confirmation_result: 三重确认结果
        
        Returns:
            信号等级 ('A', 'B', 'C', 'D')
        """
        confirmed_count = confirmation_result['confirmed_count']
        overall_score = confirmation_result['overall_score']
        
        # A级：三重确认全部满足
        if confirmed_count == 3:
            return 'A'
        # B级：满足两重确认
        elif confirmed_count == 2:
            return 'B'
        # C级：满足一重确认（仅资金强度）
        elif confirmed_count == 1 and confirmation_result['capital']['confirmed']:
            return 'C'
        # D级：不满足任何确认
        else:
            return 'D'
