"""
增强版特征工程 - 引入主力资金、北向资金等强特征
针对A股市场特点设计
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """增强版特征工程（引入主力资金、北向资金等）"""

    def __init__(self, config_path: str = "config/short_term_assault_config.json"):
        self.config = self._load_config(config_path)
        self.feature_names = []

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        import json
        from pathlib import Path

        config_file = Path(config_path)
        if not config_file.exists():
            # 默认配置
            return {
                "feature_weights": {
                    "main_capital_flow": {"weight": 0.30},  # 主力资金
                    "northbound_capital": {"weight": 0.20},  # 北向资金
                    "market_sentiment": {"weight": 0.20},  # 市场情绪
                    "technical_indicators": {"weight": 0.30}  # 技术指标
                }
            }

        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _calculate_adj_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算复权价格
        
        Args:
            df: 包含 adj_factor 列的DataFrame
            
        Returns:
            添加复权价格列的DataFrame
        """
        df = df.copy()
        
        # 检查是否有复权因子
        if 'adj_factor' in df.columns:
            # 计算前复权价格
            df['adj_close'] = df['close'] * df['adj_factor']
            df['adj_open'] = df['open'] * df['adj_factor']
            df['adj_high'] = df['high'] * df['adj_factor']
            df['adj_low'] = df['low'] * df['adj_factor']
            print("  ✓ 使用复权价格计算技术指标")
        else:
            # 如果没有复权因子，使用原始价格
            df['adj_close'] = df['close']
            df['adj_open'] = df['open']
            df['adj_high'] = df['high']
            df['adj_low'] = df['low']
            print("  ⚠ 未检测到复权因子，使用原始价格计算技术指标（可能导致信号失真）")
        
        return df

    def create_main_capital_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建主力资金流向特征（权重30%）

        优先使用真实资金流数据，如果不可用则使用代理指标
        """
        df = df.copy()

        # 检查是否有真实资金流数据（特大单买入量字段是关键）
        has_real_flow = 'buy_elg_vol' in df.columns

        if has_real_flow:
            print("  ✓ 检测到真实资金流数据，正在计算主力特征...")
            
            # 1. 主力净流入金额 (特大单+大单)
            df['main_net_inflow'] = (
                df['buy_elg_vol'].fillna(0) + df['buy_lg_vol'].fillna(0) - 
                df['sell_elg_vol'].fillna(0) - df['sell_lg_vol'].fillna(0)
            )
            
            # 2. 主力净流入率 (净流入 / 总成交量)
            df['main_flow_rate'] = df['main_net_inflow'] / (df['volume'] + 1)
            
            # 3. 主力资金连红天数
            df['main_flow_up_days'] = (df['main_net_inflow'] > 0).astype(int)
            df['main_flow_persistence'] = df['main_flow_up_days'].rolling(5).sum()
            
            # 4. 主力控盘度 (累计主力净流入 / 流通股本)
            # 使用总成交量代替流通股本作为代理
            df['main_control_proxy'] = (
                df['main_net_inflow'].rolling(20).sum() / 
                (df['volume'].rolling(20).sum() + 1)
            )
            
            # 5. 主力净流入5日均值
            df['main_net_inflow_ma5'] = df['main_net_inflow'].rolling(5).mean()
            
            # 6. 散户恐慌度 (小单净卖出占比)
            df['retail_panic_ratio'] = (
                (df['sell_sm_vol'].fillna(0) - df['buy_sm_vol'].fillna(0)) / 
                (df['volume'] + 1)
            )
            
            # 7. 大单净流入金额（基于金额数据）
            if 'buy_elg_amount' in df.columns:
                df['main_net_inflow_amount'] = (
                    df['buy_elg_amount'].fillna(0) + df['buy_lg_amount'].fillna(0) - 
                    df['sell_elg_amount'].fillna(0) - df['sell_lg_amount'].fillna(0)
                )
                df['main_net_inflow_amount_ma5'] = df['main_net_inflow_amount'].rolling(5).mean()
            
            # 8. 资金加速度（资金流入的加速度）
            df['capital_acceleration'] = df['main_flow_rate'].diff()
            
            print(f"✓ 主力资金特征已创建: 基于真实资金流数据（权重30%）")
        else:
            print("  ⚠ 未检测到资金流数据，使用量价代理指标（效果较差）...")
            
            # 回退到代理指标（基于OBV）
            price_change = df['close'].diff()
            volume = df['volume']

            # 计算OBV
            obv = (np.sign(price_change) * volume).fillna(0).cumsum()

            # OBV变化率（主力资金流向代理）
            df['main_capital_inflow_rate'] = obv.diff() / (obv.rolling(20).std() + 1)
            df['main_capital_inflow_ma5'] = df['main_capital_inflow_rate'].rolling(5).mean()
            df['main_capital_inflow_ma20'] = df['main_capital_inflow_rate'].rolling(20).mean()

            # 主力资金持续性
            df['main_capital_persistence'] = (
                (df['main_capital_inflow_rate'] > 0).astype(int).rolling(5).sum() / 5
            )

            # 大单净流入（基于量价关系的代理）
            price_up = df['close'] > df['open']
            volume_avg = df['volume'].rolling(20).mean()

            df['large_order_inflow'] = np.where(
                price_up,
                (df['volume'] / volume_avg).clip(0.5, 3),
                -(df['volume'] / volume_avg).clip(0.5, 3) * 0.3
            )

            df['large_order_inflow_ma5'] = df['large_order_inflow'].rolling(5).mean()

            # 资金集中度（成交量集中度）
            df['volume_concentration'] = (
                df['volume'] / df['volume'].rolling(5).mean()
            ).rolling(5).std()

            # 资金加速度（资金流入的加速度）
            df['capital_acceleration'] = df['main_capital_inflow_rate'].diff()
            
            # 为了保持特征名称一致性，添加别名
            df['main_net_inflow'] = df['large_order_inflow']
            df['main_flow_rate'] = df['main_capital_inflow_rate']
            df['main_flow_persistence'] = df['main_capital_persistence'] * 5
            df['retail_panic_ratio'] = 0.0  # 代理指标无法计算
            
            print(f"✓ 主力资金特征已创建: 基于代理指标（权重30%）")
        
        return df

    def create_northbound_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建北向资金特征（权重20%）

        使用个股相对强度作为北向资金的代理指标
        """
        df = df.copy()

        # 1. 个股相对强度（个股vs市场的相对表现）
        df['returns'] = df['close'].pct_change()
        df['returns_ma5'] = df['returns'].rolling(5).mean()
        df['returns_ma20'] = df['returns'].rolling(20).mean()

        # 相对强度 = 个股收益 - 市场平均收益（用板块或大盘代理）
        # 这里简化为个股自身收益的加权组合
        df['relative_strength'] = (
            0.6 * df['returns_ma5'] +
            0.4 * df['returns_ma20']
        )

        # 2. 北向资金流入信号（基于相对强度的变化）
        df['northbound_signal'] = df['relative_strength'].diff(3)

        # 3. 北向资金持续性
        df['northbound_persistence'] = (
            (df['relative_strength'] > 0).astype(int).rolling(5).sum() / 5
        )

        # 4. 北向资金强度（相对强度的强度）
        df['northbound_intensity'] = (
            df['relative_strength'].abs() *
            np.sign(df['relative_strength'])
        )

        # 5. 外资偏好度（价格稳定性 + 流动性）
        price_stability = 1 / (df['close'].pct_change().rolling(20).std() + 0.01)
        liquidity = df['volume'].rolling(20).mean()
        df['foreign_preference'] = (
            price_stability * liquidity /
            (price_stability * liquidity).rolling(60).mean()
        )

        print(f"✓ 北向资金特征已创建: 5个特征（权重20%）")
        return df

    def create_market_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建市场情绪特征（权重20%）

        包括板块热度、个股情绪、市场广度
        """
        df = df.copy()

        # 1. 涨停强度（价格涨停力度）
        df['daily_return'] = df['close'] / df['open'] - 1
        df['limit_up_signal'] = (df['daily_return'] > 0.095).astype(int)
        df['limit_up_ma5'] = df['limit_up_signal'].rolling(5).sum() / 5

        # 2. 跌停风险
        df['limit_down_signal'] = (df['daily_return'] < -0.095).astype(int)
        df['limit_down_ma5'] = df['limit_down_signal'].rolling(5).sum() / 5

        # 3. 涨跌比（上涨vs下跌的概率）
        df['up_down_ratio'] = (
            df['daily_return'].rolling(10).apply(
                lambda x: (x > 0).sum() / len(x), raw=True
            )
        )

        # 4. 个股情绪得分（综合多个指标）
        # 价格位置（0-1）
        price_range = df['high'].rolling(20).max() - df['low'].rolling(20).min()
        price_position = (df['close'] - df['low'].rolling(20).min()) / (price_range + 0.01)
        price_position = price_position.fillna(0.5)

        # 量能放大倍数
        volume_surge = df['volume'] / df['volume'].rolling(20).mean()

        # 涨幅强度
        return_strength = df['daily_return'].clip(0, 0.1) * 10

        df['stock_sentiment'] = (
            0.3 * price_position * 10 +
            0.3 * volume_surge.clip(0.5, 3).fillna(1) * 3.33 +
            0.2 * return_strength +
            0.2 * df['up_down_ratio'] * 10
        ).clip(0, 100)

        # 5. 情绪周期（基于RSI）
        rsi_14 = self._calculate_rsi(df['close'], 14)
        df['sentiment_cycle'] = rsi_14 / 100

        # 6. 情绪变化率
        df['sentiment_change'] = df['stock_sentiment'].diff(3)

        print(f"✓ 市场情绪特征已创建: 8个特征（权重20%）")
        return df

    def create_technical_indicators_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建技术指标特征（权重30%）

        包括趋势、动量、波动率、成交量等
        优先使用复权价格计算，避免除权日信号失真
        """
        df = df.copy()
        
        # 先计算复权价格
        df = self._calculate_adj_price(df)
        
        # 确定使用哪个价格列（优先复权价格）
        price_col = 'adj_close'
        open_col = 'adj_open'
        high_col = 'adj_high'
        low_col = 'adj_low'

        # 1. 趋势指标
        df['ma_5'] = df[price_col].rolling(5).mean()
        df['ma_10'] = df[price_col].rolling(10).mean()
        df['ma_20'] = df[price_col].rolling(20).mean()
        df['ma_60'] = df[price_col].rolling(60).mean()

        # 价格均线斜率
        df['ma_5_slope'] = df['ma_5'].diff(2)
        df['ma_20_slope'] = df['ma_20'].diff(2)

        # 均线多头排列
        df['ma_bullish_arrangement'] = (
            (df['ma_5'] > df['ma_10']) &
            (df['ma_10'] > df['ma_20'])
        ).astype(int)

        # 价格站上均线
        df['price_above_ma5'] = (df[price_col] > df['ma_5']).astype(int)
        df['price_above_ma20'] = (df[price_col] > df['ma_20']).astype(int)

        # 2. 动量指标（RSI多周期）
        rsi_6 = self._calculate_rsi(df[price_col], 6)
        rsi_12 = self._calculate_rsi(df[price_col], 12)
        rsi_24 = self._calculate_rsi(df[price_col], 24)

        df['rsi_6'] = rsi_6
        df['rsi_12'] = rsi_12
        df['rsi_24'] = rsi_24

        # RSI组合信号
        df['rsi_combination'] = (
            0.4 * rsi_6 +
            0.3 * rsi_12 +
            0.3 * rsi_24
        ) / 100

        # RSI背离检测（简化版）
        df['rsi_bullish_divergence'] = (
            (df['rsi_combination'] > 0.6) &
            (df['rsi_combination'] > df['rsi_combination'].shift(1))
        ).astype(int)

        # 3. MACD指标（使用复权价格）
        df['ema_12'] = df[price_col].ewm(span=12).mean()
        df['ema_26'] = df[price_col].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # MACD金叉死叉
        df['macd_golden_cross'] = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)

        df['macd_death_cross'] = (
            (df['macd'] < df['macd_signal']) &
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)

        # 4. KDJ指标（使用复权价格）
        low_9 = df[low_col].rolling(9).min()
        high_9 = df[high_col].rolling(9).max()
        rsv = (df[price_col] - low_9) / (high_9 - low_9 + 0.01) * 100
        df['k_value'] = rsv.ewm(com=2).mean()
        df['d_value'] = df['k_value'].ewm(com=2).mean()
        df['j_value'] = 3 * df['k_value'] - 2 * df['d_value']

        # KDJ金叉
        df['kdj_golden_cross'] = (
            (df['k_value'] > df['d_value']) &
            (df['k_value'].shift(1) <= df['d_value'].shift(1))
        ).astype(int)

        # 5. 波动率指标（使用复权价格）
        df['volatility_5'] = df[price_col].pct_change().rolling(5).std()
        df['volatility_20'] = df[price_col].pct_change().rolling(20).std()

        # ATR (Average True Range) 使用复权价格
        high_low = df[high_col] - df[low_col]
        high_close = np.abs(df[high_col] - df[price_col].shift())
        low_close = np.abs(df[low_col] - df[price_col].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean() / df[price_col]

        # 6. 成交量指标
        df['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
        df['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()

        # 量价关系（使用复权价格）
        price_change = df[price_col].pct_change()
        volume_change = df['volume'].pct_change()
        df['volume_price_correlation'] = price_change * volume_change

        # 7. 价格突破（使用复权价格）
        df['high_20'] = df[high_col].rolling(20).max()
        df['low_20'] = df[low_col].rolling(20).min()
        df['price_breakout_up'] = (
            df[price_col] > df['high_20'].shift(1)
        ).astype(int)

        df['price_breakdown'] = (
            df['close'] < df['low_20'].shift(1)
        ).astype(int)

        # 8. 动量指标
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = (
                df['close'] / df['close'].shift(period) - 1
            )

        # 9. 支撑阻力
        df['support_20'] = df['low'].rolling(20).min()
        df['resistance_20'] = df['high'].rolling(20).max()

        # 距离支撑阻力位的距离
        df['distance_to_support'] = (
            df['close'] - df['support_20']
        ) / (df['resistance_20'] - df['support_20'] + 0.01)
        df['distance_to_resistance'] = (
            df['resistance_20'] - df['close']
        ) / (df['resistance_20'] - df['support_20'] + 0.01)

        print(f"✓ 技术指标特征已创建: 45+个特征（权重30%）")
        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建所有增强特征

        Args:
            df: 原始数据，必须包含列: open, high, low, close, volume, stock_code

        Returns:
            包含所有特征的DataFrame
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'stock_code']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"数据缺少必要列: {col}")

        print("=" * 70)
        print("创建增强版特征工程")
        print("=" * 70)

        # 1. 主力资金特征（权重30%）
        df = self.create_main_capital_features(df)

        # 2. 北向资金特征（权重20%）
        df = self.create_northbound_features(df)

        # 3. 市场情绪特征（权重20%）
        df = self.create_market_sentiment_features(df)

        # 4. 技术指标特征（权重30%）
        df = self.create_technical_indicators_features(df)

        # 清理异常值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        # 收集特征名称
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume', 'stock_code', '股票代码',
            'future', 'return_', 'label', '日期'
        ]
        self.feature_names = [
            col for col in df.columns
            if col not in exclude_cols
        ]

        print(f"\n总计创建特征: {len(self.feature_names)}个")
        print(f"特征权重分布:")
        print(f"  - 主力资金（30%）：8个特征")
        print(f"  - 北向资金（20%）：5个特征")
        print(f"  - 市场情绪（20%）：8个特征")
        print(f"  - 技术指标（30%）：45+个特征")

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 0.001)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self.feature_names

    def get_feature_weights(self) -> Dict[str, float]:
        """获取特征权重"""
        return {
            'main_capital_flow': self.config['feature_weights']['main_capital_flow']['weight'],
            'northbound_capital': self.config['feature_weights']['northbound_capital']['weight'],
            'market_sentiment': self.config['feature_weights']['market_sentiment']['weight'],
            'technical_indicators': self.config['feature_weights']['technical_indicators']['weight']
        }
