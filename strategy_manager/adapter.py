"""
选股程序适配器 - 整合选股A/B/C到策略管理系统

功能：
1. 封装原有的选股A/B/C程序
2. 统一输出格式
3. 支持市场状态判断
4. 支持次日实盘数据对比
"""

import pandas as pd
import numpy as np
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .config import Config
from .simple_backtest import SimpleBacktestEngine

logger = logging.getLogger(__name__)


class MarketStateDetector:
    """
    市场状态检测器
    
    使用20日均线判断市场状态（牛市/震荡市/熊市）
    """

    def __init__(self, config: Config):
        """
        Args:
            config: 配置对象
        """
        self._config = config
        self._pro = None
        self._init_tushare()

    def _init_tushare(self):
        """初始化Tushare"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = self._config.tushare_token or os.getenv("TUSHARE_TOKEN")
            if token:
                import tushare as ts
                ts.set_token(token)
                self._pro = ts.pro_api(timeout=30)
        except Exception as e:
            logger.error(f"Tushare初始化失败: {e}")

    def detect_market_state(self, index_code: str = "000001.SH") -> Dict:
        """
        检测市场状态
        
        Args:
            index_code: 指数代码（默认上证指数）
            
        Returns:
            市场状态字典:
            - state: bull/bear/neutral
            - ma20: 20日均线
            - current_price: 当前价格
            - deviation_pct: 偏离度(%)
            - description: 状态描述
        """
        if self._pro is None:
            return {
                "state": "neutral",
                "ma20": 0,
                "current_price": 0,
                "deviation_pct": 0,
                "description": "数据不可用"
            }

        try:
            time.sleep(0.5)

            # 获取最近60天的指数数据
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")

            df = self._pro.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date
            )

            if df is None or len(df) < 20:
                return {"state": "neutral", "description": "数据不足"}

            df = df.sort_values("trade_date").tail(60).reset_index(drop=True)

            # 计算20日均线
            latest = df.iloc[-1]
            ma20 = df["close"].rolling(20).mean().iloc[-1]

            # 计算偏离度
            deviation_pct = (latest["close"] - ma20) / ma20 * 100

            # 判断市场状态
            if deviation_pct > 3.0:
                state = "bull"
                description = f"牛市（指数偏离均线+{deviation_pct:.2f}%）"
            elif deviation_pct < -3.0:
                state = "bear"
                description = f"熊市（指数偏离均线{deviation_pct:.2f}%）"
            else:
                state = "neutral"
                description = f"震荡市（指数偏离均线{deviation_pct:.2f}%）"

            return {
                "state": state,
                "ma20": round(ma20, 2),
                "current_price": round(latest["close"], 2),
                "deviation_pct": round(deviation_pct, 2),
                "description": description
            }

        except Exception as e:
            logger.error(f"市场状态检测失败: {e}")
            return {"state": "neutral", "description": "检测失败"}

    def recommend_strategy(self, market_state: str) -> str:
        """
        根据市场状态推荐策略
        
        Args:
            market_state: bull/bear/neutral
            
        Returns:
            推荐的策略类型
        """
        if market_state == "bull":
            return "momentum"  # 牛市用动量策略
        elif market_state == "bear":
            return "value"  # 熊市用价值策略
        else:
            return "value"  # 震荡市用价值策略


class ScreenerAdapter:
    """
    选股程序适配器
    
    整合选股A/B/C，统一输出格式
    """

    def __init__(self, config: Config):
        """
        Args:
            config: 配置对象
        """
        self._config = config
        self._market_detector = MarketStateDetector(config)
        self._backtest_engine = SimpleBacktestEngine(config)

    def run_screener_a(
        self,
        data: pd.DataFrame,
        market_state: Optional[str] = None
    ) -> pd.DataFrame:
        """
        运行选股A逻辑
        
        选股A思路：市场状态判断 → 策略选择 → 基础筛选 → 量价筛选
        
        Args:
            data: 股票数据
            market_state: 市场状态（自动检测如果为None）
            
        Returns:
            选股结果
        """
        if data.empty:
            return pd.DataFrame()

        logger.info("运行选股A逻辑...")

        # 1. 市场状态判断
        if market_state is None:
            market_info = self._market_detector.detect_market_state()
            market_state = market_info["state"]
            logger.info(f"市场状态: {market_info['description']}")

        # 2. 根据市场状态选择筛选参数
        if market_state == "bull":
            min_pct_chg = 3.0  # 牛市降低涨幅要求
        elif market_state == "bear":
            min_pct_chg = 5.0  # 熊市提高涨幅要求
        else:
            min_pct_chg = 4.0  # 震荡市

        # 3. 基础筛选
        filtered = self._basic_filter(data, min_pct_chg=min_pct_chg)

        # 4. 量价筛选
        filtered = self._price_volume_filter(filtered)

        logger.info(f"选股A: {len(data)} → {len(filtered)} 只")

        return filtered

    def run_screener_b(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        运行选股B逻辑
        
        选股B思路：风险过滤（跌停、解禁、独食）+ 基础筛选
        
        Args:
            data: 股票数据
            
        Returns:
            选股结果
        """
        if data.empty:
            return pd.DataFrame()

        logger.info("运行选股B逻辑...")

        # 1. 基础筛选
        filtered = self._basic_filter(data, min_pct_chg=5.0)

        # 2. 风险过滤
        # TODO: 获取跌停历史、解禁数据、龙虎榜数据
        # 这里简化处理

        logger.info(f"选股B: {len(data)} → {len(filtered)} 只")

        return filtered

    def run_screener_c(
        self,
        data: pd.DataFrame,
        enable_industry: bool = True
    ) -> pd.DataFrame:
        """
        运行选股C逻辑
        
        选股C思路：选股A的市场感知 + 选股B的风险过滤 + 行业板块分类
        
        Args:
            data: 股票数据
            enable_industry: 是否启用行业分类
            
        Returns:
            选股结果
        """
        if data.empty:
            return pd.DataFrame()

        logger.info("运行选股C逻辑...")

        # 1. 市场状态判断
        market_info = self._market_detector.detect_market_state()
        market_state = market_info["state"]

        # 2. 基础筛选（根据市场状态调整）
        if market_state == "bull":
            min_pct_chg = 3.0
        elif market_state == "bear":
            min_pct_chg = 5.0
        else:
            min_pct_chg = 4.0

        filtered = self._basic_filter(data, min_pct_chg=min_pct_chg)

        # 3. 量价筛选
        filtered = self._price_volume_filter(filtered)

        # 4. 行业板块分类
        if enable_industry and "industry" in filtered.columns:
            industry_counts = filtered["industry"].value_counts()
            logger.info(f"行业分布:\n{industry_counts.head(10)}")

        logger.info(f"选股C: {len(data)} → {len(filtered)} 只")

        return filtered

    def _basic_filter(
        self,
        data: pd.DataFrame,
        min_pct_chg: float = 5.0,
        min_turnover: float = 3.0,
        max_turnover: float = 20.0,
        min_price: float = 3.0,
        max_price: float = 50.0
    ) -> pd.DataFrame:
        """
        基础筛选
        
        Args:
            data: 股票数据
            min_pct_chg: 最低涨幅
            min_turnover: 最低换手率
            max_turnover: 最高换手率
            min_price: 最低价格
            max_price: 最高价格
            
        Returns:
            筛选后的数据
        """
        if data.empty:
            return data

        filtered = data.copy()

        # 涨幅筛选
        if "pct_chg" in filtered.columns:
            filtered = filtered[filtered["pct_chg"] >= min_pct_chg]

        # 换手率筛选
        if "turnover_rate" in filtered.columns:
            filtered = filtered[
                (filtered["turnover_rate"] >= min_turnover) &
                (filtered["turnover_rate"] <= max_turnover)
            ]

        # 价格筛选
        if "close" in filtered.columns:
            filtered = filtered[
                (filtered["close"] >= min_price) &
                (filtered["close"] <= max_price)
            ]

        # ST股过滤
        if "name" in filtered.columns:
            filtered = filtered[~filtered["name"].str.contains("ST|退", na=False)]

        return filtered

    def _price_volume_filter(
        self,
        data: pd.DataFrame,
        min_volume_ratio: float = 1.5
    ) -> pd.DataFrame:
        """
        量价筛选
        
        Args:
            data: 股票数据
            min_volume_ratio: 最低量比
            
        Returns:
            筛选后的数据
        """
        if data.empty:
            return data

        filtered = data.copy()

        # 量比筛选
        if "volume_ratio" in filtered.columns:
            filtered = filtered[filtered["volume_ratio"] >= min_volume_ratio]

        return filtered

    def backtest_and_compare(
        self,
        selected_df: pd.DataFrame,
        buy_date: str,
        hold_days: int = 5
    ) -> Dict:
        """
        回测并对比次日实盘数据
        
        Args:
            selected_df: 选股结果
            buy_date: 买入日期
            hold_days: 持有天数
            
        Returns:
            回测结果对比字典
        """
        if selected_df.empty:
            return {"error": "选股结果为空"}

        logger.info(f"开始回测对比: 买入日={buy_date}, 持有={hold_days}天")

        # 1. 回测
        backtest_df = self._backtest_engine.backtest_selection(
            selected_df=selected_df,
            buy_date=buy_date,
            hold_days=hold_days
        )

        if backtest_df.empty:
            return {"error": "回测失败"}

        # 2. 计算统计指标
        stats = self._backtest_engine.calculate_stats(backtest_df)

        # 3. 生成报告
        report = self._backtest_engine.generate_report(
            backtest_df=backtest_df,
            stats=stats,
            strategy_name="选股回测"
        )

        return {
            "backtest_df": backtest_df,
            "stats": stats,
            "report": report,
            "buy_date": buy_date,
            "hold_days": hold_days
        }

    def detect_and_correct_errors(
        self,
        backtest_df: pd.DataFrame,
        actual_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        检测并修正错误
        
        检查点：
        1. 胜率是否过低
        2. 平均收益是否为负
        3. 是否存在系统性偏差
        
        Args:
            backtest_df: 回测结果
            actual_df: 实际数据（如果有）
            
        Returns:
            错误检测结果和建议
        """
        if backtest_df.empty:
            return {"errors": [], "suggestions": []}

        stats = self._backtest_engine.calculate_stats(backtest_df)

        errors = []
        suggestions = []

        # 检测胜率
        if stats["win_rate"] < 40:
            errors.append(f"胜率过低: {stats['win_rate']}%")
            suggestions.append("建议提高选股门槛，如增加涨幅要求")

        # 检测平均收益
        if stats["avg_return"] < 0:
            errors.append(f"平均收益为负: {stats['avg_return']}%")
            suggestions.append("建议调整选股逻辑，避免追高")

        # 检测最差收益
        if stats["worst_return"] < -10:
            errors.append(f"存在大幅亏损: {stats['worst_return']}%")
            suggestions.append("建议增加止损条件，如跌破5日均线")

        return {
            "errors": errors,
            "suggestions": suggestions,
            "stats": stats
        }
