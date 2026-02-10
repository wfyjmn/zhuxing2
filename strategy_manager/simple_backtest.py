"""
简化回测引擎 - 适配原有选股系统逻辑

特点：
1. 不考虑实际持仓和资金管理
2. 只关注选股后的涨跌表现
3. 简单记录买入后的N天收益
4. 适配选股A/B/C的输出格式
"""

import pandas as pd
import numpy as np
import tushare as ts
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from .config import Config


logger = logging.getLogger(__name__)


class SimpleBacktestEngine:
    """
    简化回测引擎
    
    功能：
    - 对选股结果进行简单回测
    - 记录买入后N天的收益
    - 计算胜率、平均收益等指标
    """

    def __init__(self, config: Config):
        """
        初始化回测引擎
        
        Args:
            config: 配置对象
        """
        self._config = config
        self._pro = None
        self._init_tushare()

    def _init_tushare(self):
        """初始化Tushare连接"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = self._config.tushare_token or os.getenv("TUSHARE_TOKEN")
            if token:
                ts.set_token(token)
                self._pro = ts.pro_api(timeout=30)
                logger.info("Tushare连接成功")
            else:
                logger.warning("未配置Tushare Token，回测功能受限")
        except Exception as e:
            logger.error(f"Tushare初始化失败: {e}")

    def backtest_selection(
        self,
        selected_df: pd.DataFrame,
        buy_date: str,
        hold_days: int = 5,
        price_col: str = "close"
    ) -> pd.DataFrame:
        """
        回测选股结果
        
        Args:
            selected_df: 选股结果DataFrame
            buy_date: 买入日期 (YYYYMMDD)
            hold_days: 持有天数
            price_col: 价格列名
            
        Returns:
            回测结果DataFrame，包含:
            - ts_code: 股票代码
            - buy_date: 买入日期
            - buy_price: 买入价
            - sell_date: 卖出日期
            - sell_price: 卖出价
            - return_pct: 收益率(%)
            - holding_days: 实际持有天数
        """
        if selected_df.empty:
            logger.warning("选股结果为空")
            return pd.DataFrame()

        if self._pro is None:
            logger.error("Tushare未初始化，无法回测")
            return pd.DataFrame()

        logger.info(f"开始回测: {len(selected_df)} 只股票, 买入日={buy_date}, 持有={hold_days}天")

        results = []

        for idx, row in selected_df.iterrows():
            ts_code = row.get("ts_code")
            if not ts_code:
                continue

            try:
                # 获取股票价格数据
                price_df = self._get_stock_prices(
                    ts_code=ts_code,
                    start_date=buy_date,
                    end_days=hold_days + 5  # 多获取几天，防止停牌
                )

                if price_df is None or len(price_df) == 0:
                    logger.debug(f"{ts_code} 无法获取价格数据")
                    continue

                # 买入价（第1天收盘价）
                buy_price = price_df.iloc[0]["close"]

                # 计算卖出价（持有N天后）
                actual_hold = min(hold_days, len(price_df) - 1)
                if actual_hold > 0:
                    sell_price = price_df.iloc[actual_hold]["close"]
                    sell_date = price_df.iloc[actual_hold]["trade_date"]
                else:
                    # 只有一天的数据，用开盘价作为卖出价
                    sell_price = buy_price
                    sell_date = buy_date
                    actual_hold = 0

                # 计算收益率
                return_pct = (sell_price / buy_price - 1) * 100 if buy_price > 0 else 0

                results.append({
                    "ts_code": ts_code,
                    "name": row.get("name", ""),
                    "buy_date": buy_date,
                    "buy_price": round(buy_price, 2),
                    "sell_date": sell_date,
                    "sell_price": round(sell_price, 2),
                    "return_pct": round(return_pct, 2),
                    "holding_days": actual_hold,
                })

            except Exception as e:
                logger.debug(f"{ts_code} 回测失败: {e}")
                continue

        result_df = pd.DataFrame(results)
        logger.info(f"回测完成: {len(result_df)}/{len(selected_df)} 只股票成功回测")

        return result_df

    def _get_stock_prices(
        self,
        ts_code: str,
        start_date: str,
        end_days: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        获取股票价格数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_days: 获取天数
            
        Returns:
            价格DataFrame
        """
        try:
            time.sleep(0.3)  # 避免限流

            end_date_dt = datetime.strptime(start_date, "%Y%m%d") + timedelta(days=end_days * 2)
            end_date = end_date_dt.strftime("%Y%m%d")

            df = self._pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df is None or len(df) == 0:
                return None

            df = df.sort_values("trade_date").reset_index(drop=True)
            return df

        except Exception as e:
            logger.debug(f"获取{ts_code}价格失败: {e}")
            return None

    def calculate_stats(
        self,
        backtest_df: pd.DataFrame
    ) -> Dict:
        """
        计算回测统计指标
        
        Args:
            backtest_df: 回测结果DataFrame
            
        Returns:
            统计指标字典
        """
        if backtest_df.empty:
            return {
                "count": 0,
                "win_rate": 0,
                "avg_return": 0,
                "median_return": 0,
                "best_return": 0,
                "worst_return": 0,
                "positive_count": 0,
                "negative_count": 0,
            }

        returns = backtest_df["return_pct"]
        total = len(returns)
        positive = (returns > 0).sum()
        negative = (returns < 0).sum()

        stats = {
            "count": total,
            "win_rate": round(positive / total * 100, 2) if total > 0 else 0,
            "avg_return": round(returns.mean(), 2),
            "median_return": round(returns.median(), 2),
            "best_return": round(returns.max(), 2),
            "worst_return": round(returns.min(), 2),
            "positive_count": int(positive),
            "negative_count": int(negative),
        }

        return stats

    def generate_report(
        self,
        backtest_df: pd.DataFrame,
        stats: Dict,
        strategy_name: str = "未命名"
    ) -> str:
        """
        生成回测报告
        
        Args:
            backtest_df: 回测结果
            stats: 统计指标
            strategy_name: 策略名称
            
        Returns:
            报告文本
        """
        lines = [
            "=" * 60,
            f"回测报告: {strategy_name}",
            "=" * 60,
            "",
            "📊 统计摘要:",
            f"  回测股票数: {stats['count']} 只",
            f"  胜率: {stats['win_rate']}%",
            f"  平均收益: {stats['avg_return']}%",
            f"  中位数收益: {stats['median_return']}%",
            f"  最佳收益: {stats['best_return']}%",
            f"  最差收益: {stats['worst_return']}%",
            f"  盈利笔数: {stats['positive_count']}",
            f"  亏损笔数: {stats['negative_count']}",
            "",
            "📋 明细 (Top 10):",
        ]

        if not backtest_df.empty:
            top10 = backtest_df.nlargest(10, "return_pct")
            lines.append(f"{'股票代码':<12} {'股票名称':<10} {'买入价':<8} {'卖出价':<8} {'收益率':<10}")
            lines.append("-" * 60)
            for _, row in top10.iterrows():
                lines.append(
                    f"{row['ts_code']:<12} {row['name']:<10} "
                    f"{row['buy_price']:<8.2f} {row['sell_price']:<8.2f} "
                    f"{row['return_pct']:>8.2f}%"
                )

        lines.append("=" * 60)

        return "\n".join(lines)
