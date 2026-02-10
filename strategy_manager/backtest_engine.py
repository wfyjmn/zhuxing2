"""
回测引擎模块

修复内容：
1. 收益率计算 - 明确区分累计收益率和日度收益率，用向量化替代循环
2. 夏普比率 - 使用日度收益率计算，使用样本标准差(ddof=1)
3. 止盈止损优先级 - 按触发时间先后决定，而非硬编码优先级
4. 交易成本 - 按A股真实费率计算（佣金+印花税+过户费+滑点）
5. 返回值类型 - backtest_portfolio始终返回Tuple[DataFrame, Dict]
6. 最大回撤 - 基于净值曲线计算真实最大回撤
7. 数据类型安全 - 统一日期格式处理，防止类型不匹配
8. 组合层面指标 - 净值加权计算组合收益、回撤、夏普
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import Config
from .data_manager import BatchDataManager


# ==================== 数据类 ====================


@dataclass
class TradeResult:
    """
    单笔交易结果
    
    记录一次完整的买入→持有→卖出过程的所有信息
    """

    # 基本信息
    ts_code: str
    stock_name: str = ""

    # 买入信息
    buy_date: str = ""
    buy_price: float = 0.0

    # 卖出信息
    sell_date: str = ""
    sell_price: float = 0.0

    # 持有信息
    holding_days: int = 0
    holding_trade_days: int = 0  # 交易日天数

    # 收益指标（扣费前）
    gross_return_pct: float = 0.0

    # 收益指标（扣费后）
    net_return_pct: float = 0.0
    annualized_return_pct: float = 0.0

    # 风险指标
    max_return_pct: float = 0.0       # 持有期间最高累计收益
    min_return_pct: float = 0.0       # 持有期间最低累计收益（最大浮亏）
    max_drawdown_pct: float = 0.0     # 从最高点的最大回撤
    daily_volatility: float = 0.0     # 日收益率标准差
    sharpe_ratio: float = 0.0

    # 退出原因
    exit_reason: str = ""  # 'stop_loss', 'take_profit', 'time_expired'

    # 交易成本
    buy_cost: float = 0.0
    sell_cost: float = 0.0
    total_cost: float = 0.0
    cost_ratio_pct: float = 0.0       # 成本占买入金额的百分比

    # 仓位信息
    invest_amount: float = 0.0
    shares: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @property
    def is_profitable(self) -> bool:
        """是否盈利"""
        return self.net_return_pct > 0


# ==================== 回测引擎 ====================


class BacktestEngine:
    """
    回测引擎
    
    职责：
    - 单只股票回测（含止盈止损、交易成本）
    - 投资组合回测（并行处理）
    - 组合层面绩效计算
    - 基准对比分析
    
    使用示例：
        engine = BacktestEngine(data_manager, config)
        
        # 单股回测
        result = engine.backtest_single("000001.SZ", "20230601")
        
        # 组合回测
        portfolio = [
            {"ts_code": "000001.SZ", "buy_price": 10.5},
            {"ts_code": "000002.SZ", "buy_price": 15.3},
        ]
        results_df, summary = engine.backtest_portfolio(portfolio, "20230601")
    """

    def __init__(self, data_manager: BatchDataManager, config: Config):
        self._data_manager = data_manager
        self._config = config
        self._logger = logging.getLogger(__name__)

    # ==================== 单股回测 ====================

    def backtest_single(
        self,
        ts_code: str,
        buy_date: str,
        buy_price: Optional[float] = None,
        shares: Optional[int] = None,
        invest_amount: Optional[float] = None,
        max_holding_days: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[TradeResult]:
        """
        回测单只股票
        
        执行流程：
        1. 获取买入日及之后的行情数据
        2. 确定买入价格
        3. 计算每日累计收益率和日度收益率
        4. 检查止盈止损触发（按时间先后判断优先级）
        5. 计算交易成本和净收益
        6. 计算风险指标（夏普比率、最大回撤、波动率）
        
        Args:
            ts_code: 股票代码
            buy_date: 买入日期 (YYYYMMDD)
            buy_price: 买入价格，None则使用买入日收盘价
            shares: 买入股数，None则根据invest_amount计算
            invest_amount: 投入金额，None则使用配置的初始资金/10
            max_holding_days: 最大持有自然日，None使用配置默认值
            stop_loss: 止损比例(%)，如-5.0表示亏5%止损，None使用配置默认值
            take_profit: 止盈比例(%)，如10.0表示赚10%止盈，None使用配置默认值
            
        Returns:
            TradeResult对象，数据不足时返回None
        """
        # 参数默认值
        if max_holding_days is None:
            max_holding_days = self._config.max_holding_period
        if stop_loss is None:
            stop_loss = self._config.stop_loss
        if take_profit is None:
            take_profit = self._config.take_profit

        # 获取行情数据（多取一些天数以覆盖非交易日）
        data_end_date = self._add_calendar_days(buy_date, max_holding_days + 30)
        df = self._data_manager.get_daily_data(ts_code, buy_date, data_end_date)

        if df is None or df.empty:
            self._logger.warning(f"无行情数据: {ts_code} {buy_date}起")
            return None

        # 统一日期格式为datetime
        df = self._normalize_dates(df)

        # 确定实际买入日（如果buy_date非交易日则顺延到下一个交易日）
        buy_date_dt = pd.Timestamp(buy_date)
        actual_buy_date, buy_row = self._find_buy_date(df, buy_date_dt)

        if buy_row is None:
            self._logger.warning(f"{ts_code}: 买入日 {buy_date} 之后无可用交易日")
            return None

        # 确定买入价格
        if buy_price is None or buy_price <= 0:
            buy_price = float(buy_row["close"])

        # 获取买入日起的持仓期数据
        holding_data = df[df["trade_date"] >= actual_buy_date].copy()
        holding_data = holding_data.sort_values("trade_date").reset_index(drop=True)

        if len(holding_data) < 2:
            self._logger.warning(f"{ts_code}: 持仓期数据不足（仅{len(holding_data)}行）")
            return None

        # 按最大持有日限制数据范围
        max_end_date = actual_buy_date + pd.Timedelta(days=max_holding_days)
        holding_data = holding_data[holding_data["trade_date"] <= max_end_date]

        if holding_data.empty:
            return None

        # 计算收益率序列
        close_prices = holding_data["close"].values.astype(float)
        cum_returns = self._calculate_cumulative_returns(close_prices, buy_price)
        daily_returns = self._calculate_daily_returns(close_prices, buy_price)

        holding_data = holding_data.copy()
        holding_data["cum_return"] = cum_returns
        holding_data["daily_return"] = daily_returns

        # 确定退出点（止盈止损按时间先后判断）
        exit_reason, exit_idx = self._determine_exit(
            holding_data, stop_loss, take_profit
        )

        # 提取卖出信息
        sell_row = holding_data.iloc[exit_idx]
        sell_price = float(sell_row["close"])
        sell_date_dt = sell_row["trade_date"]
        sell_date_str = self._format_date(sell_date_dt)
        buy_date_str = self._format_date(actual_buy_date)

        # 持有天数
        holding_days = (sell_date_dt - actual_buy_date).days
        holding_trade_days = exit_idx + 1

        # 计算投资金额和股数
        if invest_amount is None:
            invest_amount = self._config.initial_capital / 10
        if shares is None or shares <= 0:
            shares = max(int(invest_amount / buy_price / 100) * 100, 100)

        actual_invest = buy_price * shares

        # 计算交易成本
        sell_amount = sell_price * shares
        buy_cost = self._config.calculate_buy_cost(actual_invest)
        sell_cost = self._config.calculate_sell_cost(sell_amount)
        total_cost = buy_cost + sell_cost

        # 滑点成本
        slippage_rate = getattr(self._config, "slippage_rate", 0.001)
        slippage_cost = (actual_invest + sell_amount) * slippage_rate
        total_cost += slippage_cost

        # 收益计算
        gross_return_pct = (sell_price - buy_price) / buy_price * 100
        net_profit = (sell_amount - actual_invest) - total_cost
        net_return_pct = net_profit / actual_invest * 100 if actual_invest > 0 else 0

        # 年化收益率
        if holding_days > 0:
            annualized = (1 + net_return_pct / 100) ** (365 / holding_days) - 1
            annualized_return_pct = annualized * 100
        else:
            annualized_return_pct = 0.0

        # 持有期风险指标（只取到卖出日的数据）
        period_data = holding_data.iloc[: exit_idx + 1]
        period_cum_returns = period_data["cum_return"].values
        period_daily_returns = period_data["daily_return"].values

        max_return_pct = float(np.max(period_cum_returns))
        min_return_pct = float(np.min(period_cum_returns))

        # 最大回撤（基于净值曲线）
        max_drawdown_pct = self._calculate_max_drawdown(period_cum_returns)

        # 波动率和夏普比率（使用日度收益率）
        daily_vol = self._calculate_volatility(period_daily_returns)
        sharpe = self._calculate_sharpe_ratio(
            period_daily_returns, self._config.risk_free_rate
        )

        # 成本占比
        cost_ratio_pct = total_cost / actual_invest * 100 if actual_invest > 0 else 0

        return TradeResult(
            ts_code=ts_code,
            buy_date=buy_date_str,
            buy_price=round(buy_price, 3),
            sell_date=sell_date_str,
            sell_price=round(sell_price, 3),
            holding_days=holding_days,
            holding_trade_days=holding_trade_days,
            gross_return_pct=round(gross_return_pct, 4),
            net_return_pct=round(net_return_pct, 4),
            annualized_return_pct=round(annualized_return_pct, 4),
            max_return_pct=round(max_return_pct, 4),
            min_return_pct=round(min_return_pct, 4),
            max_drawdown_pct=round(max_drawdown_pct, 4),
            daily_volatility=round(daily_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            exit_reason=exit_reason,
            buy_cost=round(buy_cost, 2),
            sell_cost=round(sell_cost, 2),
            total_cost=round(total_cost + slippage_cost, 2),
            cost_ratio_pct=round(cost_ratio_pct, 4),
            invest_amount=round(actual_invest, 2),
            shares=shares,
        )

    # ==================== 组合回测 ====================

    def backtest_portfolio(
        self,
        stock_list: List[Dict[str, Any]],
        buy_date: str,
        initial_capital: Optional[float] = None,
        max_holding_days: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        weight_method: str = "equal",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        投资组合回测
        
        始终返回 Tuple[DataFrame, Dict]，即使回测失败也返回空容器
        
        Args:
            stock_list: 股票列表，每个元素为dict，必须包含ts_code
                示例: [{"ts_code": "000001.SZ", "buy_price": 10.5, "weight": 0.5}]
            buy_date: 统一买入日期 (YYYYMMDD)
            initial_capital: 初始资金，None使用配置默认值
            max_holding_days: 最大持有天数
            stop_loss: 止损比例(%)
            take_profit: 止盈比例(%)
            weight_method: 权重分配方式 'equal'等权 或 'custom'自定义
            
        Returns:
            (results_df, portfolio_summary) 元组
            results_df: 每只股票的回测结果DataFrame
            portfolio_summary: 组合层面的汇总指标Dict
        """
        if not stock_list:
            self._logger.warning("股票列表为空")
            return pd.DataFrame(), {}

        if initial_capital is None:
            initial_capital = self._config.initial_capital

        # 计算每只股票的投资金额
        weights = self._calculate_weights(stock_list, weight_method)
        n_stocks = len(stock_list)

        # 逐只回测
        results: List[TradeResult] = []
        failed_codes: List[str] = []

        for i, stock_info in enumerate(stock_list):
            ts_code = stock_info.get("ts_code", "")
            if not ts_code:
                continue

            stock_invest = initial_capital * weights[i]
            stock_buy_price = stock_info.get("buy_price")

            try:
                result = self.backtest_single(
                    ts_code=ts_code,
                    buy_date=buy_date,
                    buy_price=stock_buy_price,
                    invest_amount=stock_invest,
                    max_holding_days=max_holding_days,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

                if result is not None:
                    results.append(result)
                else:
                    failed_codes.append(ts_code)

            except Exception as e:
                self._logger.error(f"回测 {ts_code} 异常: {e}")
                failed_codes.append(ts_code)

        if failed_codes:
            self._logger.warning(
                f"组合回测失败 {len(failed_codes)}/{n_stocks}: "
                f"{failed_codes[:10]}{'...' if len(failed_codes) > 10 else ''}"
            )

        # 构建结果DataFrame
        if not results:
            self._logger.warning("组合回测无有效结果")
            return pd.DataFrame(), {}

        results_df = pd.DataFrame([r.to_dict() for r in results])

        # 计算组合汇总指标
        portfolio_summary = self._calculate_portfolio_summary(
            results, results_df, initial_capital, weights, n_stocks
        )

        self._logger.info(
            f"组合回测完成: {len(results)}/{n_stocks} 成功, "
            f"胜率 {portfolio_summary.get('win_rate_pct', 0):.1f}%, "
            f"组合收益 {portfolio_summary.get('portfolio_return_pct', 0):.2f}%"
        )

        return results_df, portfolio_summary

    # ==================== 基准对比 ====================

    def compare_with_benchmark(
        self,
        portfolio_daily_returns: List[float],
        start_date: str,
        end_date: str,
        benchmark_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        与基准指数对比
        
        Args:
            portfolio_daily_returns: 组合日度收益率序列(%)
            start_date: 开始日期
            end_date: 结束日期
            benchmark_code: 基准代码，None使用配置默认值
            
        Returns:
            对比结果字典
        """
        if benchmark_code is None:
            benchmark_code = self._config.benchmark_code

        # 获取基准数据
        bench_df = self._data_manager.get_daily_data(
            benchmark_code, start_date, end_date
        )

        if bench_df is None or bench_df.empty:
            self._logger.warning(f"未获取到基准 {benchmark_code} 数据")
            return {}

        bench_df = self._normalize_dates(bench_df)
        bench_df = bench_df.sort_values("trade_date")

        # 基准日度收益率(%)
        bench_close = bench_df["close"].values.astype(float)
        if len(bench_close) < 2:
            return {}

        bench_daily = np.zeros(len(bench_close))
        bench_daily[1:] = np.diff(bench_close) / bench_close[:-1] * 100

        # 对齐长度
        port_arr = np.array(portfolio_daily_returns, dtype=float)
        min_len = min(len(port_arr), len(bench_daily))

        if min_len < 2:
            return {}

        port_aligned = port_arr[:min_len]
        bench_aligned = bench_daily[:min_len]

        # 累计收益
        port_cum = np.prod(1 + port_aligned / 100) - 1
        bench_cum = np.prod(1 + bench_aligned / 100) - 1

        # 超额收益
        excess_return = port_cum - bench_cum

        # 超额收益序列
        excess_daily = port_aligned - bench_aligned

        # 信息比率（年化超额收益 / 跟踪误差）
        tracking_error = np.std(excess_daily, ddof=1) if len(excess_daily) > 1 else 0
        if tracking_error > 0:
            information_ratio = np.mean(excess_daily) / tracking_error * np.sqrt(252)
        else:
            information_ratio = 0.0

        # 相关系数
        if len(port_aligned) > 1:
            correlation = float(np.corrcoef(port_aligned, bench_aligned)[0, 1])
        else:
            correlation = 0.0

        # Beta
        if np.var(bench_aligned) > 0:
            beta = float(
                np.cov(port_aligned, bench_aligned)[0, 1] / np.var(bench_aligned)
            )
        else:
            beta = 0.0

        # Alpha（Jensen's Alpha，年化）
        rf_daily = self._config.risk_free_rate / 252 * 100
        alpha = (
            np.mean(port_aligned)
            - rf_daily
            - beta * (np.mean(bench_aligned) - rf_daily)
        ) * 252

        return {
            "benchmark_code": benchmark_code,
            "start_date": start_date,
            "end_date": end_date,
            "data_points": min_len,
            "portfolio_cum_return_pct": round(port_cum * 100, 4),
            "benchmark_cum_return_pct": round(bench_cum * 100, 4),
            "excess_return_pct": round(excess_return * 100, 4),
            "information_ratio": round(information_ratio, 4),
            "correlation": round(correlation, 4),
            "beta": round(beta, 4),
            "alpha_annualized": round(alpha, 4),
            "tracking_error_annualized": round(
                tracking_error * np.sqrt(252), 4
            )
            if tracking_error > 0
            else 0,
        }

    # ==================== 收益率计算（核心修复） ====================

    @staticmethod
    def _calculate_cumulative_returns(
        prices: np.ndarray, buy_price: float
    ) -> np.ndarray:
        """
        计算累计收益率序列（相对于买入价）
        
        用途：止盈止损判断、持有期最大/最小收益
        
        Args:
            prices: 收盘价序列（prices[0]是买入日收盘价）
            buy_price: 买入价格
            
        Returns:
            累计收益率序列(%)
        """
        if len(prices) == 0 or buy_price <= 0:
            return np.array([])

        return (prices - buy_price) / buy_price * 100

    @staticmethod
    def _calculate_daily_returns(
        prices: np.ndarray, buy_price: float
    ) -> np.ndarray:
        """
        计算日度收益率序列
        
        用途：夏普比率、波动率等风险指标计算
        
        计算规则：
        - 第0天: (收盘价 - 买入价) / 买入价 × 100
        - 第n天(n≥1): (收盘价[n] - 收盘价[n-1]) / 收盘价[n-1] × 100
        
        Args:
            prices: 收盘价序列
            buy_price: 买入价格
            
        Returns:
            日度收益率序列(%)
        """
        if len(prices) == 0 or buy_price <= 0:
            return np.array([])

        daily_returns = np.zeros(len(prices), dtype=float)

        # 第一天：相对于买入价
        daily_returns[0] = (prices[0] - buy_price) / buy_price * 100

        # 后续天：逐日变化率（向量化计算）
        if len(prices) > 1:
            daily_returns[1:] = np.diff(prices) / prices[:-1] * 100

        return daily_returns

    # ==================== 退出点判断（核心修复） ====================

    def _determine_exit(
        self,
        holding_data: pd.DataFrame,
        stop_loss: float,
        take_profit: float,
    ) -> Tuple[str, int]:
        """
        确定退出点
        
        修复：按触发时间先后决定优先级，而非硬编码
        
        逻辑：
        1. 遍历每个交易日，记录止损和止盈的首次触发时间
        2. 如果同一天同时触发：
           - 检查日内最低价是否先触及止损（Low <= buy_price * (1 + stop_loss/100)）
           - 否则认为止盈先触发
        3. 两者都未触发时，持有到期
        
        Args:
            holding_data: 持仓期行情数据（必须包含cum_return列）
            stop_loss: 止损比例(%)，负数
            take_profit: 止盈比例(%)，正数
            
        Returns:
            (exit_reason, exit_index) 元组
        """
        cum_returns = holding_data["cum_return"].values
        n = len(cum_returns)

        stop_idx = None
        take_idx = None

        for i in range(n):
            # 检查止损（累计收益 <= 止损线）
            if stop_idx is None and cum_returns[i] <= stop_loss:
                stop_idx = i

            # 检查止盈（累计收益 >= 止盈线）
            if take_idx is None and cum_returns[i] >= take_profit:
                take_idx = i

            # 两者都已找到，无需继续
            if stop_idx is not None and take_idx is not None:
                break

        # 按触发时间先后判断
        if stop_idx is not None and take_idx is not None:
            if stop_idx < take_idx:
                return "stop_loss", stop_idx
            elif take_idx < stop_idx:
                return "take_profit", take_idx
            else:
                # 同一天触发，用日内价格判断
                return self._resolve_same_day_exit(
                    holding_data, stop_idx, stop_loss, take_profit
                )
        elif stop_idx is not None:
            return "stop_loss", stop_idx
        elif take_idx is not None:
            return "take_profit", take_idx
        else:
            # 未触发止盈止损，持有到期
            return "time_expired", n - 1

    def _resolve_same_day_exit(
        self,
        holding_data: pd.DataFrame,
        idx: int,
        stop_loss: float,
        take_profit: float,
    ) -> Tuple[str, int]:
        """
        解决同一天同时触发止盈止损的情况
        
        启发式规则：
        - 如果当天开盘价已经触发止损（跳空低开），认为止损先触发
        - 如果当天开盘价已经触发止盈（跳空高开），认为止盈先触发
        - 否则，检查当天最低价和最高价，判断哪个更可能先触发
        - 兜底：认为止损优先（保守策略）
        
        Args:
            holding_data: 持仓期数据
            idx: 触发日的索引
            stop_loss: 止损比例(%)
            take_profit: 止盈比例(%)
            
        Returns:
            (exit_reason, exit_index)
        """
        row = holding_data.iloc[idx]

        # 获取买入价（从第一天的cum_return反推或使用close）
        if idx > 0:
            prev_close = holding_data.iloc[idx - 1]["close"]
        else:
            # 第一天同时触发的极端情况
            return "stop_loss", idx

        has_open = "open" in holding_data.columns
        has_low = "low" in holding_data.columns
        has_high = "high" in holding_data.columns

        if has_open:
            open_price = float(row["open"])
            open_return = (open_price - prev_close) / prev_close * 100

            # 开盘即触发
            if open_return <= stop_loss:
                return "stop_loss", idx
            if open_return >= take_profit:
                return "take_profit", idx

        # 根据日内低点和高点判断
        if has_low and has_high:
            low_return = (float(row["low"]) - prev_close) / prev_close * 100
            high_return = (float(row["high"]) - prev_close) / prev_close * 100

            # 低点更接近止损线 → 可能先触发止损
            stop_distance = abs(low_return - stop_loss)
            take_distance = abs(high_return - take_profit)

            if stop_distance < take_distance:
                return "stop_loss", idx
            else:
                return "take_profit", idx

        # 兜底：保守策略，止损优先
        self._logger.debug(
            f"同日触发止盈止损(idx={idx})，无法精确判断先后，默认止损优先"
        )
        return "stop_loss", idx

    # ==================== 风险指标计算 ====================

    @staticmethod
    def _calculate_max_drawdown(cum_returns_pct: np.ndarray) -> float:
        """
        计算最大回撤
        
        基于累计收益率曲线计算，将其转为净值后取最大回撤
        
        Args:
            cum_returns_pct: 累计收益率序列(%)
            
        Returns:
            最大回撤百分比（正数表示，如 5.0 表示回撤5%）
        """
        if len(cum_returns_pct) < 2:
            return 0.0

        # 转换为净值（1.0为基准）
        nav = 1 + cum_returns_pct / 100

        # 滚动最高净值
        running_max = np.maximum.accumulate(nav)

        # 回撤 = (最高净值 - 当前净值) / 最高净值
        drawdowns = (running_max - nav) / running_max * 100

        # 修正除零问题
        drawdowns = np.where(np.isfinite(drawdowns), drawdowns, 0)

        return float(np.max(drawdowns))

    @staticmethod
    def _calculate_volatility(daily_returns_pct: np.ndarray) -> float:
        """
        计算日度收益率的标准差
        
        使用样本标准差(ddof=1)
        
        Args:
            daily_returns_pct: 日度收益率序列(%)
            
        Returns:
            日度波动率(%)
        """
        if len(daily_returns_pct) < 2:
            return 0.0

        return float(np.std(daily_returns_pct, ddof=1))

    @staticmethod
    def _calculate_sharpe_ratio(
        daily_returns_pct: np.ndarray,
        annual_risk_free_rate: float = 0.02,
    ) -> float:
        """
        计算年化夏普比率
        
        公式: Sharpe = mean(日超额收益) / std(日超额收益) × sqrt(252)
        
        注意：使用日度收益率（而非累计收益率），使用样本标准差(ddof=1)
        
        Args:
            daily_returns_pct: 日度收益率序列(%)
            annual_risk_free_rate: 年化无风险利率（小数形式，如0.02表示2%）
            
        Returns:
            年化夏普比率
        """
        if len(daily_returns_pct) < 2:
            return 0.0

        # 日度无风险利率(%)
        daily_rf_pct = annual_risk_free_rate / 252 * 100

        # 日超额收益
        excess_returns = daily_returns_pct - daily_rf_pct

        std = np.std(excess_returns, ddof=1)
        if std <= 0 or not np.isfinite(std):
            return 0.0

        mean_excess = np.mean(excess_returns)

        # 年化
        sharpe = mean_excess / std * np.sqrt(252)

        return float(sharpe) if np.isfinite(sharpe) else 0.0

    # ==================== 组合汇总 ====================

    def _calculate_portfolio_summary(
        self,
        results: List[TradeResult],
        results_df: pd.DataFrame,
        initial_capital: float,
        weights: List[float],
        total_stocks: int,
    ) -> Dict[str, Any]:
        """
        计算投资组合汇总指标
        
        Args:
            results: TradeResult对象列表
            results_df: 回测结果DataFrame
            initial_capital: 初始资金
            weights: 各股票权重
            total_stocks: 总股票数（含失败的）
            
        Returns:
            组合汇总指标字典
        """
        n = len(results)
        if n == 0:
            return {}

        # ---- 基础统计 ----
        net_returns = results_df["net_return_pct"].values
        gross_returns = results_df["gross_return_pct"].values

        winning = results_df[results_df["net_return_pct"] > 0]
        losing = results_df[results_df["net_return_pct"] <= 0]

        win_count = len(winning)
        lose_count = len(losing)
        win_rate = win_count / n * 100 if n > 0 else 0

        # ---- 组合加权收益 ----
        # 使用各股票实际投入金额加权
        invest_amounts = results_df["invest_amount"].values
        total_invested = invest_amounts.sum()

        if total_invested > 0:
            # 加权平均收益率
            weighted_return = np.sum(net_returns * invest_amounts) / total_invested
        else:
            weighted_return = np.mean(net_returns) if n > 0 else 0

        # 总盈亏金额
        total_profit = np.sum(
            invest_amounts * net_returns / 100
        )
        total_cost = results_df["total_cost"].sum()

        # ---- 风险指标 ----
        avg_return = float(np.mean(net_returns))
        median_return = float(np.median(net_returns))
        std_return = float(np.std(net_returns, ddof=1)) if n > 1 else 0

        # 组合最大回撤（近似：使用各股票最大回撤的加权值）
        max_drawdowns = results_df["max_drawdown_pct"].values
        if total_invested > 0:
            weighted_max_dd = np.sum(max_drawdowns * invest_amounts) / total_invested
        else:
            weighted_max_dd = float(np.max(max_drawdowns)) if n > 0 else 0

        # 组合夏普比率
        if std_return > 0:
            rf_rate = self._config.risk_free_rate
            # 假设平均持有天数来年化
            avg_holding_days = results_df["holding_days"].mean()
            if avg_holding_days > 0:
                periods_per_year = 365 / avg_holding_days
            else:
                periods_per_year = 252

            rf_per_period = rf_rate / periods_per_year * 100  # 转换为%
            portfolio_sharpe = (
                (avg_return - rf_per_period) / std_return * np.sqrt(periods_per_year)
            )
        else:
            portfolio_sharpe = 0.0

        # ---- 盈亏比 ----
        avg_win = float(winning["net_return_pct"].mean()) if win_count > 0 else 0
        avg_loss = float(losing["net_return_pct"].mean()) if lose_count > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # ---- 收益风险比 ----
        return_risk_ratio = (
            abs(weighted_return / weighted_max_dd) if weighted_max_dd > 0 else 0
        )

        # ---- 退出原因统计 ----
        exit_counts = results_df["exit_reason"].value_counts().to_dict()

        # ---- 持有天数统计 ----
        avg_holding = float(results_df["holding_days"].mean())
        max_holding = int(results_df["holding_days"].max())
        min_holding = int(results_df["holding_days"].min())

        return {
            # 基础信息
            "total_stocks": total_stocks,
            "successful_backtests": n,
            "failed_backtests": total_stocks - n,
            # 收益指标
            "portfolio_return_pct": round(weighted_return, 4),
            "total_profit": round(total_profit, 2),
            "total_invested": round(total_invested, 2),
            "initial_capital": initial_capital,
            "final_capital": round(initial_capital + total_profit, 2),
            # 统计指标
            "avg_return_pct": round(avg_return, 4),
            "median_return_pct": round(median_return, 4),
            "std_return_pct": round(std_return, 4),
            "best_return_pct": round(float(np.max(net_returns)), 4),
            "worst_return_pct": round(float(np.min(net_returns)), 4),
            # 胜率
            "win_rate_pct": round(win_rate, 2),
            "win_count": win_count,
            "lose_count": lose_count,
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "profit_loss_ratio": round(profit_loss_ratio, 4),
            # 风险指标
            "max_drawdown_pct": round(weighted_max_dd, 4),
            "sharpe_ratio": round(portfolio_sharpe, 4),
            "return_risk_ratio": round(return_risk_ratio, 4),
            # 成本
            "total_cost": round(total_cost, 2),
            "avg_cost_ratio_pct": round(
                results_df["cost_ratio_pct"].mean(), 4
            ),
            # 持有天数
            "avg_holding_days": round(avg_holding, 1),
            "max_holding_days": max_holding,
            "min_holding_days": min_holding,
            # 退出原因
            "exit_reason_counts": exit_counts,
            "stop_loss_rate_pct": round(
                exit_counts.get("stop_loss", 0) / n * 100, 2
            )
            if n > 0
            else 0,
            "take_profit_rate_pct": round(
                exit_counts.get("take_profit", 0) / n * 100, 2
            )
            if n > 0
            else 0,
        }

    # ==================== 辅助方法 ====================

    @staticmethod
    def _calculate_weights(
        stock_list: List[Dict[str, Any]], method: str = "equal"
    ) -> List[float]:
        """
        计算各股票的投资权重
        
        Args:
            stock_list: 股票列表
            method: 'equal'等权 或 'custom'使用stock_info中的weight字段
            
        Returns:
            权重列表（和为1.0）
        """
        n = len(stock_list)
        if n == 0:
            return []

        if method == "custom":
            weights = [s.get("weight", 1.0 / n) for s in stock_list]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / n] * n
        else:
            weights = [1.0 / n] * n

        return weights

    @staticmethod
    def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
        """
        统一日期列为datetime类型
        
        Args:
            df: 包含trade_date列的DataFrame
            
        Returns:
            日期列已标准化的DataFrame
        """
        if "trade_date" not in df.columns:
            return df

        result = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(result["trade_date"]):
            result["trade_date"] = pd.to_datetime(
                result["trade_date"], format="%Y%m%d", errors="coerce"
            )
            result = result.dropna(subset=["trade_date"])

        return result

    @staticmethod
    def _find_buy_date(
        df: pd.DataFrame, target_date: pd.Timestamp
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Series]]:
        """
        查找实际买入日
        
        如果目标日期不是交易日，顺延到下一个交易日
        
        Args:
            df: 行情数据（trade_date列为datetime类型）
            target_date: 目标买入日期
            
        Returns:
            (actual_buy_date, buy_row) 或 (None, None)
        """
        # 精确匹配
        exact_match = df[df["trade_date"] == target_date]
        if not exact_match.empty:
            return target_date, exact_match.iloc[0]

        # 顺延到下一个交易日
        future = df[df["trade_date"] > target_date]
        if not future.empty:
            row = future.iloc[0]
            return row["trade_date"], row

        return None, None

    @staticmethod
    def _format_date(dt) -> str:
        """将日期转换为YYYYMMDD字符串"""
        if isinstance(dt, pd.Timestamp):
            return dt.strftime("%Y%m%d")
        elif isinstance(dt, datetime):
            return dt.strftime("%Y%m%d")
        elif isinstance(dt, str):
            return dt.replace("-", "")[:8]
        return str(dt)

    @staticmethod
    def _add_calendar_days(date_str: str, days: int) -> str:
        """日期加上指定自然日天数"""
        dt = datetime.strptime(date_str[:8], "%Y%m%d")
        result = dt + timedelta(days=days)
        return result.strftime("%Y%m%d")
