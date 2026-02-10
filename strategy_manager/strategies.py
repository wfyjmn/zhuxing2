"""
策略模块 - 策略基类和具体策略实现

设计原则：
1. 所有策略继承BaseStrategy抽象基类，统一接口
2. 三步流程：prepare_data → select_stocks → generate_signals
3. 每个策略声明自己的必要参数和参数优化范围
4. 参数验证在基类中统一处理（含范围警告）
5. 策略内部不依赖具体数据源，只操作DataFrame
6. 列名兼容：同时支持英文列名和中文列名
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==================== 异常定义 ====================


class StrategyError(Exception):
    """策略相关异常的基类"""
    pass


class ParameterError(StrategyError):
    """参数错误"""
    pass


class DataError(StrategyError):
    """数据错误"""
    pass


# ==================== 列名映射工具 ====================


class ColumnMapper:
    """
    列名映射工具
    
    解决中英文列名不一致的问题，允许策略代码统一使用英文列名，
    内部自动查找对应的中文列名
    
    使用示例：
        mapper = ColumnMapper(df)
        pe_col = mapper.find("pe_ttm")  # 可能返回 "pe_ttm" 或 "PE(TTM)"
    """

    # 英文列名 → 可能的中文/别名列表
    ALIAS_MAP: Dict[str, List[str]] = {
        "ts_code": ["ts_code", "代码", "stock_code", "code", "symbol", "证券代码"],
        "name": ["name", "股票名称", "stock_name", "简称", "证券名称"],
        "trade_date": ["trade_date", "交易日期", "date", "日期"],
        "open": ["open", "开盘价", "开盘"],
        "high": ["high", "最高价", "最高"],
        "low": ["low", "最低价", "最低"],
        "close": ["close", "收盘价", "收盘", "现价"],
        "vol": ["vol", "成交量", "volume", "成交量(手)"],
        "amount": ["amount", "成交额", "成交额(万)"],
        "pct_chg": ["pct_chg", "涨跌幅", "涨跌幅(%)", "change_pct"],
        "pe_ttm": ["pe_ttm", "PE(TTM)", "市盈率TTM", "市盈率(TTM)", "pe"],
        "pb": ["pb", "PB", "市净率", "pb_ratio"],
        "roe": ["roe", "ROE", "近3年平均ROE(%)", "ROE(%)", "净资产收益率"],
        "total_mv": ["total_mv", "市值(亿)", "总市值", "总市值(亿)", "market_cap"],
        "circ_mv": ["circ_mv", "流通市值", "流通市值(亿)"],
        "turnover_rate": ["turnover_rate", "换手率", "换手率(%)"],
        "volume_ratio": ["volume_ratio", "量比", "成交量倍数"],
        "revenue_yoy": ["revenue_yoy", "营收同比增长(%)", "营收增长率"],
        "profit_yoy": ["profit_yoy", "净利润同比增长(%)", "利润增长率"],
        "dv_ratio": ["dv_ratio", "股息率", "股息率(%)", "dividend_yield"],
        "industry": ["industry", "行业", "所属行业", "申万行业"],
        "area": ["area", "地区", "地域"],
        "list_date": ["list_date", "上市日期"],
        "signal": ["signal", "信号", "trade_signal"],
        "signal_strength": ["signal_strength", "信号强度", "得分", "score"],
        "weight": ["weight", "权重"],
    }

    def __init__(self, df: pd.DataFrame):
        self._columns = set(df.columns.tolist())
        self._cache: Dict[str, Optional[str]] = {}

    def find(self, standard_name: str) -> Optional[str]:
        """
        查找标准列名对应的实际列名
        
        Args:
            standard_name: 标准英文列名
            
        Returns:
            DataFrame中实际存在的列名，找不到返回None
        """
        if standard_name in self._cache:
            return self._cache[standard_name]

        # 直接匹配
        if standard_name in self._columns:
            self._cache[standard_name] = standard_name
            return standard_name

        # 别名匹配
        aliases = self.ALIAS_MAP.get(standard_name, [])
        for alias in aliases:
            if alias in self._columns:
                self._cache[standard_name] = alias
                return alias

        # 大小写不敏感匹配
        lower_map = {c.lower(): c for c in self._columns}
        if standard_name.lower() in lower_map:
            actual = lower_map[standard_name.lower()]
            self._cache[standard_name] = actual
            return actual

        self._cache[standard_name] = None
        return None

    def has(self, standard_name: str) -> bool:
        """检查列是否存在"""
        return self.find(standard_name) is not None

    def get_series(
        self, df: pd.DataFrame, standard_name: str
    ) -> Optional[pd.Series]:
        """获取列数据"""
        col = self.find(standard_name)
        if col is not None and col in df.columns:
            return df[col]
        return None

    def rename_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将DataFrame的列名重命名为标准英文列名
        
        只重命名能找到映射的列，其他列保持不变
        
        Args:
            df: 原始DataFrame
            
        Returns:
            重命名后的DataFrame
        """
        rename_map = {}

        for standard_name, aliases in self.ALIAS_MAP.items():
            for alias in aliases:
                if alias in df.columns and alias != standard_name:
                    rename_map[alias] = standard_name
                    break

        if rename_map:
            return df.rename(columns=rename_map)
        return df


# ==================== 策略基类 ====================


class BaseStrategy(ABC):
    """
    策略抽象基类
    
    所有具体策略必须继承此类并实现三个抽象方法：
    - prepare_data: 数据预处理（清洗、特征计算）
    - select_stocks: 选股逻辑（过滤、排序）
    - generate_signals: 生成交易信号（买入信号、信号强度）
    
    基类提供：
    - 参数管理（验证、默认值合并）
    - 列名映射（自动适配中英文列名）
    - ST股/退市股过滤
    - 通用的过滤辅助方法
    
    使用示例：
        class MyStrategy(BaseStrategy):
            def prepare_data(self, raw_data):
                ...
            def select_stocks(self, data):
                ...
            def generate_signals(self, data):
                ...
        
        strategy = MyStrategy("my_strategy", {"param1": 10})
        prepared = strategy.prepare_data(raw_data)
        selected = strategy.select_stocks(prepared)
        signals = strategy.generate_signals(selected)
    """

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Args:
            name: 策略名称
            parameters: 策略参数字典
        """
        self.name = name
        self._logger = logging.getLogger(f"{__name__}.{name}")

        # 合并默认参数和用户参数
        defaults = self.get_default_parameters()
        self.parameters: Dict[str, Any] = {**defaults, **parameters}

        # 验证参数
        self._validate_parameters()

    # ==================== 抽象方法（子类必须实现） ====================

    @abstractmethod
    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        职责：
        - 数据清洗（类型转换、缺失值处理）
        - 特征计算（技术指标、衍生指标）
        - 数据对齐
        
        Args:
            raw_data: 原始数据
            
        Returns:
            预处理后的数据
        """
        pass

    @abstractmethod
    def select_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        选股逻辑
        
        职责：
        - 应用筛选条件
        - 过滤不合格的股票
        - 排序和截断
        
        Args:
            data: 预处理后的数据
            
        Returns:
            选中的股票数据
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        职责：
        - 确定买入/卖出信号
        - 计算信号强度（用于排序和仓位分配）
        - 添加信号相关列
        
        Args:
            data: 选中的股票数据
            
        Returns:
            包含signal和signal_strength列的DataFrame
        """
        pass

    # ==================== 可选覆盖方法 ====================

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """
        返回策略的默认参数
        
        子类应覆盖此方法提供自己的默认值
        """
        return {}

    @classmethod
    def get_required_parameters(cls) -> List[str]:
        """
        返回必要参数名列表
        
        缺少这些参数时会抛出ParameterError
        """
        return []

    @classmethod
    def get_parameter_ranges(cls) -> Dict[str, Tuple[Any, Any]]:
        """
        返回参数的优化范围
        
        用于参数优化器自动生成搜索空间
        返回格式: {"param_name": (low, high)}
        """
        return {}

    @classmethod
    def get_description(cls) -> str:
        """返回策略描述"""
        return cls.__doc__ or cls.__name__

    # ==================== 参数验证 ====================

    def _validate_parameters(self):
        """
        验证参数
        
        - 检查必要参数是否存在
        - 检查参数值是否在合理范围内（警告但不阻塞）
        """
        # 检查必要参数
        required = self.get_required_parameters()
        missing = [p for p in required if p not in self.parameters]
        if missing:
            raise ParameterError(
                f"策略 '{self.name}' 缺少必要参数: {missing}"
            )

        # 检查参数范围
        ranges = self.get_parameter_ranges()
        for param_name, (low, high) in ranges.items():
            if param_name not in self.parameters:
                continue

            value = self.parameters[param_name]
            if not isinstance(value, (int, float)):
                continue

            if not (low <= value <= high):
                self._logger.warning(
                    f"参数 '{param_name}'={value} "
                    f"超出建议范围 [{low}, {high}]"
                )

    def update_parameters(self, new_params: Dict[str, Any]):
        """
        更新参数
        
        Args:
            new_params: 新参数（只更新指定的键，不影响其他参数）
        """
        self.parameters.update(new_params)
        self._validate_parameters()
        self._logger.debug(f"参数已更新: {list(new_params.keys())}")

    # ==================== 通用过滤辅助方法 ====================

    def _filter_st_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤ST股和退市股
        
        匹配规则：名称包含"ST"、"*ST"、"退"的股票
        
        Args:
            df: 股票数据
            
        Returns:
            过滤后的数据
        """
        mapper = ColumnMapper(df)
        name_col = mapper.find("name")

        if name_col is None:
            return df

        original_len = len(df)
        mask = ~df[name_col].astype(str).str.contains(
            r"ST|退|退市", case=False, na=False
        )
        result = df[mask]

        filtered_count = original_len - len(result)
        if filtered_count > 0:
            self._logger.debug(f"过滤ST/退市股: {filtered_count} 只")

        return result

    def _filter_by_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        exclude_negative: bool = False,
        exclude_zero: bool = False,
    ) -> pd.DataFrame:
        """
        按数值范围过滤
        
        Args:
            df: 数据
            column: 标准列名（会自动查找实际列名）
            min_val: 最小值（含）
            max_val: 最大值（含）
            exclude_negative: 是否排除负值
            exclude_zero: 是否排除零值
            
        Returns:
            过滤后的数据
        """
        mapper = ColumnMapper(df)
        actual_col = mapper.find(column)

        if actual_col is None:
            self._logger.debug(f"列 '{column}' 不存在，跳过过滤")
            return df

        result = df.copy()

        # 转为数值类型
        result[actual_col] = pd.to_numeric(result[actual_col], errors="coerce")

        if exclude_negative:
            result = result[result[actual_col] >= 0]

        if exclude_zero:
            result = result[result[actual_col] != 0]

        if min_val is not None:
            result = result[result[actual_col] >= min_val]

        if max_val is not None:
            result = result[result[actual_col] <= max_val]

        filtered = len(df) - len(result)
        if filtered > 0:
            self._logger.debug(
                f"按 {column} 过滤: {filtered} 只 "
                f"(范围: [{min_val}, {max_val}])"
            )

        return result

    def _ensure_numeric(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """
        确保指定列为数值类型
        
        Args:
            df: 数据
            columns: 标准列名列表
            
        Returns:
            处理后的数据
        """
        result = df.copy()
        mapper = ColumnMapper(result)

        for col in columns:
            actual_col = mapper.find(col)
            if actual_col and actual_col in result.columns:
                result[actual_col] = pd.to_numeric(
                    result[actual_col], errors="coerce"
                )

        return result

    def _rank_and_score(
        self,
        df: pd.DataFrame,
        score_config: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        多因子打分
        
        对多个指标进行排名打分，综合计算信号强度
        
        Args:
            df: 数据
            score_config: 打分配置列表，每个元素为：
                {
                    "column": "pe_ttm",         # 标准列名
                    "weight": 0.3,              # 权重
                    "ascending": True,          # True=越小越好
                }
            
        Returns:
            添加了signal_strength列的DataFrame
        """
        if df.empty:
            return df

        result = df.copy()
        mapper = ColumnMapper(result)
        total_score = np.zeros(len(result))
        total_weight = 0.0

        for config in score_config:
            col_name = config["column"]
            weight = config.get("weight", 1.0)
            ascending = config.get("ascending", True)

            actual_col = mapper.find(col_name)
            if actual_col is None or actual_col not in result.columns:
                continue

            values = pd.to_numeric(result[actual_col], errors="coerce")
            if values.isna().all():
                continue

            # 百分位排名
            ranks = values.rank(pct=True, na_option="bottom")

            if ascending:
                # 值越小排名越高（如PE越低越好）
                ranks = 1 - ranks

            total_score += ranks.fillna(0).values * weight
            total_weight += weight

        if total_weight > 0:
            result["signal_strength"] = total_score / total_weight
        else:
            result["signal_strength"] = 0.0

        return result

    # ==================== 信息方法 ====================

    def describe(self) -> Dict[str, Any]:
        """返回策略完整描述"""
        return {
            "name": self.name,
            "class": type(self).__name__,
            "description": self.get_description(),
            "parameters": self.parameters,
            "default_parameters": self.get_default_parameters(),
            "required_parameters": self.get_required_parameters(),
            "parameter_ranges": self.get_parameter_ranges(),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name='{self.name}', "
            f"params={len(self.parameters)})"
        )


# ==================== 价值投资策略 ====================


class ValueStrategy(BaseStrategy):
    """
    价值投资策略
    
    选股逻辑：
    - 低PE（市盈率）：寻找估值偏低的公司
    - 高ROE（净资产收益率）：寻找盈利能力强的公司
    - 低PB（市净率）：寻找资产折价的公司
    - 适当市值：排除微盘股和超大盘股
    - 可选：高股息率
    
    信号强度：
    - PE越低、ROE越高、PB越低 → 信号越强
    
    适用场景：
    - 中长期投资
    - 震荡市和熊市后期
    """

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        return {
            "pe_ttm_min": 0,           # PE最小值（排除负PE）
            "pe_ttm_max": 30,          # PE最大值
            "roe_min": 10.0,           # ROE最小值(%)
            "pb_min": 0,               # PB最小值（排除负PB）
            "pb_max": 3.0,             # PB最大值
            "market_cap_min": 20.0,    # 最小市值(亿)
            "market_cap_max": 5000.0,  # 最大市值(亿)
            "dividend_yield_min": 0.0, # 最低股息率(%)，0表示不限
            "enable_st_filter": True,  # 是否过滤ST股
            "max_stocks": 30,          # 最大选股数量
            # 多因子权重
            "pe_weight": 0.35,
            "roe_weight": 0.35,
            "pb_weight": 0.20,
            "dividend_weight": 0.10,
        }

    @classmethod
    def get_required_parameters(cls) -> List[str]:
        return ["pe_ttm_max", "roe_min"]

    @classmethod
    def get_parameter_ranges(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            "pe_ttm_max": (5, 60),
            "roe_min": (3, 35),
            "pb_max": (0.5, 8.0),
            "market_cap_min": (5, 200),
            "market_cap_max": (100, 10000),
            "dividend_yield_min": (0, 6.0),
            "pe_weight": (0.1, 0.5),
            "roe_weight": (0.1, 0.5),
            "pb_weight": (0.05, 0.4),
        }

    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        if raw_data.empty:
            return raw_data

        data = raw_data.copy()

        # 确保数值类型
        data = self._ensure_numeric(
            data,
            ["pe_ttm", "roe", "pb", "total_mv", "dv_ratio", "close"],
        )

        # 去重
        mapper = ColumnMapper(data)
        code_col = mapper.find("ts_code")
        if code_col:
            data = data.drop_duplicates(subset=[code_col], keep="last")

        self._logger.debug(
            f"价值策略数据预处理完成: {len(data)} 条记录"
        )

        return data

    def select_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        价值选股
        
        筛选顺序：ST过滤 → PE过滤 → ROE过滤 → PB过滤 → 市值过滤 → 股息率过滤
        """
        if data.empty:
            return data

        params = self.parameters
        filtered = data.copy()
        initial_count = len(filtered)

        # ST股过滤
        if params.get("enable_st_filter", True):
            filtered = self._filter_st_stocks(filtered)

        # PE过滤
        filtered = self._filter_by_range(
            filtered,
            "pe_ttm",
            min_val=params.get("pe_ttm_min", 0),
            max_val=params["pe_ttm_max"],
            exclude_negative=True,
        )

        # ROE过滤
        filtered = self._filter_by_range(
            filtered,
            "roe",
            min_val=params["roe_min"],
        )

        # PB过滤
        pb_max = params.get("pb_max")
        if pb_max and pb_max > 0:
            filtered = self._filter_by_range(
                filtered,
                "pb",
                min_val=params.get("pb_min", 0),
                max_val=pb_max,
                exclude_negative=True,
            )

        # 市值过滤
        market_cap_min = params.get("market_cap_min", 0)
        market_cap_max = params.get("market_cap_max", 0)
        if market_cap_min > 0 or market_cap_max > 0:
            filtered = self._filter_by_range(
                filtered,
                "total_mv",
                min_val=market_cap_min if market_cap_min > 0 else None,
                max_val=market_cap_max if market_cap_max > 0 else None,
            )

        # 股息率过滤
        div_min = params.get("dividend_yield_min", 0)
        if div_min > 0:
            filtered = self._filter_by_range(
                filtered,
                "dv_ratio",
                min_val=div_min,
            )

        self._logger.info(
            f"价值策略选股: {initial_count} → {len(filtered)} "
            f"(PE<={params['pe_ttm_max']}, ROE>={params['roe_min']})"
        )

        return filtered

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成价值因子信号
        
        多因子打分：PE(低好) + ROE(高好) + PB(低好) + 股息率(高好)
        """
        if data.empty:
            return data

        params = self.parameters

        # 多因子打分
        score_config = [
            {
                "column": "pe_ttm",
                "weight": params.get("pe_weight", 0.35),
                "ascending": True,   # PE越低越好
            },
            {
                "column": "roe",
                "weight": params.get("roe_weight", 0.35),
                "ascending": False,  # ROE越高越好
            },
            {
                "column": "pb",
                "weight": params.get("pb_weight", 0.20),
                "ascending": True,   # PB越低越好
            },
            {
                "column": "dv_ratio",
                "weight": params.get("dividend_weight", 0.10),
                "ascending": False,  # 股息率越高越好
            },
        ]

        result = self._rank_and_score(data, score_config)

        # 添加买入信号
        result["signal"] = "buy"

        # 按信号强度排序，取前N只
        max_stocks = params.get("max_stocks", 30)
        result = result.sort_values(
            "signal_strength", ascending=False
        ).head(max_stocks)

        self._logger.info(
            f"价值策略生成 {len(result)} 个买入信号"
        )

        return result


# ==================== 动量交易策略 ====================


class MomentumStrategy(BaseStrategy):
    """
    动量交易策略
    
    选股逻辑：
    - 价格动量：近期涨幅较大的股票
    - 成交量放大：量能配合
    - 价格位置：不追过高的股票
    
    信号强度：
    - 动量越强、量能越大 → 信号越强
    
    适用场景：
    - 短中期交易
    - 趋势市（牛市初中期）
    """

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        return {
            "lookback_period": 20,          # 动量回看周期（交易日）
            "momentum_threshold": 5.0,      # 最低动量阈值(%)
            "volume_ratio_min": 1.5,        # 最低量比
            "min_price": 5.0,               # 最低价格
            "max_price": 200.0,             # 最高价格
            "min_turnover": 1.0,            # 最低换手率(%)
            "enable_st_filter": True,
            "max_stocks": 20,
            # 因子权重
            "momentum_weight": 0.50,
            "volume_weight": 0.30,
            "turnover_weight": 0.20,
        }

    @classmethod
    def get_required_parameters(cls) -> List[str]:
        return ["lookback_period", "momentum_threshold"]

    @classmethod
    def get_parameter_ranges(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            "lookback_period": (5, 60),
            "momentum_threshold": (1.0, 20.0),
            "volume_ratio_min": (1.0, 5.0),
            "min_price": (2.0, 20.0),
            "max_price": (50.0, 500.0),
            "min_turnover": (0.5, 5.0),
            "momentum_weight": (0.2, 0.7),
            "volume_weight": (0.1, 0.5),
        }

    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理：计算动量指标"""
        if raw_data.empty:
            return raw_data

        data = raw_data.copy()
        mapper = ColumnMapper(data)

        # 确保数值类型
        data = self._ensure_numeric(
            data,
            ["close", "vol", "pct_chg", "turnover_rate", "volume_ratio"],
        )

        # 计算区间动量（如果有涨跌幅序列）
        pct_col = mapper.find("pct_chg")
        close_col = mapper.find("close")

        if pct_col and pct_col in data.columns:
            # 如果每行是不同股票的截面数据（非时间序列），
            # 则动量需要从其他来源获取
            pass

        # 去重
        code_col = mapper.find("ts_code")
        if code_col:
            data = data.drop_duplicates(subset=[code_col], keep="last")

        self._logger.debug(
            f"动量策略数据预处理完成: {len(data)} 条记录"
        )

        return data

    def select_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        动量选股
        
        筛选：ST过滤 → 价格过滤 → 量比过滤 → 换手率过滤 → 动量过滤
        """
        if data.empty:
            return data

        params = self.parameters
        filtered = data.copy()
        initial_count = len(filtered)

        # ST过滤
        if params.get("enable_st_filter", True):
            filtered = self._filter_st_stocks(filtered)

        # 价格过滤
        filtered = self._filter_by_range(
            filtered,
            "close",
            min_val=params.get("min_price", 5.0),
            max_val=params.get("max_price", 200.0),
        )

        # 量比过滤
        vol_ratio_min = params.get("volume_ratio_min", 1.5)
        if vol_ratio_min > 0:
            filtered = self._filter_by_range(
                filtered,
                "volume_ratio",
                min_val=vol_ratio_min,
            )

        # 换手率过滤
        turnover_min = params.get("min_turnover", 1.0)
        if turnover_min > 0:
            filtered = self._filter_by_range(
                filtered,
                "turnover_rate",
                min_val=turnover_min,
            )

        # 动量过滤（涨跌幅）
        momentum_threshold = params.get("momentum_threshold", 5.0)
        filtered = self._filter_by_range(
            filtered,
            "pct_chg",
            min_val=momentum_threshold,
        )

        self._logger.info(
            f"动量策略选股: {initial_count} → {len(filtered)} "
            f"(动量>={momentum_threshold}%, "
            f"量比>={vol_ratio_min})"
        )

        return filtered

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成动量信号"""
        if data.empty:
            return data

        params = self.parameters

        score_config = [
            {
                "column": "pct_chg",
                "weight": params.get("momentum_weight", 0.50),
                "ascending": False,  # 涨幅越大越好
            },
            {
                "column": "volume_ratio",
                "weight": params.get("volume_weight", 0.30),
                "ascending": False,  # 量比越大越好
            },
            {
                "column": "turnover_rate",
                "weight": params.get("turnover_weight", 0.20),
                "ascending": False,  # 换手率越高越好
            },
        ]

        result = self._rank_and_score(data, score_config)
        result["signal"] = "buy"

        max_stocks = params.get("max_stocks", 20)
        result = result.sort_values(
            "signal_strength", ascending=False
        ).head(max_stocks)

        self._logger.info(
            f"动量策略生成 {len(result)} 个买入信号"
        )

        return result


# ==================== 成长投资策略 ====================


class GrowthStrategy(BaseStrategy):
    """
    成长投资策略
    
    选股逻辑：
    - 营收高增长：收入端持续扩张
    - 利润高增长：盈利能力持续提升
    - ROE良好：经营效率高
    - 合理估值：PE不超过成长性对应的合理水平
    
    适用场景：
    - 中长期投资
    - 牛市和结构性行情
    """

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        return {
            "revenue_growth_min": 15.0,    # 最低营收增长率(%)
            "profit_growth_min": 20.0,     # 最低利润增长率(%)
            "roe_min": 8.0,                # 最低ROE(%)
            "pe_ttm_max": 60,              # 最高PE
            "market_cap_min": 30.0,        # 最小市值(亿)
            "enable_st_filter": True,
            "max_stocks": 25,
            # 因子权重
            "revenue_growth_weight": 0.25,
            "profit_growth_weight": 0.35,
            "roe_weight": 0.25,
            "pe_weight": 0.15,
        }

    @classmethod
    def get_required_parameters(cls) -> List[str]:
        return ["revenue_growth_min", "profit_growth_min"]

    @classmethod
    def get_parameter_ranges(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            "revenue_growth_min": (5, 50),
            "profit_growth_min": (5, 80),
            "roe_min": (3, 25),
            "pe_ttm_max": (15, 100),
            "market_cap_min": (10, 200),
        }

    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        if raw_data.empty:
            return raw_data

        data = raw_data.copy()

        data = self._ensure_numeric(
            data,
            [
                "revenue_yoy",
                "profit_yoy",
                "roe",
                "pe_ttm",
                "total_mv",
                "close",
            ],
        )

        mapper = ColumnMapper(data)
        code_col = mapper.find("ts_code")
        if code_col:
            data = data.drop_duplicates(subset=[code_col], keep="last")

        return data

    def select_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """成长选股"""
        if data.empty:
            return data

        params = self.parameters
        filtered = data.copy()
        initial_count = len(filtered)

        if params.get("enable_st_filter", True):
            filtered = self._filter_st_stocks(filtered)

        # 营收增长过滤
        filtered = self._filter_by_range(
            filtered,
            "revenue_yoy",
            min_val=params["revenue_growth_min"],
        )

        # 利润增长过滤
        filtered = self._filter_by_range(
            filtered,
            "profit_yoy",
            min_val=params["profit_growth_min"],
        )

        # ROE过滤
        filtered = self._filter_by_range(
            filtered,
            "roe",
            min_val=params.get("roe_min", 8.0),
        )

        # PE过滤
        pe_max = params.get("pe_ttm_max", 60)
        if pe_max > 0:
            filtered = self._filter_by_range(
                filtered,
                "pe_ttm",
                max_val=pe_max,
                exclude_negative=True,
            )

        # 市值过滤
        cap_min = params.get("market_cap_min", 0)
        if cap_min > 0:
            filtered = self._filter_by_range(
                filtered,
                "total_mv",
                min_val=cap_min,
            )

        self._logger.info(
            f"成长策略选股: {initial_count} → {len(filtered)}"
        )

        return filtered

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成成长因子信号"""
        if data.empty:
            return data

        params = self.parameters

        score_config = [
            {
                "column": "revenue_yoy",
                "weight": params.get("revenue_growth_weight", 0.25),
                "ascending": False,
            },
            {
                "column": "profit_yoy",
                "weight": params.get("profit_growth_weight", 0.35),
                "ascending": False,
            },
            {
                "column": "roe",
                "weight": params.get("roe_weight", 0.25),
                "ascending": False,
            },
            {
                "column": "pe_ttm",
                "weight": params.get("pe_weight", 0.15),
                "ascending": True,
            },
        ]

        result = self._rank_and_score(data, score_config)
        result["signal"] = "buy"

        max_stocks = params.get("max_stocks", 25)
        result = result.sort_values(
            "signal_strength", ascending=False
        ).head(max_stocks)

        self._logger.info(
            f"成长策略生成 {len(result)} 个买入信号"
        )

        return result


# ==================== 均值回归策略 ====================


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略
    
    选股逻辑：
    - 短期超跌：近期跌幅较大，偏离均线
    - 基本面支撑：ROE和PE处于合理范围（非基本面恶化导致的下跌）
    - 流动性：成交量维持一定水平（非缩量阴跌）
    
    适用场景：
    - 短期交易（持有3-10天）
    - 震荡市
    """

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        return {
            "oversold_threshold": -8.0,    # 超跌阈值(%)，近N日跌幅
            "lookback_days": 5,            # 回看天数
            "pe_ttm_max": 50,              # PE上限（排除垃圾股）
            "roe_min": 5.0,                # ROE下限
            "min_price": 5.0,              # 最低价格
            "min_turnover": 0.5,           # 最低换手率(%)
            "enable_st_filter": True,
            "max_stocks": 15,
            # 因子权重
            "oversold_weight": 0.50,
            "pe_weight": 0.20,
            "roe_weight": 0.20,
            "turnover_weight": 0.10,
        }

    @classmethod
    def get_required_parameters(cls) -> List[str]:
        return ["oversold_threshold"]

    @classmethod
    def get_parameter_ranges(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            "oversold_threshold": (-20.0, -3.0),
            "lookback_days": (3, 20),
            "pe_ttm_max": (15, 80),
            "roe_min": (2, 20),
            "min_price": (2.0, 20.0),
        }

    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        if raw_data.empty:
            return raw_data

        data = raw_data.copy()

        data = self._ensure_numeric(
            data,
            ["close", "pct_chg", "pe_ttm", "roe", "turnover_rate"],
        )

        mapper = ColumnMapper(data)
        code_col = mapper.find("ts_code")
        if code_col:
            data = data.drop_duplicates(subset=[code_col], keep="last")

        return data

    def select_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """均值回归选股：寻找超跌但基本面良好的股票"""
        if data.empty:
            return data

        params = self.parameters
        filtered = data.copy()
        initial_count = len(filtered)

        if params.get("enable_st_filter", True):
            filtered = self._filter_st_stocks(filtered)

        # 价格过滤
        filtered = self._filter_by_range(
            filtered,
            "close",
            min_val=params.get("min_price", 5.0),
        )

        # 超跌过滤（涨跌幅为负且低于阈值）
        oversold = params["oversold_threshold"]
        filtered = self._filter_by_range(
            filtered,
            "pct_chg",
            max_val=oversold,
        )

        # PE过滤（排除基本面恶化的股票）
        pe_max = params.get("pe_ttm_max", 50)
        if pe_max > 0:
            filtered = self._filter_by_range(
                filtered,
                "pe_ttm",
                max_val=pe_max,
                exclude_negative=True,
            )

        # ROE过滤
        filtered = self._filter_by_range(
            filtered,
            "roe",
            min_val=params.get("roe_min", 5.0),
        )

        # 换手率过滤
        min_turnover = params.get("min_turnover", 0.5)
        if min_turnover > 0:
            filtered = self._filter_by_range(
                filtered,
                "turnover_rate",
                min_val=min_turnover,
            )

        self._logger.info(
            f"均值回归策略选股: {initial_count} → {len(filtered)} "
            f"(超跌<={oversold}%)"
        )

        return filtered

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成均值回归信号：跌幅越大、基本面越好 → 信号越强"""
        if data.empty:
            return data

        params = self.parameters

        score_config = [
            {
                "column": "pct_chg",
                "weight": params.get("oversold_weight", 0.50),
                "ascending": True,  # 跌幅越大越好（负值越小越好）
            },
            {
                "column": "pe_ttm",
                "weight": params.get("pe_weight", 0.20),
                "ascending": True,  # PE越低越好
            },
            {
                "column": "roe",
                "weight": params.get("roe_weight", 0.20),
                "ascending": False,  # ROE越高越好
            },
            {
                "column": "turnover_rate",
                "weight": params.get("turnover_weight", 0.10),
                "ascending": False,  # 换手率越高说明资金关注度越高
            },
        ]

        result = self._rank_and_score(data, score_config)
        result["signal"] = "buy"

        max_stocks = params.get("max_stocks", 15)
        result = result.sort_values(
            "signal_strength", ascending=False
        ).head(max_stocks)

        self._logger.info(
            f"均值回归策略生成 {len(result)} 个买入信号"
        )

        return result


# ==================== 策略工厂 ====================


class StrategyFactory:
    """
    策略工厂
    
    根据策略类型名称创建策略实例
    
    使用示例：
        strategy = StrategyFactory.create("value", "my_strategy", {"pe_ttm_max": 25})
        
        # 列出所有可用策略
        available = StrategyFactory.list_available()
    """

    # 策略类型注册表
    _REGISTRY: Dict[str, type] = {
        "value": ValueStrategy,
        "momentum": MomentumStrategy,
        "growth": GrowthStrategy,
        "mean_reversion": MeanReversionStrategy,
    }

    @classmethod
    def create(
        cls,
        strategy_type: str,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> BaseStrategy:
        """
        创建策略实例
        
        Args:
            strategy_type: 策略类型（value/momentum/growth/mean_reversion）
            name: 策略名称
            parameters: 策略参数（None使用默认值）
            
        Returns:
            策略实例
            
        Raises:
            ValueError: 不支持的策略类型
        """
        strategy_class = cls._REGISTRY.get(strategy_type.lower())

        if strategy_class is None:
            available = sorted(cls._REGISTRY.keys())
            raise ValueError(
                f"不支持的策略类型: '{strategy_type}'，"
                f"可用类型: {available}"
            )

        return strategy_class(name, parameters or {})

    @classmethod
    def register(cls, type_name: str, strategy_class: type):
        """
        注册自定义策略类型
        
        Args:
            type_name: 类型名称
            strategy_class: 策略类（必须继承BaseStrategy）
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(
                f"{strategy_class.__name__} 必须继承 BaseStrategy"
            )

        cls._REGISTRY[type_name.lower()] = strategy_class
        logging.getLogger(__name__).info(
            f"注册策略类型: '{type_name}' → {strategy_class.__name__}"
        )

    @classmethod
    def list_available(cls) -> List[Dict[str, Any]]:
        """
        列出所有可用的策略类型
        
        Returns:
            策略类型信息列表
        """
        result = []
        for type_name, strategy_class in sorted(cls._REGISTRY.items()):
            result.append(
                {
                    "type": type_name,
                    "class": strategy_class.__name__,
                    "description": strategy_class.get_description(),
                    "default_parameters": strategy_class.get_default_parameters(),
                    "required_parameters": strategy_class.get_required_parameters(),
                    "parameter_ranges": strategy_class.get_parameter_ranges(),
                }
            )
        return result
