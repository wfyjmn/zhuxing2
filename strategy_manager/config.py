"""
配置管理模块

修复内容：
1. 日志目录创建时序问题 - 确保目录在日志配置之前创建
2. 日志配置冲突 - 统一由Config管理，支持覆盖已有handler
3. _setup_logging不再重复创建目录
4. _init_tushare移到日志配置之后，确保异常可被记录
5. from_yaml增加字段过滤和错误处理
6. 增加配置验证逻辑
"""

import os
import json
import logging
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
import copy


@dataclass
class Config:
    """
    系统配置类
    
    职责：
    - 管理所有系统配置参数
    - 创建必要的目录结构
    - 统一管理日志配置（全局唯一入口）
    - 初始化外部数据源连接
    - 支持从YAML/JSON文件加载
    - 支持配置导出和持久化
    
    使用示例：
        # 方式1：直接创建
        config = Config(tushare_token="your_token")
        
        # 方式2：从YAML加载
        config = Config.from_yaml("config.yaml")
        
        # 方式3：从字典创建
        config = Config.from_dict({"tushare_token": "your_token"})
    """

    # ==================== 路径配置 ====================
    data_dir: str = "data"
    output_dir: str = "output"
    reports_dir: str = "reports"
    cache_dir: str = "cache"
    log_dir: str = "logs"
    db_path: str = "data/strategy_manager.db"

    # ==================== 数据源配置 ====================
    tushare_token: str = ""
    tushare_pro: Any = field(default=None, repr=False)  # 不在repr中显示

    # ==================== 回测配置 ====================
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003       # 佣金万3
    stamp_tax_rate: float = 0.001         # 印花税千1（卖出时收取）
    transfer_fee_rate: float = 0.00002    # 过户费万0.2（双向）
    slippage_rate: float = 0.001          # 滑点0.1%
    min_commission: float = 5.0           # 最低佣金5元

    # ==================== 策略参数 ====================
    max_holding_period: int = 20
    stop_loss: float = -5.0
    take_profit: float = 10.0
    max_position_pct: float = 0.1         # 单只股票最大仓位10%
    max_industry_exposure: float = 0.3    # 单行业最大暴露30%

    # ==================== 优化配置 ====================
    optimization_method: str = "bayesian"  # grid_search, random_search, bayesian
    n_iterations: int = 50
    cv_folds: int = 5

    # ==================== 线程配置 ====================
    max_workers: int = 4
    batch_size: int = 100

    # ==================== 缓存配置 ====================
    memory_cache_size: int = 2000
    disk_cache_ttl_hours: int = 24
    use_disk_cache: bool = True

    # ==================== 日志配置 ====================
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    log_max_bytes: int = 10 * 1024 * 1024  # 单个日志文件最大10MB
    log_backup_count: int = 30              # 保留30个备份

    # ==================== 基准配置 ====================
    benchmark_code: str = "000300.SH"
    risk_free_rate: float = 0.02           # 年化无风险利率2%

    # ==================== 通知配置 ====================
    send_email: bool = False
    send_wechat: bool = False
    send_dingtalk: bool = False

    # ==================== 其他配置 ====================
    market_environment: str = "normal"

    # ==================== 内部状态（不参与序列化） ====================
    _initialized: bool = field(default=False, repr=False, init=False)

    def __post_init__(self):
        """
        初始化后处理
        
        执行顺序至关重要：
        1. 创建所有必要目录
        2. 配置日志系统（此后所有操作都能被记录）
        3. 验证配置参数
        4. 初始化外部依赖（Tushare等）
        """
        # 步骤1：创建所有必要目录
        self._create_directories()

        # 步骤2：立即配置日志（后续所有操作都能正确记录）
        self._setup_logging()

        # 步骤3：验证配置参数
        self._validate_config()

        # 步骤4：初始化外部依赖
        if self.tushare_token:
            self._init_tushare()

        self._initialized = True
        logging.getLogger(__name__).info("配置初始化完成")

    # ==================== 目录管理 ====================

    def _create_directories(self):
        """
        创建所有必要的目录
        
        只在此处统一创建，其他方法不再重复创建
        """
        required_dirs = [
            self.data_dir,
            self.output_dir,
            self.reports_dir,
            self.cache_dir,
            self.log_dir,
        ]

        # 确保数据库文件所在目录也存在
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            required_dirs.append(db_dir)

        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    # ==================== 日志管理 ====================

    def _setup_logging(self):
        """
        配置日志系统
        
        作为全局唯一的日志配置入口：
        - 清除已有handler，避免与外部代码冲突
        - 同时输出到控制台和文件
        - 使用RotatingFileHandler防止日志文件过大
        - 第三方库日志级别设为WARNING减少噪音
        """
        from logging.handlers import RotatingFileHandler

        log_file = (
            Path(self.log_dir)
            / f"strategy_manager_{datetime.now().strftime('%Y%m%d')}.log"
        )

        # 获取根日志器并设置级别
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))

        # 清除所有已有handler，防止重复添加或与外部配置冲突
        root_logger.handlers.clear()

        # 日志格式
        formatter = logging.Formatter(self.log_format, datefmt=self.log_date_format)

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 文件handler（带轮转）
        try:
            file_handler = RotatingFileHandler(
                str(log_file),
                maxBytes=self.log_max_bytes,
                backupCount=self.log_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            # 文件handler创建失败时只用控制台，不能让日志配置阻塞程序
            root_logger.warning(f"无法创建日志文件 {log_file}: {e}，仅使用控制台输出")

        # 降低第三方库日志级别
        for noisy_logger in ("requests", "urllib3", "tushare", "matplotlib"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # ==================== 配置验证 ====================

    def _validate_config(self):
        """
        验证配置参数的合理性
        
        对不合理的参数发出警告但不阻塞运行，
        对严重错误的参数抛出ValueError
        """
        logger = logging.getLogger(__name__)

        # 回测参数验证
        if self.initial_capital <= 0:
            raise ValueError(f"初始资金必须为正数，当前值: {self.initial_capital}")

        if not -100 < self.stop_loss < 0:
            logger.warning(
                f"止损比例 {self.stop_loss}% 可能不合理，建议范围: (-100, 0)"
            )

        if not 0 < self.take_profit < 1000:
            logger.warning(
                f"止盈比例 {self.take_profit}% 可能不合理，建议范围: (0, 1000)"
            )

        if self.max_holding_period <= 0:
            raise ValueError(
                f"最大持有天数必须为正数，当前值: {self.max_holding_period}"
            )

        # 费率验证
        if self.commission_rate < 0 or self.commission_rate > 0.01:
            logger.warning(f"佣金费率 {self.commission_rate} 可能不合理，A股通常为万3")

        if self.stamp_tax_rate < 0 or self.stamp_tax_rate > 0.01:
            logger.warning(f"印花税率 {self.stamp_tax_rate} 可能不合理，A股通常为千1")

        # 仓位验证
        if not 0 < self.max_position_pct <= 1:
            logger.warning(
                f"单股最大仓位 {self.max_position_pct} 应在 (0, 1] 范围内"
            )

        if not 0 < self.max_industry_exposure <= 1:
            logger.warning(
                f"行业最大暴露 {self.max_industry_exposure} 应在 (0, 1] 范围内"
            )

        # 优化方法验证
        valid_methods = {"grid_search", "grid", "random_search", "random", "bayesian"}
        if self.optimization_method.lower() not in valid_methods:
            logger.warning(
                f"优化方法 '{self.optimization_method}' 不在支持列表中: {valid_methods}"
            )

        # 日志级别验证
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            logger.warning(
                f"日志级别 '{self.log_level}' 不在有效列表中: {valid_levels}，"
                f"将使用INFO"
            )

        # 线程数验证
        if self.max_workers <= 0:
            raise ValueError(f"最大工作线程数必须为正数，当前值: {self.max_workers}")

        if self.max_workers > 16:
            logger.warning(
                f"工作线程数 {self.max_workers} 较大，可能导致API限流"
            )

    # ==================== Tushare初始化 ====================

    def _init_tushare(self):
        """
        初始化Tushare连接
        
        在日志配置完成之后调用，确保异常能被正确记录
        """
        logger = logging.getLogger(__name__)

        try:
            import tushare as ts

            ts.set_token(self.tushare_token)
            self.tushare_pro = ts.pro_api()

            # 简单验证连接是否可用
            logger.info("Tushare初始化成功")

        except ImportError:
            logger.warning(
                "未安装tushare包，数据获取功能将不可用。"
                "安装方式: pip install tushare"
            )
            self.tushare_pro = None

        except Exception as e:
            logger.error(f"Tushare初始化失败: {e}")
            self.tushare_pro = None

    # ==================== 工厂方法 ====================

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        从YAML文件加载配置
        
        特性：
        - 文件不存在时返回默认配置并发出警告
        - 自动过滤YAML中Config未定义的字段
        - 支持嵌套配置的展开（paths、backtest等分组）
        
        Args:
            yaml_path: YAML文件路径
            
        Returns:
            Config实例
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            # 此时日志可能还未配置，用print作为后备
            print(f"[WARNING] 配置文件不存在: {yaml_path}，使用默认配置")
            return cls()

        try:
            import yaml
        except ImportError:
            print(
                "[WARNING] 未安装PyYAML，无法加载YAML配置文件。"
                "安装方式: pip install pyyaml"
            )
            return cls()

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            if not isinstance(raw_config, dict):
                print(f"[WARNING] 配置文件格式无效: {yaml_path}，使用默认配置")
                return cls()

            return cls._from_raw_dict(raw_config, source=str(yaml_path))

        except yaml.YAMLError as e:
            print(f"[ERROR] 解析YAML文件失败: {e}，使用默认配置")
            return cls()

        except Exception as e:
            print(f"[ERROR] 加载配置文件失败: {e}，使用默认配置")
            return cls()

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """
        从JSON文件加载配置
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            Config实例
        """
        json_path = Path(json_path)

        if not json_path.exists():
            print(f"[WARNING] 配置文件不存在: {json_path}，使用默认配置")
            return cls()

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)

            return cls._from_raw_dict(raw_config, source=str(json_path))

        except json.JSONDecodeError as e:
            print(f"[ERROR] 解析JSON文件失败: {e}，使用默认配置")
            return cls()

        except Exception as e:
            print(f"[ERROR] 加载配置文件失败: {e}，使用默认配置")
            return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            Config实例
        """
        return cls._from_raw_dict(config_dict, source="dict")

    @classmethod
    def _from_raw_dict(cls, raw_config: Dict[str, Any], source: str = "") -> "Config":
        """
        从原始字典创建Config，自动展开嵌套配置并过滤无效字段
        
        Args:
            raw_config: 原始配置字典（可能包含嵌套分组）
            source: 配置来源描述（用于日志）
            
        Returns:
            Config实例
        """
        flat_config = cls._flatten_config(raw_config)

        # 获取Config中定义的所有字段名
        valid_fields = cls._get_valid_field_names()

        # 分离有效字段和未知字段
        filtered = {}
        unknown = []

        for key, value in flat_config.items():
            if key in valid_fields:
                filtered[key] = value
            else:
                unknown.append(key)

        if unknown:
            print(f"[WARNING] 来自 {source} 的未知配置项已忽略: {unknown}")

        return cls(**filtered)

    @classmethod
    def _flatten_config(cls, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        展开嵌套配置分组
        
        支持的分组键: paths, backtest, optimization, threading, cache, logging_, notification
        分组内的字段会被提升到顶层
        
        Args:
            raw_config: 原始配置字典
            
        Returns:
            展开后的平面字典
        """
        nested_groups = {
            "paths",
            "backtest",
            "optimization",
            "threading",
            "cache",
            "logging_config",
            "notification",
            "benchmark",
        }

        flat = {}
        for key, value in raw_config.items():
            if key in nested_groups and isinstance(value, dict):
                # 展开嵌套分组
                flat.update(value)
            else:
                flat[key] = value

        return flat

    @classmethod
    def _get_valid_field_names(cls) -> Set[str]:
        """获取Config中所有可初始化的字段名"""
        return {
            f.name
            for f in dataclasses.fields(cls)
            if f.init  # 排除 init=False 的内部字段
        }

    # ==================== 序列化与导出 ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（排除不可序列化的字段）
        
        Returns:
            配置字典
        """
        exclude_fields = {"tushare_pro", "_initialized"}

        result = {}
        for f in dataclasses.fields(self):
            if f.name in exclude_fields:
                continue
            if f.name.startswith("_"):
                continue

            value = getattr(self, f.name)
            result[f.name] = value

        return result

    def save_yaml(self, filepath: str):
        """
        保存配置到YAML文件
        
        Args:
            filepath: 目标文件路径
        """
        try:
            import yaml
        except ImportError:
            logging.getLogger(__name__).error("未安装PyYAML，无法保存YAML配置")
            return

        config_dict = self.to_dict()

        # 确保目标目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logging.getLogger(__name__).info(f"配置已保存到: {filepath}")

    def save_json(self, filepath: str):
        """
        保存配置到JSON文件
        
        Args:
            filepath: 目标文件路径
        """
        config_dict = self.to_dict()

        # 确保目标目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logging.getLogger(__name__).info(f"配置已保存到: {filepath}")

    # ==================== 配置更新 ====================

    def update(self, **kwargs) -> "Config":
        """
        更新配置参数
        
        返回自身以支持链式调用：
            config.update(stop_loss=-3.0, take_profit=15.0)
        
        Args:
            **kwargs: 要更新的参数
            
        Returns:
            self
        """
        logger = logging.getLogger(__name__)
        valid_fields = self._get_valid_field_names()

        for key, value in kwargs.items():
            if key not in valid_fields:
                logger.warning(f"忽略未知配置项: {key}")
                continue

            old_value = getattr(self, key, None)
            setattr(self, key, value)
            logger.debug(f"配置更新: {key} = {old_value} -> {value}")

        # 重新验证
        self._validate_config()

        return self

    def copy(self) -> "Config":
        """
        创建配置的深拷贝
        
        Returns:
            新的Config实例
        """
        config_dict = self.to_dict()
        return Config(**config_dict)

    # ==================== 交易成本计算辅助 ====================

    def calculate_buy_cost(self, amount: float) -> float:
        """
        计算买入交易成本
        
        A股买入成本 = 佣金 + 过户费
        
        Args:
            amount: 买入金额
            
        Returns:
            买入总成本
        """
        commission = max(amount * self.commission_rate, self.min_commission)
        transfer_fee = amount * self.transfer_fee_rate
        return commission + transfer_fee

    def calculate_sell_cost(self, amount: float) -> float:
        """
        计算卖出交易成本
        
        A股卖出成本 = 佣金 + 印花税 + 过户费
        
        Args:
            amount: 卖出金额
            
        Returns:
            卖出总成本
        """
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_tax = amount * self.stamp_tax_rate
        transfer_fee = amount * self.transfer_fee_rate
        return commission + stamp_tax + transfer_fee

    def calculate_round_trip_cost(self, buy_amount: float, sell_amount: float) -> float:
        """
        计算一次完整交易（买入+卖出）的总成本
        
        Args:
            buy_amount: 买入金额
            sell_amount: 卖出金额
            
        Returns:
            总交易成本
        """
        return self.calculate_buy_cost(buy_amount) + self.calculate_sell_cost(
            sell_amount
        )

    # ==================== 信息展示 ====================

    def summary(self) -> str:
        """
        生成配置摘要字符串
        
        Returns:
            格式化的配置摘要
        """
        lines = [
            "=" * 60,
            "量化策略管理系统 - 配置摘要",
            "=" * 60,
            "",
            f"  数据目录:       {self.data_dir}",
            f"  输出目录:       {self.output_dir}",
            f"  数据库路径:     {self.db_path}",
            f"  日志目录:       {self.log_dir}",
            "",
            f"  初始资金:       {self.initial_capital:,.0f}",
            f"  佣金费率:       {self.commission_rate:.4%}",
            f"  印花税率:       {self.stamp_tax_rate:.4%}",
            f"  止损线:         {self.stop_loss}%",
            f"  止盈线:         {self.take_profit}%",
            f"  最大持有天数:   {self.max_holding_period}",
            "",
            f"  优化方法:       {self.optimization_method}",
            f"  优化迭代次数:   {self.n_iterations}",
            f"  交叉验证折数:   {self.cv_folds}",
            f"  工作线程数:     {self.max_workers}",
            "",
            f"  Tushare状态:    {'已连接' if self.tushare_pro else '未连接'}",
            f"  日志级别:       {self.log_level}",
            f"  基准指数:       {self.benchmark_code}",
            f"  无风险利率:     {self.risk_free_rate:.2%}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
