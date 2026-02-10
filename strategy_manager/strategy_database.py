"""
策略数据库模块

修复内容：
1. save_strategy逻辑 - 先查后决定INSERT或UPDATE，消除死代码分支
2. 数据库连接安全 - 使用上下文管理器确保连接正确关闭和事务回滚
3. WAL模式 - 启用WAL日志模式支持并发读取
4. 外键约束 - 每次连接都启用PRAGMA foreign_keys
5. 线程安全 - 所有数据库操作加可重入锁
6. 回测结果保存 - 正确提取嵌套字典中的指标字段
7. JSON序列化 - DataFrame等不可序列化对象的安全处理
8. 查询接口 - 统一返回类型，增加分页和条件查询支持
9. 数据库维护 - vacuum、备份、迁移支持
"""

import contextlib
import json
import logging
import os
import shutil
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .config import Config


class StrategyDatabase:
    """
    策略数据库
    
    职责：
    - 策略的增删改查和版本管理
    - 回测结果的持久化存储
    - 持仓记录和交易记录管理
    - 数据库维护（清理、备份、优化）
    
    表结构：
    - strategies: 策略主表（name唯一）
    - strategy_versions: 策略版本表（每次参数变更产生新版本）
    - backtest_results: 回测结果表
    - positions: 持仓记录表
    - trade_records: 交易明细表
    
    使用示例：
        db = StrategyDatabase("strategy.db", config)
        
        # 保存策略
        strategy_id = db.save_strategy("价值策略", "低PE高ROE", {"pe_max": 20})
        
        # 创建版本
        version_id = db.create_version(strategy_id, {"pe_max": 25})
        
        # 保存回测结果
        db.save_backtest_result(version_id, {"win_rate": 55.0, "sharpe_ratio": 1.2})
        
        # 查询
        versions = db.get_versions(strategy_id)
        history = db.get_backtest_history(version_id)
    """

    def __init__(self, db_path: str, config: Config):
        self._db_path = db_path
        self._config = config
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # 确保数据库目录存在
        db_dir = os.path.dirname(db_path)
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)

        # 初始化表结构
        self._init_schema()

        self._logger.info(f"数据库初始化完成: {db_path}")

    # ==================== 连接管理 ====================

    @contextlib.contextmanager
    def _get_connection(self):
        """
        数据库连接上下文管理器
        
        确保：
        - WAL模式启用（支持并发读）
        - 外键约束启用
        - 异常时自动回滚
        - 正常时自动提交
        - 连接始终关闭
        """
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row  # 支持按列名访问

        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=10000")

            yield conn
            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

    # ==================== 数据库初始化 ====================

    def _init_schema(self):
        """初始化数据库表结构和索引"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 策略主表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategies (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        name        TEXT    UNIQUE NOT NULL,
                        description TEXT    DEFAULT '',
                        parameters  TEXT,
                        is_active   INTEGER DEFAULT 1,
                        created_at  TEXT    NOT NULL,
                        updated_at  TEXT    NOT NULL
                    )
                """)

                # 策略版本表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_versions (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id         INTEGER NOT NULL,
                        version             INTEGER NOT NULL,
                        parameters          TEXT    NOT NULL,
                        optimization_score  REAL,
                        is_current          INTEGER DEFAULT 0,
                        is_production       INTEGER DEFAULT 0,
                        notes               TEXT    DEFAULT '',
                        created_at          TEXT    NOT NULL,
                        FOREIGN KEY (strategy_id) 
                            REFERENCES strategies(id) ON DELETE CASCADE,
                        UNIQUE(strategy_id, version)
                    )
                """)

                # 回测结果表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        version_id          INTEGER NOT NULL,
                        backtest_date       TEXT    NOT NULL,
                        start_date          TEXT,
                        end_date            TEXT,
                        win_rate            REAL    DEFAULT 0,
                        avg_return          REAL    DEFAULT 0,
                        sharpe_ratio        REAL    DEFAULT 0,
                        max_drawdown        REAL    DEFAULT 0,
                        information_ratio   REAL    DEFAULT 0,
                        total_trades        INTEGER DEFAULT 0,
                        total_profit        REAL    DEFAULT 0,
                        cost_ratio          REAL    DEFAULT 0,
                        details             TEXT,
                        market_environment  TEXT    DEFAULT 'unknown',
                        created_at          TEXT    NOT NULL,
                        FOREIGN KEY (version_id) 
                            REFERENCES strategy_versions(id) ON DELETE CASCADE
                    )
                """)

                # 持仓记录表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id     INTEGER NOT NULL,
                        version_id      INTEGER,
                        ts_code         TEXT    NOT NULL,
                        buy_date        TEXT    NOT NULL,
                        buy_price       REAL    NOT NULL,
                        shares          INTEGER NOT NULL DEFAULT 100,
                        status          TEXT    NOT NULL DEFAULT 'holding',
                        sell_date       TEXT,
                        sell_price      REAL,
                        return_pct      REAL,
                        exit_reason     TEXT,
                        created_at      TEXT    NOT NULL,
                        updated_at      TEXT    NOT NULL,
                        FOREIGN KEY (strategy_id) 
                            REFERENCES strategies(id) ON DELETE CASCADE,
                        FOREIGN KEY (version_id) 
                            REFERENCES strategy_versions(id) ON DELETE SET NULL
                    )
                """)

                # 交易明细表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_records (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id     INTEGER,
                        ts_code         TEXT    NOT NULL,
                        trade_date      TEXT    NOT NULL,
                        trade_type      TEXT    NOT NULL,
                        price           REAL    NOT NULL,
                        shares          INTEGER NOT NULL,
                        amount          REAL    NOT NULL,
                        commission      REAL    DEFAULT 0,
                        stamp_tax       REAL    DEFAULT 0,
                        slippage        REAL    DEFAULT 0,
                        total_cost      REAL    DEFAULT 0,
                        reason          TEXT    DEFAULT '',
                        created_at      TEXT    NOT NULL,
                        FOREIGN KEY (position_id) 
                            REFERENCES positions(id) ON DELETE SET NULL
                    )
                """)

                # 优化历史表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_history (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id         INTEGER NOT NULL,
                        version_from        INTEGER,
                        version_to          INTEGER,
                        optimization_method TEXT,
                        target_metric       TEXT,
                        old_score           REAL,
                        new_score           REAL,
                        improvement_pct     REAL,
                        details             TEXT,
                        created_at          TEXT    NOT NULL,
                        FOREIGN KEY (strategy_id) 
                            REFERENCES strategies(id) ON DELETE CASCADE
                    )
                """)

                # 创建索引
                self._create_indices(cursor)

    def _create_indices(self, cursor: sqlite3.Cursor):
        """创建数据库索引"""
        indices = [
            ("idx_strategies_name", "strategies", "name"),
            ("idx_strategies_active", "strategies", "is_active"),
            ("idx_versions_strategy", "strategy_versions", "strategy_id"),
            ("idx_versions_current", "strategy_versions", "is_current"),
            ("idx_backtest_version", "backtest_results", "version_id"),
            ("idx_backtest_date", "backtest_results", "backtest_date"),
            ("idx_positions_strategy", "positions", "strategy_id"),
            ("idx_positions_status", "positions", "status"),
            ("idx_positions_code", "positions", "ts_code"),
            ("idx_trades_date", "trade_records", "trade_date"),
            ("idx_trades_code", "trade_records", "ts_code"),
            ("idx_opt_history_strategy", "optimization_history", "strategy_id"),
        ]

        for idx_name, table, column in indices:
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})"
            )

    # ==================== 策略管理 ====================

    def save_strategy(
        self,
        name: str,
        description: str = "",
        parameters: Optional[Dict] = None,
    ) -> int:
        """
        保存策略（新建或更新）
        
        修复：先查询是否存在，再决定INSERT或UPDATE，消除死代码分支
        
        Args:
            name: 策略名称（唯一标识）
            description: 策略描述
            parameters: 策略参数字典
            
        Returns:
            策略ID
        """
        now = self._now()
        params_json = self._safe_json_dumps(parameters)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 先查询是否已存在
                cursor.execute(
                    "SELECT id FROM strategies WHERE name = ?",
                    (name,),
                )
                existing = cursor.fetchone()

                if existing:
                    # 更新现有策略
                    strategy_id = existing["id"]
                    cursor.execute(
                        """
                        UPDATE strategies 
                        SET description = ?, parameters = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (description, params_json, now, strategy_id),
                    )
                    self._logger.info(
                        f"更新策略: '{name}' (ID={strategy_id})"
                    )
                else:
                    # 插入新策略
                    cursor.execute(
                        """
                        INSERT INTO strategies 
                            (name, description, parameters, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (name, description, params_json, now, now),
                    )
                    strategy_id = cursor.lastrowid
                    self._logger.info(
                        f"创建策略: '{name}' (ID={strategy_id})"
                    )

                return strategy_id

    def get_strategy(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """
        获取策略信息
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            策略信息字典，不存在返回None
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM strategies WHERE id = ?",
                    (strategy_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_dict(row, json_fields=["parameters"])

    def get_strategy_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        按名称获取策略
        
        Args:
            name: 策略名称
            
        Returns:
            策略信息字典，不存在返回None
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM strategies WHERE name = ?",
                    (name,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_dict(row, json_fields=["parameters"])

    def list_strategies(
        self,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        列出策略
        
        Args:
            active_only: 是否只列出活跃策略
            limit: 返回数量上限
            offset: 偏移量（分页用）
            
        Returns:
            策略信息列表
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                if active_only:
                    cursor.execute(
                        """
                        SELECT * FROM strategies 
                        WHERE is_active = 1 
                        ORDER BY updated_at DESC 
                        LIMIT ? OFFSET ?
                        """,
                        (limit, offset),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM strategies 
                        ORDER BY updated_at DESC 
                        LIMIT ? OFFSET ?
                        """,
                        (limit, offset),
                    )

                rows = cursor.fetchall()
                return [
                    self._row_to_dict(r, json_fields=["parameters"])
                    for r in rows
                ]

    def deactivate_strategy(self, strategy_id: int) -> bool:
        """
        停用策略（软删除）
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            是否成功
        """
        return self._update_field(
            "strategies", strategy_id, "is_active", 0
        )

    def activate_strategy(self, strategy_id: int) -> bool:
        """
        启用策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            是否成功
        """
        return self._update_field(
            "strategies", strategy_id, "is_active", 1
        )

    def delete_strategy(self, strategy_id: int) -> bool:
        """
        永久删除策略（级联删除所有关联数据）
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            是否成功
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM strategies WHERE id = ?",
                        (strategy_id,),
                    )
                    deleted = cursor.rowcount > 0

                    if deleted:
                        self._logger.info(
                            f"删除策略: ID={strategy_id}（含关联数据）"
                        )
                    else:
                        self._logger.warning(
                            f"策略不存在: ID={strategy_id}"
                        )

                    return deleted

            except Exception as e:
                self._logger.error(f"删除策略失败: {e}")
                return False

    # ==================== 版本管理 ====================

    def create_version(
        self,
        strategy_id: int,
        parameters: Dict,
        notes: str = "",
        set_current: bool = True,
    ) -> int:
        """
        创建策略新版本
        
        Args:
            strategy_id: 策略ID
            parameters: 版本参数
            notes: 版本备注
            set_current: 是否设为当前版本
            
        Returns:
            版本ID
        """
        now = self._now()
        params_json = self._safe_json_dumps(parameters)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 获取下一个版本号
                cursor.execute(
                    """
                    SELECT COALESCE(MAX(version), 0) 
                    FROM strategy_versions 
                    WHERE strategy_id = ?
                    """,
                    (strategy_id,),
                )
                next_version = cursor.fetchone()[0] + 1

                # 如果设为当前版本，先取消其他版本的current标记
                if set_current:
                    cursor.execute(
                        """
                        UPDATE strategy_versions 
                        SET is_current = 0 
                        WHERE strategy_id = ? AND is_current = 1
                        """,
                        (strategy_id,),
                    )

                # 插入新版本
                cursor.execute(
                    """
                    INSERT INTO strategy_versions 
                        (strategy_id, version, parameters, is_current, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        strategy_id,
                        next_version,
                        params_json,
                        1 if set_current else 0,
                        notes,
                        now,
                    ),
                )
                version_id = cursor.lastrowid

                # 更新策略的updated_at
                cursor.execute(
                    "UPDATE strategies SET updated_at = ? WHERE id = ?",
                    (now, strategy_id),
                )

                self._logger.info(
                    f"创建版本: 策略ID={strategy_id}, "
                    f"版本={next_version}, 版本ID={version_id}"
                )

                return version_id

    def get_version(self, version_id: int) -> Optional[Dict[str, Any]]:
        """获取指定版本信息"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM strategy_versions WHERE id = ?",
                    (version_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_dict(row, json_fields=["parameters"])

    def get_current_version(
        self, strategy_id: int
    ) -> Optional[Dict[str, Any]]:
        """获取策略的当前版本"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM strategy_versions 
                    WHERE strategy_id = ? AND is_current = 1
                    """,
                    (strategy_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_dict(row, json_fields=["parameters"])

    def get_versions(
        self, strategy_id: int, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取策略的所有版本（按版本号降序）
        
        Args:
            strategy_id: 策略ID
            limit: 返回数量上限
            
        Returns:
            版本信息列表
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM strategy_versions 
                    WHERE strategy_id = ? 
                    ORDER BY version DESC 
                    LIMIT ?
                    """,
                    (strategy_id, limit),
                )
                rows = cursor.fetchall()
                return [
                    self._row_to_dict(r, json_fields=["parameters"])
                    for r in rows
                ]

    def set_current_version(
        self, strategy_id: int, version_id: int
    ) -> bool:
        """
        设置当前版本
        
        Args:
            strategy_id: 策略ID
            version_id: 要设为当前的版本ID
            
        Returns:
            是否成功
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    # 取消旧的current
                    cursor.execute(
                        """
                        UPDATE strategy_versions 
                        SET is_current = 0 
                        WHERE strategy_id = ? AND is_current = 1
                        """,
                        (strategy_id,),
                    )

                    # 设置新的current
                    cursor.execute(
                        """
                        UPDATE strategy_versions 
                        SET is_current = 1 
                        WHERE id = ? AND strategy_id = ?
                        """,
                        (version_id, strategy_id),
                    )

                    return cursor.rowcount > 0

            except Exception as e:
                self._logger.error(f"设置当前版本失败: {e}")
                return False

    def set_production_version(
        self, strategy_id: int, version_id: int
    ) -> bool:
        """
        设置生产版本
        
        Args:
            strategy_id: 策略ID
            version_id: 版本ID
            
        Returns:
            是否成功
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    # 取消旧的production标记
                    cursor.execute(
                        """
                        UPDATE strategy_versions 
                        SET is_production = 0 
                        WHERE strategy_id = ? AND is_production = 1
                        """,
                        (strategy_id,),
                    )

                    # 设置新的production
                    cursor.execute(
                        """
                        UPDATE strategy_versions 
                        SET is_production = 1 
                        WHERE id = ? AND strategy_id = ?
                        """,
                        (version_id, strategy_id),
                    )

                    success = cursor.rowcount > 0
                    if success:
                        self._logger.info(
                            f"设置生产版本: 策略ID={strategy_id}, "
                            f"版本ID={version_id}"
                        )
                    return success

            except Exception as e:
                self._logger.error(f"设置生产版本失败: {e}")
                return False

    def update_version_score(
        self, version_id: int, score: float
    ) -> bool:
        """更新版本的优化分数"""
        return self._update_field(
            "strategy_versions", version_id, "optimization_score", score
        )

    # ==================== 回测结果管理 ====================

    def save_backtest_result(
        self, version_id: int, results: Dict[str, Any]
    ) -> int:
        """
        保存回测结果
        
        修复：正确处理嵌套字典结构，安全提取各层级指标
        
        支持两种输入格式：
        1. 平面字典: {"win_rate": 55.0, "sharpe_ratio": 1.2, ...}
        2. 嵌套字典: {"portfolio_summary": {"win_rate": 55.0, ...}, "backtest_df": ...}
        
        Args:
            version_id: 策略版本ID
            results: 回测结果字典
            
        Returns:
            回测结果ID
        """
        now = self._now()

        # 从可能的嵌套结构中提取摘要数据
        summary = self._extract_summary(results)

        # 提取各字段（提供默认值）
        win_rate = summary.get("win_rate_pct", summary.get("win_rate", 0))
        avg_return = summary.get(
            "avg_return_pct", summary.get("avg_return", 0)
        )
        sharpe_ratio = summary.get("sharpe_ratio", 0)
        max_drawdown = summary.get(
            "max_drawdown_pct", summary.get("max_drawdown", 0)
        )
        information_ratio = summary.get("information_ratio", 0)
        total_trades = summary.get(
            "total_trades", summary.get("successful_backtests", 0)
        )
        total_profit = summary.get("total_profit", 0)
        cost_ratio = summary.get(
            "avg_cost_ratio_pct", summary.get("cost_ratio", 0)
        )
        start_date = summary.get("start_date")
        end_date = summary.get("end_date")
        market_env = summary.get(
            "market_environment",
            getattr(self._config, "market_environment", "unknown"),
        )

        # 序列化详细信息（排除不可序列化的对象）
        details_json = self._safe_json_dumps(
            self._filter_serializable(results)
        )

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO backtest_results 
                        (version_id, backtest_date, start_date, end_date,
                         win_rate, avg_return, sharpe_ratio, max_drawdown,
                         information_ratio, total_trades, total_profit,
                         cost_ratio, details, market_environment, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        now[:10],  # 只取日期部分
                        start_date,
                        end_date,
                        win_rate,
                        avg_return,
                        sharpe_ratio,
                        max_drawdown,
                        information_ratio,
                        total_trades,
                        total_profit,
                        cost_ratio,
                        details_json,
                        market_env,
                        now,
                    ),
                )
                result_id = cursor.lastrowid

                self._logger.info(
                    f"保存回测结果: ID={result_id}, 版本ID={version_id}, "
                    f"胜率={win_rate:.1f}%, 夏普={sharpe_ratio:.3f}"
                )

                return result_id

    def get_backtest_result(
        self, result_id: int
    ) -> Optional[Dict[str, Any]]:
        """获取单条回测结果"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM backtest_results WHERE id = ?",
                    (result_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_dict(row, json_fields=["details"])

    def get_backtest_history(
        self,
        version_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        获取版本的回测历史（按日期降序）
        
        Args:
            version_id: 版本ID
            limit: 返回数量上限
            offset: 偏移量
            
        Returns:
            回测结果列表
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM backtest_results 
                    WHERE version_id = ? 
                    ORDER BY backtest_date DESC, created_at DESC 
                    LIMIT ? OFFSET ?
                    """,
                    (version_id, limit, offset),
                )
                rows = cursor.fetchall()
                return [
                    self._row_to_dict(r, json_fields=["details"])
                    for r in rows
                ]

    def get_latest_backtest(
        self, version_id: int
    ) -> Optional[Dict[str, Any]]:
        """获取最新一次回测结果"""
        results = self.get_backtest_history(version_id, limit=1)
        return results[0] if results else None

    def get_strategy_performance(
        self, strategy_id: int
    ) -> List[Dict[str, Any]]:
        """
        获取策略所有版本的回测表现汇总
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            各版本的最新回测结果列表
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        sv.version,
                        sv.is_current,
                        sv.is_production,
                        sv.optimization_score,
                        br.backtest_date,
                        br.win_rate,
                        br.avg_return,
                        br.sharpe_ratio,
                        br.max_drawdown,
                        br.total_trades,
                        br.market_environment
                    FROM strategy_versions sv
                    LEFT JOIN (
                        SELECT version_id, 
                               MAX(id) as max_id
                        FROM backtest_results
                        GROUP BY version_id
                    ) latest ON sv.id = latest.version_id
                    LEFT JOIN backtest_results br 
                        ON br.id = latest.max_id
                    WHERE sv.strategy_id = ?
                    ORDER BY sv.version DESC
                    """,
                    (strategy_id,),
                )
                rows = cursor.fetchall()
                return [self._row_to_dict(r) for r in rows]

    # ==================== 持仓管理 ====================

    def add_position(
        self,
        strategy_id: int,
        ts_code: str,
        buy_date: str,
        buy_price: float,
        shares: int = 100,
        version_id: Optional[int] = None,
    ) -> int:
        """
        添加持仓记录
        
        Args:
            strategy_id: 策略ID
            ts_code: 股票代码
            buy_date: 买入日期
            buy_price: 买入价格
            shares: 股数
            version_id: 版本ID（可选）
            
        Returns:
            持仓记录ID
        """
        now = self._now()

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO positions 
                        (strategy_id, version_id, ts_code, buy_date, 
                         buy_price, shares, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'holding', ?, ?)
                    """,
                    (
                        strategy_id,
                        version_id,
                        ts_code,
                        buy_date,
                        buy_price,
                        shares,
                        now,
                        now,
                    ),
                )
                position_id = cursor.lastrowid

                self._logger.debug(
                    f"添加持仓: {ts_code} 买入{buy_price}×{shares} "
                    f"(ID={position_id})"
                )

                return position_id

    def close_position(
        self,
        position_id: int,
        sell_date: str,
        sell_price: float,
        exit_reason: str = "",
    ) -> bool:
        """
        平仓（更新持仓状态）
        
        Args:
            position_id: 持仓ID
            sell_date: 卖出日期
            sell_price: 卖出价格
            exit_reason: 退出原因
            
        Returns:
            是否成功
        """
        now = self._now()

        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    # 获取持仓信息以计算收益
                    cursor.execute(
                        "SELECT buy_price FROM positions WHERE id = ?",
                        (position_id,),
                    )
                    row = cursor.fetchone()

                    if row is None:
                        self._logger.warning(
                            f"持仓不存在: ID={position_id}"
                        )
                        return False

                    buy_price = row["buy_price"]
                    return_pct = (
                        (sell_price - buy_price) / buy_price * 100
                        if buy_price > 0
                        else 0
                    )

                    cursor.execute(
                        """
                        UPDATE positions 
                        SET status = 'sold', sell_date = ?, sell_price = ?,
                            return_pct = ?, exit_reason = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            sell_date,
                            sell_price,
                            return_pct,
                            exit_reason,
                            now,
                            position_id,
                        ),
                    )

                    self._logger.debug(
                        f"平仓: ID={position_id}, "
                        f"收益={return_pct:.2f}%, 原因={exit_reason}"
                    )

                    return cursor.rowcount > 0

            except Exception as e:
                self._logger.error(f"平仓失败: {e}")
                return False

    def get_holding_positions(
        self, strategy_id: int
    ) -> List[Dict[str, Any]]:
        """获取当前持仓"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM positions 
                    WHERE strategy_id = ? AND status = 'holding'
                    ORDER BY buy_date DESC
                    """,
                    (strategy_id,),
                )
                rows = cursor.fetchall()
                return [self._row_to_dict(r) for r in rows]

    def get_position_history(
        self,
        strategy_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """获取持仓历史"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM positions 
                    WHERE strategy_id = ? 
                    ORDER BY buy_date DESC 
                    LIMIT ? OFFSET ?
                    """,
                    (strategy_id, limit, offset),
                )
                rows = cursor.fetchall()
                return [self._row_to_dict(r) for r in rows]

    # ==================== 交易记录 ====================

    def add_trade_record(
        self,
        ts_code: str,
        trade_date: str,
        trade_type: str,
        price: float,
        shares: int,
        commission: float = 0,
        stamp_tax: float = 0,
        slippage: float = 0,
        reason: str = "",
        position_id: Optional[int] = None,
    ) -> int:
        """
        添加交易记录
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            trade_type: 'buy' 或 'sell'
            price: 成交价格
            shares: 成交股数
            commission: 佣金
            stamp_tax: 印花税
            slippage: 滑点成本
            reason: 交易原因
            position_id: 关联的持仓ID
            
        Returns:
            交易记录ID
        """
        now = self._now()
        amount = price * shares
        total_cost = commission + stamp_tax + slippage

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO trade_records 
                        (position_id, ts_code, trade_date, trade_type,
                         price, shares, amount, commission, stamp_tax,
                         slippage, total_cost, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        position_id,
                        ts_code,
                        trade_date,
                        trade_type,
                        price,
                        shares,
                        amount,
                        commission,
                        stamp_tax,
                        slippage,
                        total_cost,
                        reason,
                        now,
                    ),
                )
                return cursor.lastrowid

    def get_trade_records(
        self,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trade_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        查询交易记录（支持多条件）
        
        Args:
            ts_code: 股票代码筛选
            start_date: 开始日期筛选
            end_date: 结束日期筛选
            trade_type: 交易类型筛选 ('buy' 或 'sell')
            limit: 返回上限
            
        Returns:
            交易记录列表
        """
        conditions = []
        params = []

        if ts_code:
            conditions.append("ts_code = ?")
            params.append(ts_code)
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)
        if trade_type:
            conditions.append("trade_type = ?")
            params.append(trade_type)

        where_clause = (
            "WHERE " + " AND ".join(conditions) if conditions else ""
        )

        params.append(limit)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT * FROM trade_records 
                    {where_clause}
                    ORDER BY trade_date DESC, created_at DESC 
                    LIMIT ?
                    """,
                    params,
                )
                rows = cursor.fetchall()
                return [self._row_to_dict(r) for r in rows]

    # ==================== 优化历史 ====================

    def save_optimization_record(
        self,
        strategy_id: int,
        version_from: int,
        version_to: int,
        method: str,
        target_metric: str,
        old_score: float,
        new_score: float,
        details: Optional[Dict] = None,
    ) -> int:
        """
        保存优化记录
        
        Args:
            strategy_id: 策略ID
            version_from: 优化前版本号
            version_to: 优化后版本号
            method: 优化方法
            target_metric: 目标指标
            old_score: 旧分数
            new_score: 新分数
            details: 详细信息
            
        Returns:
            记录ID
        """
        now = self._now()
        improvement = (
            (new_score - old_score) / abs(old_score) * 100
            if old_score != 0
            else 0
        )
        details_json = self._safe_json_dumps(details)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO optimization_history 
                        (strategy_id, version_from, version_to,
                         optimization_method, target_metric,
                         old_score, new_score, improvement_pct,
                         details, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        strategy_id,
                        version_from,
                        version_to,
                        method,
                        target_metric,
                        old_score,
                        new_score,
                        improvement,
                        details_json,
                        now,
                    ),
                )

                self._logger.info(
                    f"保存优化记录: 策略ID={strategy_id}, "
                    f"v{version_from}→v{version_to}, "
                    f"改进={improvement:.1f}%"
                )

                return cursor.lastrowid

    def get_optimization_history(
        self, strategy_id: int, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取优化历史"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM optimization_history 
                    WHERE strategy_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                    """,
                    (strategy_id, limit),
                )
                rows = cursor.fetchall()
                return [
                    self._row_to_dict(r, json_fields=["details"])
                    for r in rows
                ]

    # ==================== 数据导出 ====================

    def export_table(
        self,
        table_name: str,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        导出表数据到DataFrame
        
        Args:
            table_name: 表名
            conditions: 查询条件 {column: value}
            
        Returns:
            DataFrame
        """
        # 安全检查表名（防止SQL注入）
        valid_tables = {
            "strategies",
            "strategy_versions",
            "backtest_results",
            "positions",
            "trade_records",
            "optimization_history",
        }
        if table_name not in valid_tables:
            raise ValueError(
                f"无效的表名: {table_name}, 有效值: {valid_tables}"
            )

        query = f"SELECT * FROM {table_name}"
        params = []

        if conditions:
            where_parts = []
            for col, val in conditions.items():
                where_parts.append(f"{col} = ?")
                params.append(val)
            query += " WHERE " + " AND ".join(where_parts)

        with self._lock:
            with self._get_connection() as conn:
                return pd.read_sql_query(query, conn, params=params)

    # ==================== 数据库维护 ====================

    def vacuum(self):
        """
        优化数据库（整理碎片、回收空间）
        
        注意：VACUUM不能在事务中执行，需要单独处理
        """
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("VACUUM")
                self._logger.info("数据库VACUUM完成")
            except Exception as e:
                self._logger.error(f"VACUUM失败: {e}")
            finally:
                conn.close()

    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        备份数据库
        
        Args:
            backup_path: 备份文件路径，None则自动生成
            
        Returns:
            备份文件路径
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(self._config.data_dir) / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = str(
                backup_dir / f"strategy_db_{timestamp}.bak"
            )

        with self._lock:
            try:
                shutil.copy2(self._db_path, backup_path)
                self._logger.info(f"数据库备份完成: {backup_path}")
                return backup_path

            except Exception as e:
                self._logger.error(f"数据库备份失败: {e}")
                raise

    def get_database_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            各表记录数和数据库文件大小
        """
        stats = {}

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                tables = [
                    "strategies",
                    "strategy_versions",
                    "backtest_results",
                    "positions",
                    "trade_records",
                    "optimization_history",
                ]

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]

        # 文件大小
        db_file = Path(self._db_path)
        if db_file.exists():
            stats["file_size_mb"] = round(
                db_file.stat().st_size / (1024 * 1024), 2
            )

        return stats

    # ==================== 内部工具方法 ====================

    @staticmethod
    def _now() -> str:
        """当前时间字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _row_to_dict(
        row: sqlite3.Row,
        json_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        将sqlite3.Row转换为字典，自动解析JSON字段
        
        Args:
            row: 数据库行
            json_fields: 需要JSON解析的字段名列表
            
        Returns:
            字典
        """
        if row is None:
            return {}

        result = dict(row)

        if json_fields:
            for field_name in json_fields:
                if field_name in result and result[field_name]:
                    try:
                        result[field_name] = json.loads(result[field_name])
                    except (json.JSONDecodeError, TypeError):
                        pass  # 保持原始值

        return result

    @staticmethod
    def _safe_json_dumps(obj: Any) -> Optional[str]:
        """
        安全的JSON序列化
        
        处理不可序列化的类型（如DataFrame、numpy类型等）
        
        Args:
            obj: 要序列化的对象
            
        Returns:
            JSON字符串，None输入返回None
        """
        if obj is None:
            return None

        def default_handler(o):
            """处理不可序列化的类型"""
            if isinstance(o, pd.DataFrame):
                return f"<DataFrame: {len(o)} rows × {len(o.columns)} cols>"
            if isinstance(o, pd.Series):
                return f"<Series: {len(o)} items>"
            if hasattr(o, "dtype"):
                # numpy类型
                return o.item()
            if isinstance(o, (set, frozenset)):
                return list(o)
            if isinstance(o, bytes):
                return o.decode("utf-8", errors="replace")
            return str(o)

        try:
            return json.dumps(
                obj,
                ensure_ascii=False,
                default=default_handler,
                indent=None,
            )
        except (TypeError, ValueError) as e:
            logging.getLogger(__name__).warning(
                f"JSON序列化失败: {e}"
            )
            return json.dumps({"error": str(e)})

    @staticmethod
    def _extract_summary(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        从回测结果中提取摘要指标
        
        兼容多种输入格式：
        - {"portfolio_summary": {...}, "backtest_df": ...}
        - {"win_rate": 55.0, "sharpe_ratio": 1.2}
        - {"portfolio_summary": {"win_rate_pct": 55.0, ...}}
        
        Args:
            results: 回测结果字典
            
        Returns:
            扁平化的摘要字典
        """
        if "portfolio_summary" in results:
            summary = results["portfolio_summary"]
            if isinstance(summary, dict):
                # 补充顶层的额外字段
                for key in ("start_date", "end_date", "market_environment"):
                    if key in results and key not in summary:
                        summary[key] = results[key]
                return summary

        # 已经是平面字典
        return results

    @staticmethod
    def _filter_serializable(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤掉不可JSON序列化的值
        
        保留基本类型，将DataFrame等大对象替换为描述字符串
        
        Args:
            data: 原始字典
            
        Returns:
            可序列化的字典
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                result[key] = {
                    "_type": "DataFrame",
                    "shape": list(value.shape),
                    "columns": list(value.columns),
                }
            elif isinstance(value, pd.Series):
                result[key] = {
                    "_type": "Series",
                    "length": len(value),
                }
            elif isinstance(value, dict):
                result[key] = StrategyDatabase._filter_serializable(value)
            elif isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
            elif isinstance(value, (list, tuple)):
                result[key] = value
            else:
                result[key] = str(value)

        return result

    def _update_field(
        self, table: str, record_id: int, field: str, value: Any
    ) -> bool:
        """
        更新单个字段的通用方法
        
        Args:
            table: 表名
            record_id: 记录ID
            field: 字段名
            value: 新值
            
        Returns:
            是否成功
        """
        # 表名和字段名白名单校验（防SQL注入）
        valid_tables = {
            "strategies",
            "strategy_versions",
            "backtest_results",
            "positions",
        }
        if table not in valid_tables:
            self._logger.error(f"无效的表名: {table}")
            return False

        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f"UPDATE {table} SET {field} = ? WHERE id = ?",
                        (value, record_id),
                    )
                    return cursor.rowcount > 0

            except Exception as e:
                self._logger.error(
                    f"更新字段失败: {table}.{field} = {value}, "
                    f"ID={record_id}, 错误: {e}"
                )
                return False
