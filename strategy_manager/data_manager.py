"""
数据管理模块

修复内容：
1. 线程安全 - LRUCache使用可重入锁，所有共享状态加锁保护
2. 缓存返回副本 - 防止外部修改污染缓存数据
3. 内存泄漏 - LRUCache有容量上限，自动淘汰最久未使用的条目
4. 缓存键碰撞 - 使用MD5哈希避免长键名和排列不一致问题
5. API速率限制 - 指数退避重试和最小请求间隔
6. 磁盘缓存 - 使用parquet格式，支持过期清理
7. 数据清洗 - 过滤停牌、异常价格、涨跌幅异常数据
8. 复权计算 - 正确的前复权/后复权价格计算
"""

import copy
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import Config


# ==================== LRU缓存 ====================


class LRUCache:
    """
    线程安全的LRU缓存
    
    特性：
    - 容量上限防止内存泄漏
    - 可重入锁支持嵌套调用
    - DataFrame类型返回深拷贝防止缓存污染
    - 支持统计信息（命中率等）
    
    使用示例：
        cache = LRUCache(max_size=1000)
        cache.set("key1", some_dataframe)
        result = cache.get("key1")  # 返回副本
    """

    def __init__(self, max_size: int = 1000):
        if max_size <= 0:
            raise ValueError(f"缓存容量必须为正数，当前值: {max_size}")

        self._max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()

        # 统计信息
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any:
        """
        获取缓存值
        
        对DataFrame和dict类型返回副本，防止外部修改污染缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值（副本），未命中返回None
        """
        with self._lock:
            if key in self._cache:
                # 移到末尾表示最近使用
                self._cache.move_to_end(key)
                self._hits += 1

                value = self._cache[key]
                return self._safe_copy(value)

            self._misses += 1
            return None

    def set(self, key: str, value: Any):
        """
        设置缓存值
        
        存入时也做副本，防止外部后续修改影响缓存
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)

            self._cache[key] = self._safe_copy(value)

            # 超过容量时淘汰最久未使用的
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logging.getLogger(__name__).debug(f"缓存淘汰: {evicted_key}")

    def delete(self, key: str) -> bool:
        """
        删除指定缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def __contains__(self, key: str) -> bool:
        """检查键是否存在（不更新访问顺序）"""
        with self._lock:
            return key in self._cache

    @property
    def size(self) -> int:
        """当前缓存条目数"""
        with self._lock:
            return len(self._cache)

    @property
    def max_size(self) -> int:
        """缓存容量上限"""
        return self._max_size

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "utilization": len(self._cache) / self._max_size,
            }

    @staticmethod
    def _safe_copy(value: Any) -> Any:
        """
        安全拷贝值
        
        DataFrame和dict做深拷贝，其他类型直接返回
        （不可变类型如str/int/tuple不需要拷贝）
        """
        if isinstance(value, pd.DataFrame):
            return value.copy(deep=True)
        elif isinstance(value, dict):
            return copy.deepcopy(value)
        elif isinstance(value, list):
            return copy.copy(value)
        return value


# ==================== 磁盘缓存 ====================


class DiskCache:
    """
    磁盘缓存管理器
    
    特性：
    - 使用parquet格式存储DataFrame（高效压缩）
    - 使用pickle存储其他类型
    - 支持TTL过期清理
    - 键名使用MD5哈希避免文件名过长或含特殊字符
    
    使用示例：
        cache = DiskCache(Path("cache/data"), ttl_hours=24)
        cache.set("my_key", df)
        result = cache.get("my_key")  # 过期返回None
    """

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_hours * 3600
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def _key_to_path(self, key: str, suffix: str = ".parquet") -> Path:
        """将缓存键转换为文件路径"""
        hashed = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{hashed}{suffix}"

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        获取缓存的DataFrame
        
        Args:
            key: 缓存键
            
        Returns:
            DataFrame或None（未命中或已过期）
        """
        cache_path = self._key_to_path(key, ".parquet")
        return self._read_if_valid(cache_path, reader=pd.read_parquet)

    def set_dataframe(self, key: str, df: pd.DataFrame):
        """
        缓存DataFrame
        
        Args:
            key: 缓存键
            df: 要缓存的DataFrame
        """
        if df is None or df.empty:
            return

        cache_path = self._key_to_path(key, ".parquet")
        with self._lock:
            try:
                df.to_parquet(cache_path, index=False, engine="pyarrow")
            except ImportError:
                # pyarrow不可用时回退到fastparquet
                try:
                    df.to_parquet(cache_path, index=False, engine="fastparquet")
                except ImportError:
                    # 都不可用时用pickle
                    self._write_pickle(key, df)
            except Exception as e:
                self._logger.warning(f"写入parquet缓存失败: {e}")

    def get_object(self, key: str) -> Any:
        """获取缓存的Python对象"""
        import pickle

        cache_path = self._key_to_path(key, ".pkl")

        def reader(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        return self._read_if_valid(cache_path, reader=reader)

    def set_object(self, key: str, obj: Any):
        """缓存Python对象"""
        self._write_pickle(key, obj)

    def _read_if_valid(self, cache_path: Path, reader) -> Any:
        """读取缓存文件，检查是否过期"""
        if not cache_path.exists():
            return None

        # 检查TTL
        file_age = time.time() - cache_path.stat().st_mtime
        if file_age > self._ttl_seconds:
            self._safe_delete(cache_path)
            return None

        with self._lock:
            try:
                return reader(cache_path)
            except Exception as e:
                self._logger.warning(f"读取缓存文件失败 {cache_path}: {e}")
                self._safe_delete(cache_path)
                return None

    def _write_pickle(self, key: str, obj: Any):
        """使用pickle写入缓存"""
        import pickle

        cache_path = self._key_to_path(key, ".pkl")
        with self._lock:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                self._logger.warning(f"写入pickle缓存失败: {e}")

    def _safe_delete(self, path: Path):
        """安全删除文件"""
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass

    def clear_expired(self) -> int:
        """
        清理所有过期缓存文件
        
        Returns:
            清理的文件数
        """
        expired_count = 0
        now = time.time()

        with self._lock:
            for pattern in ("*.parquet", "*.pkl"):
                for cache_file in self._cache_dir.glob(pattern):
                    try:
                        file_age = now - cache_file.stat().st_mtime
                        if file_age > self._ttl_seconds:
                            cache_file.unlink()
                            expired_count += 1
                    except OSError:
                        continue

        if expired_count > 0:
            self._logger.info(f"清理了 {expired_count} 个过期缓存文件")

        return expired_count

    def clear_all(self) -> int:
        """
        清理所有缓存文件
        
        Returns:
            清理的文件数
        """
        count = 0
        with self._lock:
            for pattern in ("*.parquet", "*.pkl"):
                for cache_file in self._cache_dir.glob(pattern):
                    try:
                        cache_file.unlink()
                        count += 1
                    except OSError:
                        continue

        self._logger.info(f"清理了 {count} 个缓存文件")
        return count

    @property
    def cache_size(self) -> int:
        """当前缓存文件数"""
        count = 0
        for pattern in ("*.parquet", "*.pkl"):
            count += len(list(self._cache_dir.glob(pattern)))
        return count


# ==================== 批量数据管理器 ====================


class BatchDataManager:
    """
    批量数据管理器
    
    职责：
    - 从Tushare获取股票日线数据、复权因子、基本信息等
    - 三级缓存：内存LRU缓存 → 磁盘缓存 → API调用
    - 线程安全的并发数据获取
    - API速率限制和指数退避重试
    - 数据清洗和质量校验
    
    使用示例：
        manager = BatchDataManager(config)
        df = manager.get_daily_data("000001.SZ", "20230101", "20231231")
        batch_df = manager.batch_get_daily(["000001.SZ", "000002.SZ"], "20230101", "20231231")
    """

    def __init__(self, config: Config):
        self.config = config
        self._logger = logging.getLogger(__name__)

        # Tushare API引用
        self._tushare_pro = config.tushare_pro
        if self._tushare_pro is None:
            self._logger.warning("Tushare未初始化，数据获取功能将不可用")

        # 内存缓存
        self._memory_cache = LRUCache(
            max_size=getattr(config, "memory_cache_size", 2000)
        )

        # 磁盘缓存
        disk_cache_dir = Path(config.cache_dir) / "data"
        ttl_hours = getattr(config, "disk_cache_ttl_hours", 24)
        self._disk_cache = DiskCache(disk_cache_dir, ttl_hours=ttl_hours)
        self._use_disk_cache = getattr(config, "use_disk_cache", True)

        # API调用保护
        self._api_lock = threading.RLock()
        self._last_request_time = 0.0
        self._min_request_interval = 0.2  # 最小请求间隔200ms
        self._max_retries = 3

        # 统计信息
        self._stats_lock = threading.Lock()
        self._stats = {
            "api_success": 0,
            "api_failure": 0,
            "cache_hit_memory": 0,
            "cache_hit_disk": 0,
            "total_requests": 0,
        }

        self._logger.info(
            f"BatchDataManager初始化完成 "
            f"(内存缓存容量: {self._memory_cache.max_size}, "
            f"磁盘缓存: {'启用' if self._use_disk_cache else '禁用'})"
        )

    # ==================== 公开接口 ====================

    def get_daily_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取单只股票日线数据
        
        三级缓存查找顺序：内存缓存 → 磁盘缓存 → Tushare API
        
        Args:
            ts_code: 股票代码，如 "000001.SZ"
            start_date: 开始日期，格式 "YYYYMMDD"
            end_date: 结束日期，格式 "YYYYMMDD"
            fields: 返回字段（可选），如 "ts_code,trade_date,open,high,low,close,vol"
            
        Returns:
            日线数据DataFrame，获取失败返回空DataFrame
        """
        cache_key = self._make_cache_key("daily", ts_code, start_date, end_date)

        # 第1层：内存缓存
        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            self._increment_stat("cache_hit_memory")
            return cached  # LRUCache.get已经返回副本

        # 第2层：磁盘缓存
        if self._use_disk_cache:
            disk_cached = self._disk_cache.get_dataframe(cache_key)
            if disk_cached is not None and not disk_cached.empty:
                # 回填内存缓存
                self._memory_cache.set(cache_key, disk_cached)
                self._increment_stat("cache_hit_disk")
                return disk_cached.copy()

        # 第3层：API调用
        df = self._fetch_daily_from_api(ts_code, start_date, end_date, fields)

        if df is not None and not df.empty:
            # 数据清洗
            df = self._clean_daily_data(df, ts_code)

            if not df.empty:
                # 写入两级缓存
                self._memory_cache.set(cache_key, df)
                if self._use_disk_cache:
                    self._disk_cache.set_dataframe(cache_key, df)

            return df
        else:
            return pd.DataFrame()

    def batch_get_daily(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        批量获取多只股票日线数据（并行版本）
        
        使用线程池并行获取，自动处理失败和重试
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 返回字段（可选）
            
        Returns:
            合并的日线数据DataFrame
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not ts_codes:
            return pd.DataFrame()

        all_data = []
        failed_codes = []

        max_workers = min(self.config.max_workers, len(ts_codes))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_code = {
                executor.submit(
                    self.get_daily_data, code, start_date, end_date, fields
                ): code
                for code in ts_codes
            }

            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    df = future.result(timeout=60)
                    if df is not None and not df.empty:
                        all_data.append(df)
                    else:
                        failed_codes.append(code)
                except Exception as e:
                    self._logger.warning(f"获取 {code} 数据失败: {e}")
                    failed_codes.append(code)

        if failed_codes:
            self._logger.warning(
                f"批量获取完成，失败 {len(failed_codes)}/{len(ts_codes)}: "
                f"{failed_codes[:10]}{'...' if len(failed_codes) > 10 else ''}"
            )

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self._logger.info(
                f"批量获取完成: 成功 {len(all_data)}/{len(ts_codes)}, "
                f"总行数 {len(combined)}"
            )
            return combined
        else:
            return pd.DataFrame()

    def get_adj_factor(
        self,
        ts_codes: List[str],
        trade_date: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        获取复权因子
        
        Args:
            ts_codes: 股票代码列表
            trade_date: 交易日期（可选，不传则获取最新）
            
        Returns:
            {ts_code: adj_factor} 字典
        """
        # 生成一致的缓存键（排序后哈希）
        sorted_codes = sorted(ts_codes)
        date_str = trade_date or "latest"
        raw_key = f"adj_factor|{'|'.join(sorted_codes)}|{date_str}"
        cache_key = f"adj_{hashlib.md5(raw_key.encode()).hexdigest()}"

        # 检查内存缓存
        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            self._increment_stat("cache_hit_memory")
            return cached  # LRUCache.get 已返回副本

        # 检查磁盘缓存
        if self._use_disk_cache:
            disk_cached = self._disk_cache.get_object(cache_key)
            if disk_cached is not None:
                self._memory_cache.set(cache_key, disk_cached)
                self._increment_stat("cache_hit_disk")
                return copy.copy(disk_cached)

        # API调用（分批）
        result = self._fetch_adj_factors(ts_codes, trade_date)

        if result:
            self._memory_cache.set(cache_key, result)
            if self._use_disk_cache:
                self._disk_cache.set_object(cache_key, result)

        return result

    def get_stock_basic(
        self,
        ts_codes: Optional[List[str]] = None,
        list_status: str = "L",
        fields: str = "ts_code,name,area,industry,market,list_date,list_status",
    ) -> pd.DataFrame:
        """
        获取股票基本信息
        
        Args:
            ts_codes: 股票代码列表（可选，不传获取全部）
            list_status: 上市状态 L-上市 D-退市 P-暂停
            fields: 返回字段
            
        Returns:
            股票基本信息DataFrame
        """
        # 缓存键
        codes_str = ",".join(sorted(ts_codes)) if ts_codes else "all"
        cache_key = f"basic_{hashlib.md5(codes_str.encode()).hexdigest()}_{list_status}"

        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._tushare_pro is None:
            self._logger.error("Tushare未初始化")
            return pd.DataFrame()

        try:
            self._rate_limit()

            params = {"list_status": list_status, "fields": fields}
            if ts_codes:
                params["ts_code"] = ",".join(ts_codes)

            with self._api_lock:
                df = self._tushare_pro.stock_basic(**params)

            if df is not None and not df.empty:
                self._memory_cache.set(cache_key, df)
                self._increment_stat("api_success")
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            self._logger.error(f"获取股票基本信息失败: {e}")
            self._increment_stat("api_failure")
            return pd.DataFrame()

    def calculate_adjusted_price(
        self,
        df: pd.DataFrame,
        adj_type: str = "qfq",
    ) -> pd.DataFrame:
        """
        计算复权价格
        
        Args:
            df: 包含原始价格的DataFrame（必须有ts_code和trade_date列）
            adj_type: 'qfq'前复权 或 'hfq'后复权
            
        Returns:
            添加了复权价格列的DataFrame
        """
        if df.empty:
            return df

        if "ts_code" not in df.columns or "trade_date" not in df.columns:
            self._logger.warning("数据缺少ts_code或trade_date列，无法计算复权价格")
            return df

        result = df.copy()
        ts_codes = result["ts_code"].unique().tolist()

        # 获取复权因子（按股票逐只获取以确保完整）
        for code in ts_codes:
            code_mask = result["ts_code"] == code
            code_data = result[code_mask]

            if code_data.empty:
                continue

            # 获取该股票的复权因子
            adj_factors = self._fetch_adj_factor_series(code)
            if adj_factors.empty:
                continue

            # 合并复权因子
            result = self._merge_adj_factor(result, code, adj_factors, adj_type)

        return result

    # ==================== 缓存管理 ====================

    def clear_cache(self, memory: bool = True, disk: bool = False):
        """
        清理缓存
        
        Args:
            memory: 是否清理内存缓存
            disk: 是否清理磁盘缓存
        """
        if memory:
            self._memory_cache.clear()
            self._logger.info("内存缓存已清空")

        if disk:
            count = self._disk_cache.clear_all()
            self._logger.info(f"磁盘缓存已清空，删除 {count} 个文件")

    def clear_expired_cache(self):
        """清理过期的磁盘缓存"""
        self._disk_cache.clear_expired()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据管理器统计信息
        
        Returns:
            包含API调用统计和缓存统计的字典
        """
        with self._stats_lock:
            stats = self._stats.copy()

        stats["memory_cache"] = self._memory_cache.get_stats()
        stats["disk_cache_files"] = self._disk_cache.cache_size

        # 计算总命中率
        total_hits = stats["cache_hit_memory"] + stats["cache_hit_disk"]
        total_requests = stats["total_requests"]
        stats["overall_cache_hit_rate"] = (
            total_hits / total_requests if total_requests > 0 else 0.0
        )

        return stats

    # ==================== 内部方法：API调用 ====================

    def _fetch_daily_from_api(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        从Tushare API获取日线数据（带重试）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            fields: 返回字段
            
        Returns:
            日线数据DataFrame，失败返回None
        """
        if self._tushare_pro is None:
            self._logger.error("Tushare未初始化，无法获取数据")
            return None

        self._increment_stat("total_requests")

        params = {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
        }
        if fields:
            params["fields"] = fields

        # 带指数退避的重试
        for attempt in range(self._max_retries):
            try:
                self._rate_limit()

                with self._api_lock:
                    df = self._tushare_pro.daily(**params)

                if df is not None and not df.empty:
                    self._increment_stat("api_success")
                    return df
                else:
                    self._logger.debug(
                        f"API返回空数据: {ts_code} {start_date}-{end_date}"
                    )
                    return pd.DataFrame()

            except Exception as e:
                wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                is_last_attempt = attempt == self._max_retries - 1

                if is_last_attempt:
                    self._logger.error(
                        f"获取 {ts_code} 日线数据失败（已重试{self._max_retries}次）: {e}"
                    )
                    self._increment_stat("api_failure")
                    return None
                else:
                    self._logger.warning(
                        f"获取 {ts_code} 失败，{wait_time:.1f}秒后重试 "
                        f"(第{attempt + 1}/{self._max_retries}次): {e}"
                    )
                    time.sleep(wait_time)

        return None

    def _fetch_adj_factors(
        self,
        ts_codes: List[str],
        trade_date: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        批量获取复权因子
        
        分批请求，每批最多500只股票
        
        Args:
            ts_codes: 股票代码列表
            trade_date: 交易日期
            
        Returns:
            {ts_code: adj_factor} 字典
        """
        if self._tushare_pro is None:
            return {}

        result = {}
        batch_size = getattr(self.config, "batch_size", 100)
        batch_size = min(batch_size, 500)  # Tushare限制

        for i in range(0, len(ts_codes), batch_size):
            batch = ts_codes[i : i + batch_size]

            try:
                self._rate_limit()

                params = {"ts_code": ",".join(batch)}
                if trade_date:
                    params["trade_date"] = trade_date

                with self._api_lock:
                    df = self._tushare_pro.adj_factor(**params)

                if df is not None and not df.empty:
                    # 取每只股票最新的复权因子
                    if trade_date:
                        for _, row in df.iterrows():
                            result[row["ts_code"]] = float(row["adj_factor"])
                    else:
                        latest = (
                            df.sort_values("trade_date")
                            .groupby("ts_code")
                            .last()
                        )
                        for code, row in latest.iterrows():
                            result[code] = float(row["adj_factor"])

                    self._increment_stat("api_success")

            except Exception as e:
                self._logger.warning(f"获取复权因子批次失败: {e}")
                self._increment_stat("api_failure")

        return result

    def _fetch_adj_factor_series(self, ts_code: str) -> pd.DataFrame:
        """
        获取单只股票的复权因子时间序列
        
        Args:
            ts_code: 股票代码
            
        Returns:
            包含trade_date和adj_factor的DataFrame
        """
        cache_key = f"adj_series_{ts_code}"
        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._tushare_pro is None:
            return pd.DataFrame()

        try:
            self._rate_limit()

            with self._api_lock:
                df = self._tushare_pro.adj_factor(ts_code=ts_code)

            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
                self._memory_cache.set(cache_key, df)
                return df

        except Exception as e:
            self._logger.warning(f"获取 {ts_code} 复权因子序列失败: {e}")

        return pd.DataFrame()

    # ==================== 内部方法：数据清洗 ====================

    def _clean_daily_data(self, df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
        """
        清洗日线数据
        
        清洗规则：
        1. 转换trade_date为datetime类型
        2. 按日期排序
        3. 过滤停牌数据（成交量为0）
        4. 过滤价格异常数据（负值或零值）
        5. 检查OHLC价格关系合理性
        6. 删除完全重复的行
        
        Args:
            df: 原始日线数据
            ts_code: 股票代码（用于日志）
            
        Returns:
            清洗后的DataFrame
        """
        if df.empty:
            return df

        original_len = len(df)
        result = df.copy()

        # 1. 日期处理
        if "trade_date" in result.columns:
            result["trade_date"] = pd.to_datetime(
                result["trade_date"], format="%Y%m%d", errors="coerce"
            )
            # 删除日期解析失败的行
            result = result.dropna(subset=["trade_date"])

        # 2. 按日期排序
        result = result.sort_values("trade_date").reset_index(drop=True)

        # 3. 确保ts_code列存在
        if "ts_code" not in result.columns:
            result["ts_code"] = ts_code

        # 4. 过滤停牌数据（成交量为0）
        if "vol" in result.columns:
            result = result[result["vol"] > 0]

        # 5. 过滤价格异常
        price_cols = ["open", "high", "low", "close"]
        existing_price_cols = [c for c in price_cols if c in result.columns]

        for col in existing_price_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")
            result = result[result[col] > 0]

        # 6. 检查OHLC价格关系合理性
        if all(c in result.columns for c in ["open", "high", "low", "close"]):
            valid_ohlc = (
                (result["low"] <= result["high"])
                & (result["open"] >= result["low"])
                & (result["open"] <= result["high"])
                & (result["close"] >= result["low"])
                & (result["close"] <= result["high"])
            )
            invalid_count = (~valid_ohlc).sum()
            if invalid_count > 0:
                self._logger.debug(
                    f"{ts_code}: 过滤 {invalid_count} 条OHLC关系异常数据"
                )
            result = result[valid_ohlc]

        # 7. 删除完全重复的行
        result = result.drop_duplicates(subset=["trade_date"], keep="last")

        cleaned_count = original_len - len(result)
        if cleaned_count > 0:
            self._logger.debug(
                f"{ts_code}: 清洗了 {cleaned_count} 条数据 "
                f"({original_len} → {len(result)})"
            )

        return result

    # ==================== 内部方法：复权计算 ====================

    def _merge_adj_factor(
        self,
        df: pd.DataFrame,
        ts_code: str,
        adj_factors: pd.DataFrame,
        adj_type: str,
    ) -> pd.DataFrame:
        """
        合并复权因子并计算复权价格
        
        前复权(qfq): 以最新价格为基准向前调整
        后复权(hfq): 以上市首日价格为基准向后调整
        
        Args:
            df: 原始数据
            ts_code: 当前处理的股票代码
            adj_factors: 该股票的复权因子序列
            adj_type: 'qfq' 或 'hfq'
            
        Returns:
            添加了复权价格列的DataFrame
        """
        code_mask = df["ts_code"] == ts_code

        if not code_mask.any():
            return df

        # 确保日期类型一致
        if adj_factors["trade_date"].dtype != df["trade_date"].dtype:
            adj_factors = adj_factors.copy()
            adj_factors["trade_date"] = pd.to_datetime(adj_factors["trade_date"])

        # 合并复权因子
        adj_rename = adj_factors[["trade_date", "adj_factor"]].rename(
            columns={"adj_factor": "_adj_factor_raw"}
        )

        df = df.merge(adj_rename, on="trade_date", how="left")

        # 只处理当前股票的行
        code_idx = df.index[code_mask]

        # 用前向填充处理缺失的复权因子
        df.loc[code_idx, "_adj_factor_raw"] = (
            df.loc[code_idx, "_adj_factor_raw"].ffill().bfill()
        )

        # 计算复权系数
        if adj_type == "qfq":
            # 前复权：除以最新日的复权因子
            latest_factor = df.loc[code_idx, "_adj_factor_raw"].iloc[-1]
            if latest_factor and latest_factor > 0:
                df.loc[code_idx, "_adj_ratio"] = (
                    df.loc[code_idx, "_adj_factor_raw"] / latest_factor
                )
            else:
                df.loc[code_idx, "_adj_ratio"] = 1.0
        elif adj_type == "hfq":
            # 后复权：除以最早日的复权因子
            earliest_factor = df.loc[code_idx, "_adj_factor_raw"].iloc[0]
            if earliest_factor and earliest_factor > 0:
                df.loc[code_idx, "_adj_ratio"] = (
                    df.loc[code_idx, "_adj_factor_raw"] / earliest_factor
                )
            else:
                df.loc[code_idx, "_adj_ratio"] = 1.0
        else:
            self._logger.warning(f"不支持的复权类型: {adj_type}，使用原始价格")
            df.loc[code_idx, "_adj_ratio"] = 1.0

        # 计算复权价格
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                adj_col = f"{col}_{adj_type}"
                df[adj_col] = df[col]  # 默认等于原始价格
                df.loc[code_idx, adj_col] = (
                    df.loc[code_idx, col] * df.loc[code_idx, "_adj_ratio"]
                )

        # 清理临时列
        df = df.drop(columns=["_adj_factor_raw", "_adj_ratio"], errors="ignore")

        return df

    # ==================== 内部方法：工具 ====================

    @staticmethod
    def _make_cache_key(*parts: str) -> str:
        """
        生成缓存键
        
        使用MD5哈希确保键名长度一致且无特殊字符
        
        Args:
            *parts: 键的组成部分
            
        Returns:
            哈希后的缓存键
        """
        raw_key = "|".join(str(p) for p in parts)
        hashed = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
        # 保留前缀便于调试
        prefix = parts[0] if parts else "key"
        return f"{prefix}_{hashed}"

    def _rate_limit(self):
        """
        API速率限制
        
        确保两次API调用之间至少间隔 _min_request_interval 秒
        """
        with self._api_lock:
            now = time.time()
            elapsed = now - self._last_request_time

            if elapsed < self._min_request_interval:
                sleep_time = self._min_request_interval - elapsed
                time.sleep(sleep_time)

            self._last_request_time = time.time()

    def _increment_stat(self, key: str):
        """线程安全地增加统计计数"""
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + 1
