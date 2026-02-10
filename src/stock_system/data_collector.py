"""
行情数据采集模块（修复版）
修复内容：
1. 数据合并时做「日期对齐」严格校验
2. 修复实时行情获取逻辑错误
3. 优化缓存机制，区分静态数据和动态数据
4. 实现增量数据更新机制
"""
import os
import json
import logging
import hashlib
import pickle
import time
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import pandas as pd
import numpy as np
import tushare as ts

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketDataCollector:
    """行情数据采集器（修复版）"""
    
    def __init__(self, config_path: str = None):
        """
        初始化数据采集器
        
        Args:
            config_path: tushare配置文件路径
        """
        if config_path is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            config_path = os.path.join(workspace_path, "config/tushare_config.json")
        
        self.config = self._load_config(config_path)
        
        # API请求限制配置
        self.max_workers = self.config.get('max_workers', 5)
        self.request_timeout = self.config.get('timeout', 30)
        self.retry_count = self.config.get('retry_count', 3)
        self.rate_limit_delay = self.config.get('rate_limit_delay', 0.1)
        
        # 【v3.2 新增】实时行情请求频率限制（更严格，避免限流）
        self.realtime_rate_limit_delay = self.config.get('realtime_rate_limit_delay', 0.2)
        
        # 【v3.2 新增】实时行情一致性校验参数
        self.realtime_price_change_max = self.config.get('realtime_price_change_max', 0.20)  # 单日涨跌幅上限20%
        self.realtime_price_check_enabled = self.config.get('realtime_price_check_enabled', True)  # 是否启用一致性校验
        
        # 【v3.3 新增】API请求熔断机制配置
        self.circuit_breaker_enabled = self.config.get('circuit_breaker_enabled', True)  # 是否启用熔断机制
        self.circuit_breaker_pause_minutes = self.config.get('circuit_breaker_pause_minutes', 10)  # 熔断暂停时长（分钟）
        
        # 熔断状态管理（股票代码 -> {'fail_count': 失败次数, 'pause_until': 暂停截止时间}）
        self.circuit_breaker_status = {}
        
        # 缓存配置（修复：区分静态数据和动态数据）
        self.cache_expiry_static_hours = self.config.get('cache_expiry_static_hours', 168)  # 静态数据：7天
        self.cache_expiry_dynamic_hours = self.config.get('cache_expiry_dynamic_hours', 24)  # 动态数据：1天
        
        # 【v3.2 新增】缓存统计（命中率监控）
        self.cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'static_hits': 0,
            'dynamic_hits': 0,
            'cache_misses': 0
        }
        
        # 【v3.2 新增】自动清理配置
        self.auto_clean_enabled = self.config.get('auto_clean_enabled', True)  # 是否启用自动清理
        self.auto_clean_time = self.config.get('auto_clean_time', '03:00')  # 每日自动清理时间
        self.last_clean_date = None  # 上次清理日期
        
        # 初始化tushare和缓存目录
        self.pro = self._init_tushare()
        self.cache_dir = self._init_cache_dir()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        try:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            
            # 加载 .env 文件
            from pathlib import Path
            env_locations = [
                Path(workspace_path) / ".env",
                Path(workspace_path) / "config" / ".env",
                Path(".env")
            ]
            
            for env_file in env_locations:
                if env_file.exists():
                    with open(env_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip()
                    logger.info(f"加载环境变量成功: {env_file}")
                    break
            
            # 优先从环境变量读取token
            env_token = os.getenv('TUSHARE_TOKEN')
            
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if env_token:
                config['token'] = env_token
                logger.info(f"使用环境变量中的token")
            
            logger.info(f"加载配置文件成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _init_tushare(self):
        """初始化tushare连接"""
        try:
            token = self.config.get('token', '')
            if not token:
                logger.warning("未配置tushare token，请先在环境变量TUSHARE_TOKEN中配置")
                return None
            
            ts.set_token(token)
            pro = ts.pro_api(timeout=self.request_timeout)
            logger.info("tushare连接初始化成功")
            return pro
        except Exception as e:
            logger.error(f"初始化tushare失败: {e}")
            raise
    
    def _init_cache_dir(self) -> str:
        """初始化缓存目录"""
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        cache_dir = os.path.join(workspace_path, "assets/data/market_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"缓存目录初始化: {cache_dir}")
        return cache_dir
    
    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """生成缓存key"""
        key_str = f"{prefix}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: str, cache_type: str = 'dynamic') -> bool:
        """
        检查缓存是否有效（修复：区分静态数据和动态数据）
        
        Args:
            cache_file: 缓存文件路径
            cache_type: 缓存类型，static=静态数据，dynamic=动态数据
        """
        if not os.path.exists(cache_file):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        
        # 根据缓存类型设置不同的过期时间
        if cache_type == 'static':
            expiry_hours = self.cache_expiry_static_hours  # 静态数据：7天
        else:
            expiry_hours = self.cache_expiry_dynamic_hours  # 动态数据：1天
        
        expiry_time = datetime.now() - timedelta(hours=expiry_hours)
        
        return file_time > expiry_time
    
    def _save_pickle_cache(self, data, cache_file: str) -> bool:
        """保存pickle格式的缓存"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"保存缓存成功: {cache_file}")
            return True
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
            return False
    
    def _load_pickle_cache(self, cache_file: str, cache_type: str = 'dynamic'):
        """加载pickle格式的缓存（修复：区分缓存类型 + v3.2 新增命中率监控）"""
        try:
            # 【v3.2 新增】记录总请求数
            self.cache_stats['total_requests'] += 1
            
            if not self._is_cache_valid(cache_file, cache_type):
                logger.debug(f"缓存已过期: {cache_file}")
                # 【v3.2 新增】记录缓存未命中
                self.cache_stats['cache_misses'] += 1
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # 【v3.2 新增】记录缓存命中
            self.cache_stats['cache_hits'] += 1
            if cache_type == 'static':
                self.cache_stats['static_hits'] += 1
            else:
                self.cache_stats['dynamic_hits'] += 1
            
            logger.debug(f"加载缓存成功: {cache_file}")
            return data
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            # 【v3.2 新增】记录缓存未命中
            self.cache_stats['cache_misses'] += 1
            return None
    
    @lru_cache(maxsize=100)
    def get_stock_list(self, market: str = None, status: str = 'L', 
                       use_cache: bool = True) -> pd.DataFrame:
        """
        获取股票列表（修复：使用静态缓存）
        
        Args:
            market: 市场，SSE=上海，SZSE=深圳，None=所有
            status: 状态，L=上市，D=退市，P=暂停上市
            use_cache: 是否使用缓存
            
        Returns:
            股票列表DataFrame
        """
        # 修复：使用静态缓存（7天）
        if use_cache:
            cache_key = self._get_cache_key('stock_list', market=market, status=status)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            cached_data = self._load_pickle_cache(cache_file, cache_type='static')
            if cached_data is not None:
                logger.info(f"从缓存加载股票列表，共 {len(cached_data)} 只股票")
                return cached_data
        
        try:
            if not self.pro:
                logger.error("tushare未初始化")
                return pd.DataFrame()
            
            df = self.pro.stock_basic(exchange='', list_status=status,
                                      fields='ts_code,symbol,name,area,industry,list_date,total_mv,circ_mv')
            
            # 树形筛选第一层：市场筛选
            if market:
                if market == 'SSE':
                    df = df[df['ts_code'].str.endswith('.SH')]
                elif market == 'SZSE':
                    df = df[df['ts_code'].str.endswith('.SZ')]
            
            # 树形筛选第二层：排除ST、退市、暂停上市股票
            df = df[~df['name'].str.contains('ST|退|暂停', na=False)]
            
            # 树形筛选第三层：排除新上市股票（不足30天）
            if not df.empty and 'list_date' in df.columns:
                df['list_date'] = pd.to_datetime(df['list_date'])
                min_list_date = datetime.now() - timedelta(days=30)
                df = df[df['list_date'] < min_list_date]
            
            logger.info(f"获取股票列表成功，共 {len(df)} 只股票")
            
            # 修复：保存为静态缓存
            if use_cache:
                self._save_pickle_cache(df, cache_file)
            
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def _validate_date_alignment(self, df_daily: pd.DataFrame, 
                                df_basic: pd.DataFrame, 
                                df_adj: pd.DataFrame, 
                                df_flow: pd.DataFrame,
                                ts_code: str, 
                                auto_repair: bool = True) -> Tuple[Dict, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        【v3.2 优化】数据合并时做「日期对齐」严格校验 + 自动修复逻辑
        
        Args:
            df_daily: 日线数据
            df_basic: 每日指标数据
            df_adj: 复权因子数据
            df_flow: 资金流向数据
            ts_code: 股票代码
            auto_repair: 是否自动修复缺失数据（≤10%用前后均值填充，>50%直接放弃）
            
        Returns:
            (校验结果字典, 修复后的df_basic, 修复后的df_adj, 修复后的df_flow)
            如果校验失败（缺失>50%），返回的DataFrame为None
        """
        validation_result = {
            'is_valid': True,
            'daily_dates': set(),
            'basic_dates': set(),
            'adj_dates': set(),
            'flow_dates': set(),
            'missing_in_basic': [],
            'missing_in_adj': [],
            'missing_in_flow': [],
            'duplicate_dates': [],
            'basic_filled': 0,
            'adj_filled': 0,
            'flow_filled': 0
        }
        
        if df_daily.empty:
            validation_result['is_valid'] = False
            return validation_result, None, None, None
        
        # 获取日线数据的日期集合
        validation_result['daily_dates'] = set(df_daily['trade_date'].tolist())
        total_dates = len(validation_result['daily_dates'])
        
        # 检查 daily_basic 的日期对齐
        df_basic_repaired = df_basic.copy() if not df_basic.empty else None
        if not df_basic.empty:
            validation_result['basic_dates'] = set(df_basic['trade_date'].tolist())
            missing = validation_result['daily_dates'] - validation_result['basic_dates']
            validation_result['missing_in_basic'] = list(missing)
            missing_count = len(missing)
            missing_ratio = missing_count / total_dates if total_dates > 0 else 0
            
            # 检查重复日期
            duplicates = df_basic[df_basic.duplicated(subset=['ts_code', 'trade_date'], keep=False)]
            if not duplicates.empty:
                validation_result['duplicate_dates'].extend(
                    duplicates['trade_date'].tolist()
                )
            
            # 【v3.2 新增】自动修复逻辑
            if auto_repair and missing_count > 0:
                if missing_ratio > 0.5:
                    # 缺失超过50%，直接放弃该股票数据
                    validation_result['is_valid'] = False
                    logger.warning(f"股票 {ts_code}: daily_basic 缺失 {missing_count}/{total_dates} ({missing_ratio*100:.1f}%) > 50%，放弃该股票")
                    return validation_result, None, None, None
                elif missing_ratio <= 0.1:
                    # 缺失≤10%，用前后均值填充
                    logger.info(f"股票 {ts_code}: daily_basic 缺失 {missing_count}/{total_dates} ({missing_ratio*100:.1f}%) ≤ 10%，自动修复")
                    
                    # 为缺失的日期创建数据行
                    missing_dates = sorted(list(missing))
                    filled_rows = []
                    
                    for missing_date in missing_dates:
                        # 找到最近的前后日期
                        df_basic_sorted = df_basic.sort_values('trade_date')
                        df_basic_sorted['trade_date'] = pd.to_datetime(df_basic_sorted['trade_date'])
                        missing_date_dt = pd.to_datetime(missing_date)
                        
                        # 前一个交易日
                        prev_row = df_basic_sorted[df_basic_sorted['trade_date'] < missing_date_dt].iloc[-1:] if len(df_basic_sorted[df_basic_sorted['trade_date'] < missing_date_dt]) > 0 else None
                        
                        # 后一个交易日
                        next_row = df_basic_sorted[df_basic_sorted['trade_date'] > missing_date_dt].iloc[:1] if len(df_basic_sorted[df_basic_sorted['trade_date'] > missing_date_dt]) > 0 else None
                        
                        # 创建新行
                        new_row = {'ts_code': ts_code, 'trade_date': missing_date}
                        
                        # 用前后均值填充数值列
                        numeric_cols = df_basic.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if col == 'ts_code':
                                continue
                            
                            if prev_row is not None and next_row is not None:
                                # 前后都有，用平均值
                                new_row[col] = (prev_row[col].values[0] + next_row[col].values[0]) / 2
                            elif prev_row is not None:
                                # 只有前一个，用前一个的值
                                new_row[col] = prev_row[col].values[0]
                            elif next_row is not None:
                                # 只有后一个，用后一个的值
                                new_row[col] = next_row[col].values[0]
                            else:
                                # 都没有，用0填充
                                new_row[col] = 0
                        
                        filled_rows.append(new_row)
                    
                    if filled_rows:
                        filled_df = pd.DataFrame(filled_rows)
                        df_basic_repaired = pd.concat([df_basic_repaired, filled_df], ignore_index=True)
                        validation_result['basic_filled'] = len(filled_rows)
                        logger.info(f"股票 {ts_code}: 已填充 {len(filled_rows)} 条 daily_basic 数据")
                else:
                    # 缺失10%-50%，仅警告，不自动修复
                    logger.warning(f"股票 {ts_code}: daily_basic 缺失 {missing_count}/{total_dates} ({missing_ratio*100:.1f}%)，超过10%但未超过50%，保留警告")
        
        # 检查复权因子的日期对齐
        df_adj_repaired = df_adj.copy() if not df_adj.empty else None
        if not df_adj.empty:
            validation_result['adj_dates'] = set(df_adj['trade_date'].tolist())
            missing = validation_result['daily_dates'] - validation_result['adj_dates']
            validation_result['missing_in_adj'] = list(missing)
            missing_count = len(missing)
            missing_ratio = missing_count / total_dates if total_dates > 0 else 0
            
            # 【v3.2 新增】复权因子自动修复逻辑
            if auto_repair and missing_count > 0:
                if missing_ratio > 0.5:
                    validation_result['is_valid'] = False
                    logger.warning(f"股票 {ts_code}: 复权因子缺失 {missing_count}/{total_dates} ({missing_ratio*100:.1f}%) > 50%，放弃该股票")
                    return validation_result, None, None, None
                elif missing_ratio <= 0.1:
                    logger.info(f"股票 {ts_code}: 复权因子缺失 {missing_count}/{total_dates} ({missing_ratio*100:.1f}%) ≤ 10%，自动修复")
                    
                    # 复权因子用前向填充
                    df_adj_repaired = df_adj_repaired.sort_values('trade_date')
                    df_adj_repaired = df_adj_repaired.set_index('trade_date')
                    
                    # 为缺失的日期创建行
                    full_index = pd.date_range(start=min(validation_result['daily_dates']), 
                                              end=max(validation_result['daily_dates']), 
                                              freq='D')
                    df_adj_repaired = df_adj_repaired.reindex(full_index)
                    
                    # 前向填充复权因子
                    df_adj_repaired['adj_factor'] = df_adj_repaired['adj_factor'].ffill().fillna(1.0)
                    df_adj_repaired = df_adj_repaired.reset_index().rename(columns={'index': 'trade_date'})
                    df_adj_repaired['trade_date'] = df_adj_repaired['trade_date'].dt.strftime('%Y%m%d')
                    df_adj_repaired['ts_code'] = ts_code
                    
                    filled_count = df_adj_repaired['adj_factor'].notna().sum() - len(df_adj)
                    validation_result['adj_filled'] = filled_count
                    logger.info(f"股票 {ts_code}: 已填充 {filled_count} 条复权因子数据")
        
        # 检查资金流的日期对齐
        df_flow_repaired = df_flow.copy() if not df_flow.empty else None
        if not df_flow.empty:
            validation_result['flow_dates'] = set(df_flow['trade_date'].tolist())
            missing = validation_result['daily_dates'] - validation_result['flow_dates']
            validation_result['missing_in_flow'] = list(missing)
            missing_count = len(missing)
            missing_ratio = missing_count / total_dates if total_dates > 0 else 0
            
            # 【v3.2 新增】资金流自动修复逻辑
            if auto_repair and missing_count > 0:
                if missing_ratio > 0.5:
                    validation_result['is_valid'] = False
                    logger.warning(f"股票 {ts_code}: 资金流缺失 {missing_count}/{total_dates} ({missing_ratio*100:.1f}%) > 50%，放弃该股票")
                    return validation_result, None, None, None
                elif missing_ratio <= 0.1:
                    logger.info(f"股票 {ts_code}: 资金流缺失 {missing_count}/{total_dates} ({missing_ratio*100:.1f}%) ≤ 10%，自动修复")
                    
                    # 资金流用0填充（缺失表示无资金流入/流出）
                    missing_dates = sorted(list(missing))
                    filled_rows = []
                    
                    for missing_date in missing_dates:
                        new_row = {'ts_code': ts_code, 'trade_date': missing_date}
                        # 资金流相关列全部填充为0
                        flow_cols = [
                            'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
                            'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol',
                            'net_mf_vol', 'net_mf_amount', 'buy_sm_amount', 'sell_sm_amount',
                            'buy_md_amount', 'sell_md_amount', 'buy_lg_amount', 'sell_lg_amount',
                            'buy_elg_amount', 'sell_elg_amount'
                        ]
                        for col in flow_cols:
                            if col in df_flow.columns:
                                new_row[col] = 0
                        filled_rows.append(new_row)
                    
                    if filled_rows:
                        filled_df = pd.DataFrame(filled_rows)
                        df_flow_repaired = pd.concat([df_flow_repaired, filled_df], ignore_index=True)
                        validation_result['flow_filled'] = len(filled_rows)
                        logger.info(f"股票 {ts_code}: 已填充 {len(filled_rows)} 条资金流数据")
        
        # 检查重复日期
        if len(validation_result['duplicate_dates']) > 0:
            logger.warning(f"股票 {ts_code}: 存在重复日期 {validation_result['duplicate_dates'][:10]}")
        
        # 记录校验结果
        if validation_result['is_valid']:
            fill_info = []
            if validation_result['basic_filled'] > 0:
                fill_info.append(f"basic: {validation_result['basic_filled']}")
            if validation_result['adj_filled'] > 0:
                fill_info.append(f"adj: {validation_result['adj_filled']}")
            if validation_result['flow_filled'] > 0:
                fill_info.append(f"flow: {validation_result['flow_filled']}")
            
            if fill_info:
                logger.info(f"股票 {ts_code} 数据对齐校验通过，自动修复: {', '.join(fill_info)}")
            else:
                logger.debug(f"股票 {ts_code} 数据对齐校验通过")
        else:
            logger.error(f"股票 {ts_code} 数据对齐校验失败: {validation_result}")
        
        return validation_result, df_basic_repaired, df_adj_repaired, df_flow_repaired
    
    def get_daily_data(self, ts_code: str, start_date: str, end_date: str = None,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        获取全维度日线数据（修复：添加日期对齐校验）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'，默认为今天
            use_cache: 是否使用缓存
            
        Returns:
            包含行情、指标、资金流的完整DataFrame
        """
        # 【v3.3 新增】检查熔断状态
        is_paused, pause_until = self._check_circuit_breaker(ts_code)
        if is_paused:
            logger.info(f"股票 {ts_code} 处于熔断期，跳过请求")
            return pd.DataFrame()
        
        # 修复：使用动态缓存（1天）
        if use_cache:
            cache_key = self._get_cache_key('full_data_v3', ts_code=ts_code, 
                                           start_date=start_date, end_date=end_date)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            cached_data = self._load_pickle_cache(cache_file, cache_type='dynamic')
            if cached_data is not None:
                logger.debug(f"从缓存加载股票 {ts_code} 的全量数据")
                # 【v3.3 新增】成功获取数据，重置熔断状态
                self._reset_circuit_breaker(ts_code)
                return cached_data
        
        try:
            if not self.pro:
                logger.error("tushare未初始化")
                return pd.DataFrame()
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            # 重试机制
            for retry in range(self.retry_count):
                try:
                    # 1. 基础行情
                    df_daily = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    
                    if df_daily is None or df_daily.empty:
                        logger.warning(f"获取股票 {ts_code} 的日线数据为空")
                        return pd.DataFrame()
                    
                    # 2. 每日指标
                    try:
                        df_basic = self.pro.daily_basic(
                            ts_code=ts_code, 
                            start_date=start_date, 
                            end_date=end_date,
                            fields='ts_code,trade_date,turnover_rate,turnover_rate_f,circ_mv,pe_ttm,pe,pb'
                        )
                    except Exception as e:
                        logger.debug(f"获取股票 {ts_code} 的 daily_basic 数据失败: {e}")
                        df_basic = pd.DataFrame()
                    
                    # 3. 复权因子
                    try:
                        df_adj = self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    except Exception as e:
                        logger.debug(f"获取股票 {ts_code} 的复权因子数据失败: {e}")
                        df_adj = pd.DataFrame()

                    # 4. 资金流向
                    try:
                        df_flow = self.pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    except Exception as e:
                        logger.debug(f"获取股票 {ts_code} 的资金流数据失败: {e}")
                        df_flow = pd.DataFrame()

                    # 【v3.2 优化】日期对齐校验 + 自动修复逻辑
                    validation_result, df_basic_repaired, df_adj_repaired, df_flow_repaired = self._validate_date_alignment(
                        df_daily, df_basic, df_adj, df_flow, ts_code, auto_repair=True
                    )
                    
                    if not validation_result['is_valid']:
                        logger.warning(f"股票 {ts_code} 数据对齐校验失败（缺失>50%），跳过该股票")
                        return pd.DataFrame()

                    # --- 数据合并（使用修复后的数据）---
                    # 以 daily 为主表
                    df = df_daily.copy()
                    
                    # 合并复权因子（使用修复后的数据）
                    if df_adj_repaired is not None and not df_adj_repaired.empty:
                        df = df.merge(df_adj_repaired, on=['ts_code', 'trade_date'], how='left')
                    
                    # 合并每日指标（使用修复后的数据）
                    if df_basic_repaired is not None and not df_basic_repaired.empty:
                        df = df.merge(df_basic_repaired, on=['ts_code', 'trade_date'], how='left')
                    
                    # 合并资金流向（使用修复后的数据）
                    if df_flow_repaired is not None and not df_flow_repaired.empty:
                        df = df.merge(df_flow_repaired, on=['ts_code', 'trade_date'], how='left')
                        # 填充资金流空值
                        flow_cols = [
                            'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
                            'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol',
                            'net_mf_vol', 'net_mf_amount', 'buy_sm_amount', 'sell_sm_amount',
                            'buy_md_amount', 'sell_md_amount', 'buy_lg_amount', 'sell_lg_amount',
                            'buy_elg_amount', 'sell_elg_amount'
                        ]
                        for col in flow_cols:
                            if col in df.columns:
                                df[col] = df[col].fillna(0)

                    # 填充复权因子
                    if 'adj_factor' in df.columns:
                        df['adj_factor'] = df['adj_factor'].ffill().fillna(1.0)
                    else:
                        df['adj_factor'] = 1.0

                    # 按日期排序
                    df = df.sort_values('trade_date').reset_index(drop=True)

                    # 计算涨跌幅
                    df['pct_chg'] = df['pct_chg'].round(2)
                    
                    # 重命名列以兼容特征工程
                    column_mapping = {
                        'vol': 'volume',
                        'ts_code': 'stock_code'
                    }
                    df = df.rename(columns=column_mapping)
                    
                    # 修复：保存为动态缓存
                    if use_cache:
                        self._save_pickle_cache(df, cache_file)
                    
                    logger.debug(f"获取股票 {ts_code} 的全量数据成功，共 {len(df)} 条")
                    return df
                    
                except Exception as e:
                    if retry < self.retry_count - 1:
                        logger.warning(f"获取股票 {ts_code} 数据失败，重试 {retry + 1}/{self.retry_count}: {e}")
                        time.sleep(1)
                    else:
                        # 【v3.3 新增】达到重试上限，记录熔断失败
                        self._record_circuit_breaker_failure(ts_code)
                        raise
                    
        except Exception as e:
            logger.error(f"获取股票 {ts_code} 的日线数据失败: {e}")
            return pd.DataFrame()
    
    def _validate_realtime_consistency(self, realtime_df: pd.DataFrame, 
                                       historical_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
        """
        【v3.2 新增】实时行情与历史行情的一致性校验
        
        Args:
            realtime_df: 实时行情DataFrame
            historical_data: 股票代码到历史数据的映射字典
            
        Returns:
            (校验通过的实时行情DataFrame, 校验结果字典)
        """
        validation_result = {
            'total_count': len(realtime_df),
            'valid_count': 0,
            'invalid_count': 0,
            'invalid_codes': [],
            'invalid_reasons': {}
        }
        
        if not self.realtime_price_check_enabled:
            # 如果未启用校验，直接返回原数据
            validation_result['valid_count'] = len(realtime_df)
            return realtime_df, validation_result
        
        if realtime_df.empty:
            return realtime_df, validation_result
        
        valid_rows = []
        
        for _, row in realtime_df.iterrows():
            ts_code = row.get('ts_code') or row.get('code')
            if not ts_code:
                continue
            
            is_valid = True
            invalid_reason = []
            
            # 获取该股票的历史数据
            if ts_code in historical_data and not historical_data[ts_code].empty:
                hist_df = historical_data[ts_code]
                last_hist = hist_df.iloc[-1]  # 最后一个交易日
                
                # 校验1：涨跌幅合理性（实时价格与前一交易日收盘价的涨跌幅应在合理范围内）
                if 'price' in row and 'close' in last_hist:
                    realtime_price = row['price']
                    last_close = last_hist['close']
                    
                    if last_close > 0:
                        price_change = abs((realtime_price - last_close) / last_close)
                        
                        if price_change > self.realtime_price_change_max:
                            is_valid = False
                            invalid_reason.append(
                                f"涨跌幅异常: {price_change*100:.2f}% > {self.realtime_price_change_max*100:.0f}%"
                            )
                
                # 校验2：成交量合理性（实时成交量不应超过历史最高成交量的10倍）
                if 'amount' in row and 'vol' in last_hist:
                    realtime_amount = row['amount']
                    last_vol = last_hist['vol']
                    
                    if last_vol > 0:
                        vol_ratio = realtime_amount / last_vol
                        
                        if vol_ratio > 10:
                            is_valid = False
                            invalid_reason.append(f"成交量异常: {vol_ratio:.1f}倍")
            
            if is_valid:
                valid_rows.append(row)
                validation_result['valid_count'] += 1
            else:
                validation_result['invalid_count'] += 1
                validation_result['invalid_codes'].append(ts_code)
                validation_result['invalid_reasons'][ts_code] = invalid_reason
                logger.warning(f"股票 {ts_code} 实时行情校验失败: {', '.join(invalid_reason)}")
        
        # 返回校验通过的数据
        if valid_rows:
            result_df = pd.DataFrame(valid_rows)
        else:
            result_df = pd.DataFrame()
        
        logger.info(f"实时行情一致性校验完成: {validation_result['valid_count']}/{validation_result['total_count']} 通过")
        
        if validation_result['invalid_count'] > 0:
            logger.warning(f"剔除异常实时行情: {validation_result['invalid_codes'][:10]}...")
        
        return result_df, validation_result
    
    def update_daily_data(self, ts_code: str, last_date: str = None, 
                         auto_merge: bool = True) -> pd.DataFrame:
        """
        【v3.2 优化】增量更新单个股票的日线数据 + 自动合并逻辑
        
        Args:
            ts_code: 股票代码
            last_date: 最后更新的日期，格式 'YYYYMMDD'
            auto_merge: 是否自动合并新增数据到旧缓存
            
        Returns:
            新增的数据DataFrame（如果auto_merge=True，则返回合并后的完整数据）
        """
        try:
            if not self.pro:
                logger.error("tushare未初始化")
                return pd.DataFrame()
            
            # 获取缓存文件路径
            cache_key = self._get_cache_key('full_data_v3', ts_code=ts_code, 
                                           start_date='19900101', end_date='20991231')
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            old_data = None
            
            # 如果未指定最后日期，获取该股票的最新日期
            if last_date is None:
                if os.path.exists(cache_file):
                    cached_data = self._load_pickle_cache(cache_file, cache_type='dynamic')
                    if cached_data is not None and not cached_data.empty:
                        old_data = cached_data
                        last_date = cached_data['trade_date'].max()
                        logger.info(f"从缓存获取股票 {ts_code} 的最后日期: {last_date}")
                    else:
                        # 如果缓存为空，从10天前开始获取
                        last_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
                else:
                    last_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
            
            # 计算更新起始日期（最后日期的下一个交易日）
            last_date_obj = datetime.strptime(last_date, '%Y%m%d')
            start_date = (last_date_obj + timedelta(days=1)).strftime('%Y%m%d')
            end_date = datetime.now().strftime('%Y%m%d')
            
            logger.info(f"增量更新股票 {ts_code}: {start_date} 至 {end_date}")
            
            # 获取新增数据
            new_data = self.get_daily_data(ts_code, start_date, end_date, use_cache=False)
            
            if new_data.empty:
                logger.info(f"股票 {ts_code} 无新增数据")
                return old_data if auto_merge else pd.DataFrame()
            
            # 【v3.2 新增】自动合并增量数据
            if auto_merge and old_data is not None and not old_data.empty:
                # 去除重复日期（保留新数据）
                old_dates = set(old_data['trade_date'].tolist())
                new_dates = set(new_data['trade_date'].tolist())
                overlap_dates = old_dates & new_dates
                
                if overlap_dates:
                    logger.warning(f"股票 {ts_code}: 发现重复日期 {len(overlap_dates)} 个，去除重复")
                    old_data = old_data[~old_data['trade_date'].isin(overlap_dates)]
                
                # 合并新旧数据
                merged_data = pd.concat([old_data, new_data], ignore_index=True)
                
                # 按日期排序
                merged_data = merged_data.sort_values('trade_date').reset_index(drop=True)
                
                # 保存合并后的数据到缓存
                self._save_pickle_cache(merged_data, cache_file)
                
                logger.info(f"股票 {ts_code} 增量更新并合并成功: 旧{len(old_data)} + 新{len(new_data)} = {len(merged_data)}")
                return merged_data
            else:
                # 不需要合并或没有旧数据
                if auto_merge and not new_data.empty:
                    self._save_pickle_cache(new_data, cache_file)
                
                logger.info(f"股票 {ts_code} 增量更新成功，共 {len(new_data)} 条新数据")
                return new_data
            
        except Exception as e:
            logger.error(f"增量更新股票 {ts_code} 失败: {e}")
            return pd.DataFrame()
    
    def update_batch_daily_data(self, ts_codes: List[str], last_dates: Dict[str, str] = None,
                                use_thread: bool = True, auto_merge: bool = True,
                                retry_failed: bool = True) -> Dict[str, pd.DataFrame]:
        """
        【v3.2 优化】批量增量更新日线数据 + 更新失败重试机制
        
        Args:
            ts_codes: 股票代码列表
            last_dates: 股票代码到最后日期的映射，None表示自动推断
            use_thread: 是否使用多线程
            auto_merge: 是否自动合并增量数据到旧缓存
            retry_failed: 是否对失败的股票进行二次重试
            
        Returns:
            股票代码到新增数据的映射字典
        """
        result = {}
        failed_codes = []
        
        if last_dates is None:
            last_dates = {}
        
        if use_thread and len(ts_codes) > 1:
            # 多线程批量更新
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_code = {
                    executor.submit(
                        self.update_daily_data, ts_code, last_dates.get(ts_code), auto_merge
                    ): ts_code
                    for ts_code in ts_codes
                }
                
                for future in as_completed(future_to_code):
                    ts_code = future_to_code[future]
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            result[ts_code] = df
                        else:
                            failed_codes.append(ts_code)
                    except Exception as e:
                        logger.error(f"增量更新股票 {ts_code} 失败: {e}")
                        failed_codes.append(ts_code)
        else:
            # 单线程更新
            for ts_code in ts_codes:
                try:
                    df = self.update_daily_data(ts_code, last_dates.get(ts_code), auto_merge)
                    if df is not None and not df.empty:
                        result[ts_code] = df
                    else:
                        failed_codes.append(ts_code)
                except Exception as e:
                    logger.error(f"增量更新股票 {ts_code} 失败: {e}")
                    failed_codes.append(ts_code)
                finally:
                    time.sleep(self.rate_limit_delay)
        
        success_count = len(result)
        total_count = len(ts_codes)
        logger.info(f"批量增量更新完成，成功 {success_count}/{total_count} 只股票")
        
        # 【v3.2 新增】更新失败重试机制
        if retry_failed and failed_codes and len(failed_codes) > 0:
            logger.info(f"开始对 {len(failed_codes)} 只失败股票进行二次重试...")
            
            retry_result = {}
            retry_failed = []
            
            # 单线程重试（降低并发，避免进一步限流）
            for ts_code in failed_codes:
                try:
                    logger.info(f"重试股票 {ts_code}...")
                    df = self.update_daily_data(ts_code, last_dates.get(ts_code), auto_merge)
                    
                    if df is not None and not df.empty:
                        retry_result[ts_code] = df
                        logger.info(f"重试成功: {ts_code}")
                    else:
                        retry_failed.append(ts_code)
                        logger.warning(f"重试失败（数据为空）: {ts_code}")
                except Exception as e:
                    retry_failed.append(ts_code)
                    logger.error(f"重试失败（异常）: {ts_code}, {e}")
                
                # 重试时使用更长的延迟
                time.sleep(self.rate_limit_delay * 2)
            
            # 合并重试成功的结果
            result.update(retry_result)
            
            retry_success = len(retry_result)
            final_failed = len(retry_failed)
            
            logger.info(f"二次重试完成: 成功 {retry_success}/{len(failed_codes)}，最终失败 {final_failed}/{total_count}")
            
            if retry_failed:
                logger.warning(f"最终失败的股票代码: {retry_failed[:10]}...")
        
        return result
    
    def get_realtime_quotes_batch(self, ts_codes: List[str], 
                                  batch_size: int = 100,
                                  enable_consistency_check: bool = True,
                                  historical_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        【v3.2 优化】批量获取实时行情（使用正确的API接口 + 一致性校验 + 请求频率限制）
        
        Args:
            ts_codes: 股票代码列表
            batch_size: 每批获取的数量
            enable_consistency_check: 是否启用一致性校验
            historical_data: 历史数据字典（用于一致性校验）
            
        Returns:
            实时行情DataFrame（已剔除异常数据）
        """
        try:
            if not self.pro:
                logger.error("tushare未初始化")
                return pd.DataFrame()
            
            all_data = []
            
            # 【v3.2 优化】使用更严格的实时行情请求频率限制
            rate_delay = self.realtime_rate_limit_delay
            
            # 分批获取
            for i in range(0, len(ts_codes), batch_size):
                batch_codes = ts_codes[i:i + batch_size]
                ts_code_str = ','.join(batch_codes)
                
                try:
                    # 修复：使用正确的实时行情接口
                    df = self.pro.realtime_quotes(ts_code_str)
                    
                    if df is not None and not df.empty:
                        all_data.append(df)
                    
                    # 【v3.2 优化】使用实时行情专用的请求频率限制
                    time.sleep(rate_delay)
                    
                except Exception as e:
                    logger.error(f"获取批量 {i}-{i+batch_size} 实时行情失败: {e}")
                    continue
            
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                logger.info(f"获取实时行情成功，共 {len(result)} 只股票")
                
                # 【v3.2 新增】一致性校验
                if enable_consistency_check and historical_data:
                    result, validation_result = self._validate_realtime_consistency(
                        result, historical_data
                    )
                
                return result
            else:
                logger.warning("获取实时行情为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            return pd.DataFrame()
    
    def get_batch_daily_data(self, ts_codes: List[str], start_date: str, 
                            end_date: str = None, use_cache: bool = True,
                            use_thread: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的日线数据（多线程+缓存）
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            use_thread: 是否使用多线程
            
        Returns:
            股票代码到日线数据的映射字典
        """
        result = {}
        failed_codes = []
        
        # 【v3.3 新增】计算最优批次大小
        optimal_batch_size = self.get_optimal_batch_size(len(ts_codes))
        
        if use_thread and len(ts_codes) > 1:
            # 【v3.3 优化】多线程批量获取，使用最优批次大小
            with ThreadPoolExecutor(max_workers=optimal_batch_size) as executor:
                # 记录请求开始时间
                start_time = time.time()
                
                future_to_code = {
                    executor.submit(
                        self.get_daily_data, ts_code, start_date, end_date, use_cache
                    ): ts_code
                    for ts_code in ts_codes
                }
                
                for future in as_completed(future_to_code):
                    ts_code = future_to_code[future]
                    
                    # 【v3.3 新增】记录请求结束时间，计算响应时间
                    request_time = time.time() - start_time
                    
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            result[ts_code] = df
                        else:
                            failed_codes.append(ts_code)
                    except Exception as e:
                        logger.error(f"获取股票 {ts_code} 的数据失败: {e}")
                        failed_codes.append(ts_code)
                    
                    # 【v3.3 新增】动态调整请求间隔
                    dynamic_delay = self.get_dynamic_request_delay(
                        base_delay=self.rate_limit_delay,
                        response_time=request_time
                    )
                    time.sleep(dynamic_delay)
        else:
            # 单线程获取
            for ts_code in ts_codes:
                try:
                    df = self.get_daily_data(ts_code, start_date, end_date, use_cache)
                    if df is not None and not df.empty:
                        result[ts_code] = df
                    else:
                        failed_codes.append(ts_code)
                except Exception as e:
                    logger.error(f"获取股票 {ts_code} 的数据失败: {e}")
                    failed_codes.append(ts_code)
                finally:
                    # 【v3.3 优化】使用动态请求间隔
                    time.sleep(self.rate_limit_delay)
        
        success_count = len(result)
        total_count = len(ts_codes)
        logger.info(f"批量获取日线数据完成，成功 {success_count}/{total_count} 只股票 (批次大小: {optimal_batch_size})")
        
        if failed_codes:
            logger.warning(f"失败的股票代码: {failed_codes[:10]}...")
        
        return result
    
    def get_stock_pool(self, pool_size: int = 100, market: str = None) -> List[str]:
        """获取股票池（向后兼容方法）"""
        logger.warning("get_stock_pool 方法已弃用，建议使用 get_stock_pool_tree")
        return self.get_stock_pool_tree(
            pool_size=pool_size,
            market=market,
            exclude_st=True,
            min_days_listed=30
        )
    
    def get_stock_pool_tree(self, pool_size: int = 100, market: str = None,
                           industries: List[str] = None, exclude_st: bool = True,
                           min_days_listed: int = 30, exclude_markets: List[str] = None,
                           exclude_board_types: List[str] = None, use_cache: bool = True) -> List[str]:
        """获取股票池（树形筛选）"""
        try:
            df = self.get_stock_list(market=market, use_cache=use_cache)
            
            if df.empty:
                logger.warning("获取股票列表为空，返回空池")
                return []
            
            # 排除ST股票
            if exclude_st:
                df = df[~df['name'].str.contains('ST|退|暂停', na=False)]
                logger.info(f"排除ST后剩余: {len(df)} 只股票")
            
            # 排除指定市场
            if exclude_markets:
                df = df[~df['ts_code'].str.contains('|'.join([f'\\.{m}$' for m in exclude_markets]), regex=True)]
                logger.info(f"排除市场 {exclude_markets} 后剩余: {len(df)} 只股票")
            
            # 排除指定板块
            if exclude_board_types:
                patterns = []
                for bt in exclude_board_types:
                    if bt == '300' or bt == '301':
                        patterns.append(f'^{bt}')
                    else:
                        patterns.append(f'^{bt}')
                exclude_pattern = '|'.join(patterns)
                df = df[~df['ts_code'].str.match(exclude_pattern)]
                logger.info(f"排除板块 {exclude_board_types} 后剩余: {len(df)} 只股票")
            
            # 排除新上市股票
            if min_days_listed > 0 and 'list_date' in df.columns:
                df['list_date'] = pd.to_datetime(df['list_date'])
                min_list_date = datetime.now() - timedelta(days=min_days_listed)
                df = df[df['list_date'] < min_list_date]
                logger.info(f"排除新股后剩余: {len(df)} 只股票")
            
            # 行业筛选
            if industries:
                df = df[df['industry'].isin(industries)]
                logger.info(f"行业筛选后剩余: {len(df)} 只股票")
            
            # 按行业均匀采样
            if 'industry' in df.columns:
                industry_groups = df.groupby('industry')
                selected_stocks = []
                
                per_industry = max(1, int(pool_size / len(industry_groups)))
                
                for industry, group in industry_groups:
                    group_sorted = group.sort_values('list_date', ascending=False)
                    selected = group_sorted.head(per_industry)
                    selected_stocks.extend(selected['ts_code'].tolist())
                
                if len(selected_stocks) < pool_size:
                    remaining = pool_size - len(selected_stocks)
                    available = df[~df['ts_code'].isin(selected_stocks)]
                    extra = available.head(remaining)
                    selected_stocks.extend(extra['ts_code'].tolist())
            else:
                selected_stocks = df['ts_code'].tolist()
            
            selected_stocks = selected_stocks[:pool_size]
            
            logger.info(f"获取股票池成功，共 {len(selected_stocks)} 只股票")
            return selected_stocks
            
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            return []
    
    def clear_cache(self, older_than_hours: int = None, cache_type: str = None) -> int:
        """
        修复：清理缓存（支持按类型清理）
        
        Args:
            older_than_hours: 清理多少小时前的缓存，None=清理全部
            cache_type: 清理类型，static=静态数据，dynamic=动态数据，None=全部
            
        Returns:
            清理的文件数量
        """
        try:
            cleared_count = 0
            current_time = datetime.now()
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    should_delete = False
                    
                    # 检查缓存类型
                    if cache_type:
                        if cache_type == 'static' and 'stock_list' not in filename:
                            continue
                        if cache_type == 'dynamic' and 'stock_list' in filename:
                            continue
                    
                    # 检查时间
                    if older_than_hours is None:
                        should_delete = True
                    else:
                        expiry_time = current_time - timedelta(hours=older_than_hours)
                        if file_time < expiry_time:
                            should_delete = True
                    
                    if should_delete:
                        os.remove(file_path)
                        cleared_count += 1
                        logger.debug(f"删除缓存文件: {filename}")
            
            cache_type_str = f" ({cache_type})" if cache_type else ""
            logger.info(f"清理缓存完成{cache_type_str}，删除 {cleared_count} 个文件")
            return cleared_count
            
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            return 0
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """检查数据质量"""
        result = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'abnormal_values': {},
            'data_range': {}
        }
        
        # 检查异常值
        numeric_cols = ['open', 'high', 'low', 'close', 'vol', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    result['abnormal_values'][f'{col}_negative'] = negative_count
                
                if not df[col].empty:
                    result['data_range'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean())
                    }
        
        logger.info(f"数据质量检查完成: {result}")
        return result
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息（修复：区分静态和动态）"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in cache_files
            )
            
            static_files = [f for f in cache_files if 'stock_list' in f]
            dynamic_files = [f for f in cache_files if 'stock_list' not in f]
            
            static_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in static_files
            )
            dynamic_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in dynamic_files
            )
            
            return {
                'total_files': len(cache_files),
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'static_files': len(static_files),
                'static_size_mb': round(static_size / 1024 / 1024, 2),
                'dynamic_files': len(dynamic_files),
                'dynamic_size_mb': round(dynamic_size / 1024 / 1024, 2)
            }
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}

    def get_cache_hit_stats(self) -> Dict:
        """
        【v3.2 新增】获取缓存命中统计信息（命中率监控）
        
        Returns:
            缓存命中统计字典
        """
        total = self.cache_stats['total_requests']
        hits = self.cache_stats['cache_hits']
        misses = self.cache_stats['cache_misses']
        
        stats = self.cache_stats.copy()
        
        if total > 0:
            stats['hit_rate'] = hits / total
            stats['miss_rate'] = misses / total
        else:
            stats['hit_rate'] = 0.0
            stats['miss_rate'] = 0.0
        
        return stats
    
    def print_cache_hit_stats(self):
        """【v3.2 新增】打印缓存命中统计信息"""
        stats = self.get_cache_hit_stats()
        
        print("\n" + "="*60)
        print("缓存命中统计信息")
        print("="*60)
        print(f"总请求次数: {stats['total_requests']}")
        print(f"缓存命中次数: {stats['cache_hits']} (静态: {stats['static_hits']}, 动态: {stats['dynamic_hits']})")
        print(f"缓存未命中次数: {stats['cache_misses']}")
        print(f"缓存命中率: {stats['hit_rate']*100:.2f}%")
        print(f"缓存未命中率: {stats['miss_rate']*100:.2f}%")
        print("="*60 + "\n")
        
        logger.info(f"缓存命中统计: 命中率={stats['hit_rate']*100:.2f}%, 总请求={stats['total_requests']}")
    
    def auto_clean_expired_cache(self) -> bool:
        """
        【v3.2 新增】自动清理过期缓存（每日凌晨触发）
        
        Returns:
            是否执行了清理
        """
        if not self.auto_clean_enabled:
            return False
        
        current_time = datetime.now()
        current_date = current_time.date()
        
        # 检查是否已经今天清理过
        if self.last_clean_date == current_date:
            return False
        
        # 检查是否到达清理时间
        clean_time = datetime.strptime(self.auto_clean_time, '%H:%M').time()
        
        if current_time.time() >= clean_time:
            logger.info(f"触发自动清理缓存: {current_time}")
            
            # 清理动态缓存（超过有效期）
            cleared_dynamic = self.clear_cache(
                older_than_hours=self.cache_expiry_dynamic_hours,
                cache_type='dynamic'
            )
            
            # 清理静态缓存（超过有效期）
            cleared_static = self.clear_cache(
                older_than_hours=self.cache_expiry_static_hours,
                cache_type='static'
            )
            
            total_cleared = cleared_dynamic + cleared_static
            self.last_clean_date = current_date
            
            logger.info(f"自动清理缓存完成: 清理 {total_cleared} 个文件 (动态: {cleared_dynamic}, 静态: {cleared_static})")
            
            return True
        
        return False
    
    def refresh_stock_list_cache(self, market: str = None, status: str = 'L') -> pd.DataFrame:
        """
        【v3.2 新增】手动刷新股票列表缓存（适配股票上市/退市变动）
        
        Args:
            market: 市场，SSE=上海，SZSE=深圳，None=所有
            status: 状态，L=上市，D=退市，P=暂停上市
            
        Returns:
            最新的股票列表DataFrame
        """
        logger.info(f"手动刷新股票列表缓存: market={market}, status={status}")
        
        # 强制不使用缓存，重新获取
        df = self.get_stock_list(market=market, status=status, use_cache=False)
        
        # 手动保存缓存
        if not df.empty:
            cache_key = self._get_cache_key('stock_list', market=market, status=status)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            self._save_pickle_cache(df, cache_file)
            logger.info(f"股票列表缓存已刷新: {len(df)} 只股票")
        
        return df

    
    # ==================== 【v3.3 新增】API请求熔断机制 ====================
    
    def _check_circuit_breaker(self, ts_code: str) -> Tuple[bool, Optional[datetime]]:
        """
        【v3.3 新增】检查股票是否被熔断
        
        Args:
            ts_code: 股票代码
            
        Returns:
            (是否熔断, 恢复时间)
        """
        if not self.circuit_breaker_enabled:
            return False, None
        
        if ts_code in self.circuit_breaker_status:
            status = self.circuit_breaker_status[ts_code]
            pause_until = status.get('pause_until')
            
            if pause_until and datetime.now() < pause_until:
                # 仍在熔断期内
                logger.warning(f"股票 {ts_code} 处于熔断期，暂停至 {pause_until.strftime('%H:%M:%S')}")
                return True, pause_until
            else:
                # 熔断期已过，重置状态
                logger.info(f"股票 {ts_code} 熔断期已过，重置状态")
                del self.circuit_breaker_status[ts_code]
        
        return False, None
    
    def _record_circuit_breaker_failure(self, ts_code: str):
        """
        【v3.3 新增】记录股票请求失败
        
        Args:
            ts_code: 股票代码
        """
        if not self.circuit_breaker_enabled:
            return
        
        if ts_code not in self.circuit_breaker_status:
            self.circuit_breaker_status[ts_code] = {
                'fail_count': 0,
                'pause_until': None
            }
        
        # 增加失败计数
        self.circuit_breaker_status[ts_code]['fail_count'] += 1
        fail_count = self.circuit_breaker_status[ts_code]['fail_count']
        
        logger.warning(f"股票 {ts_code} 请求失败 ({fail_count}/{self.retry_count})")
        
        # 检查是否达到熔断阈值
        if fail_count >= self.retry_count:
            # 触发熔断，暂停请求
            pause_until = datetime.now() + timedelta(minutes=self.circuit_breaker_pause_minutes)
            self.circuit_breaker_status[ts_code]['pause_until'] = pause_until
            
            logger.error(f"股票 {ts_code} 请求失败达 {fail_count} 次，触发熔断，暂停至 {pause_until.strftime('%H:%M:%S')}")
    
    def _reset_circuit_breaker(self, ts_code: str):
        """
        【v3.3 新增】重置股票的熔断状态
        
        Args:
            ts_code: 股票代码
        """
        if ts_code in self.circuit_breaker_status:
            del self.circuit_breaker_status[ts_code]
            logger.info(f"股票 {ts_code} 熔断状态已重置")
    
    def get_circuit_breaker_status(self) -> Dict:
        """
        【v3.3 新增】获取熔断状态统计
        
        Returns:
            熔断状态字典
        """
        paused_count = 0
        status_list = []
        
        for ts_code, status in self.circuit_breaker_status.items():
            is_paused, pause_until = self._check_circuit_breaker(ts_code)
            if is_paused:
                paused_count += 1
            
            status_list.append({
                'ts_code': ts_code,
                'fail_count': status['fail_count'],
                'pause_until': pause_until.strftime('%H:%M:%S') if pause_until else None,
                'is_paused': is_paused
            })
        
        return {
            'total_paused': paused_count,
            'total_tracked': len(self.circuit_breaker_status),
            'enabled': self.circuit_breaker_enabled,
            'pause_minutes': self.circuit_breaker_pause_minutes,
            'retry_count': self.retry_count,
            'status_list': status_list
        }
    
    def print_circuit_breaker_status(self):
        """【v3.3 新增】打印熔断状态"""
        status = self.get_circuit_breaker_status()
        
        print("\n" + "="*60)
        print("API请求熔断状态")
        print("="*60)
        print(f"熔断机制: {'启用' if status['enabled'] else '禁用'}")
        print(f"重试阈值: {status['retry_count']} 次")
        print(f"暂停时长: {status['pause_minutes']} 分钟")
        print(f"追踪股票数: {status['total_tracked']}")
        print(f"熔断中股票数: {status['total_paused']}")
        
        if status['status_list']:
            print("\n熔断状态详情:")
            for item in status['status_list']:
                status_str = "🔴 熔断中" if item['is_paused'] else "🟢 正常"
                pause_info = f" (暂停至 {item['pause_until']})" if item['is_paused'] else ""
                print(f"  {status_str} {item['ts_code']}: 失败 {item['fail_count']} 次{pause_info}")
        
        print("="*60 + "\n")
    
    # ==================== 【v3.3 新增】数据质量校验增强 ====================
    
    def check_data_quality_enhanced(self, df: pd.DataFrame, 
                                    auto_filter: bool = True,
                                    auto_fix: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        【v3.3 新增】增强数据质量校验（自动过滤/修正异常值）
        
        Args:
            df: 行情数据DataFrame
            auto_filter: 是否自动过滤异常值
            auto_fix: 是否自动修正可修正的异常值
            
        Returns:
            (校验后的DataFrame, 校验结果字典)
        """
        if df.empty:
            return df, {'is_valid': False, 'reason': '数据为空'}
        
        result = {
            'is_valid': True,
            'original_count': len(df),
            'filtered_count': 0,
            'fixed_count': 0,
            'issues': [],
            'fixes': []
        }
        
        df_clean = df.copy()
        
        # 1. 过滤异常值
        if auto_filter:
            # 过滤成交量为负的行
            if 'volume' in df_clean.columns:
                neg_volume_count = (df_clean['volume'] < 0).sum()
                if neg_volume_count > 0:
                    df_clean = df_clean[df_clean['volume'] >= 0]
                    result['filtered_count'] += neg_volume_count
                    result['issues'].append(f"成交量为负: {neg_volume_count} 行")
            
            # 过滤成交量为0但有成交金额的行（异常数据）
            if 'volume' in df_clean.columns and 'amount' in df_clean.columns:
                invalid_volume_amount = ((df_clean['volume'] == 0) & (df_clean['amount'] > 0)).sum()
                if invalid_volume_amount > 0:
                    df_clean = df_clean[~((df_clean['volume'] == 0) & (df_clean['amount'] > 0))]
                    result['filtered_count'] += invalid_volume_amount
                    result['issues'].append(f"成交量为0但有成交金额: {invalid_volume_amount} 行")
            
            # 过滤市盈率异常偏高（> 1000）的行
            if 'pe_ttm' in df_clean.columns:
                abnormal_pe = (df_clean['pe_ttm'] > 1000).sum()
                if abnormal_pe > 0:
                    df_clean = df_clean[df_clean['pe_ttm'] <= 1000]
                    result['filtered_count'] += abnormal_pe
                    result['issues'].append(f"市盈率异常偏高(>1000): {abnormal_pe} 行")
            
            # 过滤市盈率为负但股价为正的行（异常数据）
            if 'pe_ttm' in df_clean.columns and 'close' in df_clean.columns:
                invalid_pe = ((df_clean['pe_ttm'] < 0) & (df_clean['close'] > 0)).sum()
                if invalid_pe > 0:
                    df_clean = df_clean[~((df_clean['pe_ttm'] < 0) & (df_clean['close'] > 0))]
                    result['filtered_count'] += invalid_pe
                    result['issues'].append(f"市盈率为负但股价为正: {invalid_pe} 行")
        
        # 2. 修正可修正的异常值
        if auto_fix:
            # 修正成交量为空值的情况（填充为0）
            if 'volume' in df_clean.columns:
                volume_na_count = df_clean['volume'].isna().sum()
                if volume_na_count > 0:
                    df_clean['volume'] = df_clean['volume'].fillna(0)
                    result['fixed_count'] += volume_na_count
                    result['fixes'].append(f"成交量为空: 填充为0 ({volume_na_count} 行)")
            
            # 修正成交金额为空值的情况（填充为0）
            if 'amount' in df_clean.columns:
                amount_na_count = df_clean['amount'].isna().sum()
                if amount_na_count > 0:
                    df_clean['amount'] = df_clean['amount'].fillna(0)
                    result['fixed_count'] += amount_na_count
                    result['fixes'].append(f"成交金额为空: 填充为0 ({amount_na_count} 行)")
        
        # 判断数据是否有效
        if df_clean.empty:
            result['is_valid'] = False
            result['reason'] = '过滤后数据为空'
        
        # 输出校验结果
        if result['issues'] or result['fixes']:
            logger.info(f"数据质量校验: 过滤 {result['filtered_count']} 行, 修正 {result['fixed_count']} 行")
            if result['issues']:
                logger.warning(f"  异常: {', '.join(result['issues'][:3])}")
            if result['fixes']:
                logger.info(f"  修正: {', '.join(result['fixes'][:3])}")
        
        return df_clean, result
    
    def validate_data_format(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        【v3.3 新增】数据格式统一校验（日期YYYYMMDD、股票代码标准化）
        
        Args:
            df: 行情数据DataFrame
            
        Returns:
            (格式化后的DataFrame, 校验结果字典)
        """
        if df.empty:
            return df, {'is_valid': False, 'reason': '数据为空'}
        
        result = {
            'is_valid': True,
            'format_fixes': []
        }
        
        df_formatted = df.copy()
        
        # 1. 日期格式统一校验（YYYYMMDD）
        if 'trade_date' in df_formatted.columns:
            # 尝试转换为标准日期格式
            try:
                # 检查日期格式
                date_sample = df_formatted['trade_date'].iloc[0] if len(df_formatted) > 0 else None
                
                if date_sample:
                    # 如果是字符串格式，统一转换为YYYYMMDD
                    if isinstance(date_sample, str):
                        # 尝试解析日期
                        try:
                            # 使用format='mixed'自动识别格式
                            df_formatted['trade_date'] = pd.to_datetime(df_formatted['trade_date'], format='mixed').dt.strftime('%Y%m%d')
                            result['format_fixes'].append('日期格式: 统一 -> YYYYMMDD')
                        except Exception as e:
                            logger.warning(f"日期格式解析失败: {e}")
                            result['is_valid'] = False
                            result['reason'] = f"日期格式解析失败: {e}"
                    elif isinstance(date_sample, datetime):
                        # datetime格式，转换为YYYYMMDD
                        df_formatted['trade_date'] = df_formatted['trade_date'].dt.strftime('%Y%m%d')
                        result['format_fixes'].append('日期格式: datetime -> YYYYMMDD')
            except Exception as e:
                logger.warning(f"日期格式校验失败: {e}")
                result['is_valid'] = False
                result['reason'] = f"日期格式校验失败: {e}"
        
        # 2. 股票代码格式标准化
        if 'ts_code' in df_formatted.columns:
            # 检查是否有未标准化的股票代码
            unstandardized_count = df_formatted['ts_code'].apply(
                lambda x: isinstance(x, str) and not ('.SZ' in x or '.SH' in x or '.BJ' in x)
            ).sum()
            
            if unstandardized_count > 0:
                # 对所有未标准化的股票代码添加交易所后缀
                def standardize_code(code):
                    if not isinstance(code, str) or '.SZ' in code or '.SH' in code or '.BJ' in code:
                        return code
                    if not code.isdigit() or len(code) != 6:
                        return code
                    if code.startswith('6'):
                        return f"{code}.SH"
                    elif code.startswith(('0', '3')):
                        return f"{code}.SZ"
                    elif code.startswith('8'):
                        return f"{code}.BJ"
                    return code
                
                df_formatted['ts_code'] = df_formatted['ts_code'].apply(standardize_code)
                result['format_fixes'].append('股票代码: 添加交易所后缀')
        
        # 输出校验结果
        if result['format_fixes']:
            logger.info(f"数据格式校验: {', '.join(result['format_fixes'])}")
        
        return df_formatted, result
    
    # ==================== 【v3.3 新增】多线程优化 ====================
    
    def get_dynamic_request_delay(self, base_delay: float = None, 
                                  response_time: float = None) -> float:
        """
        【v3.3 新增】根据API响应速度动态调整请求间隔
        
        Args:
            base_delay: 基础延迟（秒），默认为 rate_limit_delay
            response_time: API响应时间（秒）
            
        Returns:
            动态调整后的延迟（秒）
        """
        if base_delay is None:
            base_delay = self.rate_limit_delay
        
        if response_time is None:
            return base_delay
        
        # 如果API响应较慢，增加延迟
        if response_time > 1.0:
            # 响应时间 > 1秒，延迟加倍
            return base_delay * 2.0
        elif response_time > 0.5:
            # 响应时间 > 0.5秒，延迟增加50%
            return base_delay * 1.5
        elif response_time < 0.1:
            # 响应很快（< 0.1秒），可以稍微减少延迟
            return max(base_delay * 0.8, 0.05)
        else:
            # 响应时间正常，使用基础延迟
            return base_delay
    
    def get_optimal_batch_size(self, total_stocks: int) -> int:
        """
        【v3.3 新增】根据股票数量动态计算最优批次大小
        
        Args:
            total_stocks: 总股票数量
            
        Returns:
            最优批次大小
        """
        # 根据配置的最大线程数和股票数量动态调整
        optimal_size = min(self.max_workers, total_stocks)
        
        # 如果股票数量很多，可以考虑分批次处理
        if total_stocks > 100:
            # 每50只股票作为一个批次
            optimal_size = min(optimal_size, 50)
        
        return optimal_size
    
    def get_concept_stocks(self, concept_name: str = None) -> pd.DataFrame:
        """
        【新增】获取概念板块成分股
        
        Args:
            concept_name: 概念板块名称（可选，不传则返回所有概念板块）
            
        Returns:
            包含概念板块和成分股信息的DataFrame
        """
        try:
            # 获取概念板块列表
            concept_list = self.pro.concept()
            
            if concept_name:
                # 筛选指定概念板块
                concept_list = concept_list[concept_list['name'] == concept_name]
                
                if len(concept_list) == 0:
                    logger.warning(f"未找到概念板块: {concept_name}")
                    return pd.DataFrame()
            
            logger.info(f"获取到 {len(concept_list)} 个概念板块")
            
            # 获取每个概念板块的成分股
            concept_stocks = []
            for _, row in concept_list.iterrows():
                concept_code = row['code']
                concept_name = row['name']
                
                try:
                    # 获取成分股
                    members = self.pro.concept_member(concept_code=concept_code)
                    members['concept_code'] = concept_code
                    members['concept_name'] = concept_name
                    concept_stocks.append(members)
                except Exception as e:
                    logger.warning(f"获取概念板块 {concept_name} 成分股失败: {str(e)}")
                    continue
            
            if concept_stocks:
                result_df = pd.concat(concept_stocks, ignore_index=True)
                logger.info(f"获取到 {len(result_df)} 条概念板块成分股记录")
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取概念板块数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_industry_stocks(self, industry_name: str = None) -> pd.DataFrame:
        """
        【新增】获取行业板块成分股（申万行业）
        
        Args:
            industry_name: 行业名称（可选，不传则返回所有行业）
            
        Returns:
            包含行业板块和成分股信息的DataFrame
        """
        try:
            # 获取行业分类（申万一级）
            industry_list = self.pro.index_classify(level='L1', source='SW2021')
            
            if industry_name:
                # 筛选指定行业
                industry_list = industry_list[industry_list['industry_name'] == industry_name]
                
                if len(industry_list) == 0:
                    logger.warning(f"未找到行业: {industry_name}")
                    return pd.DataFrame()
            
            logger.info(f"获取到 {len(industry_list)} 个行业")
            
            # 获取每个行业的成分股
            industry_stocks = []
            for _, row in industry_list.iterrows():
                index_code = row['index_code']
                industry_name = row['industry_name']
                
                try:
                    # 获取成分股
                    members = self.pro.index_member(index_code=index_code)
                    members['industry_code'] = index_code
                    members['industry_name'] = industry_name
                    industry_stocks.append(members)
                except Exception as e:
                    logger.warning(f"获取行业 {industry_name} 成分股失败: {str(e)}")
                    continue
            
            if industry_stocks:
                result_df = pd.concat(industry_stocks, ignore_index=True)
                logger.info(f"获取到 {len(result_df)} 条行业板块成分股记录")
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取行业板块数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_sector_data(self, ts_code: str, trade_date: str = None) -> Dict:
        """
        【新增】获取股票所属板块数据
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期（YYYYMMDD，可选，默认最新）
            
        Returns:
            包含股票所属板块信息的字典
        """
        try:
            result = {
                'ts_code': ts_code,
                'trade_date': trade_date,
                'industries': [],
                'concepts': [],
                'sector_count': 0
            }
            
            # 获取行业归属（申万）
            try:
                # 获取所有行业成分股
                industry_stocks = self.get_industry_stocks()
                
                # 筛选该股票所属的行业
                if not industry_stocks.empty and 'con_code' in industry_stocks.columns:
                    stock_industries = industry_stocks[industry_stocks['con_code'] == ts_code]
                    
                    if not stock_industries.empty:
                        result['industries'] = stock_industries[['industry_code', 'industry_name']].to_dict('records')
            except Exception as e:
                logger.warning(f"获取股票 {ts_code} 行业归属失败: {str(e)}")
            
            # 获取概念归属
            try:
                # 获取所有概念成分股
                concept_stocks = self.get_concept_stocks()
                
                # 筛选该股票所属的概念
                if not concept_stocks.empty and 'ts_code' in concept_stocks.columns:
                    stock_concepts = concept_stocks[concept_stocks['ts_code'] == ts_code]
                    
                    if not stock_concepts.empty:
                        result['concepts'] = stock_concepts[['concept_code', 'concept_name']].to_dict('records')
            except Exception as e:
                logger.warning(f"获取股票 {ts_code} 概念归属失败: {str(e)}")
            
            result['sector_count'] = len(result['industries']) + len(result['concepts'])
            
            return result
            
        except Exception as e:
            logger.error(f"获取股票 {ts_code} 板块数据失败: {str(e)}")
            return {
                'ts_code': ts_code,
                'trade_date': trade_date,
                'industries': [],
                'concepts': [],
                'sector_count': 0
            }
    
    def get_sector_ranking(self, ts_codes: List[str], trade_date: str = None, 
                          rank_by: str = 'market_cap') -> pd.DataFrame:
        """
        【新增】获取板块内排名数据
        
        Args:
            ts_codes: 股票代码列表
            trade_date: 交易日期（YYYYMMDD，可选）
            rank_by: 排名依据（market_cap: 市值, pct_chg: 涨跌幅, amount: 成交额）
            
        Returns:
            包含板块排名信息的DataFrame
        """
        try:
            # 批量获取股票基本信息和行情数据
            if not ts_codes:
                return pd.DataFrame()
            
            # 获取股票基本信息（包含市值）
            stock_info = self.get_stock_list()
            if stock_info is None or stock_info.empty:
                return pd.DataFrame()
            
            # 筛选目标股票
            target_stocks = stock_info[stock_info['ts_code'].isin(ts_codes)].copy()
            
            # 获取行情数据
            if trade_date:
                daily_data = self.get_batch_daily_data(ts_codes, trade_date, trade_date)
            else:
                # 获取最新交易日数据
                today = datetime.now().strftime('%Y%m%d')
                daily_data = self.get_batch_daily_data(ts_codes, today, today)
            
            if daily_data is None or daily_data.empty:
                # 如果没有行情数据，仅使用基本信息
                target_stocks['pct_chg'] = 0
                target_stocks['amount'] = 0
            else:
                # 合并数据
                target_stocks = target_stocks.merge(
                    daily_data[['ts_code', 'pct_chg', 'amount']],
                    on='ts_code',
                    how='left'
                )
                target_stocks['pct_chg'] = target_stocks['pct_chg'].fillna(0)
                target_stocks['amount'] = target_stocks['amount'].fillna(0)
            
            # 计算排名
            if rank_by == 'market_cap':
                # 按市值排名（前10%为龙头）
                target_stocks['market_cap_rank'] = target_stocks['total_mv'].rank(ascending=False)
                target_stocks['market_cap_rank_pct'] = target_stocks['market_cap_rank'] / len(target_stocks)
            elif rank_by == 'pct_chg':
                # 按涨跌幅排名（前5%为龙头）
                target_stocks['pct_chg_rank'] = target_stocks['pct_chg'].rank(ascending=False)
                target_stocks['pct_chg_rank_pct'] = target_stocks['pct_chg_rank'] / len(target_stocks)
            elif rank_by == 'amount':
                # 按成交额排名
                target_stocks['amount_rank'] = target_stocks['amount'].rank(ascending=False)
                target_stocks['amount_rank_pct'] = target_stocks['amount_rank'] / len(target_stocks)
            
            logger.info(f"获取到 {len(target_stocks)} 只股票的板块排名数据（按{rank_by}排序）")
            
            return target_stocks
            
        except Exception as e:
            logger.error(f"获取板块排名数据失败: {str(e)}")
            return pd.DataFrame()
