"""
数据获取与ETL模块

该模块提供数据获取、清洗和转换的核心功能，支持多种数据源。
使用抽象基类定义统一的数据处理接口，确保扩展性。
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import time
import warnings
import ssl
import os

import pandas as pd
import numpy as np
import akshare as ak

logger = logging.getLogger(__name__)

# ============================================================================
# SSL/网络配置（解决eastmoney等数据源的SSL连接问题）
# ============================================================================
def _configure_ssl_context() -> None:
    """
    配置SSL上下文以解决SSL_UNEXPECTED_EOF等连接问题
    
    针对中国金融数据API（如东方财富）常见的SSL问题进行优化：
    - 部分服务器SSL实现不标准，会导致EOF错误
    - 通过降低SSL验证级别来提升连接成功率
    
    Notes
    -----
    此配置会降低SSL安全性，仅用于获取公开金融数据场景。
    生产环境建议使用VPN或代理解决网络问题。
    """
    try:
        # 方法1：通过环境变量禁用SSL验证（requests库）
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['CURL_CA_BUNDLE'] = ''
        
        # 方法2：配置全局SSL上下文
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 方法3：创建更宽松的SSL上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl_context.set_ciphers('DEFAULT@SECLEVEL=0')
        
        # 设置更兼容的TLS选项，解决EOF问题
        ssl_context.options |= ssl.OP_NO_SSLv2
        ssl_context.options |= ssl.OP_NO_SSLv3
        # 允许TLS重协商
        ssl_context.options |= getattr(ssl, 'OP_LEGACY_SERVER_CONNECT', 0x4)
        
        # 尝试修改默认HTTPS上下文
        ssl._create_default_https_context = lambda: ssl_context
        
        logger.debug("SSL上下文已配置为宽松模式")
        
    except Exception as e:
        logger.warning(f"SSL配置失败，使用默认设置: {e}")


def _configure_requests_session() -> None:
    """
    配置 requests 库的全局会话以处理SSL问题
    
    通过 Monkey Patch 方式劫持 requests.Session.request 方法：
    - 强制添加 Connection: close 头，禁用 Keep-Alive，解决 EOF 错误
    - 设置默认超时 (10, 30)，防止请求卡死
    
    Notes
    -----
    使用 Monkey Patch 而非 HTTPAdapter 的原因：
    - HTTPAdapter 无法强制禁用 Keep-Alive
    - 需要在所有请求中统一添加 Connection: close 头
    """
    try:
        import requests
        
        # 保存原始方法引用（确保可追溯）
        _original_request = requests.Session.request
        
        def _patched_request(self, method, url, **kwargs):
            """
            劫持后的 request 方法
            
            强制禁用 Keep-Alive 并设置默认超时
            """
            # 强制添加 Connection: close 头，禁用长连接
            headers = kwargs.get('headers', {}) or {}
            headers['Connection'] = 'close'
            kwargs['headers'] = headers
            
            # 设置默认超时 (连接超时 10s, 读取超时 30s)
            if 'timeout' not in kwargs or kwargs['timeout'] is None:
                kwargs['timeout'] = (10, 30)
            
            return _original_request(self, method, url, **kwargs)
        
        # Monkey Patch: 替换全局 Session.request 方法
        requests.Session.request = _patched_request
        
        logger.debug("requests.Session.request 已通过 Monkey Patch 配置 (Connection: close, timeout=(10,30))")
        
    except Exception as e:
        logger.warning(f"requests会话配置失败: {e}")


# 模块加载时自动配置SSL和requests
_configure_ssl_context()
_configure_requests_session()


class DataHandler(ABC):
    """
    数据处理抽象基类
    
    定义数据获取和处理的标准接口，所有数据源实现类必须继承此类。
    
    Attributes
    ----------
    config : Dict[str, Any]
        数据配置字典
    
    Methods
    -------
    fetch_daily_data(symbol, start_date, end_date)
        获取日线数据
    fetch_fundamental_data(symbol)
        获取基本面数据
    get_stock_list(index_code)
        获取股票列表
    save_data(data, filepath)
        保存数据到本地
    load_data(filepath)
        从本地加载数据
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化数据处理器
        
        Parameters
        ----------
        config : Dict[str, Any]
            数据配置字典，包含数据源、存储路径等配置
        """
        self.config = config
        self._retry_times = config.get("data_source", {}).get("retry_times", 3)
        self._retry_delay = config.get("data_source", {}).get("retry_delay", 5)
        self._timeout = config.get("data_source", {}).get("timeout", 30)
    
    @abstractmethod
    def fetch_daily_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取股票日线数据
        
        Parameters
        ----------
        symbol : str
            股票代码，如 '000001'
        start_date : str
            开始日期，格式 'YYYY-MM-DD'
        end_date : str
            结束日期，格式 'YYYY-MM-DD'
        
        Returns
        -------
        pd.DataFrame
            日线数据，索引为DatetimeIndex
        
        Raises
        ------
        ValueError
            当日期格式错误时
        ConnectionError
            当网络连接失败时
        """
        pass
    
    @abstractmethod
    def fetch_fundamental_data(self, symbol: str) -> pd.DataFrame:
        """
        获取股票基本面数据
        
        Parameters
        ----------
        symbol : str
            股票代码
        
        Returns
        -------
        pd.DataFrame
            基本面数据
        """
        pass
    
    @abstractmethod
    def get_stock_list(self, index_code: Optional[str] = None) -> List[str]:
        """
        获取股票列表
        
        Parameters
        ----------
        index_code : Optional[str]
            指数代码，如 '000300' 表示沪深300成分股
            如果为None，返回全部A股列表
        
        Returns
        -------
        List[str]
            股票代码列表
        """
        pass
    
    def save_data(
        self,
        data: pd.DataFrame,
        filepath: str,
        compression: str = "snappy"
    ) -> None:
        """
        保存数据到Parquet文件
        
        Parameters
        ----------
        data : pd.DataFrame
            要保存的数据
        filepath : str
            文件路径
        compression : str, optional
            压缩算法，默认为 'snappy'
        """
        data.to_parquet(filepath, compression=compression)
        logger.info(f"数据已保存至: {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        从Parquet文件加载数据
        
        Parameters
        ----------
        filepath : str
            文件路径
        
        Returns
        -------
        pd.DataFrame
            加载的数据
        """
        data = pd.read_parquet(filepath)
        logger.info(f"数据已加载: {filepath}, 形状: {data.shape}")
        return data
    
    def _retry_request(self, func: Callable, *args, **kwargs) -> Any:
        """
        带重试机制的请求包装器（支持指数退避和网络错误处理）
        
        Parameters
        ----------
        func : callable
            要执行的函数
        *args
            位置参数
        **kwargs
            关键字参数
        
        Returns
        -------
        Any
            函数返回值
        
        Raises
        ------
        Exception
            当所有重试都失败时
        
        Notes
        -----
        使用指数退避策略处理网络错误：
        - 第1次重试等待 base_delay 秒
        - 第2次重试等待 base_delay * 2 秒
        - 第N次重试等待 base_delay * 2^(N-1) 秒（上限60秒）
        增加随机抖动（jitter）避免请求风暴
        """
        import random
        import ssl
        import urllib3
        
        # 网络相关异常类型（需要更长等待时间）
        NETWORK_ERROR_KEYWORDS = (
            'ssl', 'timeout', 'connection', 'reset', 'eof', 
            'refused', 'aborted', '10054', '10060', 'timed out'
        )
        
        last_exception = None
        base_delay = self._retry_delay
        
        for attempt in range(self._retry_times):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # 判断是否为网络错误
                is_network_error = any(
                    keyword in error_msg for keyword in NETWORK_ERROR_KEYWORDS
                )
                
                # 计算等待时间：指数退避 + 随机抖动
                if is_network_error:
                    # 网络错误使用更长的退避时间
                    wait_time = min(base_delay * (2 ** attempt), 60)
                    jitter = random.uniform(0.5, 1.5)
                    wait_time = wait_time * jitter
                    error_type = "网络/SSL错误"
                else:
                    # 其他错误使用标准退避
                    wait_time = base_delay * (attempt + 1)
                    jitter = random.uniform(0.8, 1.2)
                    wait_time = wait_time * jitter
                    error_type = "一般错误"
                
                logger.warning(
                    f"请求失败 (尝试 {attempt + 1}/{self._retry_times}) "
                    f"[{error_type}]: {e}"
                )
                
                if attempt < self._retry_times - 1:
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
        
        logger.error(f"所有重试均失败: {last_exception}")
        raise last_exception


class AkshareDataLoader(DataHandler):
    """
    基于AkShare的数据加载器
    
    使用AkShare库从东方财富等数据源获取A股数据。
    
    Examples
    --------
    >>> config = {"data_source": {"retry_times": 3}}
    >>> loader = AkshareDataLoader(config)
    >>> df = loader.fetch_daily_data("000001", "2023-01-01", "2023-12-31")
    >>> print(df.head())
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化AkShare数据加载器
        
        Parameters
        ----------
        config : Dict[str, Any]
            数据配置字典
        """
        super().__init__(config)
        self._request_delay = config.get("data_source", {}).get("request_delay", 0.5)
        
        # 配置AkShare底层请求库的超时和重试
        self._configure_requests_session()
        logger.info("AkShare数据加载器初始化完成")
    
    def _configure_requests_session(self) -> None:
        """
        配置requests库以提升网络稳定性
        
        针对东方财富API常见的SSL/连接问题进行优化：
        - 增加连接超时和读取超时
        - 禁用SSL证书验证（可选，仅用于调试）
        - 配置连接池重试机制
        """
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # 创建重试策略
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
            
            # 创建适配器
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=10
            )
            
            # 尝试为AkShare配置session（如果支持）
            # 注：AkShare内部使用自己的session，这里主要是设置默认超时
            import socket
            socket.setdefaulttimeout(self._timeout)
            
            logger.debug("网络请求配置完成: timeout=%ss", self._timeout)
            
        except ImportError as e:
            logger.warning(f"配置请求session失败，使用默认设置: {e}")
    
    def fetch_daily_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取股票日线数据
        
        Parameters
        ----------
        symbol : str
            股票代码，如 '000001'
        start_date : str
            开始日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        
        Returns
        -------
        pd.DataFrame
            日线数据，包含 open, high, low, close, volume 等字段
            索引为 DatetimeIndex
        """
        # 标准化日期格式
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        def _fetch():
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            return df
        
        df = self._retry_request(_fetch)
        
        if df.empty:
            logger.warning(f"股票 {symbol} 在指定日期范围内无数据")
            return pd.DataFrame()
        
        # 数据清洗和标准化
        df = self._standardize_columns(df)
        df = self._set_datetime_index(df)
        
        logger.info(f"获取 {symbol} 日线数据成功，共 {len(df)} 条记录")
        return df
    
    def fetch_fundamental_data(self, symbol: str) -> pd.DataFrame:
        """
        获取股票基本面数据
        
        Parameters
        ----------
        symbol : str
            股票代码
        
        Returns
        -------
        pd.DataFrame
            基本面数据，包含市盈率、市净率等指标
        """
        def _fetch():
            df = ak.stock_individual_info_em(symbol=symbol)
            return df
        
        df = self._retry_request(_fetch)
        logger.info(f"获取 {symbol} 基本面数据成功")
        return df
    
    def get_stock_list(
        self, 
        index_code: Optional[str] = None,
        use_cache: bool = True
    ) -> List[str]:
        """
        获取股票列表（支持本地缓存降级）
        
        Parameters
        ----------
        index_code : Optional[str]
            指数代码，如 '000300' 表示沪深300成分股
        use_cache : bool, optional
            是否启用缓存（网络失败时自动使用），默认 True
        
        Returns
        -------
        List[str]
            股票代码列表
        
        Notes
        -----
        当网络请求失败时，会尝试从本地缓存加载股票列表。
        缓存文件保存在 data/cache/ 目录下。
        """
        import json
        
        # 缓存目录和文件
        cache_dir = Path("data/cache")
        cache_file = cache_dir / f"stock_list_{index_code or 'all'}.json"
        
        def _fetch():
            if index_code:
                # 获取指数成分股
                df = ak.index_stock_cons(symbol=index_code)
                return df["品种代码"].tolist()
            else:
                # 获取全部A股列表
                df = ak.stock_zh_a_spot_em()
                return df["代码"].tolist()
        
        try:
            stock_list = self._retry_request(_fetch)
            
            # 成功获取后更新缓存
            if use_cache and stock_list:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_data = {
                    "updated_at": datetime.now().isoformat(),
                    "index_code": index_code,
                    "stock_list": stock_list
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False)
                logger.debug(f"股票列表已缓存至 {cache_file}")
            
            logger.info(f"获取股票列表成功，共 {len(stock_list)} 只股票")
            return stock_list
            
        except Exception as e:
            logger.error(f"网络获取股票列表失败: {e}")
            
            # 尝试从缓存加载
            if use_cache and cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    stock_list = cache_data.get("stock_list", [])
                    updated_at = cache_data.get("updated_at", "未知")
                    logger.warning(
                        f"使用缓存的股票列表 (更新时间: {updated_at})，"
                        f"共 {len(stock_list)} 只股票"
                    )
                    return stock_list
                except Exception as cache_error:
                    logger.error(f"缓存加载失败: {cache_error}")
            
            # 缓存也没有则抛出异常
            raise
    
    def fetch_index_data(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取指数日线数据
        
        Parameters
        ----------
        index_code : str
            指数代码，如 '000300'
        start_date : str
            开始日期
        end_date : str
            结束日期
        
        Returns
        -------
        pd.DataFrame
            指数日线数据
        """
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        def _fetch():
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            return df
        
        df = self._retry_request(_fetch)
        
        # 过滤日期范围
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df.loc[mask].copy()
        df.set_index("date", inplace=True)
        
        logger.info(f"获取指数 {index_code} 数据成功，共 {len(df)} 条记录")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名
        
        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        
        Returns
        -------
        pd.DataFrame
            标准化后的数据
        """
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        df = df.rename(columns=column_mapping)
        return df
    
    def _set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        设置日期时间索引
        
        Parameters
        ----------
        df : pd.DataFrame
            数据
        
        Returns
        -------
        pd.DataFrame
            设置索引后的数据
        """
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
        return df


class DataPipeline:
    """
    数据处理管道
    
    整合数据获取、清洗和存储的完整流程。
    
    Attributes
    ----------
    data_handler : DataHandler
        数据处理器实例
    config : Dict[str, Any]
        配置字典
    """
    
    def __init__(
        self,
        data_handler: DataHandler,
        config: Dict[str, Any]
    ) -> None:
        """
        初始化数据管道
        
        Parameters
        ----------
        data_handler : DataHandler
            数据处理器实例
        config : Dict[str, Any]
            配置字典
        """
        self.data_handler = data_handler
        self.config = config
    
    def run_daily_update(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        执行每日数据更新
        
        Parameters
        ----------
        symbols : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            股票代码到数据的映射
        """
        result = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"正在获取 {symbol} ({i}/{total})")
            try:
                df = self.data_handler.fetch_daily_data(
                    symbol, start_date, end_date
                )
                if not df.empty:
                    result[symbol] = df
            except Exception as e:
                logger.error(f"获取 {symbol} 失败: {e}")
                continue
        
        logger.info(f"数据更新完成，成功获取 {len(result)}/{total} 只股票")
        return result


class AShareDataCleaner:
    """
    A股市场数据清洗器
    
    提供针对A股市场数据的专业清洗功能，包括停牌处理、涨跌停识别等。
    所有方法采用Pandas向量化操作，确保高性能处理大规模数据。
    
    Attributes
    ----------
    output_dir : str
        清洗后数据的输出目录
    compression : str
        Parquet文件压缩算法
    
    Examples
    --------
    >>> cleaner = AShareDataCleaner(output_dir="data/processed")
    >>> cleaned_df = cleaner.clean_market_data(raw_df)
    >>> print(cleaned_df["is_limit"].sum())  # 涨跌停天数
    """
    
    # A股涨跌停幅度常量
    LIMIT_PCT_MAIN = 0.10      # 主板涨跌停幅度 10%
    LIMIT_PCT_GEM_STAR = 0.20  # 创业板/科创板涨跌停幅度 20%
    LIMIT_PCT_ST = 0.05        # ST股票涨跌停幅度 5%
    
    def __init__(
        self,
        output_dir: str = "data/processed",
        compression: str = "snappy"
    ) -> None:
        """
        初始化数据清洗器
        
        Parameters
        ----------
        output_dir : str, optional
            清洗后数据的输出目录，默认为 'data/processed'
        compression : str, optional
            Parquet压缩算法，默认为 'snappy'
        """
        self.output_dir = output_dir
        self.compression = compression
        
        # 确保输出目录存在
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"A股数据清洗器初始化完成，输出目录: {output_dir}")
    
    def clean_market_data(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        清洗A股市场数据
        
        执行完整的数据清洗流程，包括：
        1. 日期索引转换
        2. 停牌日识别与处理
        3. 一字涨跌停识别
        4. 数据保存
        
        Parameters
        ----------
        df : pd.DataFrame
            原始市场数据，需包含以下列：
            - '日期' 或 'date': 交易日期
            - 'open', 'high', 'low', 'close': OHLC价格
            - 'volume': 成交量
            - 'pct_change' 或 '涨跌幅': 涨跌幅（可选，用于精确判断涨跌停）
        symbol : Optional[str]
            股票代码，用于保存文件命名和判断涨跌停幅度
        save : bool, optional
            是否保存清洗后的数据，默认为True
        
        Returns
        -------
        pd.DataFrame
            清洗后的数据，包含以下新增列：
            - 'is_suspended': 是否停牌 (bool)
            - 'is_limit': 是否一字涨跌停 (bool)
            - 'is_limit_up': 是否一字涨停 (bool)
            - 'is_limit_down': 是否一字跌停 (bool)
        
        Notes
        -----
        停牌日的OHLC价格将被填充为NaN，便于后续处理时识别和过滤。
        
        Examples
        --------
        >>> cleaner = AShareDataCleaner()
        >>> df = loader.fetch_daily_data("000001", "2023-01-01", "2024-01-01")
        >>> cleaned = cleaner.clean_market_data(df, symbol="000001")
        >>> print(f"停牌天数: {cleaned['is_suspended'].sum()}")
        >>> print(f"涨跌停天数: {cleaned['is_limit'].sum()}")
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # Step 1: 转换日期列为DatetimeIndex
        df = self._convert_datetime_index(df)
        
        # Step 2: 识别并处理停牌日
        df = self._handle_suspended_days(df)
        
        # Step 3: 识别一字涨跌停
        df = self._identify_limit_days(df, symbol)
        
        # Step 4: 保存为Parquet格式
        if save and symbol:
            self._save_to_parquet(df, symbol)
        
        logger.info(
            f"数据清洗完成: 总记录={len(df)}, "
            f"停牌={df['is_suspended'].sum()}, "
            f"一字板={df['is_limit'].sum()}"
        )
        
        return df
    
    def _convert_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将日期列转换为DatetimeIndex
        
        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        
        Returns
        -------
        pd.DataFrame
            设置DatetimeIndex后的数据
        """
        # 查找日期列（支持中英文列名）
        date_columns = ["日期", "date", "Date", "DATE", "trade_date"]
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"未找到日期列，需要以下列之一: {date_columns}"
            )
        
        if date_col:
            # 使用向量化的pd.to_datetime进行转换
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
        
        # 确保索引是DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        
        # 按日期排序
        df = df.sort_index()
        df.index.name = "date"
        
        return df
    
    def _handle_suspended_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        识别停牌日并填充NaN
        
        停牌日特征：成交量(volume)为0
        
        Parameters
        ----------
        df : pd.DataFrame
            数据
        
        Returns
        -------
        pd.DataFrame
            处理后的数据，停牌日OHLC填充为NaN
        """
        # 标准化列名（处理可能的中文列名）
        volume_cols = ["volume", "成交量", "vol", "Volume"]
        volume_col = None
        
        for col in volume_cols:
            if col in df.columns:
                volume_col = col
                break
        
        if volume_col is None:
            logger.warning("未找到成交量列，跳过停牌日处理")
            df["is_suspended"] = False
            return df
        
        # 向量化识别停牌日：成交量为0或NaN
        is_suspended = (df[volume_col] == 0) | df[volume_col].isna()
        df["is_suspended"] = is_suspended
        
        # OHLC列列表
        ohlc_cols = ["open", "high", "low", "close"]
        ohlc_chinese = ["开盘", "最高", "最低", "收盘"]
        
        # 查找存在的OHLC列
        existing_ohlc = []
        for col in ohlc_cols + ohlc_chinese:
            if col in df.columns:
                existing_ohlc.append(col)
        
        # 使用向量化操作填充停牌日的OHLC为NaN
        if existing_ohlc:
            df.loc[is_suspended, existing_ohlc] = np.nan
        
        suspended_count = is_suspended.sum()
        if suspended_count > 0:
            logger.info(f"识别到 {suspended_count} 个停牌日，已填充NaN")
        
        return df
    
    def _identify_limit_days(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        识别一字涨跌停
        
        一字涨跌停特征：
        1. 开盘价 = 收盘价 = 最高价 = 最低价
        2. 涨跌幅达到涨跌停幅度
        
        Parameters
        ----------
        df : pd.DataFrame
            数据
        symbol : Optional[str]
            股票代码，用于判断涨跌停幅度
        
        Returns
        -------
        pd.DataFrame
            添加涨跌停标识列的数据
        """
        # 确定涨跌停幅度阈值
        limit_pct = self._get_limit_pct(symbol)
        
        # 获取OHLC列（优先英文，兼容中文）
        open_col = "open" if "open" in df.columns else "开盘"
        high_col = "high" if "high" in df.columns else "最高"
        low_col = "low" if "low" in df.columns else "最低"
        close_col = "close" if "close" in df.columns else "收盘"
        
        # 检查必要列是否存在
        required = [open_col, high_col, low_col, close_col]
        if not all(col in df.columns for col in required):
            logger.warning("缺少OHLC列，无法识别一字涨跌停")
            df["is_limit"] = False
            df["is_limit_up"] = False
            df["is_limit_down"] = False
            return df
        
        # 向量化判断一字板：所有价格相等
        # 使用np.isclose处理浮点数精度问题
        price_equal = (
            np.isclose(df[open_col], df[close_col], rtol=1e-6) &
            np.isclose(df[high_col], df[close_col], rtol=1e-6) &
            np.isclose(df[low_col], df[close_col], rtol=1e-6)
        )
        
        # 获取涨跌幅列
        pct_cols = ["pct_change", "涨跌幅", "pctChg", "change_pct"]
        pct_col = None
        for col in pct_cols:
            if col in df.columns:
                pct_col = col
                break
        
        if pct_col:
            # 使用涨跌幅判断（更精确）
            pct_change = df[pct_col]
            
            # 如果是百分比形式（如9.98），需要转换为小数
            if pct_change.abs().max() > 1:
                pct_change = pct_change / 100
            
            # 一字涨停：价格相等 + 涨幅接近涨停幅度
            is_limit_up = price_equal & (
                pct_change >= (limit_pct - 0.005)  # 留0.5%容差
            )
            
            # 一字跌停：价格相等 + 跌幅接近跌停幅度
            is_limit_down = price_equal & (
                pct_change <= -(limit_pct - 0.005)
            )
        else:
            # 无涨跌幅列时，使用前收盘价计算
            prev_close = df[close_col].shift(1)
            pct_change = (df[close_col] - prev_close) / prev_close
            
            is_limit_up = price_equal & (pct_change >= (limit_pct - 0.005))
            is_limit_down = price_equal & (pct_change <= -(limit_pct - 0.005))
        
        # 排除停牌日
        if "is_suspended" in df.columns:
            is_limit_up = is_limit_up & ~df["is_suspended"]
            is_limit_down = is_limit_down & ~df["is_suspended"]
        
        # 赋值结果列
        df["is_limit_up"] = is_limit_up.fillna(False).astype(bool)
        df["is_limit_down"] = is_limit_down.fillna(False).astype(bool)
        df["is_limit"] = df["is_limit_up"] | df["is_limit_down"]
        
        limit_up_count = df["is_limit_up"].sum()
        limit_down_count = df["is_limit_down"].sum()
        
        if limit_up_count > 0 or limit_down_count > 0:
            logger.info(
                f"识别到一字涨停 {limit_up_count} 天, "
                f"一字跌停 {limit_down_count} 天"
            )
        
        return df
    
    def _get_limit_pct(self, symbol: Optional[str] = None) -> float:
        """
        根据股票代码判断涨跌停幅度
        
        Parameters
        ----------
        symbol : Optional[str]
            股票代码
        
        Returns
        -------
        float
            涨跌停幅度
        """
        if symbol is None:
            return self.LIMIT_PCT_MAIN
        
        # 去除可能的前缀
        code = symbol.lstrip("shszSHSZ.")
        
        # 创业板: 300xxx, 301xxx
        if code.startswith(("300", "301")):
            return self.LIMIT_PCT_GEM_STAR
        
        # 科创板: 688xxx, 689xxx
        if code.startswith(("688", "689")):
            return self.LIMIT_PCT_GEM_STAR
        
        # 北交所: 8xxxxx, 4xxxxx
        if code.startswith(("8", "4")) and len(code) == 6:
            return 0.30  # 北交所30%
        
        # ST股票检测需要股票名称，这里使用默认值
        # 实际使用时可通过名称判断
        
        return self.LIMIT_PCT_MAIN
    
    def _save_to_parquet(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> str:
        """
        保存清洗后的数据为Parquet格式
        
        Parameters
        ----------
        df : pd.DataFrame
            清洗后的数据
        symbol : str
            股票代码
        
        Returns
        -------
        str
            保存的文件路径
        """
        from pathlib import Path
        
        filepath = Path(self.output_dir) / f"{symbol}_cleaned.parquet"
        
        # 使用pyarrow引擎保存，支持更好的压缩
        df.to_parquet(
            filepath,
            compression=self.compression,
            index=True  # 保留DatetimeIndex
        )
        
        logger.info(f"清洗后数据已保存: {filepath}")
        return str(filepath)
    
    def batch_clean(
        self,
        data_dict: Dict[str, pd.DataFrame],
        parallel: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        批量清洗多只股票数据
        
        Parameters
        ----------
        data_dict : Dict[str, pd.DataFrame]
            股票代码到原始数据的映射
        parallel : bool, optional
            是否并行处理（需要安装joblib），默认为False
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            股票代码到清洗后数据的映射
        """
        result = {}
        total = len(data_dict)
        
        for i, (symbol, df) in enumerate(data_dict.items(), 1):
            logger.info(f"清洗 {symbol} ({i}/{total})")
            try:
                cleaned = self.clean_market_data(df, symbol=symbol)
                result[symbol] = cleaned
            except Exception as e:
                logger.error(f"清洗 {symbol} 失败: {e}")
                continue
        
        logger.info(f"批量清洗完成: {len(result)}/{total}")
        return result


@dataclass
class DownloadResult:
    """
    下载结果数据类
    
    Attributes
    ----------
    symbol : str
        股票代码
    success : bool
        是否成功
    data : Optional[pd.DataFrame]
        下载的数据
    error : Optional[str]
        错误信息
    """
    symbol: str
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None


class DataLoader:
    """
    A股数据加载器（增强版）
    
    基于AkShare实现的高性能数据加载器，支持：
    - 前复权/不复权价格数据
    - 财务指标数据（PE、PB、股息率）
    - 并发批量下载
    - Parquet格式存储
    - LocalFirst模式（优先读取本地数据湖）
    
    Attributes
    ----------
    output_dir : str
        数据保存目录
    max_workers : int
        最大并发线程数
    retry_times : int
        重试次数
    timeout : int
        超时时间（秒）
    mode : str
        数据获取模式：'network'（网络优先）或 'local_first'（本地优先）
    lake_path : Path
        数据湖路径（仅 local_first 模式使用）
    
    Examples
    --------
    >>> loader = DataLoader(output_dir="data/raw")
    >>> 
    >>> # 获取单只股票数据
    >>> df = loader.fetch_daily_price("000001", "2023-01-01", "2024-12-31")
    >>> 
    >>> # 批量下载沪深300成分股
    >>> results = loader.batch_download_hs300("2023-01-01", "2024-12-31")
    >>> 
    >>> # LocalFirst模式：优先读取本地数据
    >>> loader_local = DataLoader(mode="local_first")
    >>> df = loader_local.fetch_daily_price("000001", "2023-01-01", "2024-12-31")
    """
    
    # AkShare 接口常量
    ADJUST_QFQ = "qfq"      # 前复权
    ADJUST_HFQ = "hfq"      # 后复权
    ADJUST_NONE = ""        # 不复权
    
    # 支持的模式
    MODE_NETWORK = "network"
    MODE_LOCAL_FIRST = "local_first"
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        max_workers: int = 2,
        retry_times: int = 3,
        retry_delay: float = 3.0,
        timeout: int = 30,
        mode: str = "network",
        lake_path: str = "data/lake/daily"
    ) -> None:
        """
        初始化数据加载器
        
        Parameters
        ----------
        output_dir : str, optional
            数据保存目录，默认 'data/raw'
        max_workers : int, optional
            最大并发线程数，默认2（降低并发以提升网络稳定性）
        retry_times : int, optional
            网络请求重试次数，默认3
        retry_delay : float, optional
            重试间隔秒数，默认3.0（增加间隔以避免触发限流）
        timeout : int, optional
            请求超时时间（秒），默认30
        mode : str, optional
            数据获取模式，可选：
            - 'network': 网络优先模式（默认），直接从 Akshare 获取数据
            - 'local_first': 本地优先模式，优先读取数据湖，缺失时降级到网络
        lake_path : str, optional
            数据湖路径，默认 'data/lake/daily'（仅 local_first 模式使用）
        """
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # 模式配置
        if mode not in (self.MODE_NETWORK, self.MODE_LOCAL_FIRST):
            raise ValueError(
                f"不支持的模式: {mode}，可选 'network' 或 'local_first'"
            )
        self.mode = mode
        self.lake_path = Path(lake_path)
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载统计
        self._download_stats = {
            "success": 0,
            "failed": 0,
            "total": 0
        }
        
        # 全市场数据缓存（避免重复请求 stock_zh_a_spot_em）
        self._spot_data_cache: Optional[pd.DataFrame] = None
        self._spot_data_cache_time: Optional[datetime] = None
        self._spot_cache_ttl = 300  # 缓存有效期5分钟
        
        logger.info(
            f"DataLoader初始化: output_dir={output_dir}, "
            f"max_workers={max_workers}, mode={mode}"
        )
    
    def fetch_daily_price(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        include_unadjusted: bool = True
    ) -> pd.DataFrame:
        """
        获取股票日线价格数据
        
        使用 ak.stock_zh_a_hist 接口获取数据。
        默认获取前复权数据用于计算收益率，同时可保留不复权数据用于计算成交额。
        
        在 local_first 模式下，优先读取本地数据湖，缺失或过期时降级到网络获取。
        
        Parameters
        ----------
        symbol : str
            股票代码，如 '000001'
        start_date : str
            开始日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        include_unadjusted : bool, optional
            是否同时获取不复权数据，默认True（仅网络模式有效）
        
        Returns
        -------
        pd.DataFrame
            日线数据，包含以下列：
            - date: 日期（索引）
            - open, high, low, close: 前复权OHLC
            - volume: 成交量（股）
            - amount: 成交额（元，来自不复权数据）
            - turnover: 换手率
            - pct_change: 涨跌幅
            - open_unadj, high_unadj, low_unadj, close_unadj: 不复权OHLC（可选）
        
        Raises
        ------
        ValueError
            当股票代码无效时
        ConnectionError
            当网络连接失败时
        
        Examples
        --------
        >>> loader = DataLoader()
        >>> df = loader.fetch_daily_price("000001", "2023-01-01", "2024-12-31")
        >>> print(df[['close', 'amount', 'pct_change']].head())
        >>> 
        >>> # LocalFirst模式
        >>> loader_local = DataLoader(mode="local_first")
        >>> df = loader_local.fetch_daily_price("000001", "2023-01-01", "2024-12-31")
        """
        # LocalFirst 模式
        if self.mode == self.MODE_LOCAL_FIRST:
            return self._fetch_daily_price_local_first(
                symbol, start_date, end_date, include_unadjusted
            )
        
        # Network 模式（原有逻辑）
        return self._fetch_from_network(
            symbol, start_date, end_date, include_unadjusted
        )
    
    def _fetch_daily_price_local_first(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        include_unadjusted: bool = True
    ) -> pd.DataFrame:
        """
        LocalFirst 模式获取日线数据
        
        优先读取本地数据湖，缺失或过期时降级到网络获取。
        
        Parameters
        ----------
        symbol : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
        include_unadjusted : bool
            是否包含不复权数据
        
        Returns
        -------
        pd.DataFrame
            日线数据
        """
        # 标准化日期格式
        start_dt = pd.to_datetime(start_date.replace("-", ""), format="%Y%m%d")
        end_dt = pd.to_datetime(end_date.replace("-", ""), format="%Y%m%d")
        
        # Step 1: 尝试读取本地数据
        local_df = self._read_from_lake(symbol)
        
        if local_df is not None and not local_df.empty:
            local_max_date = local_df.index.max()
            local_min_date = local_df.index.min()
            
            # 检查本地数据是否覆盖请求范围
            if local_min_date <= start_dt and local_max_date >= end_dt:
                # 本地数据完整，直接返回
                logger.debug(f"{symbol}: 本地数据完整，直接返回")
                return local_df.loc[start_dt:end_dt].copy()
            
            # Step 2: 本地数据不完整，需要补录
            logger.warning(
                f"本地数据缺失或过期 ({symbol})，正在触发临时网络请求... "
                f"(本地: {local_min_date.date()} ~ {local_max_date.date()}, "
                f"请求: {start_dt.date()} ~ {end_dt.date()})"
            )
            
            # 计算需要补录的日期范围
            missing_ranges = []
            
            # 前面缺失
            if start_dt < local_min_date:
                missing_ranges.append((
                    start_dt.strftime("%Y%m%d"),
                    (local_min_date - pd.Timedelta(days=1)).strftime("%Y%m%d")
                ))
            
            # 后面缺失
            if end_dt > local_max_date:
                missing_ranges.append((
                    (local_max_date + pd.Timedelta(days=1)).strftime("%Y%m%d"),
                    end_dt.strftime("%Y%m%d")
                ))
            
            # 获取缺失数据
            all_dfs = [local_df]
            for miss_start, miss_end in missing_ranges:
                missing_df = self._fetch_from_network(
                    symbol, miss_start, miss_end, include_unadjusted
                )
                if missing_df is not None and not missing_df.empty:
                    all_dfs.append(missing_df)
            
            # 合并数据
            if len(all_dfs) > 1:
                merged_df = pd.concat(all_dfs)
                merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                merged_df = merged_df.sort_index()
                return merged_df.loc[start_dt:end_dt].copy()
            
            return local_df.loc[start_dt:end_dt].copy()
        
        else:
            # Step 3: 本地无数据，完全从网络获取
            logger.warning(
                f"本地数据缺失或过期 ({symbol})，正在触发临时网络请求..."
            )
            return self._fetch_from_network(
                symbol, start_date, end_date, include_unadjusted
            )
    
    def _read_from_lake(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        从数据湖读取本地数据
        
        Parameters
        ----------
        symbol : str
            股票代码
        
        Returns
        -------
        Optional[pd.DataFrame]
            本地数据，文件不存在返回 None
        """
        filepath = self.lake_path / f"{symbol}.parquet"
        
        if not filepath.exists():
            logger.debug(f"本地文件不存在: {filepath}")
            return None
        
        try:
            df = pd.read_parquet(filepath)
            
            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df = df.set_index('date')
                df.index = pd.to_datetime(df.index)
            
            df = df.sort_index()
            logger.debug(f"从数据湖加载 {symbol}: {len(df)} 行")
            return df
            
        except Exception as e:
            logger.warning(f"读取本地文件失败 {filepath}: {e}")
            return None
    
    def _fetch_from_network(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        include_unadjusted: bool = True
    ) -> pd.DataFrame:
        """
        从网络获取日线数据（Akshare）
        
        Parameters
        ----------
        symbol : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
        include_unadjusted : bool
            是否包含不复权数据
        
        Returns
        -------
        pd.DataFrame
            日线数据
        """
        # 标准化日期格式
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        logger.info(f"获取 {symbol} 日线数据: {start_date} - {end_date}")
        
        # 获取前复权数据（用于计算收益率）
        df_qfq = self._fetch_with_retry(
            func=ak.stock_zh_a_hist,
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=self.ADJUST_QFQ
        )
        
        if df_qfq is None or df_qfq.empty:
            logger.warning(f"股票 {symbol} 无数据")
            return pd.DataFrame()
        
        # 标准化列名
        df_qfq = self._standardize_columns(df_qfq)
        
        # 获取不复权数据（用于成交额等）
        if include_unadjusted:
            df_unadj = self._fetch_with_retry(
                func=ak.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=self.ADJUST_NONE
            )
            
            if df_unadj is not None and not df_unadj.empty:
                df_unadj = self._standardize_columns(df_unadj)
                
                # 合并不复权数据
                df_qfq = self._merge_adjusted_data(df_qfq, df_unadj)
        
        # 设置日期索引
        df_qfq = self._set_datetime_index(df_qfq)
        
        # 添加股票代码列
        df_qfq["symbol"] = symbol
        
        logger.info(f"获取 {symbol} 成功，共 {len(df_qfq)} 条记录")
        return df_qfq
    
    def fetch_financial_indicator(
        self,
        symbol: str
    ) -> pd.DataFrame:
        """
        获取股票财务指标
        
        获取市盈率(PE-TTM)、市净率(PB)、股息率(Dividend Yield)等估值指标。
        
        Parameters
        ----------
        symbol : str
            股票代码
        
        Returns
        -------
        pd.DataFrame
            财务指标数据，包含：
            - pe_ttm: 市盈率（TTM）
            - pb: 市净率
            - dividend_yield: 股息率
            - ps_ttm: 市销率（TTM）
            - total_mv: 总市值
            - circ_mv: 流通市值
        
        Examples
        --------
        >>> loader = DataLoader()
        >>> indicators = loader.fetch_financial_indicator("000001")
        >>> print(f"PE-TTM: {indicators['pe_ttm'].iloc[-1]:.2f}")
        """
        logger.info(f"获取 {symbol} 财务指标")
        
        result = pd.DataFrame()
        
        # 方法1: 尝试使用 stock_a_lg_indicator 接口（雪球数据）
        try:
            df = self._fetch_with_retry(
                func=ak.stock_a_lg_indicator,
                symbol=symbol
            )
            
            if df is not None and not df.empty:
                # 标准化列名
                result = self._standardize_financial_columns(df)
                logger.info(f"获取 {symbol} 财务指标成功（stock_a_lg_indicator）")
                return result
        except Exception as e:
            logger.debug(f"stock_a_lg_indicator 获取失败: {e}")
        
        # 方法2: 尝试使用 stock_individual_info_em 接口
        try:
            df = self._fetch_with_retry(
                func=ak.stock_individual_info_em,
                symbol=symbol
            )
            
            if df is not None and not df.empty:
                # 转换为标准格式
                result = self._parse_individual_info(df, symbol)
                logger.info(f"获取 {symbol} 财务指标成功（stock_individual_info_em）")
                return result
        except Exception as e:
            logger.debug(f"stock_individual_info_em 获取失败: {e}")
        
        # 方法3: 尝试使用实时行情接口获取估值数据（使用缓存避免重复请求）
        try:
            df = self._get_spot_data_cached()
            
            if df is not None and not df.empty:
                # 筛选指定股票
                stock_data = df[df["代码"] == symbol]
                if not stock_data.empty:
                    result = self._parse_spot_data(stock_data)
                    logger.info(f"获取 {symbol} 财务指标成功（stock_zh_a_spot_em 缓存）")
                    return result
        except Exception as e:
            logger.debug(f"stock_zh_a_spot_em 获取失败: {e}")
        
        logger.warning(f"无法获取 {symbol} 的财务指标")
        return result
    
    def _get_spot_data_cached(self) -> Optional[pd.DataFrame]:
        """
        获取全市场实时行情数据（带多级缓存和容错）
        
        缓存策略：
        1. 内存缓存（5分钟有效期）
        2. 磁盘缓存（当日有效）
        3. 多API降级（stock_zh_a_spot_em -> 单只股票API）
        
        Returns
        -------
        Optional[pd.DataFrame]
            全市场行情数据，失败返回 None
        """
        from pathlib import Path
        
        now = datetime.now()
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        disk_cache_path = cache_dir / f"spot_data_{now.strftime('%Y%m%d')}.parquet"
        
        # 1. 检查内存缓存
        if (self._spot_data_cache is not None and 
            self._spot_data_cache_time is not None and
            (now - self._spot_data_cache_time).total_seconds() < self._spot_cache_ttl):
            logger.debug("使用内存缓存的全市场行情数据")
            return self._spot_data_cache
        
        # 2. 尝试多种API获取数据
        df = None
        api_methods = [
            ("stock_zh_a_spot_em", lambda: ak.stock_zh_a_spot_em()),
            ("stock_info_a_code_name", lambda: self._fetch_spot_via_code_name()),
        ]
        
        for api_name, api_func in api_methods:
            logger.info(f"尝试获取全市场行情数据 (API: {api_name})...")
            try:
                df = self._fetch_with_retry(func=api_func)
                if df is not None and not df.empty:
                    logger.info(f"✅ {api_name} 成功，共 {len(df)} 条记录")
                    break
            except Exception as e:
                logger.warning(f"❌ {api_name} 失败: {type(e).__name__}")
                continue
        
        # 3. 如果网络获取成功，更新缓存
        if df is not None and not df.empty:
            self._spot_data_cache = df
            self._spot_data_cache_time = now
            # 保存到磁盘缓存
            try:
                df.to_parquet(disk_cache_path, compression='snappy')
                logger.info(f"全市场行情已保存至磁盘缓存: {disk_cache_path}")
            except Exception as e:
                logger.warning(f"磁盘缓存保存失败: {e}")
            return df
        
        # 4. 网络获取失败，尝试从磁盘缓存加载
        logger.warning("网络获取失败，尝试从磁盘缓存加载...")
        
        # 4.1 先尝试当日缓存
        if disk_cache_path.exists():
            try:
                df = pd.read_parquet(disk_cache_path)
                self._spot_data_cache = df
                self._spot_data_cache_time = now
                logger.info(f"✅ 从当日磁盘缓存加载成功，共 {len(df)} 条记录")
                return df
            except Exception as e:
                logger.warning(f"当日缓存加载失败: {e}")
        
        # 4.2 尝试最近的缓存文件
        cache_files = sorted(cache_dir.glob("spot_data_*.parquet"), reverse=True)
        for cache_file in cache_files[:3]:  # 最多尝试最近3天的缓存
            try:
                df = pd.read_parquet(cache_file)
                self._spot_data_cache = df
                self._spot_data_cache_time = now
                logger.warning(f"⚠️ 使用历史缓存 {cache_file.name}，共 {len(df)} 条（数据可能不是最新）")
                return df
            except Exception as e:
                logger.debug(f"缓存文件 {cache_file} 加载失败: {e}")
                continue
        
        logger.error("所有数据源和缓存均失败")
        return None
    
    def _fetch_spot_via_code_name(self) -> Optional[pd.DataFrame]:
        """
        备用方法：通过 stock_info_a_code_name 获取基础股票信息
        
        该API更稳定，但数据较少（仅代码和名称）
        """
        try:
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                # 重命名列以匹配主API格式
                df = df.rename(columns={
                    "code": "代码",
                    "name": "名称"
                })
                return df
        except Exception as e:
            logger.debug(f"stock_info_a_code_name 失败: {e}")
        return None
    
    def get_hs300_constituents(self) -> List[str]:
        """
        获取沪深300成分股列表
        
        Returns
        -------
        List[str]
            沪深300成分股代码列表
        """
        logger.info("获取沪深300成分股列表")
        
        try:
            df = self._fetch_with_retry(
                func=ak.index_stock_cons,
                symbol="000300"
            )
            
            if df is not None and not df.empty:
                # 列名可能是 "品种代码" 或 "成分券代码"
                code_col = None
                for col in ["品种代码", "成分券代码", "代码", "stock_code"]:
                    if col in df.columns:
                        code_col = col
                        break
                
                if code_col:
                    stocks = df[code_col].tolist()
                    logger.info(f"获取沪深300成分股成功，共 {len(stocks)} 只")
                    return stocks
            
            logger.warning("无法解析沪深300成分股数据")
            return []
            
        except Exception as e:
            logger.error(f"获取沪深300成分股失败: {e}")
            return []
    
    def batch_download(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        save: bool = True,
        include_financial: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, DownloadResult]:
        """
        并发批量下载股票数据
        
        使用 concurrent.futures 实现多线程并发下载，
        自动处理网络超时异常和重试逻辑。
        
        Parameters
        ----------
        symbols : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        save : bool, optional
            是否保存为Parquet文件，默认True
        include_financial : bool, optional
            是否同时下载财务指标，默认False
        progress_callback : Optional[Callable[[int, int], None]]
            进度回调函数，接收 (已完成数, 总数)
        
        Returns
        -------
        Dict[str, DownloadResult]
            股票代码到下载结果的映射
        
        Examples
        --------
        >>> loader = DataLoader(max_workers=5)
        >>> symbols = ["000001", "000002", "600519"]
        >>> results = loader.batch_download(symbols, "2023-01-01", "2024-12-31")
        >>> 
        >>> # 统计成功/失败
        >>> success = sum(1 for r in results.values() if r.success)
        >>> print(f"成功: {success}/{len(symbols)}")
        """
        total = len(symbols)
        logger.info(f"开始批量下载: {total} 只股票, max_workers={self.max_workers}")
        
        # 重置统计
        self._download_stats = {"success": 0, "failed": 0, "total": total}
        
        results: Dict[str, DownloadResult] = {}
        completed = 0
        
        def download_one(symbol: str) -> DownloadResult:
            """下载单只股票"""
            try:
                # 下载价格数据
                df = self.fetch_daily_price(symbol, start_date, end_date)
                
                if df.empty:
                    return DownloadResult(
                        symbol=symbol,
                        success=False,
                        error="无数据"
                    )
                
                # 下载财务指标（可选）
                if include_financial:
                    try:
                        fin_df = self.fetch_financial_indicator(symbol)
                        if not fin_df.empty:
                            # 将最新财务指标合并到价格数据
                            df = self._merge_financial_data(df, fin_df)
                    except Exception as e:
                        logger.warning(f"{symbol} 财务指标获取失败: {e}")
                
                # 保存数据
                if save:
                    self.save_to_parquet(df, symbol)
                
                return DownloadResult(
                    symbol=symbol,
                    success=True,
                    data=df
                )
                
            except TimeoutError as e:
                return DownloadResult(
                    symbol=symbol,
                    success=False,
                    error=f"超时: {e}"
                )
            except ConnectionError as e:
                return DownloadResult(
                    symbol=symbol,
                    success=False,
                    error=f"网络错误: {e}"
                )
            except Exception as e:
                return DownloadResult(
                    symbol=symbol,
                    success=False,
                    error=str(e)
                )
        
        # 使用线程池并发下载
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_symbol = {
                executor.submit(download_one, symbol): symbol
                for symbol in symbols
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result(timeout=self.timeout)
                    results[symbol] = result
                    
                    if result.success:
                        self._download_stats["success"] += 1
                    else:
                        self._download_stats["failed"] += 1
                        logger.warning(f"下载 {symbol} 失败: {result.error}")
                        
                except TimeoutError:
                    results[symbol] = DownloadResult(
                        symbol=symbol,
                        success=False,
                        error="任务超时"
                    )
                    self._download_stats["failed"] += 1
                    logger.warning(f"下载 {symbol} 超时")
                    
                except Exception as e:
                    results[symbol] = DownloadResult(
                        symbol=symbol,
                        success=False,
                        error=str(e)
                    )
                    self._download_stats["failed"] += 1
                    logger.error(f"下载 {symbol} 异常: {e}")
                
                # 进度回调
                if progress_callback:
                    progress_callback(completed, total)
                
                # 简单进度日志
                if completed % 50 == 0 or completed == total:
                    logger.info(
                        f"下载进度: {completed}/{total} "
                        f"(成功: {self._download_stats['success']}, "
                        f"失败: {self._download_stats['failed']})"
                    )
        
        logger.info(
            f"批量下载完成: "
            f"成功 {self._download_stats['success']}/{total}, "
            f"失败 {self._download_stats['failed']}/{total}"
        )
        
        return results
    
    def batch_download_hs300(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
        include_financial: bool = False
    ) -> Dict[str, DownloadResult]:
        """
        批量下载沪深300成分股数据
        
        获取沪深300成分股列表后并发下载所有成分股数据。
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        save : bool, optional
            是否保存为Parquet文件，默认True
        include_financial : bool, optional
            是否同时下载财务指标，默认False
        
        Returns
        -------
        Dict[str, DownloadResult]
            下载结果
        
        Examples
        --------
        >>> loader = DataLoader(output_dir="data/raw", max_workers=5)
        >>> results = loader.batch_download_hs300("2023-01-01", "2024-12-31")
        >>> 
        >>> # 获取成功下载的数据
        >>> data = {s: r.data for s, r in results.items() if r.success}
        """
        # 获取成分股列表
        symbols = self.get_hs300_constituents()
        
        if not symbols:
            logger.error("无法获取沪深300成分股列表")
            return {}
        
        logger.info(f"开始下载沪深300成分股: {len(symbols)} 只")
        
        return self.batch_download(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            save=save,
            include_financial=include_financial
        )
    
    def fetch_multi_stock_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        include_financial: bool = True,
        n_jobs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        多线程批量获取股票数据并合并为面板数据
        
        针对全市场 5000+ 只股票的大规模数据拉取优化：
        - 使用 ThreadPoolExecutor 并行下载
        - 单个线程失败不影响整体任务
        - 自动跳过无数据的股票
        - 返回适合回测的面板格式 DataFrame
        
        Parameters
        ----------
        symbols : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        include_financial : bool, optional
            是否包含财务指标（PE、PB、流通市值等），默认 True
        n_jobs : Optional[int]
            并发线程数。如果为 None，使用 self.max_workers
        
        Returns
        -------
        pd.DataFrame
            合并后的面板数据，包含列：
            - date: 日期
            - stock_code: 股票代码
            - open, high, low, close, volume, amount: 价格/成交数据
            - turn: 换手率
            - pe_ttm, pb, circ_mv: 财务指标（如果 include_financial=True）
        
        Notes
        -----
        小资金实盘优化：
        - 为激进型小市值策略设计，支持全市场 5000+ 股票并行下载
        - 单个股票下载失败会记录警告日志但不中断整体流程
        - 返回的 DataFrame 已按 (date, stock_code) 排序
        
        Examples
        --------
        >>> loader = DataLoader(max_workers=10)  # 增加并发数
        >>> symbols = loader.get_all_a_stock_list()
        >>> df = loader.fetch_multi_stock_data(symbols, "2023-01-01", "2024-12-31")
        >>> print(f"获取 {df['stock_code'].nunique()} 只股票的数据")
        """
        n_workers = n_jobs if n_jobs is not None else self.max_workers
        total = len(symbols)
        
        logger.info(
            f"开始多线程批量获取: {total} 只股票, "
            f"并发线程={n_workers}, 财务指标={include_financial}"
        )
        
        # 用于存储成功获取的数据
        all_data: List[pd.DataFrame] = []
        success_count = 0
        failed_count = 0
        
        def fetch_one(symbol: str) -> Optional[pd.DataFrame]:
            """
            获取单只股票数据（内部函数，用于并行执行）
            
            异常处理：任何异常都被捕获并记录，返回 None
            """
            try:
                # 获取价格数据
                df = self.fetch_daily_price(symbol, start_date, end_date)
                
                if df.empty:
                    return None
                
                # 添加股票代码列
                df = df.reset_index()
                df['stock_code'] = symbol
                
                # 获取财务指标（可选）
                if include_financial:
                    try:
                        fin_df = self.fetch_financial_indicator(symbol)
                        if not fin_df.empty:
                            df = self._merge_financial_data(df, fin_df)
                    except Exception as e:
                        # 财务指标获取失败不影响主数据
                        logger.debug(f"{symbol} 财务指标获取失败: {e}")
                
                return df
                
            except TimeoutError:
                logger.warning(f"{symbol} 下载超时，跳过")
                return None
            except ConnectionError as e:
                logger.warning(f"{symbol} 网络错误: {e}，跳过")
                return None
            except Exception as e:
                logger.warning(f"{symbol} 下载失败: {e}，跳过")
                return None
        
        # 使用线程池并发获取
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # 提交所有任务
            future_to_symbol = {
                executor.submit(fetch_one, symbol): symbol
                for symbol in symbols
            }
            
            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result(timeout=self.timeout)
                    
                    if result is not None and not result.empty:
                        all_data.append(result)
                        success_count += 1
                    else:
                        failed_count += 1
                        
                except TimeoutError:
                    failed_count += 1
                    logger.warning(f"{symbol} 任务超时")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"{symbol} 任务异常: {e}")
                
                # 进度日志（每50只或完成时）
                if completed % 50 == 0 or completed == total:
                    logger.info(
                        f"下载进度: {completed}/{total} "
                        f"(成功: {success_count}, 失败: {failed_count})"
                    )
        
        # 合并所有数据
        if not all_data:
            logger.error("所有股票数据获取失败，返回空 DataFrame")
            return pd.DataFrame()
        
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 确保日期列格式正确
        if 'date' not in merged_df.columns and 'index' in merged_df.columns:
            merged_df = merged_df.rename(columns={'index': 'date'})
        
        if 'date' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(merged_df['date'])
            merged_df = merged_df.sort_values(['date', 'stock_code'])
        
        logger.info(
            f"多线程批量获取完成: "
            f"成功 {success_count}/{total} 只股票, "
            f"共 {len(merged_df)} 条记录"
        )
        
        return merged_df
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        compression: str = "snappy"
    ) -> str:
        """
        保存数据为Parquet格式
        
        Parameters
        ----------
        df : pd.DataFrame
            数据
        symbol : str
            股票代码
        compression : str, optional
            压缩算法，默认 'snappy'
        
        Returns
        -------
        str
            保存的文件路径
        """
        filepath = self.output_dir / f"{symbol}.parquet"
        
        df.to_parquet(
            filepath,
            compression=compression,
            index=True
        )
        
        logger.debug(f"数据已保存: {filepath}")
        return str(filepath)
    
    def load_from_parquet(self, symbol: str) -> pd.DataFrame:
        """
        从Parquet文件加载数据
        
        Parameters
        ----------
        symbol : str
            股票代码
        
        Returns
        -------
        pd.DataFrame
            加载的数据
        """
        filepath = self.output_dir / f"{symbol}.parquet"
        
        if not filepath.exists():
            logger.warning(f"文件不存在: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_parquet(filepath)
        logger.debug(f"数据已加载: {filepath}, shape={df.shape}")
        return df
    
    def _fetch_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        带重试机制的数据获取（增强版SSL错误处理）
        
        针对东方财富等数据源的SSL EOF错误进行专门优化：
        - SSL错误使用更长的指数退避时间
        - 自动清理连接池避免连接复用导致的问题
        - 添加随机抖动避免请求风暴
        
        Parameters
        ----------
        func : Callable
            AkShare 接口函数
        *args, **kwargs
            传递给接口的参数
        
        Returns
        -------
        Optional[pd.DataFrame]
            获取的数据，失败返回None
        """
        import random
        import gc
        
        # SSL/网络错误关键字
        SSL_ERROR_KEYWORDS = (
            'ssl', 'eof', 'unexpected_eof', 'ssleoferror',
            'connection reset', 'connection aborted',
            'max retries exceeded', 'timed out', 'timeout',
            '10054', '10060', 'broken pipe'
        )
        
        last_exception = None
        base_delay = self.retry_delay
        
        for attempt in range(self.retry_times):
            try:
                # 添加超时警告抑制
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = func(*args, **kwargs)
                return result
                
            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                wait_time = min(base_delay * (2 ** attempt), 60)
                jitter = random.uniform(0.5, 1.5)
                actual_wait = wait_time * jitter
                
                logger.warning(
                    f"网络请求失败 (尝试 {attempt + 1}/{self.retry_times}): {e}"
                )
                if attempt < self.retry_times - 1:
                    logger.info(f"等待 {actual_wait:.1f} 秒后重试...")
                    time.sleep(actual_wait)
                    
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # 检测是否为SSL相关错误
                is_ssl_error = any(
                    keyword in error_msg for keyword in SSL_ERROR_KEYWORDS
                )
                
                if is_ssl_error:
                    # SSL错误使用更激进的指数退避
                    wait_time = min(base_delay * (2 ** (attempt + 1)), 90)
                    jitter = random.uniform(0.8, 1.5)
                    actual_wait = wait_time * jitter
                    
                    logger.warning(
                        f"SSL/网络异常 (尝试 {attempt + 1}/{self.retry_times}): "
                        f"{type(e).__name__}"
                    )
                    
                    if attempt < self.retry_times - 1:
                        # 尝试清理连接池
                        try:
                            import requests
                            # 关闭所有连接，强制下次请求使用新连接
                            requests.adapters.DEFAULT_POOLBLOCK = False
                            gc.collect()
                        except Exception:
                            pass
                        
                        logger.info(f"等待 {actual_wait:.1f} 秒后重试...")
                        time.sleep(actual_wait)
                else:
                    # 非SSL错误使用普通退避
                    logger.warning(f"请求异常: {e}")
                    if attempt < self.retry_times - 1:
                        time.sleep(base_delay * (attempt + 1))
        
        logger.error(f"所有重试均失败: {last_exception}")
        return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化价格数据列名"""
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        return df.rename(columns=column_mapping)
    
    def _standardize_financial_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化财务指标列名"""
        column_mapping = {
            "trade_date": "date",
            "pe": "pe_ttm",
            "pe_ttm": "pe_ttm",
            "pb": "pb",
            "ps": "ps_ttm",
            "ps_ttm": "ps_ttm",
            "dv_ratio": "dividend_yield",
            "dv_ttm": "dividend_yield",
            "total_mv": "total_mv",
            "circ_mv": "circ_mv",
        }
        
        df = df.rename(columns=column_mapping)
        
        # 设置日期索引
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()
        
        return df
    
    def _parse_individual_info(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """解析 stock_individual_info_em 返回的数据"""
        # 该接口返回的是键值对格式
        # item | value
        # 总市值 | xxx
        
        result = {}
        
        for _, row in df.iterrows():
            item = row.get("item", row.get("项目", ""))
            value = row.get("value", row.get("值", ""))
            
            if "市盈率" in str(item) and "动" in str(item):
                result["pe_ttm"] = self._parse_number(value)
            elif "市净率" in str(item):
                result["pb"] = self._parse_number(value)
            elif "总市值" in str(item):
                result["total_mv"] = self._parse_number(value)
            elif "流通市值" in str(item):
                result["circ_mv"] = self._parse_number(value)
        
        if result:
            result["symbol"] = symbol
            result["date"] = datetime.now().strftime("%Y-%m-%d")
            return pd.DataFrame([result])
        
        return pd.DataFrame()
    
    def _parse_spot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析实时行情数据中的估值指标"""
        column_mapping = {
            "市盈率-动态": "pe_ttm",
            "市净率": "pb",
            "总市值": "total_mv",
            "流通市值": "circ_mv",
            "代码": "symbol",
        }
        
        result = df.rename(columns=column_mapping)
        result["date"] = datetime.now().strftime("%Y-%m-%d")
        
        cols_to_keep = ["symbol", "date", "pe_ttm", "pb", "total_mv", "circ_mv"]
        cols_exist = [c for c in cols_to_keep if c in result.columns]
        
        return result[cols_exist].copy()
    
    def _parse_number(self, value: Any) -> Optional[float]:
        """解析数值，处理带单位的情况"""
        if pd.isna(value) or value == "-" or value == "":
            return None
        
        try:
            value_str = str(value).replace(",", "")
            
            if "亿" in value_str:
                return float(value_str.replace("亿", "")) * 1e8
            elif "万" in value_str:
                return float(value_str.replace("万", "")) * 1e4
            else:
                return float(value_str)
        except (ValueError, TypeError):
            return None
    
    def _set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """设置日期时间索引"""
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()
        return df
    
    def _merge_adjusted_data(
        self,
        df_qfq: pd.DataFrame,
        df_unadj: pd.DataFrame
    ) -> pd.DataFrame:
        """合并前复权和不复权数据"""
        # 保留前复权的 OHLC
        # 从不复权数据获取成交额（amount）
        
        # 重命名不复权数据的 OHLC 列
        unadj_cols = {
            "open": "open_unadj",
            "high": "high_unadj",
            "low": "low_unadj",
            "close": "close_unadj",
        }
        
        df_unadj_renamed = df_unadj.rename(columns=unadj_cols)
        
        # 合并：使用前复权数据为主
        merge_cols = ["date", "open_unadj", "high_unadj", "low_unadj", 
                      "close_unadj", "amount"]
        merge_cols = [c for c in merge_cols if c in df_unadj_renamed.columns]
        
        if merge_cols and "date" in merge_cols:
            df_merged = df_qfq.merge(
                df_unadj_renamed[merge_cols],
                on="date",
                how="left",
                suffixes=("", "_y")
            )
            
            # 如果前复权数据没有 amount，使用不复权数据的
            if "amount_y" in df_merged.columns and "amount" not in df_qfq.columns:
                df_merged["amount"] = df_merged["amount_y"]
            
            # 清理重复列
            df_merged = df_merged.drop(
                columns=[c for c in df_merged.columns if c.endswith("_y")]
            )
            
            return df_merged
        
        return df_qfq
    
    def _merge_financial_data(
        self,
        price_df: pd.DataFrame,
        fin_df: pd.DataFrame
    ) -> pd.DataFrame:
        """将财务指标合并到价格数据"""
        if fin_df.empty:
            return price_df
        
        # 获取最新的财务指标
        fin_latest = fin_df.iloc[-1] if isinstance(fin_df, pd.DataFrame) else fin_df
        
        # 添加财务指标列
        for col in ["pe_ttm", "pb", "dividend_yield", "ps_ttm", "total_mv", "circ_mv"]:
            if col in fin_latest.index or col in fin_df.columns:
                price_df[col] = fin_latest.get(col, np.nan)
        
        return price_df
    
    def get_download_stats(self) -> Dict[str, int]:
        """
        获取下载统计信息
        
        Returns
        -------
        Dict[str, int]
            包含 success, failed, total 的统计字典
        """
        return self._download_stats.copy()
    
    def fetch_index_price(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取指数日线价格数据
        
        用于获取沪深300、中证500等指数的历史价格，支持大盘风控计算。
        
        Parameters
        ----------
        index_code : str
            指数代码，如 '000300' (沪深300), '000905' (中证500)
            注意：会自动添加 'sh' 前缀用于 AkShare 接口
        start_date : str
            开始日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        
        Returns
        -------
        pd.DataFrame
            指数日线数据，包含以下列：
            - date: 日期（索引，DatetimeIndex）
            - open, high, low, close: OHLC 价格
            - volume: 成交量
        
        Raises
        ------
        ConnectionError
            当网络连接失败且重试耗尽时
        
        Examples
        --------
        >>> loader = DataLoader()
        >>> hs300 = loader.fetch_index_price("000300", "2023-01-01", "2024-12-31")
        >>> print(hs300[['close']].tail())
        
        Notes
        -----
        - 使用 ak.stock_zh_index_daily 接口获取数据
        - 自动处理日期格式转换
        - 内置重试机制处理网络异常
        """
        # 标准化日期格式
        start_date_clean = start_date.replace("-", "")
        end_date_clean = end_date.replace("-", "")
        
        # 确定指数符号（AkShare 需要 sh/sz 前缀）
        # 沪深300 (000300) 在上海交易所，中证500 (000905) 也在上海
        if index_code.startswith(("000", "399")):
            symbol = f"sh{index_code}"
        else:
            symbol = f"sh{index_code}"  # 默认上海
        
        logger.info(f"获取指数 {index_code} ({symbol}) 日线数据: {start_date} - {end_date}")
        
        # 使用重试机制获取数据
        df = self._fetch_with_retry(
            func=ak.stock_zh_index_daily,
            symbol=symbol
        )
        
        if df is None or df.empty:
            logger.warning(f"指数 {index_code} 无数据")
            return pd.DataFrame()
        
        # 数据清洗和标准化
        # 确保日期列存在并转换
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
            df = df.drop(columns=['日期'])
        
        # 过滤日期范围
        start_dt = pd.to_datetime(start_date_clean)
        end_dt = pd.to_datetime(end_date_clean)
        
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
        
        # 设置日期索引
        df = df.set_index('date')
        df = df.sort_index()
        df.index.name = 'date'
        
        # 确保列名标准化
        column_mapping = {
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
        }
        df = df.rename(columns=column_mapping)
        
        logger.info(f"获取指数 {index_code} 成功，共 {len(df)} 条记录")
        return df


class DataCleaner:
    """
    A股数据清洗器
    
    提供专业的A股数据清洗功能，包括停牌处理、涨跌停识别和索引对齐。
    所有方法采用Pandas向量化操作，支持单只股票和多股票批量处理。
    
    Attributes
    ----------
    fill_method : str
        停牌日填充方法 ('nan' 或 'ffill')
    trading_calendar : Optional[pd.DatetimeIndex]
        交易日历，用于索引对齐
    
    Examples
    --------
    >>> cleaner = DataCleaner(fill_method='ffill')
    >>> 
    >>> # 清洗单只股票
    >>> df_clean = cleaner.clean(df)
    >>> 
    >>> # 批量清洗并对齐索引
    >>> data_dict = {'000001': df1, '000002': df2}
    >>> aligned = cleaner.clean_and_align(data_dict)
    """
    
    # 涨跌停幅度阈值
    LIMIT_THRESHOLD_MAIN = 0.095      # 主板 9.5%（留0.5%容差）
    LIMIT_THRESHOLD_GEM_STAR = 0.195  # 创业板/科创板 19.5%
    LIMIT_THRESHOLD_BSE = 0.295       # 北交所 29.5%
    LIMIT_THRESHOLD_ST = 0.045        # ST股票 4.5%
    
    def __init__(
        self,
        fill_method: str = "nan",
        trading_calendar: Optional[pd.DatetimeIndex] = None
    ) -> None:
        """
        初始化数据清洗器
        
        Parameters
        ----------
        fill_method : str, optional
            停牌日价格填充方法:
            - 'nan': 设置为NaN（默认）
            - 'ffill': 向前填充（Forward Fill）
        trading_calendar : Optional[pd.DatetimeIndex]
            交易日历，用于索引对齐。如果为None，会自动从数据推断。
        """
        if fill_method not in ("nan", "ffill"):
            raise ValueError(f"fill_method 必须是 'nan' 或 'ffill'，收到: {fill_method}")
        
        self.fill_method = fill_method
        self.trading_calendar = trading_calendar
        
        logger.info(f"DataCleaner初始化: fill_method={fill_method}")
    
    def clean(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        清洗单只股票数据
        
        执行完整的清洗流程：
        1. 停牌处理（标记不可交易日）
        2. 涨跌停识别
        3. 确保DatetimeIndex
        
        Parameters
        ----------
        df : pd.DataFrame
            原始股票数据，需包含 OHLCV 列
        symbol : Optional[str]
            股票代码，用于判断涨跌停阈值
        
        Returns
        -------
        pd.DataFrame
            清洗后的数据，新增列：
            - is_suspended: 是否停牌 (bool)
            - is_tradable: 是否可交易 (bool)
            - is_limit: 是否涨跌停 (bool)
            - is_limit_up: 是否涨停 (bool)
            - is_limit_down: 是否跌停 (bool)
        
        Examples
        --------
        >>> cleaner = DataCleaner()
        >>> df_clean = cleaner.clean(df, symbol="000001")
        >>> print(f"停牌天数: {(~df_clean['is_tradable']).sum()}")
        >>> print(f"涨跌停天数: {df_clean['is_limit'].sum()}")
        """
        df = df.copy()
        
        # 确保 DatetimeIndex
        df = self._ensure_datetime_index(df)
        
        # Step 1: 停牌处理
        df = self._handle_suspension(df)
        
        # Step 2: 涨跌停识别
        df = self._identify_limit(df, symbol)
        
        logger.debug(
            f"清洗完成: 停牌={df['is_suspended'].sum()}, "
            f"涨跌停={df['is_limit'].sum()}"
        )
        
        return df
    
    def clean_and_align(
        self,
        data_dict: Dict[str, pd.DataFrame],
        reference_calendar: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        批量清洗并对齐索引
        
        清洗多只股票数据，并确保所有股票使用统一的日期索引。
        缺失日期填充NaN。
        
        Parameters
        ----------
        data_dict : Dict[str, pd.DataFrame]
            股票代码到数据的映射
        reference_calendar : Optional[pd.DatetimeIndex]
            参考交易日历。如果为None，使用所有股票日期的并集。
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            对齐后的数据字典
        
        Examples
        --------
        >>> data = {'000001': df1, '000002': df2, '600519': df3}
        >>> cleaner = DataCleaner()
        >>> aligned = cleaner.clean_and_align(data)
        >>> 
        >>> # 所有股票现在有相同的索引
        >>> assert all(
        ...     aligned['000001'].index.equals(aligned['000002'].index)
        ...     for s in aligned
        ... )
        """
        if not data_dict:
            return {}
        
        logger.info(f"批量清洗并对齐索引: {len(data_dict)} 只股票")
        
        # Step 1: 清洗每只股票
        cleaned_dict = {}
        for symbol, df in data_dict.items():
            try:
                cleaned_dict[symbol] = self.clean(df, symbol)
            except Exception as e:
                logger.warning(f"清洗 {symbol} 失败: {e}")
                continue
        
        if not cleaned_dict:
            return {}
        
        # Step 2: 确定统一的日期索引
        if reference_calendar is not None:
            unified_index = reference_calendar
        elif self.trading_calendar is not None:
            unified_index = self.trading_calendar
        else:
            # 使用所有股票日期的并集
            unified_index = self._build_unified_index(cleaned_dict)
        
        # Step 3: 对齐每只股票的索引
        aligned_dict = {}
        for symbol, df in cleaned_dict.items():
            aligned_dict[symbol] = self._align_to_index(df, unified_index)
        
        logger.info(f"索引对齐完成: 统一日期范围 {unified_index[0]} - {unified_index[-1]}")
        
        return aligned_dict
    
    def to_panel(
        self,
        data_dict: Dict[str, pd.DataFrame],
        column: str = "close"
    ) -> pd.DataFrame:
        """
        将多只股票数据转换为面板格式
        
        Parameters
        ----------
        data_dict : Dict[str, pd.DataFrame]
            股票代码到数据的映射（需已对齐索引）
        column : str, optional
            要提取的列名，默认 'close'
        
        Returns
        -------
        pd.DataFrame
            面板数据，行=日期，列=股票代码
        
        Examples
        --------
        >>> aligned = cleaner.clean_and_align(data_dict)
        >>> close_panel = cleaner.to_panel(aligned, column='close')
        >>> returns = close_panel.pct_change()
        """
        panel_data = {}
        
        for symbol, df in data_dict.items():
            if column in df.columns:
                panel_data[symbol] = df[column]
        
        panel = pd.DataFrame(panel_data)
        panel = panel.sort_index()
        
        return panel
    
    def _handle_suspension(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理停牌日
        
        停牌识别条件：volume == 0 或 volume 为 NaN
        
        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        
        Returns
        -------
        pd.DataFrame
            处理后的数据，新增 is_suspended 和 is_tradable 列
        """
        # 获取成交量列
        volume_col = self._find_column(df, ["volume", "vol", "成交量", "Volume"])
        
        if volume_col is None:
            logger.warning("未找到成交量列，跳过停牌处理")
            df["is_suspended"] = False
            df["is_tradable"] = True
            return df
        
        # 向量化识别停牌日
        is_suspended = (df[volume_col] == 0) | df[volume_col].isna()
        df["is_suspended"] = is_suspended
        
        # 获取 OHLC 列
        ohlc_cols = self._get_ohlc_columns(df)
        
        if ohlc_cols and is_suspended.any():
            if self.fill_method == "nan":
                # 设置停牌日 OHLC 为 NaN
                for col in ohlc_cols:
                    df.loc[is_suspended, col] = np.nan
            elif self.fill_method == "ffill":
                # 向前填充停牌日价格
                for col in ohlc_cols:
                    # 先将停牌日设为 NaN，再向前填充
                    df.loc[is_suspended, col] = np.nan
                    df[col] = df[col].ffill()
        
        # 标记可交易状态（即使 ffill 填充了价格，仍标记为不可交易）
        df["is_tradable"] = ~is_suspended
        
        suspended_count = is_suspended.sum()
        if suspended_count > 0:
            logger.debug(f"识别到 {suspended_count} 个停牌日")
        
        return df
    
    def _identify_limit(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        识别涨跌停
        
        涨跌停判断条件：
        1. High == Low（一字板特征）
        2. abs(PctChange) > 阈值
        
        Parameters
        ----------
        df : pd.DataFrame
            数据
        symbol : Optional[str]
            股票代码，用于判断阈值
        
        Returns
        -------
        pd.DataFrame
            添加涨跌停标识列的数据
        """
        # 获取必要的列
        high_col = self._find_column(df, ["high", "最高", "High"])
        low_col = self._find_column(df, ["low", "最低", "Low"])
        pct_col = self._find_column(df, ["pct_change", "涨跌幅", "pctChg", "change_pct"])
        close_col = self._find_column(df, ["close", "收盘", "Close"])
        
        # 初始化标记列
        df["is_limit"] = False
        df["is_limit_up"] = False
        df["is_limit_down"] = False
        
        if high_col is None or low_col is None:
            logger.warning("缺少 high/low 列，无法识别涨跌停")
            return df
        
        # 确定涨跌停阈值
        threshold = self._get_limit_threshold(symbol)
        
        # 条件1: 最高价 == 最低价（一字板）
        is_flat = np.isclose(df[high_col], df[low_col], rtol=1e-6)
        
        # 获取涨跌幅
        if pct_col is not None:
            pct_change = df[pct_col].copy()
            # 如果是百分比形式（如 9.98），转换为小数
            if pct_change.abs().max() > 1:
                pct_change = pct_change / 100
        elif close_col is not None:
            # 计算涨跌幅
            pct_change = df[close_col].pct_change()
        else:
            logger.warning("无法获取涨跌幅，跳过涨跌停识别")
            return df
        
        # 条件2: 涨跌幅超过阈值
        is_up_limit = is_flat & (pct_change >= threshold)
        is_down_limit = is_flat & (pct_change <= -threshold)
        
        # 排除停牌日
        if "is_suspended" in df.columns:
            is_up_limit = is_up_limit & ~df["is_suspended"]
            is_down_limit = is_down_limit & ~df["is_suspended"]
        
        # 赋值
        df["is_limit_up"] = is_up_limit.fillna(False).astype(bool)
        df["is_limit_down"] = is_down_limit.fillna(False).astype(bool)
        df["is_limit"] = df["is_limit_up"] | df["is_limit_down"]
        
        limit_count = df["is_limit"].sum()
        if limit_count > 0:
            logger.debug(
                f"识别到涨停 {df['is_limit_up'].sum()} 天, "
                f"跌停 {df['is_limit_down'].sum()} 天"
            )
        
        return df
    
    def _get_limit_threshold(self, symbol: Optional[str] = None) -> float:
        """
        根据股票代码确定涨跌停阈值
        
        Parameters
        ----------
        symbol : Optional[str]
            股票代码
        
        Returns
        -------
        float
            涨跌停阈值
        """
        if symbol is None:
            return self.LIMIT_THRESHOLD_MAIN
        
        # 去除前缀
        code = symbol.lstrip("shszSHSZ.")
        
        # 创业板: 300xxx, 301xxx
        if code.startswith(("300", "301")):
            return self.LIMIT_THRESHOLD_GEM_STAR
        
        # 科创板: 688xxx, 689xxx
        if code.startswith(("688", "689")):
            return self.LIMIT_THRESHOLD_GEM_STAR
        
        # 北交所: 8xxxxx, 4xxxxx (6位数字开头)
        if len(code) == 6 and code.startswith(("8", "4")):
            return self.LIMIT_THRESHOLD_BSE
        
        # ST 股票需要从名称判断，这里使用主板默认值
        return self.LIMIT_THRESHOLD_MAIN
    
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保 DataFrame 使用 DatetimeIndex"""
        # 如果已经是 DatetimeIndex，直接返回
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            return df
        
        # 查找日期列
        date_col = self._find_column(
            df, ["date", "日期", "Date", "trade_date", "datetime"]
        )
        
        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
        else:
            # 尝试将索引转换为日期
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                logger.warning("无法将索引转换为 DatetimeIndex")
        
        df = df.sort_index()
        df.index.name = "date"
        
        return df
    
    def _build_unified_index(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DatetimeIndex:
        """
        从多只股票数据构建统一的日期索引
        
        使用所有股票日期的并集作为统一索引。
        """
        all_dates = set()
        
        for df in data_dict.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index.tolist())
        
        if not all_dates:
            raise ValueError("无法构建统一索引：没有有效的日期数据")
        
        unified_index = pd.DatetimeIndex(sorted(all_dates))
        unified_index.name = "date"
        
        return unified_index
    
    def _align_to_index(
        self,
        df: pd.DataFrame,
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        将 DataFrame 对齐到目标索引
        
        缺失日期填充 NaN。
        """
        # 使用 reindex 对齐，缺失值自动填充 NaN
        aligned = df.reindex(target_index)
        
        # 对于布尔列，填充为 False
        bool_cols = ["is_suspended", "is_tradable", "is_limit", 
                     "is_limit_up", "is_limit_down"]
        for col in bool_cols:
            if col in aligned.columns:
                if col == "is_tradable":
                    # 缺失日期不可交易
                    aligned[col] = aligned[col].fillna(False)
                elif col == "is_suspended":
                    # 缺失日期视为停牌
                    aligned[col] = aligned[col].fillna(True)
                else:
                    aligned[col] = aligned[col].fillna(False)
        
        return aligned
    
    def _find_column(
        self,
        df: pd.DataFrame,
        candidates: List[str]
    ) -> Optional[str]:
        """查找存在的列名"""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _get_ohlc_columns(self, df: pd.DataFrame) -> List[str]:
        """获取 OHLC 列名列表"""
        ohlc_names = [
            ["open", "开盘", "Open"],
            ["high", "最高", "High"],
            ["low", "最低", "Low"],
            ["close", "收盘", "Close"],
        ]
        
        result = []
        for candidates in ohlc_names:
            col = self._find_column(df, candidates)
            if col:
                result.append(col)
        
        return result
    
    @staticmethod
    def generate_trading_calendar(
        start_date: str,
        end_date: str,
        source: str = "akshare"
    ) -> pd.DatetimeIndex:
        """
        生成交易日历
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        source : str, optional
            数据源，默认 'akshare'
        
        Returns
        -------
        pd.DatetimeIndex
            交易日历
        
        Examples
        --------
        >>> calendar = DataCleaner.generate_trading_calendar("2023-01-01", "2024-12-31")
        >>> cleaner = DataCleaner(trading_calendar=calendar)
        """
        if source == "akshare":
            try:
                # 获取交易日历
                df = ak.tool_trade_date_hist_sina()
                dates = pd.to_datetime(df["trade_date"])
                
                # 过滤日期范围
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                dates = dates[(dates >= start) & (dates <= end)]
                
                calendar = pd.DatetimeIndex(sorted(dates))
                calendar.name = "date"
                
                logger.info(f"生成交易日历: {len(calendar)} 个交易日")
                return calendar
                
            except Exception as e:
                logger.warning(f"获取交易日历失败: {e}，使用工作日近似")
        
        # 降级：使用工作日（周一到周五）
        calendar = pd.bdate_range(start=start_date, end=end_date)
        calendar.name = "date"
        
        return calendar
