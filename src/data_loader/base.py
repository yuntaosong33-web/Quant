"""
数据处理抽象基类

定义数据获取和处理的标准接口，所有数据源实现类必须继承此类。
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
import logging
import time
import random

import pandas as pd

logger = logging.getLogger(__name__)


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
        
        Notes
        -----
        使用指数退避策略处理网络错误
        """
        # 网络相关异常类型
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
                
                is_network_error = any(
                    keyword in error_msg for keyword in NETWORK_ERROR_KEYWORDS
                )
                
                if is_network_error:
                    wait_time = min(base_delay * (2 ** attempt), 60)
                    jitter = random.uniform(0.5, 1.5)
                    wait_time = wait_time * jitter
                    error_type = "网络/SSL错误"
                else:
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


__all__ = ['DataHandler']

