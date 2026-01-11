"""
因子计算引擎抽象基类

定义因子计算的标准接口，所有因子计算类必须继承此类。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Callable
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngine(ABC):
    """
    因子计算引擎抽象基类
    
    定义因子计算的标准接口，所有因子计算类必须继承此类。
    
    Methods
    -------
    calculate(data)
        计算所有因子
    add_feature(name, func)
        添加自定义因子
    get_feature_names()
        获取所有因子名称
    """
    
    def __init__(self) -> None:
        """初始化因子计算引擎"""
        self._features: Dict[str, Callable] = {}
        self._register_default_features()
    
    @abstractmethod
    def _register_default_features(self) -> None:
        """注册默认因子，子类必须实现"""
        pass
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有注册的因子
        
        Parameters
        ----------
        data : pd.DataFrame
            原始OHLCV数据，索引为DatetimeIndex
        
        Returns
        -------
        pd.DataFrame
            包含所有因子的数据框
        """
        pass
    
    def add_feature(self, name: str, func: Callable) -> None:
        """
        添加自定义因子
        
        Parameters
        ----------
        name : str
            因子名称
        func : Callable
            因子计算函数，接收DataFrame返回Series
        """
        self._features[name] = func
        logger.info(f"添加自定义因子: {name}")
    
    def get_feature_names(self) -> List[str]:
        """
        获取所有因子名称
        
        Returns
        -------
        List[str]
            因子名称列表
        """
        return list(self._features.keys())


__all__ = ['FeatureEngine']

