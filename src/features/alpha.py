"""
Alpha因子计算器

计算量价类Alpha因子，用于选股和择时。
"""

import logging

import pandas as pd
import numpy as np

from .base import FeatureEngine

logger = logging.getLogger(__name__)


class AlphaFeatures(FeatureEngine):
    """
    Alpha因子计算器
    
    计算量价类Alpha因子，用于选股和择时。
    """
    
    def __init__(self) -> None:
        """初始化Alpha因子计算器"""
        super().__init__()
    
    def _register_default_features(self) -> None:
        """注册默认Alpha因子"""
        self._features = {
            "alpha_001": self._alpha_001,
            "alpha_002": self._alpha_002,
            "alpha_003": self._alpha_003,
            "alpha_004": self._alpha_004,
            "alpha_005": self._alpha_005,
        }
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有Alpha因子
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV数据
        
        Returns
        -------
        pd.DataFrame
            包含Alpha因子的数据框
        """
        result = data.copy()
        
        for name, func in self._features.items():
            try:
                result[name] = func(data)
            except Exception as e:
                logger.warning(f"计算Alpha因子 {name} 失败: {e}")
                result[name] = np.nan
        
        return result
    
    @staticmethod
    def _alpha_001(data: pd.DataFrame) -> pd.Series:
        """
        Alpha#001: 成交量加权平均价格动量
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV数据
        
        Returns
        -------
        pd.Series
            因子值
        """
        vwap = (data["amount"] / data["volume"]).replace([np.inf, -np.inf], np.nan)
        return (data["close"] - vwap) / vwap
    
    @staticmethod
    def _alpha_002(data: pd.DataFrame) -> pd.Series:
        """
        Alpha#002: 价格振幅因子
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV数据
        
        Returns
        -------
        pd.Series
            因子值
        """
        return (data["high"] - data["low"]) / data["close"]
    
    @staticmethod
    def _alpha_003(data: pd.DataFrame) -> pd.Series:
        """
        Alpha#003: 量价背离因子
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV数据
        
        Returns
        -------
        pd.Series
            因子值
        """
        price_change = data["close"].pct_change(5)
        volume_change = data["volume"].pct_change(5)
        
        return price_change - volume_change
    
    @staticmethod
    def _alpha_004(data: pd.DataFrame) -> pd.Series:
        """
        Alpha#004: 成交量加速因子
        
        衡量近期成交量相对中期成交量的变化程度。
        正值表示成交量放大，负值表示成交量萎缩。
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV数据，需包含 volume 列
        
        Returns
        -------
        pd.Series
            因子值: (vol_5d - vol_20d) / vol_20d
        """
        vol_5d = data['volume'].rolling(5, min_periods=1).mean()
        vol_20d = data['volume'].rolling(20, min_periods=5).mean()
        return (vol_5d - vol_20d) / vol_20d.replace(0, np.nan)
    
    @staticmethod
    def _alpha_005(data: pd.DataFrame) -> pd.Series:
        """
        Alpha#005: 尾盘强度因子 (Intraday Momentum)
        
        衡量收盘价在当日振幅中的位置。
        接近 1 表示收盘价接近最高价（多头强势），接近 0 表示接近最低价（空头强势）。
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV数据，需包含 high, low, close 列
        
        Returns
        -------
        pd.Series
            因子值: (close - low) / (high - low)，范围 [0, 1]
        """
        range_hl = data['high'] - data['low']
        return (data['close'] - data['low']) / range_hl.replace(0, np.nan)


__all__ = ['AlphaFeatures']

