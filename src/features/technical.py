"""
技术指标因子计算器

计算常用的技术分析指标，包括均线、动量、波动率等。
使用 Numba JIT 编译加速计算。
"""

from typing import Dict, Any, Optional
import logging

import pandas as pd
import numpy as np

from .base import FeatureEngine
from .numba_utils import (
    calculate_rsi_numba,
    calculate_atr_numba,
    calculate_volatility_numba,
    calculate_kdj_numba,
)

logger = logging.getLogger(__name__)


class TechnicalFeatures(FeatureEngine):
    """
    技术指标因子计算器
    
    计算常用的技术分析指标，包括均线、动量、波动率等。
    
    Examples
    --------
    >>> engine = TechnicalFeatures()
    >>> df_with_features = engine.calculate(ohlcv_data)
    >>> print(df_with_features.columns.tolist())
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化技术指标计算器
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            配置参数，如均线周期等
        """
        self.config = config or {}
        super().__init__()
    
    def _register_default_features(self) -> None:
        """注册默认技术指标"""
        self._features = {
            "sma_5": lambda df: self.sma(df["close"], 5),
            "sma_10": lambda df: self.sma(df["close"], 10),
            "sma_20": lambda df: self.sma(df["close"], 20),
            "sma_60": lambda df: self.sma(df["close"], 60),
            "ema_12": lambda df: self.ema(df["close"], 12),
            "ema_26": lambda df: self.ema(df["close"], 26),
            "rsi_14": lambda df: self.rsi(df["close"], 14),
            "macd": lambda df: self.macd(df["close"])[0],
            "macd_signal": lambda df: self.macd(df["close"])[1],
            "macd_hist": lambda df: self.macd(df["close"])[2],
            "boll_upper": lambda df: self.bollinger_bands(df["close"])[0],
            "boll_middle": lambda df: self.bollinger_bands(df["close"])[1],
            "boll_lower": lambda df: self.bollinger_bands(df["close"])[2],
            "atr_14": lambda df: self.atr(df["high"], df["low"], df["close"], 14),
            "volatility_20": lambda df: self.volatility(df["close"], 20),
            "momentum_10": lambda df: self.momentum(df["close"], 10),
            "roc_10": lambda df: self.roc(df["close"], 10),
            "roc_20": lambda df: self.roc(df["close"], 20),
            "williams_r": lambda df: self.williams_r(df["high"], df["low"], df["close"]),
            "kdj_k": lambda df: self.kdj(df["high"], df["low"], df["close"])[0],
            "kdj_d": lambda df: self.kdj(df["high"], df["low"], df["close"])[1],
            "kdj_j": lambda df: self.kdj(df["high"], df["low"], df["close"])[2],
            "vol_20": lambda df: self.volatility(df["close"], 20),
            "ivol_20": lambda df: self.calculate_ivol(20),
            "sharpe_20": lambda df: self.rolling_sharpe(df["close"], 20),
            "efficiency_20": lambda df: self.path_efficiency(df["close"], 20),
            "williams_r_14": lambda df: self.williams_r(df["high"], df["low"], df["close"], 14),
        }
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV数据，必须包含 open, high, low, close, volume 列
        
        Returns
        -------
        pd.DataFrame
            原始数据加上所有技术指标列
        """
        result = data.copy()
        
        for name, func in self._features.items():
            try:
                result[name] = func(data)
            except Exception as e:
                logger.warning(f"计算因子 {name} 失败: {e}")
                result[name] = np.nan
        
        logger.info(f"技术指标计算完成，共 {len(self._features)} 个因子")
        return result
    
    def calculate_ivol(self, period: int = 20) -> pd.Series:
        """
        计算特质波动率 (IVOL)
        
        Parameters
        ----------
        period : int
            计算周期
        
        Returns
        -------
        pd.Series
            特质波动率
        
        Notes
        -----
        需要市场收益率数据，此处返回 NaN 占位
        """
        return pd.Series(dtype=np.float64)
    
    @staticmethod
    def rolling_sharpe(series: pd.Series, period: int = 20) -> pd.Series:
        """
        滚动夏普比率 (Rolling Sharpe Ratio)
        
        衡量单位波动风险下的超额收益（假设无风险利率为0简化计算）。
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int
            计算周期
        
        Returns
        -------
        pd.Series
            年化夏普比率
        """
        returns = series.pct_change()
        mean = returns.rolling(window=period, min_periods=period//2).mean()
        std = returns.rolling(window=period, min_periods=period//2).std()
        sharpe = (mean / std.replace(0, np.nan)) * np.sqrt(252)
        return sharpe

    @staticmethod
    def path_efficiency(series: pd.Series, period: int = 20) -> pd.Series:
        """
        路径效率 (Path Efficiency / Efficiency Ratio)
        
        衡量价格走势的平滑程度。
        ER = Net Change / Sum of Absolute Changes
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int
            计算周期
        
        Returns
        -------
        pd.Series
            效率系数 (0-1)
        """
        net_change = (series - series.shift(period)).abs()
        sum_abs_change = series.diff().abs().rolling(window=period, min_periods=period//2).sum()
        er = net_change / sum_abs_change.replace(0, np.nan)
        return er

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """简单移动平均线 (Simple Moving Average)"""
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """指数移动平均线 (Exponential Moving Average)"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        相对强弱指标 (Relative Strength Index)
        
        使用 Numba JIT 编译加速计算。
        """
        rsi_values = calculate_rsi_numba(series.values.astype(np.float64), period)
        return pd.Series(rsi_values, index=series.index)
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> tuple:
        """MACD指标 (Moving Average Convergence Divergence)"""
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """布林带 (Bollinger Bands)"""
        middle = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        平均真实波幅 (Average True Range)
        
        使用 Numba JIT 编译加速计算。
        """
        atr_values = calculate_atr_numba(
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            period
        )
        return pd.Series(atr_values, index=close.index)
    
    @staticmethod
    def volatility(series: pd.Series, period: int = 20) -> pd.Series:
        """
        历史波动率
        
        使用 Numba JIT 编译加速计算。
        """
        vol_values = calculate_volatility_numba(
            series.values.astype(np.float64),
            period
        )
        return pd.Series(vol_values, index=series.index)
    
    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """动量指标"""
        return series - series.shift(period)
    
    @staticmethod
    def roc(series: pd.Series, period: int = 10) -> pd.Series:
        """变动率指标 (Rate of Change)"""
        return ((series - series.shift(period)) / series.shift(period)) * 100
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """威廉指标 (Williams %R)"""
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def kdj(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 9
    ) -> tuple:
        """
        KDJ随机指标
        
        使用 Numba JIT 编译加速计算。
        """
        k_values, d_values, j_values = calculate_kdj_numba(
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            period
        )
        
        k = pd.Series(k_values, index=close.index)
        d = pd.Series(d_values, index=close.index)
        j = pd.Series(j_values, index=close.index)
        
        return k, d, j


__all__ = ['TechnicalFeatures']

