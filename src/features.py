"""
因子计算引擎模块

该模块提供技术指标和因子计算功能，支持自定义因子扩展。
所有计算采用向量化操作以保证性能。

Performance Notes
-----------------
- 使用 numba JIT 编译加速滚动窗口计算
- 避免使用 groupby().apply()，优先使用 transform() 或向量化操作
- 对于大规模数据（全市场A股），使用 numba 优化的底层函数
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Callable
import logging

import pandas as pd
import numpy as np
from numba import jit, prange

logger = logging.getLogger(__name__)


# ==================== Numba 优化的底层计算函数 ====================

@jit(nopython=True, cache=True)
def _rolling_std_1d(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Numba 加速的一维滚动标准差计算
    
    Parameters
    ----------
    arr : np.ndarray
        输入数组
    window : int
        滚动窗口大小
    min_periods : int
        最小有效数据点数
    
    Returns
    -------
    np.ndarray
        滚动标准差结果
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        
        # 收集窗口内的有效值
        valid_count = 0
        sum_val = 0.0
        sum_sq = 0.0
        
        for j in range(start, end):
            if not np.isnan(arr[j]):
                valid_count += 1
                sum_val += arr[j]
                sum_sq += arr[j] * arr[j]
        
        if valid_count >= min_periods:
            mean_val = sum_val / valid_count
            variance = (sum_sq / valid_count) - (mean_val * mean_val)
            # 使用样本标准差 (ddof=1)
            if valid_count > 1:
                variance = variance * valid_count / (valid_count - 1)
            if variance > 0:
                result[i] = np.sqrt(variance)
            else:
                result[i] = 0.0
    
    return result


@jit(nopython=True, cache=True)
def _rolling_mean_1d(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Numba 加速的一维滚动均值计算
    
    Parameters
    ----------
    arr : np.ndarray
        输入数组
    window : int
        滚动窗口大小
    min_periods : int
        最小有效数据点数
    
    Returns
    -------
    np.ndarray
        滚动均值结果
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        
        valid_count = 0
        sum_val = 0.0
        
        for j in range(start, end):
            if not np.isnan(arr[j]):
                valid_count += 1
                sum_val += arr[j]
        
        if valid_count >= min_periods:
            result[i] = sum_val / valid_count
    
    return result


@jit(nopython=True, cache=True)
def _calculate_rsi_numba(close: np.ndarray, period: int) -> np.ndarray:
    """
    Numba 加速的 RSI 计算
    
    Parameters
    ----------
    close : np.ndarray
        收盘价数组
    period : int
        计算周期
    
    Returns
    -------
    np.ndarray
        RSI 值数组 (0-100)
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if n < period + 1:
        return result
    
    # 计算价格变动
    delta = np.empty(n, dtype=np.float64)
    delta[0] = np.nan
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    # 分离涨跌
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if delta[i] > 0:
            gain[i] = delta[i]
        elif delta[i] < 0:
            loss[i] = -delta[i]
    
    # 使用 Wilder's Smoothing (alpha = 1/period)
    alpha = 1.0 / period
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    # 初始化：使用前 period 个数据的简单平均
    sum_gain = 0.0
    sum_loss = 0.0
    for i in range(1, period + 1):
        sum_gain += gain[i]
        sum_loss += loss[i]
    avg_gain[period] = sum_gain / period
    avg_loss[period] = sum_loss / period
    
    # EMA 递推
    for i in range(period + 1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    
    # 计算 RSI
    for i in range(period, n):
        if avg_loss[i] == 0:
            if avg_gain[i] == 0:
                result[i] = 50.0  # 无变动
            else:
                result[i] = 100.0  # 全涨
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


@jit(nopython=True, cache=True)
def _calculate_returns_1d(close: np.ndarray) -> np.ndarray:
    """
    Numba 加速的收益率计算
    
    Parameters
    ----------
    close : np.ndarray
        收盘价数组
    
    Returns
    -------
    np.ndarray
        日收益率数组
    """
    n = len(close)
    returns = np.empty(n, dtype=np.float64)
    returns[0] = np.nan
    
    for i in range(1, n):
        if close[i - 1] != 0 and not np.isnan(close[i - 1]):
            returns[i] = (close[i] - close[i - 1]) / close[i - 1]
        else:
            returns[i] = np.nan
    
    return returns


@jit(nopython=True, parallel=True, cache=True)
def _grouped_rolling_std(
    values: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
    window: int,
    min_periods: int
) -> np.ndarray:
    """
    Numba 加速的分组滚动标准差计算（并行版本）
    
    Parameters
    ----------
    values : np.ndarray
        值数组
    group_ids : np.ndarray
        分组 ID 数组（从0开始的整数）
    n_groups : int
        分组数量
    window : int
        滚动窗口大小
    min_periods : int
        最小有效数据点数
    
    Returns
    -------
    np.ndarray
        分组滚动标准差结果
    """
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # 为每个组分配索引
    # 首先计算每个组的大小
    group_sizes = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        gid = group_ids[i]
        if gid >= 0:
            group_sizes[gid] += 1
    
    # 计算每个组的起始位置
    group_starts = np.zeros(n_groups + 1, dtype=np.int64)
    for i in range(n_groups):
        group_starts[i + 1] = group_starts[i] + group_sizes[i]
    
    # 创建组内索引数组
    group_indices = np.empty(n, dtype=np.int64)
    group_positions = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        gid = group_ids[i]
        if gid >= 0:
            pos = group_starts[gid] + group_positions[gid]
            group_indices[pos] = i
            group_positions[gid] += 1
    
    # 并行处理每个组
    for g in prange(n_groups):
        start_pos = group_starts[g]
        end_pos = group_starts[g + 1]
        group_size = end_pos - start_pos
        
        if group_size == 0:
            continue
        
        # 提取该组的值和原始索引
        for k in range(group_size):
            original_idx = group_indices[start_pos + k]
            
            # 计算该位置的滚动标准差
            win_start = max(0, k - window + 1)
            win_end = k + 1
            
            valid_count = 0
            sum_val = 0.0
            sum_sq = 0.0
            
            for w in range(win_start, win_end):
                val_idx = group_indices[start_pos + w]
                val = values[val_idx]
                if not np.isnan(val):
                    valid_count += 1
                    sum_val += val
                    sum_sq += val * val
            
            if valid_count >= min_periods:
                mean_val = sum_val / valid_count
                variance = (sum_sq / valid_count) - (mean_val * mean_val)
                if valid_count > 1:
                    variance = variance * valid_count / (valid_count - 1)
                if variance > 0:
                    result[original_idx] = np.sqrt(variance)
                else:
                    result[original_idx] = 0.0
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def _grouped_rsi(
    close: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
    period: int
) -> np.ndarray:
    """
    Numba 加速的分组 RSI 计算（并行版本）
    
    Parameters
    ----------
    close : np.ndarray
        收盘价数组
    group_ids : np.ndarray
        分组 ID 数组
    n_groups : int
        分组数量
    period : int
        RSI 周期
    
    Returns
    -------
    np.ndarray
        RSI 结果数组
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # 计算每个组的大小和起始位置
    group_sizes = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        gid = group_ids[i]
        if gid >= 0:
            group_sizes[gid] += 1
    
    group_starts = np.zeros(n_groups + 1, dtype=np.int64)
    for i in range(n_groups):
        group_starts[i + 1] = group_starts[i] + group_sizes[i]
    
    group_indices = np.empty(n, dtype=np.int64)
    group_positions = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        gid = group_ids[i]
        if gid >= 0:
            pos = group_starts[gid] + group_positions[gid]
            group_indices[pos] = i
            group_positions[gid] += 1
    
    # 并行处理每个组
    for g in prange(n_groups):
        start_pos = group_starts[g]
        end_pos = group_starts[g + 1]
        group_size = end_pos - start_pos
        
        if group_size < period + 1:
            continue
        
        # 提取该组的收盘价
        group_close = np.empty(group_size, dtype=np.float64)
        for k in range(group_size):
            group_close[k] = close[group_indices[start_pos + k]]
        
        # 计算 RSI
        group_rsi = _calculate_rsi_numba(group_close, period)
        
        # 写回结果
        for k in range(group_size):
            original_idx = group_indices[start_pos + k]
            result[original_idx] = group_rsi[k]
    
    return result


@jit(nopython=True, cache=True)
def _calculate_atr_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Numba 加速的 ATR 计算
    
    Parameters
    ----------
    high : np.ndarray
        最高价数组
    low : np.ndarray
        最低价数组
    close : np.ndarray
        收盘价数组
    period : int
        计算周期
    
    Returns
    -------
    np.ndarray
        ATR 值数组
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if n < 2:
        return result
    
    # 计算 True Range
    true_range = np.empty(n, dtype=np.float64)
    true_range[0] = high[0] - low[0]  # 第一天只有 HL
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        true_range[i] = max(tr1, tr2, tr3)
    
    # 计算滚动均值
    result = _rolling_mean_1d(true_range, period, 1)
    
    return result


@jit(nopython=True, cache=True)
def _calculate_volatility_numba(close: np.ndarray, period: int) -> np.ndarray:
    """
    Numba 加速的年化波动率计算
    
    Parameters
    ----------
    close : np.ndarray
        收盘价数组
    period : int
        计算周期
    
    Returns
    -------
    np.ndarray
        年化波动率数组
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if n < 2:
        return result
    
    # 计算对数收益率
    log_returns = np.empty(n, dtype=np.float64)
    log_returns[0] = np.nan
    
    for i in range(1, n):
        if close[i - 1] > 0 and close[i] > 0:
            log_returns[i] = np.log(close[i] / close[i - 1])
        else:
            log_returns[i] = np.nan
    
    # 计算滚动标准差
    volatility = _rolling_std_1d(log_returns, period, 1)
    
    # 年化（252 个交易日）
    annualization_factor = np.sqrt(252.0)
    for i in range(n):
        if not np.isnan(volatility[i]):
            result[i] = volatility[i] * annualization_factor
    
    return result


@jit(nopython=True, cache=True)
def _calculate_kdj_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 9
) -> tuple:
    """
    Numba 加速的 KDJ 指标计算
    
    Parameters
    ----------
    high : np.ndarray
        最高价数组
    low : np.ndarray
        最低价数组
    close : np.ndarray
        收盘价数组
    period : int
        计算周期，默认9
    
    Returns
    -------
    tuple
        (K值数组, D值数组, J值数组)
    """
    n = len(close)
    k_values = np.empty(n, dtype=np.float64)
    d_values = np.empty(n, dtype=np.float64)
    j_values = np.empty(n, dtype=np.float64)
    k_values[:] = 50.0  # 初始值
    d_values[:] = 50.0
    j_values[:] = 50.0
    
    # 计算 RSV
    rsv = np.empty(n, dtype=np.float64)
    rsv[:] = 50.0  # 默认值
    
    for i in range(period - 1, n):
        # 计算过去 period 天的最高价和最低价
        highest = high[i]
        lowest = low[i]
        for j in range(i - period + 1, i):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]
        
        # 计算 RSV
        if highest != lowest:
            rsv[i] = 100.0 * (close[i] - lowest) / (highest - lowest)
        else:
            rsv[i] = 50.0
    
    # 计算 K、D、J（使用 EMA，com=2 对应 alpha=1/3）
    alpha = 1.0 / 3.0
    k_values[0] = rsv[0]
    d_values[0] = k_values[0]
    
    for i in range(1, n):
        k_values[i] = alpha * rsv[i] + (1 - alpha) * k_values[i - 1]
        d_values[i] = alpha * k_values[i] + (1 - alpha) * d_values[i - 1]
        j_values[i] = 3 * k_values[i] - 2 * d_values[i]
    
    return k_values, d_values, j_values


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
            "williams_r": lambda df: self.williams_r(df["high"], df["low"], df["close"]),
            "kdj_k": lambda df: self.kdj(df["high"], df["low"], df["close"])[0],
            "kdj_d": lambda df: self.kdj(df["high"], df["low"], df["close"])[1],
            "kdj_j": lambda df: self.kdj(df["high"], df["low"], df["close"])[2],
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
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        简单移动平均线 (Simple Moving Average)
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int
            计算周期
        
        Returns
        -------
        pd.Series
            SMA值
        """
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        指数移动平均线 (Exponential Moving Average)
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int
            计算周期
        
        Returns
        -------
        pd.Series
            EMA值
        """
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        相对强弱指标 (Relative Strength Index)
        
        使用 numba JIT 编译加速计算。
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int, optional
            计算周期，默认14
        
        Returns
        -------
        pd.Series
            RSI值 (0-100)
        
        Notes
        -----
        - 使用 Wilder's Smoothing Method
        - 内部调用 numba 优化的 _calculate_rsi_numba 函数
        """
        rsi_values = _calculate_rsi_numba(series.values.astype(np.float64), period)
        return pd.Series(rsi_values, index=series.index)
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> tuple:
        """
        MACD指标 (Moving Average Convergence Divergence)
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        fast_period : int, optional
            快线周期，默认12
        slow_period : int, optional
            慢线周期，默认26
        signal_period : int, optional
            信号线周期，默认9
        
        Returns
        -------
        tuple
            (MACD线, 信号线, 柱状图)
        """
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
        """
        布林带 (Bollinger Bands)
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int, optional
            计算周期，默认20
        std_dev : float, optional
            标准差倍数，默认2.0
        
        Returns
        -------
        tuple
            (上轨, 中轨, 下轨)
        """
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
        
        使用 numba JIT 编译加速计算。
        
        Parameters
        ----------
        high : pd.Series
            最高价序列
        low : pd.Series
            最低价序列
        close : pd.Series
            收盘价序列
        period : int, optional
            计算周期，默认14
        
        Returns
        -------
        pd.Series
            ATR值
        
        Notes
        -----
        - 内部调用 numba 优化的 _calculate_atr_numba 函数
        """
        atr_values = _calculate_atr_numba(
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
        
        使用 numba JIT 编译加速计算。
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int, optional
            计算周期，默认20
        
        Returns
        -------
        pd.Series
            年化波动率
        
        Notes
        -----
        - 内部调用 numba 优化的 _calculate_volatility_numba 函数
        - 年化因子为 sqrt(252)
        """
        vol_values = _calculate_volatility_numba(
            series.values.astype(np.float64),
            period
        )
        return pd.Series(vol_values, index=series.index)
    
    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """
        动量指标
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int, optional
            计算周期，默认10
        
        Returns
        -------
        pd.Series
            动量值
        """
        return series - series.shift(period)
    
    @staticmethod
    def roc(series: pd.Series, period: int = 10) -> pd.Series:
        """
        变动率指标 (Rate of Change)
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int, optional
            计算周期，默认10
        
        Returns
        -------
        pd.Series
            ROC百分比
        """
        return ((series - series.shift(period)) / series.shift(period)) * 100
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        威廉指标 (Williams %R)
        
        Parameters
        ----------
        high : pd.Series
            最高价序列
        low : pd.Series
            最低价序列
        close : pd.Series
            收盘价序列
        period : int, optional
            计算周期，默认14
        
        Returns
        -------
        pd.Series
            Williams %R值 (-100 到 0)
        """
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
        
        使用 numba JIT 编译加速计算。
        
        Parameters
        ----------
        high : pd.Series
            最高价序列
        low : pd.Series
            最低价序列
        close : pd.Series
            收盘价序列
        period : int, optional
            计算周期，默认9
        
        Returns
        -------
        tuple
            (K值, D值, J值)
        
        Notes
        -----
        - 内部调用 numba 优化的 _calculate_kdj_numba 函数
        """
        k_values, d_values, j_values = _calculate_kdj_numba(
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            period
        )
        
        k = pd.Series(k_values, index=close.index)
        d = pd.Series(d_values, index=close.index)
        j = pd.Series(j_values, index=close.index)
        
        return k, d, j


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


class FactorCalculator:
    """
    多因子计算器
    
    接收清洗后的 OHLCV 和财务数据，计算价值因子、质量因子和技术/动量因子。
    支持行业中性化的 Z-Score 标准化处理。
    
    重要：为避免前视偏差（Look-Ahead Bias），财务数据在使用前需要进行滞后处理。
    A股财报发布时间规则：
    - 年报：次年4月30日前发布（滞后约4个月）
    - 半年报：8月31日前发布（滞后约2个月）
    - 季报：1个月内发布（滞后约1个月）
    
    默认滞后 3 个月（约 60 个交易日），确保只使用历史上可获得的数据。
    
    Parameters
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV 数据，索引为 DatetimeIndex，必须包含 open, high, low, close, volume 列
        可选包含 stock_code（股票代码）列用于多股票场景
    financial_data : pd.DataFrame
        财务数据，包含 pe_ttm, dividend_yield, roe 等列
        可选包含 stock_code（股票代码）和 report_date（报告日期）列
    industry_data : Optional[pd.DataFrame]
        行业分类数据，包含 stock_code 和 sw_industry_l1（申万一级行业）列
    lag_financial_data : bool
        是否对财务数据进行滞后处理，默认 True
    lag_days : int
        财务数据滞后天数，默认 60（约3个月）
    
    Attributes
    ----------
    ohlcv_data : pd.DataFrame
        OHLCV 数据副本
    financial_data : pd.DataFrame
        财务数据副本（可能已滞后）
    industry_data : Optional[pd.DataFrame]
        行业分类数据副本
    
    Examples
    --------
    >>> calculator = FactorCalculator(ohlcv_df, financial_df, industry_df)
    >>> factors = calculator.calculate_all_factors()
    >>> normalized_factors = calculator.z_score_normalize(factors, ['ep_ttm', 'rsi_20'])
    """
    
    # 默认财务数据滞后天数（约3个月）
    DEFAULT_LAG_DAYS = 60
    
    def __init__(
        self,
        ohlcv_data: pd.DataFrame,
        financial_data: pd.DataFrame,
        industry_data: Optional[pd.DataFrame] = None,
        lag_financial_data: bool = True,
        lag_days: int = DEFAULT_LAG_DAYS
    ) -> None:
        """
        初始化多因子计算器
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV 数据
        financial_data : pd.DataFrame
            财务数据
        industry_data : Optional[pd.DataFrame]
            行业分类数据
        lag_financial_data : bool
            是否对财务数据进行滞后处理，默认 True
        lag_days : int
            财务数据滞后天数，默认 60（约3个月）
        """
        self.ohlcv_data = ohlcv_data.copy()
        self.industry_data = industry_data.copy() if industry_data is not None else None
        
        # 对财务数据进行滞后处理（避免前视偏差）
        if lag_financial_data and not financial_data.empty:
            self.financial_data = self._lag_financial_data(financial_data.copy(), lag_days)
            logger.info(f"财务数据已滞后 {lag_days} 天（避免前视偏差）")
        else:
            self.financial_data = financial_data.copy()
        
        self._validate_data()
        logger.info("FactorCalculator 初始化完成")
    
    def _lag_financial_data(
        self,
        financial_data: pd.DataFrame,
        lag_days: int
    ) -> pd.DataFrame:
        """
        对财务数据进行滞后处理，避免前视偏差
        
        将财务数据的日期索引向后移动指定天数，模拟财报发布滞后。
        
        Parameters
        ----------
        financial_data : pd.DataFrame
            原始财务数据
        lag_days : int
            滞后天数
        
        Returns
        -------
        pd.DataFrame
            滞后后的财务数据
        
        Notes
        -----
        - 如果数据有 'report_date' 或 'pub_date' 列，优先使用发布日期
        - 否则对 DatetimeIndex 进行偏移
        - 使用 pd.Timedelta 进行日期偏移，确保精确性
        """
        df = financial_data.copy()
        
        # 检查是否有发布日期列（更精确的方式）
        pub_date_cols = ['pub_date', 'publish_date', 'announcement_date']
        pub_date_col = next((c for c in pub_date_cols if c in df.columns), None)
        
        if pub_date_col is not None:
            # 使用实际发布日期，并额外滞后1天确保数据可获得
            logger.info(f"使用发布日期列 '{pub_date_col}' 进行滞后处理")
            df[pub_date_col] = pd.to_datetime(df[pub_date_col])
            # 发布日T+1日才能使用
            df[pub_date_col] = df[pub_date_col] + pd.Timedelta(days=1)
            # 如果需要，重新设置索引
            if pub_date_col not in df.index.names:
                if 'date' in df.columns or isinstance(df.index, pd.DatetimeIndex):
                    # 用发布日期替换交易日期
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index(drop=True)
                    df = df.set_index(pub_date_col)
            return df
        
        # 方式2：对日期索引进行固定天数偏移
        if isinstance(df.index, pd.DatetimeIndex):
            # 将索引向后移动 lag_days 天
            df.index = df.index + pd.Timedelta(days=lag_days)
            logger.debug(f"DatetimeIndex 已向后偏移 {lag_days} 天")
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']) + pd.Timedelta(days=lag_days)
            logger.debug(f"date 列已向后偏移 {lag_days} 天")
        elif 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date']) + pd.Timedelta(days=lag_days)
            logger.debug(f"trade_date 列已向后偏移 {lag_days} 天")
        else:
            logger.warning("财务数据无日期列，无法进行滞后处理")
        
        return df
    
    def _validate_data(self) -> None:
        """
        验证输入数据的有效性
        
        Raises
        ------
        ValueError
            当必要列缺失时抛出
        """
        required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_ohlcv_cols if col not in self.ohlcv_data.columns]
        if missing_cols:
            raise ValueError(f"OHLCV 数据缺少必要列: {missing_cols}")
        
        if not isinstance(self.ohlcv_data.index, pd.DatetimeIndex):
            logger.warning("OHLCV 数据索引不是 DatetimeIndex，尝试转换")
            self.ohlcv_data.index = pd.to_datetime(self.ohlcv_data.index)
    
    # ==================== 价值因子 ====================
    
    def calculate_ep_ttm(self) -> pd.Series:
        """
        计算 EP_TTM (Earnings to Price, TTM)
        
        EP_TTM = 1 / PE_TTM，即市盈率的倒数。
        EP 越高表示股票越便宜（价值越高）。
        
        Returns
        -------
        pd.Series
            EP_TTM 因子值
        
        Raises
        ------
        ValueError
            当财务数据缺少 pe_ttm 列时抛出
        
        Notes
        -----
        - PE_TTM 为 0 或负数的情况会被设为 NaN
        - 使用向量化操作保证计算效率
        """
        if 'pe_ttm' not in self.financial_data.columns:
            raise ValueError("财务数据缺少 pe_ttm 列")
        
        # 向量化计算：1/PE_TTM
        pe_ttm = self.financial_data['pe_ttm'].replace(0, np.nan)
        # 负 PE 表示亏损，EP 设为 NaN
        pe_ttm = pe_ttm.where(pe_ttm > 0, np.nan)
        ep_ttm = 1.0 / pe_ttm
        
        # 处理异常值
        ep_ttm = ep_ttm.replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"EP_TTM 计算完成，有效值数量: {ep_ttm.notna().sum()}")
        return ep_ttm
    
    def calculate_dividend_yield(self) -> pd.Series:
        """
        计算股息率 (Dividend Yield)
        
        股息率 = 每股股息 / 股价，表示投资回报率。
        
        Returns
        -------
        pd.Series
            股息率因子值（百分比形式）
        
        Raises
        ------
        ValueError
            当财务数据缺少必要列时抛出
        
        Notes
        -----
        - 优先使用 dividend_yield 列
        - 若不存在则尝试从 dps（每股股息）和 close（收盘价）计算
        """
        if 'dividend_yield' in self.financial_data.columns:
            dividend_yield = self.financial_data['dividend_yield'].copy()
            logger.debug("直接使用 dividend_yield 列")
            return dividend_yield.replace([np.inf, -np.inf], np.nan)
        
        # 尝试从每股股息和股价计算
        if 'dps' in self.financial_data.columns:
            if 'close' in self.financial_data.columns:
                close = self.financial_data['close']
            elif 'close' in self.ohlcv_data.columns:
                # 尝试从 OHLCV 数据获取收盘价
                close = self.ohlcv_data['close']
            else:
                raise ValueError("无法获取收盘价数据")
            
            close = close.replace(0, np.nan)
            dividend_yield = self.financial_data['dps'] / close
            logger.debug("从 dps 和 close 计算 dividend_yield")
            return dividend_yield.replace([np.inf, -np.inf], np.nan)
        
        raise ValueError("财务数据缺少 dividend_yield 或 dps 列")
    
    # ==================== 质量因子 ====================
    
    def calculate_roe_stability(self, window: int = 12) -> pd.Series:
        """
        计算 ROE 稳定性因子
        
        ROE_Stability = 1 / std(ROE_过去N个季度)
        ROE 标准差越小，稳定性越高，因子值越大。
        
        使用 numba 加速的分组滚动标准差计算。
        
        Parameters
        ----------
        window : int, optional
            滚动窗口大小（季度数），默认12
        
        Returns
        -------
        pd.Series
            ROE 稳定性因子值
        
        Raises
        ------
        ValueError
            当财务数据缺少 roe 列时抛出
        
        Notes
        -----
        - 至少需要 4 个季度数据才开始计算
        - 标准差为 0 的情况会被设为 NaN（避免除零）
        - 使用 numba 优化的 _grouped_rolling_std 函数
        """
        if 'roe' not in self.financial_data.columns:
            raise ValueError("财务数据缺少 roe 列")
        
        min_periods = max(4, window // 3)  # 至少需要 1/3 窗口的数据
        
        # 按股票分组计算滚动标准差
        if 'stock_code' in self.financial_data.columns:
            # 使用 numba 优化的分组滚动标准差
            stock_codes = self.financial_data['stock_code'].values
            unique_codes = np.unique(stock_codes)
            code_to_id = {code: i for i, code in enumerate(unique_codes)}
            group_ids = np.array([code_to_id[code] for code in stock_codes], dtype=np.int64)
            n_groups = len(unique_codes)
            
            roe_std_values = _grouped_rolling_std(
                self.financial_data['roe'].values.astype(np.float64),
                group_ids,
                n_groups,
                window,
                min_periods
            )
            roe_std = pd.Series(roe_std_values, index=self.financial_data.index)
        else:
            # 单股票情况：使用 numba 优化的一维滚动标准差
            roe_std_values = _rolling_std_1d(
                self.financial_data['roe'].values.astype(np.float64),
                window,
                min_periods
            )
            roe_std = pd.Series(roe_std_values, index=self.financial_data.index)
        
        # 稳定性 = 1 / 标准差
        roe_std = roe_std.replace(0, np.nan)
        roe_stability = 1.0 / roe_std
        
        # 处理异常值
        roe_stability = roe_stability.replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"ROE_Stability 计算完成，窗口: {window} 季度（numba 加速）")
        return roe_stability
    
    # ==================== 技术/动量因子 ====================
    
    def calculate_rsi(self, period: int = 20) -> pd.Series:
        """
        计算 RSI 相对强弱指标
        
        RSI = 100 - 100 / (1 + RS)
        其中 RS = 平均上涨幅度 / 平均下跌幅度
        
        使用 numba 加速的计算实现。
        
        Parameters
        ----------
        period : int, optional
            计算周期，默认20
        
        Returns
        -------
        pd.Series
            RSI 因子值 (0-100)
        
        Notes
        -----
        - 使用 Wilder's Smoothing Method (EMA with alpha=1/period)
        - RSI > 70 通常视为超买，RSI < 30 通常视为超卖
        - 使用 numba 优化的 _grouped_rsi 函数替代 groupby().transform(lambda)
        """
        if 'stock_code' in self.ohlcv_data.columns:
            # 使用 numba 优化的分组 RSI 计算
            stock_codes = self.ohlcv_data['stock_code'].values
            unique_codes = np.unique(stock_codes)
            code_to_id = {code: i for i, code in enumerate(unique_codes)}
            group_ids = np.array([code_to_id[code] for code in stock_codes], dtype=np.int64)
            n_groups = len(unique_codes)
            
            rsi_values = _grouped_rsi(
                self.ohlcv_data['close'].values.astype(np.float64),
                group_ids,
                n_groups,
                period
            )
            rsi = pd.Series(rsi_values, index=self.ohlcv_data.index)
        else:
            # 单股票情况：使用 numba 优化的 RSI 计算
            rsi_values = _calculate_rsi_numba(
                self.ohlcv_data['close'].values.astype(np.float64),
                period
            )
            rsi = pd.Series(rsi_values, index=self.ohlcv_data.index)
        
        logger.debug(f"RSI_{period} 计算完成（numba 加速）")
        return rsi
    
    @staticmethod
    def _calculate_rsi_series(series: pd.Series, period: int) -> pd.Series:
        """
        计算单个序列的 RSI（向量化实现，兼容性方法）
        
        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int
            计算周期
        
        Returns
        -------
        pd.Series
            RSI 值
        
        Notes
        -----
        此方法保留用于兼容性，推荐使用 numba 优化的版本。
        """
        # 使用 numba 加速的实现
        rsi_values = _calculate_rsi_numba(series.values.astype(np.float64), period)
        return pd.Series(rsi_values, index=series.index)
    
    def calculate_ivol(self, period: int = 20) -> pd.Series:
        """
        计算特质波动率 (Idiosyncratic Volatility)
        
        简化版实现：过去 N 日收益率的标准差（年化）。
        使用 numba 加速的分组滚动标准差计算。
        
        Parameters
        ----------
        period : int, optional
            计算周期，默认20
        
        Returns
        -------
        pd.Series
            IVOL 因子值（年化波动率）
        
        Notes
        -----
        - 使用日收益率的滚动标准差
        - 年化因子为 sqrt(252)
        - IVOL 越低，股票风险越小
        - 使用 numba 优化的 _grouped_rolling_std 函数替代 groupby().apply()
        """
        min_periods = max(period // 2, 5)  # 至少需要一半窗口的数据
        
        if 'stock_code' in self.ohlcv_data.columns:
            # 使用 transform 计算分组收益率（避免 apply）
            returns = self.ohlcv_data.groupby('stock_code')['close'].transform(
                lambda x: x.pct_change()
            )
            
            # 使用 numba 优化的分组滚动标准差
            # 将 stock_code 转换为整数 ID
            stock_codes = self.ohlcv_data['stock_code'].values
            unique_codes = np.unique(stock_codes)
            code_to_id = {code: i for i, code in enumerate(unique_codes)}
            group_ids = np.array([code_to_id[code] for code in stock_codes], dtype=np.int64)
            n_groups = len(unique_codes)
            
            # 调用 numba 加速的函数
            ivol_values = _grouped_rolling_std(
                returns.values.astype(np.float64),
                group_ids,
                n_groups,
                period,
                min_periods
            )
            ivol = pd.Series(ivol_values, index=self.ohlcv_data.index)
        else:
            # 单股票情况：使用 numba 优化的一维滚动标准差
            returns = _calculate_returns_1d(self.ohlcv_data['close'].values.astype(np.float64))
            ivol_values = _rolling_std_1d(returns, period, min_periods)
            ivol = pd.Series(ivol_values, index=self.ohlcv_data.index)
        
        # 年化波动率（假设252个交易日）
        ivol_annualized = ivol * np.sqrt(252)
        
        logger.debug(f"IVOL_{period} 计算完成（年化，numba 加速）")
        return ivol_annualized
    
    # ==================== 激进型因子 ====================
    
    def calculate_log_circ_mv(self) -> pd.Series:
        """
        计算小市值因子 (Small Cap Factor)
        
        small_cap = -log(circ_mv)
        市值越小，因子值越大（取负号），适用于激进型小市值策略。
        
        Returns
        -------
        pd.Series
            小市值因子值
        
        Raises
        ------
        ValueError
            当财务数据缺少 circ_mv 列时抛出
        
        Notes
        -----
        - circ_mv (流通市值) 必须大于 0
        - 使用自然对数并取负值，确保市值越小分数越高
        - 处理 inf/-inf 的数学边界情况
        """
        if 'circ_mv' not in self.financial_data.columns:
            raise ValueError("财务数据缺少 circ_mv (流通市值) 列")
        
        # 向量化计算：-log(circ_mv)
        circ_mv = self.financial_data['circ_mv'].replace(0, np.nan)
        # 负市值不合法，设为 NaN
        circ_mv = circ_mv.where(circ_mv > 0, np.nan)
        
        # 计算自然对数
        log_mv = np.log(circ_mv)
        
        # 取负值：市值越小，因子值越大
        small_cap = -log_mv
        
        # 处理异常值
        small_cap = small_cap.replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"Small_Cap 因子计算完成，有效值数量: {small_cap.notna().sum()}")
        return small_cap
    
    def calculate_turnover_momentum(self, window: int = 5) -> pd.Series:
        """
        计算高换手情绪因子 (Active Turnover Factor)
        
        turnover_5d = rolling_mean(turn, window)
        换手率代表资金活跃度，激进策略中该值越高越好。
        
        Parameters
        ----------
        window : int, optional
            滚动窗口大小，默认 5 日
        
        Returns
        -------
        pd.Series
            换手率动量因子值
        
        Raises
        ------
        ValueError
            当 OHLCV 数据缺少 turn 列时抛出
        
        Notes
        -----
        - turn (换手率) 是日度数据
        - 使用滚动均值平滑短期波动
        - 按股票分组计算，避免跨股票数据污染
        """
        if 'turn' not in self.ohlcv_data.columns:
            raise ValueError("OHLCV 数据缺少 turn (换手率) 列")
        
        if 'stock_code' in self.ohlcv_data.columns:
            # 多股票情况：按股票分组计算滚动均值
            turnover_5d = self.ohlcv_data.groupby('stock_code')['turn'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        else:
            # 单股票情况：直接计算滚动均值
            turnover_5d = self.ohlcv_data['turn'].rolling(
                window=window, min_periods=1
            ).mean()
        
        logger.debug(f"Turnover_{window}d 因子计算完成")
        return turnover_5d
    
    # ==================== 计算所有因子 ====================
    
    def calculate_all_factors(self) -> pd.DataFrame:
        """
        计算所有因子并返回合并后的 DataFrame
        
        Returns
        -------
        pd.DataFrame
            包含所有因子的数据框，包括：
            - ep_ttm: EP_TTM 价值因子
            - dividend_yield: 股息率价值因子
            - roe_stability: ROE 稳定性质量因子
            - rsi_20: 20日 RSI 动量因子
            - ivol: 特质波动率因子
            - small_cap: 小市值因子（激进型）
            - turnover_5d: 5日换手率因子（激进型）
        
        Notes
        -----
        - 各因子计算失败不会影响其他因子
        - 失败的因子会记录警告日志
        """
        result = self.ohlcv_data.copy()
        
        # 价值因子
        try:
            result['ep_ttm'] = self.calculate_ep_ttm()
            logger.info("EP_TTM 因子计算完成")
        except Exception as e:
            logger.warning(f"EP_TTM 因子计算失败: {e}")
            result['ep_ttm'] = np.nan
        
        try:
            result['dividend_yield'] = self.calculate_dividend_yield()
            logger.info("Dividend_Yield 因子计算完成")
        except Exception as e:
            logger.warning(f"Dividend_Yield 因子计算失败: {e}")
            result['dividend_yield'] = np.nan
        
        # 质量因子
        try:
            result['roe_stability'] = self.calculate_roe_stability()
            logger.info("ROE_Stability 因子计算完成")
        except Exception as e:
            logger.warning(f"ROE_Stability 因子计算失败: {e}")
            result['roe_stability'] = np.nan
        
        # 技术/动量因子
        try:
            result['rsi_20'] = self.calculate_rsi(period=20)
            logger.info("RSI_20 因子计算完成")
        except Exception as e:
            logger.warning(f"RSI_20 因子计算失败: {e}")
            result['rsi_20'] = np.nan
        
        try:
            result['ivol'] = self.calculate_ivol(period=20)
            logger.info("IVOL 因子计算完成")
        except Exception as e:
            logger.warning(f"IVOL 因子计算失败: {e}")
            result['ivol'] = np.nan
        
        # 激进型因子
        try:
            result['small_cap'] = self.calculate_log_circ_mv()
            logger.info("Small_Cap 因子计算完成")
        except Exception as e:
            logger.warning(f"Small_Cap 因子计算失败: {e}")
            result['small_cap'] = np.nan
        
        try:
            result['turnover_5d'] = self.calculate_turnover_momentum(window=5)
            logger.info("Turnover_5d 因子计算完成")
        except Exception as e:
            logger.warning(f"Turnover_5d 因子计算失败: {e}")
            result['turnover_5d'] = np.nan
        
        logger.info("所有因子计算完成")
        return result
    
    # ==================== 标准化处理 ====================
    
    def z_score_normalize(
        self,
        data: pd.DataFrame,
        factor_cols: Union[str, List[str]],
        industry_neutral: bool = True,
        date_col: Optional[str] = None,
        stock_col: str = 'stock_code',
        industry_col: str = 'sw_industry_l1'
    ) -> pd.DataFrame:
        """
        对因子进行 Z-Score 标准化处理（支持行业中性化）
        
        对每一天的截面数据（Cross-section）进行标准化。
        行业中性化在申万一级行业内部进行标准化，消除行业效应。
        
        Parameters
        ----------
        data : pd.DataFrame
            包含因子数据的 DataFrame
        factor_cols : Union[str, List[str]]
            需要标准化的因子列名
        industry_neutral : bool, optional
            是否进行行业中性化，默认 True
        date_col : Optional[str]
            日期列名，如果为 None 则使用索引
        stock_col : str
            股票代码列名，默认 'stock_code'
        industry_col : str
            行业分类列名，默认 'sw_industry_l1'
        
        Returns
        -------
        pd.DataFrame
            标准化后的数据框，新增 '{factor}_zscore' 列
        
        Raises
        ------
        ValueError
            当行业中性化时缺少行业数据
        
        Notes
        -----
        - Z-Score = (x - mean) / std
        - 行业中性化：在每天每个行业内部计算 Z-Score
        - 处理标准差为 0 的情况（设为 0 而非 NaN）
        
        Examples
        --------
        >>> normalized = calculator.z_score_normalize(
        ...     factors, ['ep_ttm', 'rsi_20'], industry_neutral=True
        ... )
        >>> print(normalized[['ep_ttm_zscore', 'rsi_20_zscore']].head())
        """
        if isinstance(factor_cols, str):
            factor_cols = [factor_cols]
        
        result = data.copy()
        
        # 确定日期列
        date_col_name = date_col
        if date_col_name is None:
            if isinstance(result.index, pd.DatetimeIndex):
                result = result.reset_index()
                date_col_name = result.columns[0]
            elif 'date' in result.columns:
                date_col_name = 'date'
            elif 'trade_date' in result.columns:
                date_col_name = 'trade_date'
            else:
                raise ValueError("未指定日期列且无法自动推断")
        
        # 合并行业数据（如果需要行业中性化）
        if industry_neutral:
            if self.industry_data is None:
                raise ValueError("行业中性化需要提供行业分类数据（industry_data）")
            
            if industry_col not in result.columns:
                if stock_col not in result.columns:
                    raise ValueError(f"数据中缺少股票代码列: {stock_col}")
                
                # 合并行业分类
                industry_cols_to_merge = [stock_col, industry_col]
                available_cols = [c for c in industry_cols_to_merge if c in self.industry_data.columns]
                
                if industry_col not in self.industry_data.columns:
                    raise ValueError(f"行业数据中缺少行业分类列: {industry_col}")
                
                result = result.merge(
                    self.industry_data[available_cols].drop_duplicates(),
                    on=stock_col,
                    how='left'
                )
                logger.info("已合并行业分类数据")
        
        # 对每个因子进行标准化
        for col in factor_cols:
            if col not in result.columns:
                logger.warning(f"因子列 {col} 不存在，跳过")
                continue
            
            zscore_col = f'{col}_zscore'
            
            if industry_neutral:
                # 行业中性化：在每天每个行业内部进行 Z-Score 标准化
                result[zscore_col] = result.groupby(
                    [date_col_name, industry_col]
                )[col].transform(
                    lambda x: self._safe_zscore(x)
                )
                logger.info(f"{col} 行业中性化 Z-Score 标准化完成")
            else:
                # 普通标准化：在每天的截面上进行 Z-Score 标准化
                result[zscore_col] = result.groupby(date_col_name)[col].transform(
                    lambda x: self._safe_zscore(x)
                )
                logger.info(f"{col} 截面 Z-Score 标准化完成")
        
        return result
    
    @staticmethod
    def _safe_zscore(x: pd.Series) -> pd.Series:
        """
        安全的 Z-Score 计算，处理标准差为 0 的情况
        
        Parameters
        ----------
        x : pd.Series
            输入序列
        
        Returns
        -------
        pd.Series
            Z-Score 标准化后的序列
        """
        mean_val = x.mean()
        std_val = x.std()
        
        if std_val == 0 or pd.isna(std_val):
            return pd.Series(0.0, index=x.index)
        
        return (x - mean_val) / std_val


def z_score_normalize(
    data: pd.DataFrame,
    factor_cols: Union[str, List[str]],
    date_col: str,
    industry_col: Optional[str] = None,
    industry_neutral: bool = True
) -> pd.DataFrame:
    """
    独立的 Z-Score 标准化函数（支持行业中性化）
    
    对每一天的截面数据进行 Z-Score 标准化。
    若指定行业列，则在行业内部进行标准化（行业中性化）。
    
    Parameters
    ----------
    data : pd.DataFrame
        包含因子数据的 DataFrame
    factor_cols : Union[str, List[str]]
        需要标准化的因子列名
    date_col : str
        日期列名
    industry_col : Optional[str]
        行业分类列名（申万一级行业），默认 None
    industry_neutral : bool
        是否进行行业中性化，默认 True。仅当 industry_col 不为 None 时生效
    
    Returns
    -------
    pd.DataFrame
        标准化后的数据框，新增 '{factor}_zscore' 列
    
    Examples
    --------
    >>> # 普通截面标准化
    >>> df_norm = z_score_normalize(df, ['ep_ttm'], date_col='date', industry_neutral=False)
    
    >>> # 行业中性化标准化
    >>> df_norm = z_score_normalize(
    ...     df, ['ep_ttm', 'rsi_20'],
    ...     date_col='date',
    ...     industry_col='sw_industry_l1',
    ...     industry_neutral=True
    ... )
    """
    if isinstance(factor_cols, str):
        factor_cols = [factor_cols]
    
    result = data.copy()
    
    def safe_zscore(x: pd.Series) -> pd.Series:
        """安全的 Z-Score 计算"""
        mean_val = x.mean()
        std_val = x.std()
        if std_val == 0 or pd.isna(std_val):
            return pd.Series(0.0, index=x.index)
        return (x - mean_val) / std_val
    
    for col in factor_cols:
        if col not in result.columns:
            logger.warning(f"因子列 {col} 不存在，跳过")
            continue
        
        zscore_col = f'{col}_zscore'
        
        if industry_neutral and industry_col is not None:
            # 行业中性化：在每天每个行业内部进行标准化
            result[zscore_col] = result.groupby([date_col, industry_col])[col].transform(
                safe_zscore
            )
        else:
            # 普通截面标准化
            result[zscore_col] = result.groupby(date_col)[col].transform(
                safe_zscore
            )
    
    return result


def lag_fundamental_data(
    data: pd.DataFrame,
    lag_days: int = 60,
    date_col: Optional[str] = None,
    stock_col: str = "stock_code"
) -> pd.DataFrame:
    """
    对财务/基本面数据进行滞后处理，避免前视偏差（Look-Ahead Bias）
    
    A股财报发布时间规则决定了财务数据的实际可获得时间：
    - 年报（Q4）：次年4月30日前发布（滞后约4个月）
    - 半年报（Q2）：8月31日前发布（滞后约2个月）  
    - 一季报（Q1）：4月30日前发布（滞后约1个月）
    - 三季报（Q3）：10月31日前发布（滞后约1个月）
    
    默认滞后 60 个自然日（约 3 个月 / ~45 个交易日），
    这是一个保守估计，确保只使用历史上实际可获得的数据。
    
    Parameters
    ----------
    data : pd.DataFrame
        财务/基本面数据。可以是：
        - 单股票时序数据（索引为 DatetimeIndex）
        - 多股票面板数据（包含 stock_code 和 date 列）
    lag_days : int, optional
        滞后的自然日天数，默认 60（约3个月）
    date_col : Optional[str]
        日期列名。如果为 None，会尝试自动识别：
        - 优先使用 DatetimeIndex
        - 其次查找 'date', 'trade_date', 'report_date' 列
    stock_col : str
        股票代码列名，默认 'stock_code'
    
    Returns
    -------
    pd.DataFrame
        滞后后的财务数据，日期已向后移动 lag_days 天
    
    Notes
    -----
    - 此函数仅移动日期，不改变数据本身
    - 移动后的日期表示"该数据最早可被使用的日期"
    - 对于有发布日期（pub_date）的数据，建议使用 FactorCalculator 的内置方法
    
    Examples
    --------
    >>> # 对单股票财务数据进行滞后处理
    >>> fin_data_lagged = lag_fundamental_data(fin_data, lag_days=60)
    
    >>> # 对多股票面板数据进行滞后处理
    >>> fin_panel_lagged = lag_fundamental_data(
    ...     fin_panel, lag_days=45, date_col='report_date'
    ... )
    
    >>> # 在回测中使用滞后数据
    >>> from src.features import lag_fundamental_data
    >>> fin_data = loader.fetch_financial_data(symbols)
    >>> fin_data_lagged = lag_fundamental_data(fin_data, lag_days=60)
    >>> merged_data = pd.merge(price_data, fin_data_lagged, on=['date', 'stock_code'])
    """
    if data.empty:
        logger.warning("输入数据为空，直接返回")
        return data
    
    df = data.copy()
    lag_delta = pd.Timedelta(days=lag_days)
    
    # 自动识别日期列
    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            # 直接修改索引
            df.index = df.index + lag_delta
            df.index.name = 'date'
            logger.info(f"财务数据 DatetimeIndex 已向后偏移 {lag_days} 天")
            return df
        
        # 查找日期列
        date_candidates = ['date', 'trade_date', 'report_date', '日期']
        date_col = next((c for c in date_candidates if c in df.columns), None)
        
        if date_col is None:
            raise ValueError(
                f"无法识别日期列。请指定 date_col 参数或确保数据包含以下列之一: {date_candidates}"
            )
    
    # 对日期列进行偏移
    if date_col not in df.columns:
        raise ValueError(f"指定的日期列 '{date_col}' 不存在于数据中")
    
    df[date_col] = pd.to_datetime(df[date_col]) + lag_delta
    
    logger.info(f"财务数据 '{date_col}' 列已向后偏移 {lag_days} 天")
    return df


def merge_price_and_fundamental(
    price_data: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    price_date_col: str = "date",
    fund_date_col: str = "date",
    stock_col: str = "stock_code",
    lag_days: int = 60,
    apply_lag: bool = True
) -> pd.DataFrame:
    """
    合并价格数据和财务数据，自动处理前视偏差
    
    使用 asof merge 将财务数据按最近可用日期合并到价格数据上。
    
    Parameters
    ----------
    price_data : pd.DataFrame
        价格数据，包含 OHLCV 和日期
    fundamental_data : pd.DataFrame
        财务数据，包含基本面指标
    price_date_col : str
        价格数据的日期列名，默认 'date'
    fund_date_col : str
        财务数据的日期列名，默认 'date'
    stock_col : str
        股票代码列名，默认 'stock_code'
    lag_days : int
        财务数据滞后天数，默认 60
    apply_lag : bool
        是否应用滞后，默认 True
    
    Returns
    -------
    pd.DataFrame
        合并后的数据，每个价格日期使用当时最新可获得的财务数据
    
    Examples
    --------
    >>> merged = merge_price_and_fundamental(
    ...     price_data, fundamental_data,
    ...     lag_days=60, apply_lag=True
    ... )
    """
    if fundamental_data.empty:
        logger.warning("财务数据为空，仅返回价格数据")
        return price_data
    
    # 复制数据
    price_df = price_data.copy()
    fund_df = fundamental_data.copy()
    
    # 应用滞后
    if apply_lag:
        fund_df = lag_fundamental_data(fund_df, lag_days=lag_days, date_col=fund_date_col)
    
    # 确保日期列是 datetime 类型
    price_df[price_date_col] = pd.to_datetime(price_df[price_date_col])
    fund_df[fund_date_col] = pd.to_datetime(fund_df[fund_date_col])
    
    # 排序
    price_df = price_df.sort_values([stock_col, price_date_col])
    fund_df = fund_df.sort_values([stock_col, fund_date_col])
    
    # 如果两个数据的日期列名不同，统一为 'date'
    if price_date_col != 'date':
        price_df = price_df.rename(columns={price_date_col: 'date'})
    if fund_date_col != 'date':
        fund_df = fund_df.rename(columns={fund_date_col: 'date'})
    
    # 使用 merge_asof 按股票和日期合并
    # 对每个价格日期，找到最近的（但不晚于该日期的）财务数据
    merged = pd.merge_asof(
        price_df,
        fund_df,
        on='date',
        by=stock_col,
        direction='backward',  # 向后查找（只用历史数据）
        suffixes=('', '_fund')
    )
    
    logger.info(
        f"价格与财务数据合并完成: "
        f"价格数据 {len(price_df)} 行, "
        f"合并后 {len(merged)} 行"
    )
    
    return merged


# ==================== 情绪分析引擎 ====================

class SentimentEngine:
    """
    情绪分析引擎
    
    使用 LLM 分析股票新闻情绪，用于风险过滤。
    支持本地缓存避免重复调用 API。
    
    Parameters
    ----------
    llm_config : Optional[Dict[str, Any]]
        LLM 配置字典，包含：
        - provider: 提供商 ("deepseek", "openai")
        - model: 模型名称
        - base_url: API 端点
        - api_key: API 密钥（可选，优先使用环境变量）
        - cache_path: 缓存文件路径
        - request_timeout: 请求超时时间
        - max_retries: 最大重试次数
    
    Attributes
    ----------
    llm_client : LLMClient
        LLM 客户端实例
    cache : Dict[str, float]
        情绪分数缓存
    cache_path : Path
        缓存文件路径
    
    Examples
    --------
    >>> config = {"model": "deepseek-chat", "cache_path": "data/processed/sentiment_cache.json"}
    >>> engine = SentimentEngine(config)
    >>> scores = engine.calculate_sentiment(["000001", "000002"], "2024-01-15")
    >>> print(scores)
    
    Notes
    -----
    - 缓存键格式: "{stock_code}_{date}"
    - API 调用失败时返回 0.0（中性）
    - 新闻获取失败时返回 0.0（中性）
    """
    
    # 默认缓存路径
    DEFAULT_CACHE_PATH = "data/processed/sentiment_cache.json"
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化情绪分析引擎
        
        Parameters
        ----------
        llm_config : Optional[Dict[str, Any]]
            LLM 配置字典
        """
        from pathlib import Path
        
        self.config = llm_config or {}
        self.cache: Dict[str, float] = {}
        self.cache_path = Path(self.config.get("cache_path", self.DEFAULT_CACHE_PATH))
        
        # 初始化 LLM 客户端
        self.llm_client = self._create_llm_client()
        
        # 加载缓存
        self._load_cache()
        
        logger.info(
            f"SentimentEngine 初始化完成: "
            f"cache_path={self.cache_path}, "
            f"cached_entries={len(self.cache)}"
        )
    
    def _create_llm_client(self) -> Any:
        """
        创建 LLM 客户端
        
        Returns
        -------
        LLMClient
            LLM 客户端实例
        """
        try:
            from src.llm_client import LLMClient
        except ImportError:
            try:
                from llm_client import LLMClient
            except ImportError:
                logger.warning("无法导入 LLMClient，情绪分析功能不可用")
                return None
        
        return LLMClient(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("base_url"),
            model=self.config.get("model", "deepseek-chat"),
            timeout=self.config.get("request_timeout", 30),
            max_retries=self.config.get("max_retries", 3)
        )
    
    def _load_cache(self) -> None:
        """
        从文件加载缓存
        """
        import json
        
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.debug(f"加载情绪缓存: {len(self.cache)} 条记录")
            except Exception as e:
                logger.warning(f"加载情绪缓存失败: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def _save_cache(self) -> None:
        """
        保存缓存到文件
        """
        import json
        
        try:
            # 确保目录存在
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.debug(f"保存情绪缓存: {len(self.cache)} 条记录")
        except Exception as e:
            logger.warning(f"保存情绪缓存失败: {e}")
    
    def _get_cache_key(self, stock_code: str, date: str) -> str:
        """
        生成缓存键
        
        Parameters
        ----------
        stock_code : str
            股票代码
        date : str
            日期字符串
        
        Returns
        -------
        str
            缓存键
        """
        # 标准化日期格式
        date_str = str(date)[:10]  # 取前10个字符（YYYY-MM-DD）
        return f"{stock_code}_{date_str}"
    
    def _fetch_news(self, stock_code: str, date: str) -> str:
        """
        获取股票新闻
        
        使用 AkShare 获取指定股票在指定日期附近的新闻。
        
        Parameters
        ----------
        stock_code : str
            股票代码（如 "000001"）
        date : str
            日期字符串
        
        Returns
        -------
        str
            新闻内容摘要，如果获取失败返回空字符串
        
        Notes
        -----
        - 使用 akshare.stock_news_em() 获取东方财富新闻
        - 获取失败时返回空字符串，不影响策略运行
        - 新闻内容会被截断以控制 API 调用成本
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安装，无法获取新闻")
            return ""
        
        try:
            # 标准化股票代码（去除前缀后缀）
            clean_code = stock_code.replace(".", "").replace("SZ", "").replace("SH", "")
            if len(clean_code) > 6:
                clean_code = clean_code[:6]
            
            # 获取股票新闻
            # stock_news_em 返回最近的新闻，不需要日期参数
            news_df = ak.stock_news_em(symbol=clean_code)
            
            if news_df is None or news_df.empty:
                logger.debug(f"未找到股票新闻: {stock_code}")
                return ""
            
            # 过滤指定日期附近的新闻（前后3天）
            target_date = pd.to_datetime(date)
            if "发布时间" in news_df.columns:
                news_df["发布时间"] = pd.to_datetime(news_df["发布时间"], errors="coerce")
                date_mask = (
                    (news_df["发布时间"] >= target_date - pd.Timedelta(days=3)) &
                    (news_df["发布时间"] <= target_date + pd.Timedelta(days=1))
                )
                filtered_news = news_df[date_mask]
            else:
                # 如果没有日期列，取最新的5条
                filtered_news = news_df.head(5)
            
            if filtered_news.empty:
                # 没有指定日期的新闻，取最新的3条
                filtered_news = news_df.head(3)
            
            # 提取新闻标题和内容
            news_texts = []
            title_col = "新闻标题" if "新闻标题" in filtered_news.columns else None
            content_col = "新闻内容" if "新闻内容" in filtered_news.columns else None
            
            for _, row in filtered_news.head(5).iterrows():
                text_parts = []
                if title_col and pd.notna(row.get(title_col)):
                    text_parts.append(str(row[title_col]))
                if content_col and pd.notna(row.get(content_col)):
                    # 限制内容长度
                    content = str(row[content_col])[:200]
                    text_parts.append(content)
                if text_parts:
                    news_texts.append("; ".join(text_parts))
            
            combined_news = " | ".join(news_texts)
            
            # 限制总长度
            if len(combined_news) > 1500:
                combined_news = combined_news[:1500] + "..."
            
            logger.debug(
                f"获取新闻成功: stock={stock_code}, "
                f"news_count={len(filtered_news)}, "
                f"text_len={len(combined_news)}"
            )
            
            return combined_news
            
        except Exception as e:
            logger.warning(f"获取股票新闻失败 (stock={stock_code}): {e}")
            return ""
    
    def get_sentiment_score(
        self,
        stock_code: str,
        date: str,
        use_cache: bool = True
    ) -> float:
        """
        获取单只股票的情绪分数
        
        Parameters
        ----------
        stock_code : str
            股票代码
        date : str
            日期字符串
        use_cache : bool, optional
            是否使用缓存，默认 True
        
        Returns
        -------
        float
            情绪分数 [-1.0, 1.0]，失败时返回 0.0
        """
        cache_key = self._get_cache_key(stock_code, date)
        
        # 检查缓存
        if use_cache and cache_key in self.cache:
            logger.debug(f"命中缓存: {cache_key} = {self.cache[cache_key]}")
            return self.cache[cache_key]
        
        # 检查 LLM 客户端
        if self.llm_client is None or not self.llm_client.is_available:
            logger.debug(f"LLM 不可用，返回中性分数: {stock_code}")
            return 0.0
        
        # 获取新闻
        news_content = self._fetch_news(stock_code, date)
        
        if not news_content:
            # 没有新闻，返回中性
            score = 0.0
        else:
            # 调用 LLM 分析
            score = self.llm_client.get_sentiment_score(news_content, stock_code)
        
        # 更新缓存
        self.cache[cache_key] = score
        self._save_cache()
        
        return score
    
    def calculate_sentiment(
        self,
        stock_list: List[str],
        date: str,
        use_cache: bool = True
    ) -> pd.Series:
        """
        批量计算股票情绪分数
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        date : str
            日期字符串
        use_cache : bool, optional
            是否使用缓存，默认 True
        
        Returns
        -------
        pd.Series
            情绪分数序列，索引为股票代码
        
        Examples
        --------
        >>> engine = SentimentEngine()
        >>> scores = engine.calculate_sentiment(
        ...     ["000001", "000002", "600000"],
        ...     "2024-01-15"
        ... )
        >>> print(scores)
        000001    0.2
        000002   -0.5
        600000    0.0
        dtype: float64
        
        Notes
        -----
        - 缓存命中的股票不会重复调用 API
        - API 调用失败的股票返回 0.0（中性）
        - 批量调用后自动保存缓存
        """
        scores: Dict[str, float] = {}
        cache_hits = 0
        api_calls = 0
        
        for stock_code in stock_list:
            cache_key = self._get_cache_key(stock_code, date)
            
            # 检查缓存
            if use_cache and cache_key in self.cache:
                scores[stock_code] = self.cache[cache_key]
                cache_hits += 1
                continue
            
            # 需要调用 API
            score = self.get_sentiment_score(stock_code, date, use_cache=False)
            scores[stock_code] = score
            api_calls += 1
        
        logger.info(
            f"情绪分析完成: stocks={len(stock_list)}, "
            f"cache_hits={cache_hits}, "
            f"api_calls={api_calls}"
        )
        
        return pd.Series(scores, name="sentiment_score")
    
    def clear_cache(self) -> None:
        """
        清空缓存
        """
        self.cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.info("情绪缓存已清空")
    
    def get_cached_scores(
        self,
        stock_list: Optional[List[str]] = None,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取缓存的分数
        
        Parameters
        ----------
        stock_list : Optional[List[str]]
            过滤的股票列表
        date : Optional[str]
            过滤的日期
        
        Returns
        -------
        pd.DataFrame
            缓存的分数数据，包含 stock_code, date, score 列
        """
        records = []
        
        for key, score in self.cache.items():
            parts = key.rsplit("_", 1)
            if len(parts) != 2:
                continue
            
            stock_code, cache_date = parts
            
            # 应用过滤
            if stock_list is not None and stock_code not in stock_list:
                continue
            if date is not None and cache_date != str(date)[:10]:
                continue
            
            records.append({
                "stock_code": stock_code,
                "date": cache_date,
                "score": score
            })
        
        return pd.DataFrame(records)