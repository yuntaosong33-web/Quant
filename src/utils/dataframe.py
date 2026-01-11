"""
DataFrame 数据操作工具模块

提供 DataFrame 验证、重采样、收益率计算、标准化等功能。
"""

from typing import List, Union
from pathlib import Path
import logging

import pandas as pd
import numpy as np


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    check_index: bool = True
) -> bool:
    """
    验证DataFrame格式
    
    Parameters
    ----------
    df : pd.DataFrame
        要验证的数据框
    required_columns : List[str]
        必需的列名列表
    check_index : bool, optional
        是否检查索引为DatetimeIndex，默认为True
    
    Returns
    -------
    bool
        验证通过返回True
    
    Raises
    ------
    ValueError
        当验证失败时
    """
    # 检查空数据
    if df.empty:
        raise ValueError("DataFrame不能为空")
    
    # 检查必需列
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")
    
    # 检查索引类型
    if check_index and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame索引必须为DatetimeIndex类型")
    
    return True


def resample_ohlcv(
    df: pd.DataFrame,
    freq: str = "W"
) -> pd.DataFrame:
    """
    重采样OHLCV数据
    
    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据，必须包含 open, high, low, close, volume 列
    freq : str, optional
        目标频率，默认为周 'W'
        可选: 'D'(日), 'W'(周), 'M'(月), 'Q'(季), 'Y'(年)
    
    Returns
    -------
    pd.DataFrame
        重采样后的数据
    
    Examples
    --------
    >>> weekly_data = resample_ohlcv(daily_data, freq='W')
    """
    resampled = df.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    
    # 如果有成交额列
    if "amount" in df.columns:
        resampled["amount"] = df["amount"].resample(freq).sum()
    
    return resampled.dropna()


def calculate_returns(
    prices: pd.Series,
    method: str = "simple"
) -> pd.Series:
    """
    计算收益率
    
    Parameters
    ----------
    prices : pd.Series
        价格序列
    method : str, optional
        计算方法，'simple' 或 'log'，默认为 'simple'
    
    Returns
    -------
    pd.Series
        收益率序列
    """
    if method == "simple":
        return prices.pct_change()
    elif method == "log":
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"不支持的计算方法: {method}")


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    计算滚动Z-Score
    
    Parameters
    ----------
    series : pd.Series
        输入序列
    window : int, optional
        滚动窗口大小，默认20
    
    Returns
    -------
    pd.Series
        Z-Score序列
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    zscore = (series - rolling_mean) / rolling_std
    return zscore.replace([np.inf, -np.inf], np.nan)


def winsorize(
    series: pd.Series,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> pd.Series:
    """
    缩尾处理（Winsorization）
    
    Parameters
    ----------
    series : pd.Series
        输入序列
    lower_percentile : float, optional
        下界百分位，默认0.01
    upper_percentile : float, optional
        上界百分位，默认0.99
    
    Returns
    -------
    pd.Series
        缩尾后的序列
    """
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    
    return series.clip(lower=lower, upper=upper)


def neutralize(
    factor: pd.Series,
    groups: pd.Series
) -> pd.Series:
    """
    因子中性化（组内去均值）
    
    Parameters
    ----------
    factor : pd.Series
        因子值序列
    groups : pd.Series
        分组序列（如行业分类）
    
    Returns
    -------
    pd.Series
        中性化后的因子值
    """
    group_mean = factor.groupby(groups).transform("mean")
    return factor - group_mean


def create_dir_structure(base_path: Union[str, Path]) -> None:
    """
    创建项目目录结构
    
    Parameters
    ----------
    base_path : Union[str, Path]
        项目根目录
    """
    base_path = Path(base_path)
    
    directories = [
        "config",
        "data/raw",
        "data/processed",
        "src",
        "notebooks",
        "tests",
        "logs",
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"创建目录: {full_path}")

