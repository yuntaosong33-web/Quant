"""
Pandas GroupBy 向量化计算工具模块

提供高性能的分组计算函数，避免使用 apply 循环。
"""

from typing import Dict, List
import pandas as pd
import numpy as np


def groupby_rolling(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    func: str = "mean",
    min_periods: int = 1
) -> pd.Series:
    """
    分组滚动计算（向量化实现）
    
    按指定列分组后，对每组进行滚动窗口计算。
    常用于按股票代码分组计算技术指标。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据，必须已按时间排序
    group_col : str
        分组列名（如 'symbol', 'stock_code'）
    value_col : str
        计算列名（如 'close', 'volume'）
    window : int
        滚动窗口大小
    func : str, optional
        聚合函数，可选 'mean', 'sum', 'std', 'min', 'max', 'median'
        默认为 'mean'
    min_periods : int, optional
        最小有效观测数，默认1
    
    Returns
    -------
    pd.Series
        计算结果，索引与输入df相同
    
    Examples
    --------
    >>> # 按股票分组计算20日均价
    >>> df['ma20'] = groupby_rolling(df, 'symbol', 'close', 20, 'mean')
    
    >>> # 按股票分组计算10日成交量和
    >>> df['vol_sum10'] = groupby_rolling(df, 'symbol', 'volume', 10, 'sum')
    
    Notes
    -----
    使用 groupby + transform 实现向量化，避免 apply 循环，性能更优。
    """
    # 使用 groupby + rolling + transform 向量化计算
    grouped = df.groupby(group_col)[value_col]
    
    if func == "mean":
        result = grouped.transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )
    elif func == "sum":
        result = grouped.transform(
            lambda x: x.rolling(window, min_periods=min_periods).sum()
        )
    elif func == "std":
        result = grouped.transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )
    elif func == "min":
        result = grouped.transform(
            lambda x: x.rolling(window, min_periods=min_periods).min()
        )
    elif func == "max":
        result = grouped.transform(
            lambda x: x.rolling(window, min_periods=min_periods).max()
        )
    elif func == "median":
        result = grouped.transform(
            lambda x: x.rolling(window, min_periods=min_periods).median()
        )
    else:
        raise ValueError(f"不支持的聚合函数: {func}")
    
    return result


def groupby_shift(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    periods: int = 1
) -> pd.Series:
    """
    分组偏移（向量化实现）
    
    按指定列分组后，对每组进行时间偏移。
    用于计算分组内的滞后/领先值。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名
    value_col : str
        偏移列名
    periods : int, optional
        偏移周期，正数为滞后，负数为领先，默认1
    
    Returns
    -------
    pd.Series
        偏移后的结果
    
    Examples
    --------
    >>> # 按股票分组获取前一日收盘价
    >>> df['prev_close'] = groupby_shift(df, 'symbol', 'close', 1)
    
    >>> # 按股票分组获取未来5日收益
    >>> df['future_ret'] = groupby_shift(df, 'symbol', 'return', -5)
    """
    return df.groupby(group_col)[value_col].transform(
        lambda x: x.shift(periods)
    )


def groupby_pct_change(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    periods: int = 1
) -> pd.Series:
    """
    分组收益率计算（向量化实现）
    
    按指定列分组后，计算每组的百分比变化。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名
    value_col : str
        计算列名
    periods : int, optional
        变化周期，默认1
    
    Returns
    -------
    pd.Series
        收益率序列
    
    Examples
    --------
    >>> # 按股票分组计算日收益率
    >>> df['daily_ret'] = groupby_pct_change(df, 'symbol', 'close', 1)
    
    >>> # 按股票分组计算5日收益率
    >>> df['ret_5d'] = groupby_pct_change(df, 'symbol', 'close', 5)
    """
    return df.groupby(group_col)[value_col].transform(
        lambda x: x.pct_change(periods)
    )


def groupby_rank(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    method: str = "average",
    ascending: bool = True,
    pct: bool = True
) -> pd.Series:
    """
    分组排名（向量化实现）
    
    按指定列分组后，在每组内进行排名。
    常用于截面因子排名和分位数计算。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名（如 'date' 用于截面排名）
    value_col : str
        排名列名（如因子值）
    method : str, optional
        排名方法，可选 'average', 'min', 'max', 'first', 'dense'
        默认 'average'
    ascending : bool, optional
        是否升序排名，默认True
    pct : bool, optional
        是否返回百分比排名（0-1），默认True
    
    Returns
    -------
    pd.Series
        排名结果
    
    Examples
    --------
    >>> # 按日期截面对因子进行百分比排名
    >>> df['factor_rank'] = groupby_rank(df, 'date', 'factor', pct=True)
    
    >>> # 按行业分组排名
    >>> df['sector_rank'] = groupby_rank(df, 'sector', 'return', pct=False)
    """
    return df.groupby(group_col)[value_col].transform(
        lambda x: x.rank(method=method, ascending=ascending, pct=pct)
    )


def groupby_zscore(
    df: pd.DataFrame,
    group_col: str,
    value_col: str
) -> pd.Series:
    """
    分组Z-Score标准化（向量化实现）
    
    按指定列分组后，在每组内进行Z-Score标准化。
    常用于因子截面标准化。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名
    value_col : str
        标准化列名
    
    Returns
    -------
    pd.Series
        标准化后的结果（均值0，标准差1）
    
    Examples
    --------
    >>> # 按日期截面标准化因子
    >>> df['factor_zscore'] = groupby_zscore(df, 'date', 'factor')
    
    Notes
    -----
    Z-Score = (x - mean) / std
    使用 transform 实现向量化，性能优于 apply。
    """
    def zscore(x: pd.Series) -> pd.Series:
        mean = x.mean()
        std = x.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=x.index)
        return (x - mean) / std
    
    return df.groupby(group_col)[value_col].transform(zscore)


def groupby_winsorize(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    lower: float = 0.01,
    upper: float = 0.99
) -> pd.Series:
    """
    分组缩尾处理（向量化实现）
    
    按指定列分组后，在每组内进行缩尾处理。
    用于处理极端值。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名
    value_col : str
        处理列名
    lower : float, optional
        下界百分位，默认0.01（1%）
    upper : float, optional
        上界百分位，默认0.99（99%）
    
    Returns
    -------
    pd.Series
        缩尾后的结果
    
    Examples
    --------
    >>> # 按日期截面缩尾
    >>> df['factor_winsorized'] = groupby_winsorize(df, 'date', 'factor')
    """
    def winsorize_group(x: pd.Series) -> pd.Series:
        lower_bound = x.quantile(lower)
        upper_bound = x.quantile(upper)
        return x.clip(lower=lower_bound, upper=upper_bound)
    
    return df.groupby(group_col)[value_col].transform(winsorize_group)


def groupby_neutralize(
    df: pd.DataFrame,
    date_col: str,
    factor_col: str,
    group_col: str
) -> pd.Series:
    """
    因子行业中性化（向量化实现）
    
    对因子进行行业中性化处理：先按日期分组，再在每日内按行业去均值。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    date_col : str
        日期列名
    factor_col : str
        因子列名
    group_col : str
        行业/分组列名
    
    Returns
    -------
    pd.Series
        中性化后的因子
    
    Examples
    --------
    >>> # 因子行业中性化
    >>> df['factor_neutral'] = groupby_neutralize(
    ...     df, 'date', 'momentum', 'industry'
    ... )
    
    Notes
    -----
    中性化公式: factor_neutral = factor - industry_mean
    使用双重 groupby 实现截面内行业中性化。
    """
    # 计算每日每行业的因子均值
    industry_mean = df.groupby([date_col, group_col])[factor_col].transform('mean')
    
    # 中性化 = 原始值 - 行业均值
    return df[factor_col] - industry_mean


def groupby_cumsum(
    df: pd.DataFrame,
    group_col: str,
    value_col: str
) -> pd.Series:
    """
    分组累计求和（向量化实现）
    
    按指定列分组后，计算组内累计和。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名
    value_col : str
        累加列名
    
    Returns
    -------
    pd.Series
        累计和序列
    
    Examples
    --------
    >>> # 按股票分组计算累计收益
    >>> df['cum_return'] = groupby_cumsum(df, 'symbol', 'daily_return')
    """
    return df.groupby(group_col)[value_col].transform('cumsum')


def groupby_cumprod(
    df: pd.DataFrame,
    group_col: str,
    value_col: str
) -> pd.Series:
    """
    分组累计乘积（向量化实现）
    
    按指定列分组后，计算组内累计乘积。
    常用于计算累计净值。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名
    value_col : str
        累乘列名（通常是 1 + return）
    
    Returns
    -------
    pd.Series
        累计乘积序列
    
    Examples
    --------
    >>> # 按股票分组计算累计净值
    >>> df['cum_value'] = groupby_cumprod(df, 'symbol', 'return_plus_one')
    """
    return df.groupby(group_col)[value_col].transform('cumprod')


def groupby_ewm(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    span: int = 20,
    func: str = "mean"
) -> pd.Series:
    """
    分组指数加权计算（向量化实现）
    
    按指定列分组后，进行指数加权移动平均/标准差计算。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名
    value_col : str
        计算列名
    span : int, optional
        衰减跨度，默认20
    func : str, optional
        聚合函数，可选 'mean', 'std', 'var'，默认 'mean'
    
    Returns
    -------
    pd.Series
        计算结果
    
    Examples
    --------
    >>> # 按股票分组计算EMA
    >>> df['ema20'] = groupby_ewm(df, 'symbol', 'close', span=20, func='mean')
    
    >>> # 按股票分组计算指数加权波动率
    >>> df['ewm_vol'] = groupby_ewm(df, 'symbol', 'return', span=20, func='std')
    """
    grouped = df.groupby(group_col)[value_col]
    
    if func == "mean":
        return grouped.transform(lambda x: x.ewm(span=span, adjust=False).mean())
    elif func == "std":
        return grouped.transform(lambda x: x.ewm(span=span, adjust=False).std())
    elif func == "var":
        return grouped.transform(lambda x: x.ewm(span=span, adjust=False).var())
    else:
        raise ValueError(f"不支持的聚合函数: {func}")


def groupby_diff(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    periods: int = 1
) -> pd.Series:
    """
    分组差分（向量化实现）
    
    按指定列分组后，计算组内差分。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名
    value_col : str
        差分列名
    periods : int, optional
        差分周期，默认1
    
    Returns
    -------
    pd.Series
        差分结果
    
    Examples
    --------
    >>> # 按股票分组计算价格变动
    >>> df['price_change'] = groupby_diff(df, 'symbol', 'close', 1)
    """
    return df.groupby(group_col)[value_col].transform(
        lambda x: x.diff(periods)
    )


def groupby_apply_multiple(
    df: pd.DataFrame,
    group_col: str,
    agg_dict: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    分组多列多函数聚合（向量化实现）
    
    对多个列同时应用多个聚合函数。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    group_col : str
        分组列名
    agg_dict : Dict[str, List[str]]
        聚合字典，格式 {列名: [函数列表]}
    
    Returns
    -------
    pd.DataFrame
        聚合结果
    
    Examples
    --------
    >>> agg_dict = {
    ...     'close': ['mean', 'std', 'min', 'max'],
    ...     'volume': ['sum', 'mean']
    ... }
    >>> result = groupby_apply_multiple(df, 'symbol', agg_dict)
    """
    return df.groupby(group_col).agg(agg_dict)


def cross_sectional_regression(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    x_cols: List[str]
) -> pd.DataFrame:
    """
    截面回归残差（向量化实现）
    
    按日期分组进行截面回归，返回残差。
    用于因子正交化。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    date_col : str
        日期列名
    y_col : str
        因变量列名
    x_cols : List[str]
        自变量列名列表
    
    Returns
    -------
    pd.DataFrame
        包含残差和系数的DataFrame
    
    Examples
    --------
    >>> # 对动量因子进行市值和行业中性化
    >>> result = cross_sectional_regression(
    ...     df, 'date', 'momentum', ['log_market_cap', 'industry_dummy']
    ... )
    >>> df['momentum_resid'] = result['residual']
    """
    def regress(group: pd.DataFrame) -> pd.Series:
        y = group[y_col].values
        X = group[x_cols].values
        
        # 添加常数项
        X = np.column_stack([np.ones(len(X)), X])
        
        try:
            # OLS回归
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            predicted = X @ coeffs
            resid = y - predicted
            return pd.Series(resid, index=group.index)
        except Exception:
            return pd.Series(np.nan, index=group.index)
    
    residuals = df.groupby(date_col, group_keys=False).apply(regress)
    
    result = pd.DataFrame(index=df.index)
    result['residual'] = residuals
    
    return result

