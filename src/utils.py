"""
通用工具函数模块

提供日志配置、配置文件加载、数据验证等通用功能。
"""

from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import logging
import sys

import yaml
import pandas as pd
import numpy as np


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    配置日志系统
    
    Parameters
    ----------
    level : int, optional
        日志级别，默认为 INFO
    log_file : Optional[str]
        日志文件路径，如果为None则只输出到控制台
    format_string : Optional[str]
        日志格式字符串
    
    Returns
    -------
    logging.Logger
        配置好的根日志器
    
    Examples
    --------
    >>> logger = setup_logging(level=logging.DEBUG)
    >>> logger.info("系统启动")
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        )
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Parameters
    ----------
    config_path : Union[str, Path]
        配置文件路径
    
    Returns
    -------
    Dict[str, Any]
        配置字典
    
    Raises
    ------
    FileNotFoundError
        当配置文件不存在时
    yaml.YAMLError
        当YAML解析失败时
    
    Examples
    --------
    >>> config = load_config("config/strategy_config.yaml")
    >>> print(config["strategy"]["name"])
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    logging.info(f"配置文件加载成功: {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    保存配置到YAML文件
    
    Parameters
    ----------
    config : Dict[str, Any]
        配置字典
    config_path : Union[str, Path]
        保存路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    logging.info(f"配置文件保存成功: {config_path}")


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


class Timer:
    """
    计时器上下文管理器
    
    Examples
    --------
    >>> with Timer("数据加载"):
    ...     data = load_data()
    数据加载 耗时: 1.23秒
    """
    
    def __init__(self, name: str = "操作") -> None:
        """
        初始化计时器
        
        Parameters
        ----------
        name : str
            操作名称
        """
        self.name = name
        self._start_time = None
    
    def __enter__(self) -> "Timer":
        """进入上下文"""
        import time
        self._start_time = time.time()
        return self
    
    def __exit__(self, *args) -> None:
        """退出上下文"""
        import time
        elapsed = time.time() - self._start_time
        logging.info(f"{self.name} 耗时: {elapsed:.2f}秒")


def send_pushplus_msg(
    token: str,
    title: str,
    content: str,
    template: str = "html"
) -> bool:
    """
    使用 PushPlus 发送微信消息
    
    通过 PushPlus (http://www.pushplus.plus/) 将消息推送到微信。
    支持 HTML、Markdown 等多种格式。
    
    Parameters
    ----------
    token : str
        PushPlus 的用户 token，在官网注册后获取
    title : str
        消息标题
    content : str
        消息内容（支持 HTML/Markdown 格式）
    template : str, optional
        模板类型，可选：
        - 'html': HTML 格式（默认）
        - 'txt': 纯文本
        - 'json': JSON 格式
        - 'markdown': Markdown 格式
    
    Returns
    -------
    bool
        发送是否成功
    
    Examples
    --------
    >>> token = "your_pushplus_token"
    >>> send_pushplus_msg(token, "交易提醒", "<h1>今日需买入 5 只股票</h1>")
    True
    
    Notes
    -----
    - PushPlus 免费版每天限制 200 条消息
    - 如果 token 为空或无效，函数会记录警告并返回 False
    - 网络异常不会导致程序崩溃
    """
    import requests
    
    # 检查 token
    if not token or token.strip() == "":
        logging.warning("PushPlus token 未配置，跳过消息推送")
        return False
    
    url = "http://www.pushplus.plus/send"
    
    payload = {
        "token": token.strip(),
        "title": title,
        "content": content,
        "template": template
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()
        
        if result.get("code") == 200:
            logging.info(f"PushPlus 消息发送成功: {title}")
            return True
        else:
            error_msg = result.get("msg", "未知错误")
            logging.error(f"PushPlus 消息发送失败: {error_msg}")
            return False
            
    except requests.exceptions.Timeout:
        logging.error("PushPlus 请求超时")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"PushPlus 网络请求异常: {e}")
        return False
    except ValueError as e:
        logging.error(f"PushPlus 响应解析失败: {e}")
        return False
    except Exception as e:
        logging.error(f"PushPlus 发送异常: {e}")
        return False


def format_number(
    value: float,
    precision: int = 2,
    as_percentage: bool = False
) -> str:
    """
    格式化数字
    
    Parameters
    ----------
    value : float
        数值
    precision : int, optional
        小数位数，默认2
    as_percentage : bool, optional
        是否格式化为百分比，默认False
    
    Returns
    -------
    str
        格式化后的字符串
    """
    if as_percentage:
        return f"{value * 100:.{precision}f}%"
    
    if abs(value) >= 1e8:
        return f"{value / 1e8:.{precision}f}亿"
    elif abs(value) >= 1e4:
        return f"{value / 1e4:.{precision}f}万"
    else:
        return f"{value:.{precision}f}"


# =============================================================================
# Pandas GroupBy 向量化计算工具
# =============================================================================

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
    from scipy import stats
    
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


# =============================================================================
# PyPortfolioOpt 投资组合优化工具
# =============================================================================

def calculate_shrinkage_covariance(
    prices: pd.DataFrame,
    shrinkage_type: str = "ledoit_wolf"
) -> pd.DataFrame:
    """
    计算压缩协方差矩阵
    
    使用 PyPortfolioOpt 的 CovarianceShrinkage 计算更稳健的协方差矩阵。
    
    Parameters
    ----------
    prices : pd.DataFrame
        价格数据，索引为日期，列为股票代码
    shrinkage_type : str, optional
        压缩方法，可选：
        - 'ledoit_wolf': Ledoit-Wolf 压缩（默认）
        - 'oracle_approximating': Oracle Approximating Shrinkage
        - 'exponential': 指数加权协方差
    
    Returns
    -------
    pd.DataFrame
        压缩后的协方差矩阵
    
    Examples
    --------
    >>> cov_matrix = calculate_shrinkage_covariance(prices_df)
    >>> print(cov_matrix.shape)
    
    Notes
    -----
    Ledoit-Wolf 压缩通过将样本协方差矩阵向结构化目标收缩，
    可以显著提高协方差估计的稳定性，特别是在样本量较小时。
    """
    try:
        from pypfopt import risk_models
    except ImportError:
        raise ImportError(
            "请安装 pypfopt: pip install pyportfolioopt"
        )
    
    if shrinkage_type == "ledoit_wolf":
        cov_matrix = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    elif shrinkage_type == "oracle_approximating":
        cov_matrix = risk_models.CovarianceShrinkage(prices).oracle_approximating()
    elif shrinkage_type == "exponential":
        cov_matrix = risk_models.exp_cov(prices)
    else:
        raise ValueError(f"不支持的压缩方法: {shrinkage_type}")
    
    logging.debug(f"协方差矩阵计算完成，形状: {cov_matrix.shape}")
    return cov_matrix


def calculate_expected_returns_mean(
    prices: pd.DataFrame,
    method: str = "mean_historical",
    frequency: int = 252
) -> pd.Series:
    """
    计算预期收益率
    
    Parameters
    ----------
    prices : pd.DataFrame
        价格数据，索引为日期，列为股票代码
    method : str, optional
        计算方法，可选：
        - 'mean_historical': 历史平均收益率（默认）
        - 'ema_historical': 指数加权平均收益率
        - 'capm': CAPM 模型（需要市场收益率）
    frequency : int, optional
        年化频率，默认252（日频）
    
    Returns
    -------
    pd.Series
        预期年化收益率
    """
    try:
        from pypfopt import expected_returns
    except ImportError:
        raise ImportError(
            "请安装 pypfopt: pip install pyportfolioopt"
        )
    
    if method == "mean_historical":
        mu = expected_returns.mean_historical_return(prices, frequency=frequency)
    elif method == "ema_historical":
        mu = expected_returns.ema_historical_return(prices, frequency=frequency)
    else:
        raise ValueError(f"不支持的计算方法: {method}")
    
    logging.debug(f"预期收益率计算完成，股票数: {len(mu)}")
    return mu


def optimize_weights(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.02,
    max_weight: float = 0.05,
    min_weight: float = 0.0,
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
    sector_mapper: Optional[Dict[str, str]] = None,
    sector_lower: Optional[Dict[str, float]] = None,
    sector_upper: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    使用 PyPortfolioOpt 优化投资组合权重
    
    支持最大夏普比率、最小波动率等多种优化目标。
    
    Parameters
    ----------
    expected_returns : pd.Series
        预期收益率，索引为股票代码
    cov_matrix : pd.DataFrame
        协方差矩阵
    objective : str, optional
        优化目标，可选：
        - 'max_sharpe': 最大夏普比率（默认）
        - 'min_volatility': 最小波动率
        - 'efficient_return': 给定目标收益的最小风险
        - 'efficient_risk': 给定目标波动率的最大收益
    risk_free_rate : float, optional
        无风险利率，默认0.02（2%）
    max_weight : float, optional
        单只股票最大权重，默认0.05（5%）
    min_weight : float, optional
        单只股票最小权重，默认0.0
    target_return : Optional[float]
        目标收益率（用于 efficient_return）
    target_volatility : Optional[float]
        目标波动率（用于 efficient_risk）
    sector_mapper : Optional[Dict[str, str]]
        股票到行业的映射字典
    sector_lower : Optional[Dict[str, float]]
        行业权重下限
    sector_upper : Optional[Dict[str, float]]
        行业权重上限
    
    Returns
    -------
    Dict[str, Any]
        优化结果，包含：
        - 'weights': 优化后的权重字典
        - 'clean_weights': 清理后的权重（去除接近零的权重）
        - 'performance': (预期收益, 波动率, 夏普比率)
        - 'n_assets': 有效资产数量
    
    Examples
    --------
    >>> result = optimize_weights(
    ...     expected_returns, cov_matrix,
    ...     objective='max_sharpe',
    ...     max_weight=0.05
    ... )
    >>> print(result['clean_weights'])
    >>> print(f"夏普比率: {result['performance'][2]:.2f}")
    
    Notes
    -----
    - 权重之和约束为 1（全仓投资）
    - 使用 clean_weights 方法去除小于阈值的权重
    - 支持行业权重约束
    """
    try:
        from pypfopt import EfficientFrontier
        from pypfopt import objective_functions
    except ImportError:
        raise ImportError(
            "请安装 pypfopt: pip install pyportfolioopt"
        )
    
    # 确保索引一致
    common_assets = expected_returns.index.intersection(cov_matrix.index)
    if len(common_assets) < len(expected_returns):
        logging.warning(
            f"预期收益率和协方差矩阵的资产不完全匹配，"
            f"使用共同的 {len(common_assets)} 个资产"
        )
    
    mu = expected_returns.loc[common_assets]
    S = cov_matrix.loc[common_assets, common_assets]
    
    # 创建有效边界优化器
    ef = EfficientFrontier(
        mu, S,
        weight_bounds=(min_weight, max_weight)
    )
    
    # 添加行业约束（如果提供）
    if sector_mapper is not None:
        ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    
    # 执行优化
    try:
        if objective == "max_sharpe":
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif objective == "min_volatility":
            weights = ef.min_volatility()
        elif objective == "efficient_return":
            if target_return is None:
                raise ValueError("efficient_return 需要指定 target_return")
            weights = ef.efficient_return(target_return=target_return)
        elif objective == "efficient_risk":
            if target_volatility is None:
                raise ValueError("efficient_risk 需要指定 target_volatility")
            weights = ef.efficient_risk(target_volatility=target_volatility)
        else:
            raise ValueError(f"不支持的优化目标: {objective}")
    except Exception as e:
        logging.error(f"优化失败: {e}")
        # 失败时返回等权重
        n_assets = len(common_assets)
        equal_weight = 1.0 / n_assets
        weights = {asset: equal_weight for asset in common_assets}
        return {
            'weights': weights,
            'clean_weights': weights,
            'performance': (0, 0, 0),
            'n_assets': n_assets,
            'status': 'failed',
            'error': str(e)
        }
    
    # 清理权重（去除接近零的权重）
    clean_weights = ef.clean_weights(cutoff=1e-4, rounding=4)
    
    # 计算组合绩效
    performance = ef.portfolio_performance(
        verbose=False,
        risk_free_rate=risk_free_rate
    )
    
    # 计算有效资产数量
    n_effective = sum(1 for w in clean_weights.values() if w > 1e-4)
    
    logging.info(
        f"优化完成 [{objective}]: "
        f"预期收益 {performance[0]:.2%}, "
        f"波动率 {performance[1]:.2%}, "
        f"夏普比率 {performance[2]:.2f}, "
        f"有效资产 {n_effective}"
    )
    
    return {
        'weights': dict(weights),
        'clean_weights': clean_weights,
        'performance': performance,
        'n_assets': n_effective,
        'status': 'success'
    }


def optimize_portfolio_from_prices(
    prices: pd.DataFrame,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.02,
    max_weight: float = 0.05,
    min_weight: float = 0.0,
    shrinkage_type: str = "ledoit_wolf",
    returns_method: str = "mean_historical"
) -> Dict[str, Any]:
    """
    从价格数据直接优化投资组合（便捷函数）
    
    整合预期收益率计算、协方差矩阵估计和优化过程。
    
    Parameters
    ----------
    prices : pd.DataFrame
        价格数据，索引为日期，列为股票代码
    objective : str, optional
        优化目标，默认 'max_sharpe'
    risk_free_rate : float, optional
        无风险利率，默认0.02
    max_weight : float, optional
        单只股票最大权重，默认0.05
    min_weight : float, optional
        单只股票最小权重，默认0.0
    shrinkage_type : str, optional
        协方差压缩方法，默认 'ledoit_wolf'
    returns_method : str, optional
        预期收益率计算方法，默认 'mean_historical'
    
    Returns
    -------
    Dict[str, Any]
        优化结果
    
    Examples
    --------
    >>> result = optimize_portfolio_from_prices(
    ...     prices_df,
    ...     objective='max_sharpe',
    ...     max_weight=0.05
    ... )
    >>> weights = result['clean_weights']
    """
    # 计算预期收益率
    mu = calculate_expected_returns_mean(prices, method=returns_method)
    
    # 计算压缩协方差矩阵
    S = calculate_shrinkage_covariance(prices, shrinkage_type=shrinkage_type)
    
    # 优化权重
    result = optimize_weights(
        mu, S,
        objective=objective,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        min_weight=min_weight
    )
    
    # 添加额外信息
    result['expected_returns'] = mu
    result['cov_matrix'] = S
    
    return result


class PortfolioWeightOptimizer:
    """
    投资组合权重优化器
    
    封装 PyPortfolioOpt 的功能，提供灵活的权重优化接口。
    
    Parameters
    ----------
    risk_free_rate : float
        无风险利率，默认0.02
    max_weight : float
        单只股票最大权重，默认0.05
    min_weight : float
        单只股票最小权重，默认0.0
    
    Attributes
    ----------
    expected_returns : pd.Series
        预期收益率
    cov_matrix : pd.DataFrame
        协方差矩阵
    weights : Dict[str, float]
        优化后的权重
    
    Examples
    --------
    >>> optimizer = PortfolioWeightOptimizer(max_weight=0.05)
    >>> optimizer.fit(prices_df)
    >>> weights = optimizer.optimize('max_sharpe')
    >>> print(optimizer.get_portfolio_stats())
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_weight: float = 0.05,
        min_weight: float = 0.0,
        shrinkage_type: str = "ledoit_wolf"
    ) -> None:
        """
        初始化优化器
        
        Parameters
        ----------
        risk_free_rate : float
            无风险利率
        max_weight : float
            单只股票最大权重
        min_weight : float
            单只股票最小权重
        shrinkage_type : str
            协方差压缩方法
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.shrinkage_type = shrinkage_type
        
        self.expected_returns: Optional[pd.Series] = None
        self.cov_matrix: Optional[pd.DataFrame] = None
        self.weights: Optional[Dict[str, float]] = None
        self._performance: Optional[tuple] = None
        self._ef: Optional[Any] = None
    
    def fit(
        self,
        prices: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> "PortfolioWeightOptimizer":
        """
        拟合优化器
        
        可以从价格数据自动计算，或直接提供预期收益率和协方差矩阵。
        
        Parameters
        ----------
        prices : Optional[pd.DataFrame]
            价格数据
        expected_returns : Optional[pd.Series]
            预期收益率（如果提供则优先使用）
        cov_matrix : Optional[pd.DataFrame]
            协方差矩阵（如果提供则优先使用）
        
        Returns
        -------
        self
            返回自身以支持链式调用
        """
        if expected_returns is not None:
            self.expected_returns = expected_returns
        elif prices is not None:
            self.expected_returns = calculate_expected_returns_mean(prices)
        else:
            raise ValueError("必须提供 prices 或 expected_returns")
        
        if cov_matrix is not None:
            self.cov_matrix = cov_matrix
        elif prices is not None:
            self.cov_matrix = calculate_shrinkage_covariance(
                prices, self.shrinkage_type
            )
        else:
            raise ValueError("必须提供 prices 或 cov_matrix")
        
        logging.info(f"优化器拟合完成，资产数: {len(self.expected_returns)}")
        return self
    
    def optimize(
        self,
        objective: str = "max_sharpe",
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """
        执行优化
        
        Parameters
        ----------
        objective : str
            优化目标
        target_return : Optional[float]
            目标收益率
        target_volatility : Optional[float]
            目标波动率
        
        Returns
        -------
        Dict[str, float]
            清理后的权重字典
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError("请先调用 fit 方法")
        
        result = optimize_weights(
            self.expected_returns,
            self.cov_matrix,
            objective=objective,
            risk_free_rate=self.risk_free_rate,
            max_weight=self.max_weight,
            min_weight=self.min_weight,
            target_return=target_return,
            target_volatility=target_volatility
        )
        
        self.weights = result['clean_weights']
        self._performance = result['performance']
        
        return self.weights
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """
        获取组合统计信息
        
        Returns
        -------
        Dict[str, float]
            包含预期收益、波动率、夏普比率的字典
        """
        if self._performance is None:
            raise ValueError("请先调用 optimize 方法")
        
        return {
            'expected_return': self._performance[0],
            'volatility': self._performance[1],
            'sharpe_ratio': self._performance[2]
        }
    
    def get_weights_series(self) -> pd.Series:
        """
        获取权重 Series
        
        Returns
        -------
        pd.Series
            权重序列，索引为股票代码
        """
        if self.weights is None:
            raise ValueError("请先调用 optimize 方法")
        
        return pd.Series(self.weights).sort_values(ascending=False)
    
    def get_effective_assets(self, threshold: float = 1e-4) -> List[str]:
        """
        获取有效资产列表
        
        Parameters
        ----------
        threshold : float
            权重阈值
        
        Returns
        -------
        List[str]
            有效资产列表
        """
        if self.weights is None:
            raise ValueError("请先调用 optimize 方法")
        
        return [asset for asset, weight in self.weights.items() if weight > threshold]