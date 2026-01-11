"""
Tushare 数据加载器模块

该模块使用 Mixin 模式组合多个功能模块，提供完整的 TushareDataLoader 类。

Usage
-----
>>> from src.tushare import TushareDataLoader
>>> loader = TushareDataLoader()
>>> stocks = loader.fetch_index_constituents("hs300")
>>> news = loader.fetch_news_multi_source()
"""

from .base import TushareDataLoaderBase, create_tushare_loader
from .news import TushareNewsMixin
from .moneyflow import TushareMoneyflowMixin
from .limit_factor import TushareLimitFactorMixin


class TushareDataLoader(
    TushareNewsMixin,
    TushareMoneyflowMixin,
    TushareLimitFactorMixin,
    TushareDataLoaderBase
):
    """
    完整的 Tushare 数据加载器（组合所有功能模块）
    
    继承自：
    - TushareDataLoaderBase: 核心类，包含行情、财务、指数等基础功能
    - TushareNewsMixin: 新闻资讯获取
    - TushareMoneyflowMixin: 资金流向与融资融券
    - TushareLimitFactorMixin: 涨跌停与龙头因子
    
    Parameters
    ----------
    api_token : Optional[str]
        Tushare API Token
    cache_dir : str
        缓存目录，默认 "data/tushare_cache"
    
    Examples
    --------
    >>> from src.tushare import TushareDataLoader
    >>> loader = TushareDataLoader()
    >>> 
    >>> # 获取指数成分股
    >>> stocks = loader.fetch_index_constituents("hs300")
    >>> 
    >>> # 获取新闻数据
    >>> news = loader.fetch_news_multi_source()
    >>> 
    >>> # 计算主力资金因子
    >>> smart_money = loader.calculate_smart_money_score(
    ...     stocks[:50], "20240101", "20240115"
    ... )
    >>> 
    >>> # 获取龙头候选股
    >>> dragons = loader.get_dragon_candidates("20240115")
    """
    pass


# 导出符号
__all__ = [
    "TushareDataLoader",
    "TushareDataLoaderBase",
    "TushareNewsMixin",
    "TushareMoneyflowMixin",
    "TushareLimitFactorMixin",
    "create_tushare_loader",
]

