"""
数据获取与ETL模块

该模块提供数据获取、清洗和转换的核心功能。
使用抽象基类定义统一的数据处理接口，确保扩展性。

主要组件:
    - DataHandler: 数据处理抽象基类
    - AShareDataCleaner: A股市场数据清洗器

Usage
-----
    from src.data_loader import DataHandler, AShareDataCleaner
    
    cleaner = AShareDataCleaner()
    cleaned = cleaner.clean_market_data(df)
"""

# 已拆分的模块
from .base import DataHandler
from .cleaner import AShareDataCleaner

__all__ = [
    'DataHandler',
    'AShareDataCleaner',
]
