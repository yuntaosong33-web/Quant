"""
数据获取与ETL模块

该模块提供数据获取、清洗和转换的核心功能。
使用抽象基类定义统一的数据处理接口，确保扩展性。

主要组件:
    - DataHandler: 数据处理抽象基类
    - AShareDataCleaner: A股市场数据清洗器 (已拆分)
    - DataLoader: A股数据加载器（Tushare Pro）
    - DataPipeline: 数据处理管道
    - DownloadResult: 下载结果数据类
    - DataCleaner: 通用数据清洗器

Usage
-----
    from src.data_loader import DataLoader, AShareDataCleaner
    
    loader = DataLoader(mode="local_first")
    cleaner = AShareDataCleaner()
    
    # 获取数据
    df = loader.fetch_daily_price("000001", "2023-01-01", "2024-01-01")
    
    # 清洗数据
    cleaned = cleaner.clean_market_data(df)

Notes
-----
- AShareDataCleaner 已拆分到独立模块
- 其他大型类从原始 data_loader.py 导入（向后兼容）
"""

# 已拆分的模块
from .base import DataHandler
from .cleaner import AShareDataCleaner

# 从原始 data_loader.py 导入大型类（向后兼容）
try:
    from ..data_loader import (
        DataPipeline,
        DownloadResult,
        DataLoader,
        DataCleaner,
    )
except ImportError:
    try:
        from data_loader import (
            DataPipeline,
            DownloadResult,
            DataLoader,
            DataCleaner,
        )
    except ImportError:
        DataPipeline = None
        DownloadResult = None
        DataLoader = None
        DataCleaner = None

__all__ = [
    'DataHandler',
    'DataPipeline',
    'AShareDataCleaner',
    'DownloadResult',
    'DataLoader',
    'DataCleaner',
]

